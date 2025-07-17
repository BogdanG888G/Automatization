# create_table_and_upload.py  --- X5 raw loader
# -*- coding: utf-8 -*-
import os
import re
import logging
import time
from contextlib import closing
from io import StringIO
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import text, event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Month dictionary (рус / англ / сокращения)                        #
# ------------------------------------------------------------------ #
X5_MONTHS: Dict[str, int] = {
    # рус полный
    "январь": 1, "февраль": 2, "март": 3, "апрель": 4, "май": 5, "июнь": 6,
    "июль": 7, "август": 8, "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12,
    # рус короткий
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "май_": 5, "май": 5, "июн": 6,
    "июл": 7, "авг": 8, "сен": 9, "сент": 9, "окт": 10, "ноя": 11, "дек": 12,
    # англ
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "december": 12,
}


# ------------------------------------------------------------------ #
# Column canonical mapping (raw рус → нормализованное тех имя)       #
# ------------------------------------------------------------------ #
COLUMN_MAPPING: Dict[str, str] = {
    # орг
    "сеть": "retail_chain",
    "филиал": "branch",
    "регион": "region",
    "город": "city",
    "адрес": "address",
    "завод": "factory",
    "завод_1": "factory",
    "завод_2": "factory2",
    # продуктовая иерархия (варианты)
    "тов_иер_ур_2": "prod_level_2",
    "тов_иер_ур_2_на": "prod_level_2",
    "товиерур2": "prod_level_2",
    "товиерур2_на": "prod_level_2",
    "тов_иер_ур_3": "prod_level_3",
    "тов_иер_ур_3_на": "prod_level_3",
    "товиерур3": "prod_level_3",
    "товиерур3_на": "prod_level_3",
    "тов_иер_ур_4": "prod_level_4",
    "тов_иер_ур_4_на": "prod_level_4",
    "товиерур4": "prod_level_4",
    "товиерур4_на": "prod_level_4",
    # материал / дубликаты
    "материал": "material",
    "материал_1": "material",
    "материал_2": "material2",
    # прочие атрибуты
    "бренд": "brand",
    "вендор": "vendor",
    "основной_поставщик": "main_supplier",
    "поставщик_склада": "warehouse_supplier",
    "поставщик_склада_рц": "warehouse_supplier",
    "поставщик_склада_рц_": "warehouse_supplier",
    # метрики
    "количество": "quantity",
    "количество_без_ед_изм": "quantity",
    "оборот_с_ндс": "gross_turnover",
    "оборот_с_ндс_без_ед_изм": "gross_turnover",
    "общая_себестоимость": "gross_cost",
    "общая_себестоимость_с_ндс_без_ед_изм": "gross_cost",
    "средняя_цена_по_себестоимости": "avg_cost_price",
    "средняя_цена_по_себестоимости_с_ндс": "avg_cost_price",
    "средняя_цена_продажи": "avg_sell_price",
    "средняя_цена_продачи_с_ндс": "avg_sell_price",  # опечаточный вариант
    "средняя_цена_продажи_с_ндс": "avg_sell_price",
}


NUMERIC_TARGETS: List[str] = [
    "quantity", "gross_turnover", "gross_cost", "avg_cost_price", "avg_sell_price",
]


# ------------------------------------------------------------------ #
# Utilities                                                          #
# ------------------------------------------------------------------ #
_pre_ws = re.compile(r"\s+", re.MULTILINE)
_non_word = re.compile(r"[^\w]+", re.UNICODE)


def _clean_header_cell(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip()
    s = _pre_ws.sub(" ", s)
    s = s.replace('"', '').replace("'", "")
    s = re.sub(r"\(.*?\)", "", s)
    s = s.strip()
    s = s.replace(".", "_")
    s = s.replace(" ", "_")
    s = _non_word.sub("_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()


def _is_header_row(values: List[str]) -> bool:
    keys = {"сеть", "филиал", "регион", "город", "адрес", "бренд", "вендор", "количество"}
    norm = {str(v).strip().lower() for v in values if str(v).strip()}
    return len(keys.intersection(norm)) >= 2


def _collapse_to_header_and_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    bad_header = any(str(c).lower().startswith("unnamed") for c in df.columns)

    if bad_header:
        for i in range(min(10, len(df))):
            row_vals = df.iloc[i].tolist()
            if _is_header_row(row_vals):
                new_cols = [_clean_header_cell(x) for x in row_vals]
                df = df.iloc[i + 1:].reset_index(drop=True).copy()
                df.columns = new_cols
                return df
        new_cols = [_clean_header_cell(x) for x in df.iloc[0].tolist()]
        df = df.iloc[1:].reset_index(drop=True).copy()
        df.columns = new_cols
        return df

    first_row = df.iloc[0].tolist()
    if _is_header_row(first_row):
        new_cols = [_clean_header_cell(x) for x in first_row]
        df = df.iloc[1:].reset_index(drop=True).copy()
        df.columns = new_cols
        return df

    df.columns = [_clean_header_cell(c) for c in df.columns]
    return df


def _rename_using_mapping(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_map = { _clean_header_cell(k): v for k, v in COLUMN_MAPPING.items() }
    new_cols = []
    for c in df.columns:
        ck = _clean_header_cell(c)
        new_cols.append(cleaned_map.get(ck, ck))
    out = df.copy()
    out.columns = new_cols
    return out


def _make_unique_columns(cols: List[str]) -> List[str]:
    seen = {}
    out = []
    for c in cols:
        base = c or "col"
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def _drop_all_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how="all").dropna(axis=1, how="all")


def _extract_month_year_from_filename(file_name: str):
    """
    Пытаемся достать месяц/год из имени файла.
    Работает и для русских, и для англ. форм (полных или коротких).
    """
    fn = str(file_name).lower()

    # токены: либо год, либо буквенный блок
    toks = re.findall(r"[a-zа-я]+|\d{4}", fn)

    year = None
    month = None

    for t in toks:
        if t.isdigit() and len(t) == 4:
            year = int(t)
            continue

        # сначала пробуем полное совпадение
        if t in X5_MONTHS:
            month = X5_MONTHS[t]
            continue

        # иначе берём первые 3 буквы и матчим
        t3 = t[:3]
        if t3 in X5_MONTHS:
            month = X5_MONTHS[t3]
            continue

    return month, year


# ------------------------------------------------------------------ #
# File readers                                                       #
# ------------------------------------------------------------------ #
def read_csv_safely(file_path: str, nrows: Optional[int], sep_default: str = ";") -> pd.DataFrame:
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            sample_lines = [f.readline() for _ in range(5)]
        sample = "".join(sample_lines)
    except Exception:
        sample = ""

    counts = {ch: sample.count(ch) for ch in (";", "\t", ",")}
    sep = max(counts, key=counts.get)
    if counts.get(sep, 0) == 0:
        sep = sep_default

    try:
        return pd.read_csv(
            file_path,
            dtype="string",
            sep=sep,
            quotechar='"',
            on_bad_lines="warn",
            encoding="utf-8-sig",
            decimal=",",
            thousands=" ",
            nrows=nrows,
            header=0,
        )
    except Exception as e:
        logger.warning(f"[X5] Основное чтение CSV не удалось ({e}), fallback plain read.")
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            text_block = "".join(f.readlines(nrows if nrows else None))
        text_block = re.sub(r'(?<!\\)"(?!\\)', '', text_block)
        return pd.read_csv(
            StringIO(text_block),
            dtype="string",
            sep=sep,
            quotechar='"',
            decimal=",",
            thousands=" ",
            header=0,
        )


def read_excel_safely(file_path: str, nrows: Optional[int]) -> Dict[str, pd.DataFrame]:
    return pd.read_excel(
        file_path,
        sheet_name=None,
        dtype="string",
        engine="openpyxl",
        decimal=",",
        thousands=" ",
        nrows=nrows,
        header=0,
    )


# ------------------------------------------------------------------ #
# Main processor                                                     #
# ------------------------------------------------------------------ #
class X5TableProcessor:
    MAX_ROWS = 10_000   # ограничение чтения
    BATCH_SIZE = 10_000 # для bulk insert
    CHUNKSIZE = 100_000 # резерв

    @classmethod
    def process_x5_file(cls, file_path: str, engine: Engine) -> str:
        start = time.time()
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        table_name = re.sub(r"\W+", "_", base_name.lower())

        logger.info(f"Начинаем обработку X5 файла {file_name}, максимум {cls.MAX_ROWS} строк.")

        # --- Read
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            sheets = read_excel_safely(file_path, nrows=cls.MAX_ROWS)
        elif ext == ".csv":
            sheets = {"data": read_csv_safely(file_path, nrows=cls.MAX_ROWS)}
        else:
            raise ValueError(f"Неподдерживаемый формат: {file_path}")

        processed_frames: List[pd.DataFrame] = []

        for sh_name, df in sheets.items():
            if df is None or df.empty:
                logger.warning(f"[X5] Лист {sh_name} пуст — пропуск.")
                continue

            logger.info(f"[X5] Обрабатываем лист {sh_name}, строк: {len(df)}.")

            # 1) корректные заголовки (пропуск pre-header)
            df = _collapse_to_header_and_data(df)

            # 2) удалить пустые ряды/колонки
            df = _drop_all_nulls(df)

            # 3) нормализация + маппинг
            df = _rename_using_mapping(df)

            # 4) уникализация
            before_cols = df.columns.tolist()
            uniq_cols = _make_unique_columns(before_cols)
            if uniq_cols != before_cols:
                logger.info(f"[X5] Переименованы дубли колонок: {list(zip(before_cols, uniq_cols))}")
            df.columns = uniq_cols

            # 5) month/year из имени файла
            m, y = _extract_month_year_from_filename(file_name)
            if m:
                df["sale_month"] = str(m).zfill(2)
            if y:
                df["sale_year"] = str(y)

            # 6) все к строкам
            df = df.replace([np.nan, None], '')
            for col in df.columns:
                try:
                    df[col] = df[col].astype(str).str.strip()
                except Exception as e:
                    logger.warning(f"[X5] Не удалось привести колонку {col} к строке: {e}")

            # 7) убрать строку-заголовок, если просочилась
            if "retail_chain" in df.columns:
                df = df[df["retail_chain"].fillna("").str.lower() != "сеть"]

            processed_frames.append(df)

        if not processed_frames:
            raise ValueError("[X5] Ни один лист не дал данных после обработки.")

        final_df = pd.concat(processed_frames, ignore_index=True)
        if final_df.empty:
            raise ValueError("[X5] Итоговый DataFrame пуст.")

        # --- create raw table if not exists
        with engine.begin() as conn:
            exists = conn.execute(
                text("""
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema='raw' AND table_name=:tbl
                """),
                {"tbl": table_name},
            ).scalar()

            if not exists:
                cols_sql = []
                for col in final_df.columns:
                    clean = col.replace("]", "").replace("[", "")
                    cols_sql.append(f"[{clean}] NVARCHAR(255)")
                create_sql = f"CREATE TABLE raw.{table_name} ({', '.join(cols_sql)})"
                logger.debug(f"[X5] CREATE raw SQL: {create_sql}")
                conn.execute(text(create_sql))

        # --- bulk insert
        cls.bulk_insert_x5(final_df, table_name, engine)

        dur = time.time() - start
        logger.info(f"Файл {file_name} загружен в raw.{table_name} за {dur:.2f} сек ({len(final_df)} строк).")
        return table_name

    @classmethod
    def bulk_insert_x5(cls, df: pd.DataFrame, table_name: str, engine: Engine):
        @event.listens_for(engine, "before_cursor_execute")
        def _set_fast(conn, cursor, statement, parameters, context, executemany):  # pragma: no cover
            if executemany:
                cursor.fast_executemany = True

        str_df = df.copy()
        for c in str_df.columns:
            str_df[c] = str_df[c].astype("string").fillna("").str.slice(0, 255)

        rows = list(str_df.itertuples(index=False, name=None))
        if not rows:
            logger.warning(f"[X5] bulk_insert_x5: нет строк для вставки в raw.{table_name}")
            return

        with closing(engine.raw_connection()) as raw_conn:
            with raw_conn.cursor() as cursor:
                cols = ", ".join(f"[{c.replace(']','').replace('[','')}]" for c in str_df.columns)
                params = ", ".join(["?"] * len(str_df.columns))
                sql = f"INSERT INTO raw.{table_name} ({cols}) VALUES ({params})"

                logger.debug(f"[X5] Пример строки для вставки: {rows[0]}")

                for i in range(0, len(rows), cls.BATCH_SIZE):
                    batch = rows[i:i + cls.BATCH_SIZE]
                    try:
                        cursor.executemany(sql, batch)
                        raw_conn.commit()
                    except Exception as e:  # pragma: no cover
                        raw_conn.rollback()
                        logger.error(f"[X5] Ошибка вставки батча {i}-{i+len(batch)}: {e}")
                        raise


# ------------------------------------------------------------------ #
# Public wrapper                                                     #
# ------------------------------------------------------------------ #
def create_x5_table_and_upload(file_path: str, engine: Engine) -> str:
    try:
        return X5TableProcessor.process_x5_file(file_path, engine)
    except Exception as e:
        logger.error(f"[X5] Критическая ошибка при обработке {file_path}: {e}", exc_info=True)
        raise
