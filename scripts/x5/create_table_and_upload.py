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
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator

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
    "материал2": "product_name",
    "товар": "product_name",
    "наименование": "product_name",
    "товар": "product_name",
    "наименование_товара": "product_name",
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
    MAX_ROWS = 10_000_000
    BATCH_SIZE = 100_000
    CHUNKSIZE = 100_000
    enrichment_models_loaded = False

    @classmethod
    def _load_enrichment_models(cls):
        def load_model_and_vectorizer(model_name: str, folder: str):
            with open(f"{folder}/{model_name}_model.pkl", "rb") as f_model:
                model = pickle.load(f_model)
            with open(f"{folder}/{model_name}_vectorizer.pkl", "rb") as f_vec:
                vectorizer = pickle.load(f_vec)
            return model, vectorizer

        product_dir = "ml_models/product_enrichment"
        address_dir = "ml_models/address_enrichment"

        # Модели по product_name
        cls.brand_model, cls.brand_vectorizer = load_model_and_vectorizer("brand", product_dir)
        cls.flavor_model, cls.flavor_vectorizer = load_model_and_vectorizer("flavor", product_dir)
        cls.weight_model, cls.weight_vectorizer = load_model_and_vectorizer("weight", product_dir)
        cls.type_model, cls.type_vectorizer = load_model_and_vectorizer("type", product_dir)

        # Модели по адресу
        cls.city_model, cls.city_vectorizer = load_model_and_vectorizer("city", address_dir)
        cls.region_model, cls.region_vectorizer = load_model_and_vectorizer("region", address_dir)
        cls.branch_model, cls.branch_vectorizer = load_model_and_vectorizer("branch", address_dir)

        cls.enrichment_models_loaded = True


    @classmethod
    def _enrich_product_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        if not cls.enrichment_models_loaded:
            cls._load_enrichment_models()

        # Регулярка для извлечения веса из product_name
        weight_pattern = re.compile(
            r'(\d+(?:[.,]\d+)?)[ ]?(г(?:р|рамм)?|кг|кг\.|гр|грамм)',
            flags=re.IGNORECASE
        )

        def extract_weight(text):
            match = weight_pattern.search(text)
            if match:
                value = match.group(1).replace(',', '.')
                unit = match.group(2).lower()
                try:
                    value_float = float(value)
                    if 'кг' in unit:
                        return value_float * 1000  # граммы
                    else:
                        return value_float
                except Exception as e:
                    logger.warning(f"Ошибка преобразования веса: {e}")
                    return None
            return None

        # Обогащение по product_name
        if 'product_name' in df.columns:
            product_names = df['product_name'].fillna("")

            def predict_product_attr(model, vectorizer):
                try:
                    X_vec = vectorizer.transform(product_names)
                    return model.predict(X_vec)
                except Exception as e:
                    logger.warning(f"Ошибка предсказания по продукту: {e}")
                    return [""] * len(product_names)

            df['brand_predicted'] = predict_product_attr(cls.brand_model, cls.brand_vectorizer)
            df['flavor_predicted'] = predict_product_attr(cls.flavor_model, cls.flavor_vectorizer)
            df['weight_predicted'] = predict_product_attr(cls.weight_model, cls.weight_vectorizer)
            df['type_predicted'] = predict_product_attr(cls.type_model, cls.type_vectorizer)

            # Добавляем извлечённый из текста вес
            df['weight_extracted'] = product_names.apply(extract_weight)

        # Обогащение по адресу — ищем колонку, где в названии есть "add" (например, "address", "Адрес")
        address_col_candidates = [c for c in df.columns if 'add' in c.lower()]
        if address_col_candidates:
            address_col = address_col_candidates[0]
            addresses = df[address_col].fillna("")

            try:
                X_city = cls.city_vectorizer.transform(addresses)
                df['city_predicted'] = cls.city_model.predict(X_city)
            except Exception as e:
                logger.warning(f"Ошибка предсказания города: {e}")
                df['city_predicted'] = [""] * len(addresses)

            city_inputs = df['city_predicted'].fillna("").astype(str)

            try:
                X_region = cls.region_vectorizer.transform(city_inputs)
                df['region_predicted'] = cls.region_model.predict(X_region)
            except Exception as e:
                logger.warning(f"Ошибка предсказания региона: {e}")
                df['region_predicted'] = [""] * len(city_inputs)

            try:
                X_branch = cls.branch_vectorizer.transform(city_inputs)
                df['branch_predicted'] = cls.branch_model.predict(X_branch)
            except Exception as e:
                logger.warning(f"Ошибка предсказания филиала: {e}")
                df['branch_predicted'] = [""] * len(city_inputs)

        return df

    @classmethod
    def process_data_chunk(cls, df: pd.DataFrame, file_name: str, is_first_chunk: bool) -> pd.DataFrame:
        """Обработка чанка данных"""
        # Нормализация данных
        if is_first_chunk:
            df = _collapse_to_header_and_data(df)
        
        df = _rename_using_mapping(df)
        df.columns = _make_unique_columns(df.columns.tolist())
        
        # Добавление временных меток
        m, y = _extract_month_year_from_filename(file_name)
        if m: df["sale_month"] = str(m).zfill(2)
        if y: df["sale_year"] = str(y)
        
        # Очистка данных
        df = _drop_all_nulls(df)
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.slice(0, 255)
            
        df = cls._enrich_product_data(df)
        return df.replace("nan", "")


    @staticmethod
    def read_csv_safely_chunked(file_path: str, chunksize: int) -> pd.DataFrame:
        # Определение разделителя
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(50000)
        
        counts = {sep: sample.count(sep) for sep in (";", ",", "\t")}
        sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ";"
        
        # Чтение по чанкам
        return pd.read_csv(
            file_path,
            sep=sep,
            dtype="string",
            chunksize=chunksize,
            encoding="utf-8-sig",
            on_bad_lines="warn",
            decimal=",",
            thousands=" ",
            engine="c"
        )

    @classmethod
    def create_table_if_not_exists(cls, df: pd.DataFrame, table_name: str, engine: Engine):
        try:
            with engine.begin() as conn:
                if not conn.execute(text(f"""
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema='raw' AND table_name='{table_name}'
                """)).scalar():
                    columns_sql = ", ".join(
                        f"[{col.replace(']', '').replace('[', '')}] NVARCHAR(255)"
                        for col in df.columns
                    )
                    conn.execute(text(f"CREATE TABLE raw.{table_name} ({columns_sql})"))
                    logger.info(f"Создана таблица raw.{table_name}")
        except Exception as e:
            logger.error(f"Ошибка при создании таблицы: {e}")
            raise

    @classmethod
    def process_x5_file(cls, file_path: str, engine: Engine) -> str:
        start = time.time()
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        table_name = re.sub(r"\W+", "_", base_name.lower())

        logger.info(f"Начинаем обработку X5 файла {file_name}, максимум {cls.MAX_ROWS} строк.")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            sheets = read_excel_safely(file_path, nrows=cls.MAX_ROWS)
            processed_frames: List[pd.DataFrame] = []

            for sh_name, df in sheets.items():
                if df is None or df.empty:
                    logger.warning(f"[X5] Лист {sh_name} пуст — пропуск.")
                    continue

                logger.info(f"[X5] Обрабатываем лист {sh_name}, строк: {len(df)}.")
                df = _collapse_to_header_and_data(df)
                df = _drop_all_nulls(df)
                df = _rename_using_mapping(df)

                before_cols = df.columns.tolist()
                uniq_cols = _make_unique_columns(before_cols)
                if uniq_cols != before_cols:
                    logger.info(f"[X5] Переименованы дубли колонок: {list(zip(before_cols, uniq_cols))}")
                df.columns = uniq_cols

                m, y = _extract_month_year_from_filename(file_name)
                if m: df["sale_month"] = str(m).zfill(2)
                if y: df["sale_year"] = str(y)

                df = df.replace([np.nan, None], '')
                for col in df.columns:
                    try:
                        df[col] = df[col].astype(str).str.strip()
                    except Exception as e:
                        logger.warning(f"[X5] Не удалось привести колонку {col} к строке: {e}")

                if "retail_chain" in df.columns:
                    df = df[df["retail_chain"].fillna("").str.lower() != "сеть"]

                df = cls._enrich_product_data(df)
                processed_frames.append(df)

            if not processed_frames:
                raise ValueError("[X5] Ни один лист не дал данных после обработки.")

            final_df = pd.concat(processed_frames, ignore_index=True)
            if final_df.empty:
                raise ValueError("[X5] Итоговый DataFrame пуст.")

            with engine.begin() as conn:
                exists = conn.execute(
                    text("""
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema='raw' AND table_name=:tbl
                    """),
                    {"tbl": table_name},
                ).scalar()
                if not exists:
                    cols_sql = [f"[{col.replace('[','').replace(']','')}] NVARCHAR(255)" for col in final_df.columns]
                    create_sql = f"CREATE TABLE raw.{table_name} ({', '.join(cols_sql)})"
                    logger.debug(f"[X5] CREATE raw SQL: {create_sql}")
                    conn.execute(text(create_sql))

            cls.bulk_insert_x5(final_df, table_name, engine)
            dur = time.time() - start
            logger.info(f"Файл {file_name} загружен в raw.{table_name} за {dur:.2f} сек ({len(final_df)} строк).")
            return table_name

        elif ext == ".csv":
            chunk_iter = cls.read_csv_safely_chunked(file_path, cls.CHUNKSIZE)
            first_chunk = True
            created_table = False
            total_rows = 0

            for df_chunk in chunk_iter:
                if df_chunk.empty:
                    continue

                processed_chunk = cls.process_data_chunk(df_chunk, file_name, first_chunk)
                processed_chunk = cls._enrich_product_data(processed_chunk)

                if processed_chunk.empty:
                    logger.info("Чанк пуст, пропускаем.")
                    continue

                if first_chunk and not created_table:
                    cls.create_table_if_not_exists(processed_chunk, table_name, engine)
                    created_table = True
                    first_chunk = False

                cls.bulk_insert_x5(processed_chunk, table_name, engine)
                total_rows += len(processed_chunk)
                logger.info(f"Вставлено {len(processed_chunk)} строк (всего: {total_rows})")

            logger.info(f"Всего загружено {total_rows} строк в raw.{table_name}")
            return table_name

        else:
            raise ValueError(f"Неподдерживаемый формат: {file_path}")


    @classmethod
    def bulk_insert_x5(cls, df: pd.DataFrame, table_name: str, engine: Engine):
        if df.empty:
            logger.info("DataFrame пуст, вставка не требуется.")
            return

        # 1. Узнаём текущие колонки таблицы
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema='raw' AND table_name=:tbl
                """), {"tbl": table_name})
                table_columns = [row[0] for row in result]
        except Exception as e:
            logger.error(f"Ошибка при получении колонок: {e}")
            return

        # 2. Добавляем недостающие колонки
        new_columns = [col for col in df.columns if col not in table_columns]
        if new_columns:
            try:
                with engine.begin() as conn:
                    for col in new_columns:
                        clean_col = col.replace("]", "").replace("[", "")
                        conn.execute(text(f"ALTER TABLE raw.{table_name} ADD [{clean_col}] NVARCHAR(255)"))
                table_columns += new_columns
            except Exception as e:
                logger.error(f"Ошибка добавления колонок: {e}")
                return

        # 3. Перестраиваем df по колонкам таблицы
        df = df.reindex(columns=table_columns, fill_value="")

        if df.shape[1] == 0:
            logger.error("Нет колонок для вставки.")
            return

        # 4. Готовим данные (всё в Python str или None)
        records = []
        for row in df.itertuples(index=False, name=None):
            rec = []
            for v in row:
                if pd.isna(v) or v == "nan":
                    rec.append(None)
                else:
                    rec.append(str(v)[:255])
            records.append(tuple(rec))

        if not records:
            logger.warning("Нет данных для вставки после фильтрации.")
            return

        batch_size = 10_000
        num_batches = (len(records) + batch_size - 1) // batch_size

        columns_sql = ", ".join(f"[{c}]" for c in table_columns)
        placeholders = ", ".join(["?"] * len(table_columns))
        insert_sql = f"INSERT INTO raw.{table_name} ({columns_sql}) VALUES ({placeholders})"

        try:
            raw_conn = engine.raw_connection()
            try:
                cursor = raw_conn.cursor()
                cursor.fast_executemany = True

                for i in range(num_batches):
                    start = i * batch_size
                    end = min((i + 1) * batch_size, len(records))
                    batch = records[start:end]
                    logger.info(f"Вставка батча {i+1}/{num_batches} ({len(batch)} строк)")
                    cursor.executemany(insert_sql, batch)

                raw_conn.commit()
                logger.info(f"Успешно вставлено {len(records)} строк.")
            finally:
                raw_conn.close()
        except Exception as e:
            logger.error(f"Ошибка вставки (raw pyodbc): {e}", exc_info=True)
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
