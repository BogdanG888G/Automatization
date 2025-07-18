import os
import re
import logging
import time
from contextlib import closing
from io import StringIO
from typing import Optional, Tuple, Dict, List

import pyodbc  # noqa: F401  # required for SQL Server ODBC connections via raw_connection()
import pandas as pd
import numpy as np
from sqlalchemy import text, event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class PerekrestokTableProcessor:
    """Raw‑layer loader for Perekrestok retail files.

    Adapted from OkeyTableProcessor. Supports at least two known layouts:
    1) Period summary (Период / Сеть / Категория / ... / Продажи, руб / Себест., руб)
    2) Regional store‑level detail (Регион / Город / Адрес / РЦ / Сеть / Категория / ... / ТТ)

    We read up to MAX_ROWS rows per sheet, normalise RU headers -> canonical raw names,
    infer (sale_year, sale_month) from filename tokens like `perekrestok_ceptember_2024`
    or from period columns like `сен.24`, then create/append to raw.<table_name>
    (all NVARCHAR(255)).

    Downstream stage code is expected to parse numerics.
    """

    CHUNKSIZE = 100_000
    BATCH_SIZE = 10_000
    MAX_ROWS = None  # safety cap

    # Month map: full, short, dotted, digit mix, latin translit, common typos
    _MONTH_MAP: Dict[str, int] = {
        # full rus
        'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6,
        'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12,
        # short rus (various endings)
        'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'июн': 6, 'июл': 7, 'авг': 8,
        'сен': 9, 'сент': 9, 'окт': 10, 'ноя': 11, 'дек': 12,
        # dotted short tokens often seen in period column (сен.24)
        'янв.': 1, 'фев.': 2, 'мар.': 3, 'апр.': 4, 'май.': 5, 'июн.': 6,
        'июл.': 7, 'авг.': 8, 'сен.': 9, 'окт.': 10, 'ноя.': 11, 'дек.': 12,
        # latin translit & typos
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'sept': 9, 'september': 9, 'ceptember': 9,
        'oct': 10, 'nov': 11, 'dec': 12,
    }

    # ------------------------------------------------------------------ #
    # Period / metadata extraction                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def extract_perek_metadata(cls, source: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract (month, year) from filename or period string.

        Looks for alpha tokens mapped in _MONTH_MAP and 4‑digit year tokens.
        Accepts period formats like:
            'сен.24', 'Сен.24', 'сен24', 'сен-24', 'sept2024', 'perekrestok_december_2024'
        If 2‑digit year found in a period cell we expand heuristically: 00‑79 => 2000‑2079, 80‑99 => 1980‑1999.
        """
        src = str(source).lower().strip()

        # quick try: period tokens like 'сен.24'
        m = re.match(r'([a-zа-я\.]+)[\s\-_]?(?:\'?)(\d{2,4})$', src)
        if m:
            m_tok, y_tok = m.groups()
            m_tok = m_tok.strip()
            if m_tok in cls._MONTH_MAP:
                month = cls._MONTH_MAP[m_tok]
                year = cls._coerce_year(y_tok)
                return month, year

        tokens = re.findall(r"[a-zа-я]+|\d{2,4}", src)
        year = None
        month = None
        for t in tokens:
            if t in cls._MONTH_MAP and month is None:
                month = cls._MONTH_MAP[t]
            elif t.isdigit():
                if len(t) == 4 and year is None:
                    year = int(t)
                elif len(t) == 2 and year is None:
                    year = cls._coerce_year(t)
        return month, year

    @staticmethod
    def _coerce_year(y_tok: str) -> Optional[int]:
        try:
            y = int(y_tok)
        except Exception:  # noqa: BLE001
            return None
        if y < 100:  # 2‑digit year heuristic
            return 2000 + y if y < 80 else 1900 + y
        return y

    # ------------------------------------------------------------------ #
    # Header normalisation & RU→EN mapping                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_headers(cols: List[str]) -> List[str]:
        """Lowercase, strip, collapse whitespace/punct to underscore, resolve duplicates."""
        normed = []
        for c in cols:
            c = str(c).strip().lower()
            c = re.sub(r"[\\s\\./,;]+", "_", c)
            c = re.sub(r"_+", "_", c)
            c = c.strip("_")
            normed.append(c)
        # resolve duplicates
        seen: Dict[str, int] = {}
        out = []
        for c in normed:
            if c in seen:
                seen[c] += 1
                out.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                out.append(c)
        return out

    @classmethod
    def normalize_perek_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Map Perekrestok headers to canonical raw names."""
        df = df.copy()
        df.columns = cls._normalise_headers(df.columns.tolist())

        # mapping after normalisation
        column_mapping = {
            # period / chain
            'период': 'period',
            'сеть': 'retail_chain',
            'перекрёсток': 'retail_chain',   # guard rare header row containing value instead?
            'перекресток04': 'retail_chain', # sometimes header broken; fallback

            # regional
            'регион': 'region',
            'город': 'city',
            'адрес': 'address',
            'рц': 'rc_code',
            'тт': 'tt_code',  # store / trading point code

            # product hierarchy
            'категория': 'category',
            'категория_2': 'category_2',
            'тип_основы': 'base_type',

            # commercial
            'поставщик': 'supplier',
            'бренд': 'brand',

            # product names
            'наименование': 'product_name',
            'уни_наименование': 'unified_product_name',
            'уни_наименование_наименование': 'unified_product_name',  # guard weird merges

            # pack / taste
            'граммовка': 'weight_g',
            'вкусы': 'flavor',
            'вкус': 'flavor',

            # metrics
            'продажи_шт': 'sales_units',
            'продажи_шт_': 'sales_units',
            'продажи_руб': 'sales_rub',
            'продажи_тонн': 'sales_tonnes',
            'себест_руб': 'cost_rub',
            'себест__руб': 'cost_rub',
            'себест_руб_': 'cost_rub',
            'себест_руб__': 'cost_rub',

            # alt separators from sample (commas, spaces converted in _normalise_headers)
            'продажи_шт_': 'sales_units',
            'продажи_руб_': 'sales_rub',
            'продажи_тонн_': 'sales_tonnes',
            'себест_руб': 'cost_rub',
        }

        df.columns = [column_mapping.get(col, col) for col in df.columns]
        return df

    # ------------------------------------------------------------------ #
    # Readers                                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def _safe_read_perek_csv(cls, file_path: str) -> pd.DataFrame:
        """CSV reader (try ; then tab). Reads only first MAX_ROWS rows."""
        try:
            df = pd.read_csv(
                file_path,
                dtype='string',
                sep=';',
                quotechar='"',
                on_bad_lines='warn',
                encoding='utf-8-sig',
                decimal=',',
                thousands=' ',
                nrows=cls.MAX_ROWS,
            )
            if df.shape[1] == 1:  # wrong sep?
                df = pd.read_csv(
                    file_path,
                    dtype='string',
                    sep='\t',
                    quotechar='"',
                    on_bad_lines='warn',
                    encoding='utf-8-sig',
                    decimal=',',
                    thousands=' ',
                    nrows=cls.MAX_ROWS,
                )
            return df
        except pd.errors.ParserError:
            logger.warning("[Perekrestok] CSV parse error, attempting sanitised re‑read…")
            with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                lines = [f.readline() for _ in range(cls.MAX_ROWS + 1)]
            content = ''.join(lines)
            clean_content = re.sub(r'(?<!\\)\"(?!\\)', '', content)
            return pd.read_csv(
                StringIO(clean_content),
                dtype='string',
                sep=';',
                quotechar='"',
                decimal=',',
                thousands=' ',
            )

    @classmethod
    def _read_xlsb_file(cls, file_path: str) -> Dict[str, pd.DataFrame]:
        """Read XLSB (subset MAX_ROWS)."""
        try:
            import pyxlsb  # local import
            out: Dict[str, pd.DataFrame] = {}
            with pyxlsb.open_workbook(file_path) as wb:
                for sheet_name in wb.sheets:
                    with wb.get_sheet(sheet_name) as sheet:
                        data = []
                        headers = None
                        for i, row in enumerate(sheet.rows()):
                            if i > cls.MAX_ROWS:
                                break
                            if i == 0:
                                headers = [str(c.v) if c.v is not None else f"none_{idx}" for idx, c in enumerate(row)]
                            else:
                                data.append([str(c.v) if c.v is not None else '' for c in row])
                        if headers and data:
                            out[sheet_name] = pd.DataFrame(data, columns=headers)
            if not out:
                raise ValueError("XLSB file contains no data")
            return out
        except Exception as e:  # noqa: BLE001
            logger.error("[Perekrestok] XLSB read error: %s", e)
            raise

    @classmethod
    def _read_excel_file(cls, file_path: str) -> Dict[str, pd.DataFrame]:
        """Read XLSX (subset MAX_ROWS)."""
        try:
            return pd.read_excel(
                file_path,
                sheet_name=None,
                dtype='string',
                engine='openpyxl',
                decimal=',',
                thousands=' ',
                nrows=cls.MAX_ROWS,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("[Perekrestok] Excel read error: %s", e)
            raise

    # ------------------------------------------------------------------ #
    # Main processing                                                     #
    # ------------------------------------------------------------------ #
    @classmethod
    def process_perek_file(cls, file_path: str, engine: Engine) -> str:
        """Read Perekrestok file -> normalise -> load to raw schema.

        Returns created table name in *raw*.
        """
        start_time = time.time()
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        table_name = re.sub(r"\\W+", "_", file_name.lower())  # keep extension‑safe base or full? use full for uniqueness
        table_name = re.sub(r"\\W+", "_", base_name.lower())

        logger.info("[Perekrestok] Начало обработки файла: %s (<=%s строк)", file_name, cls.MAX_ROWS)

        # --- read
        if file_path.endswith('.xlsx'):
            reader = cls._read_excel_file(file_path)
        elif file_path.endswith('.xlsb'):
            reader = cls._read_xlsb_file(file_path)
        elif file_path.endswith('.csv'):
            reader = {'data': cls._safe_read_perek_csv(file_path)}
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # infer metadata from filename once
        fname_month, fname_year = cls.extract_perek_metadata(file_name)

        processed_chunks: List[pd.DataFrame] = []
        for sheet_name, df in reader.items():
            if df.empty:
                logger.warning("[Perekrestok] Лист %s пуст", sheet_name)
                continue
            logger.info("[Perekrestok] Обработка листа %s, строк: %s", sheet_name, len(df))

            try:
                df = cls.normalize_perek_columns(df)

                # fallback: if no file metadata, try period column
                month = fname_month
                year = fname_year
                if (month is None or year is None) and 'period' in df.columns and not df['period'].isna().all():
                    m2, y2 = cls.extract_perek_metadata(str(df['period'].iloc[0]))
                    month = month or m2
                    year = year or y2

                if month and year:
                    df = df.assign(sale_year=str(year), sale_month=str(month).zfill(2))

                # clean values (all strings in raw)
                df = df.replace([np.nan, None], '')
                for col in df.columns:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.strip()
                        .str.replace('\\u200b', '', regex=False)
                        .str.replace('\\r\\n', ' ', regex=False)
                        .str.replace('\\n', ' ', regex=False)
                    )

                processed_chunks.append(df)
            except Exception as e:  # noqa: BLE001
                logger.error("[Perek] Ошибка обработки листа %s: %s", sheet_name, e, exc_info=True)
                continue

        if not processed_chunks:
            raise ValueError("File contains no valid data")

        final_df = pd.concat(processed_chunks, ignore_index=True)
        if final_df.empty:
            raise ValueError("No data after processing")

            # --- ensure raw table exists (all NVARCHAR(255))
        with engine.begin() as conn:
            exists = conn.execute(
                text("""
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'raw' AND table_name = :table
                """),
                {"table": table_name},
            ).scalar()

            if not exists:
                cols_sql = []
                for col in final_df.columns:
                    clean_col = col.replace('"', '').replace("'", '')
                    cols_sql.append(f"[{clean_col}] NVARCHAR(255)")
                create_table_sql = f"CREATE TABLE raw.{table_name} ({', '.join(cols_sql)})"
                logger.debug("[Perekrestok] SQL создания таблицы: %s", create_table_sql)
                conn.execute(text(create_table_sql))

        # --- insert
        cls.bulk_insert_perek(final_df, table_name, engine)

        duration = time.time() - start_time
        logger.info(
            "[Perekrestok] Файл %s успешно загружен в raw.%s за %.2f сек (%s строк)",
            file_name, table_name, duration, len(final_df)
        )
        return table_name


    # ------------------------------------------------------------------ #
    # Bulk insert                                                        #
    # ------------------------------------------------------------------ #
    @classmethod
    def bulk_insert_perek(cls, df: pd.DataFrame, table_name: str, engine: Engine):
        """Batch insert into raw.<table_name> (all values as strings)."""

        @event.listens_for(engine, 'before_cursor_execute')
        def _set_fast_executemany(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
            if executemany:
                cursor.fast_executemany = True

        # prepare data rows
        data: List[tuple] = []
        for row in df.itertuples(index=False, name=None):
            clean_row = []
            for value in row:
                if pd.isna(value) or value in ('', None):
                    clean_row.append('')
                else:
                    s = str(value).strip()
                    if len(s) > 255:
                        s = s[:255]
                    clean_row.append(s)
            data.append(tuple(clean_row))

        cols = ', '.join(f'[{col}]' for col in df.columns)
        params = ', '.join(['?'] * len(df.columns))
        insert_sql = f"INSERT INTO raw.{table_name} ({cols}) VALUES ({params})"

        with closing(engine.raw_connection()) as conn:
            with conn.cursor() as cursor:
                logger.debug("[Perekrestok] Пример данных для вставки: %s", data[0] if data else None)
                try:
                    cursor.executemany(insert_sql, data)
                    conn.commit()
                    logger.debug("[Perek] Успешно вставлено %s записей", len(data))
                except Exception as e:  # noqa: BLE001
                    conn.rollback()
                    logger.error(
                        "[Perekrestok] Ошибка вставки. Первая строка: %s. Ошибка: %s", data[0] if data else None, e
                    )
                    raise


def create_perek_table_and_upload(file_path: str, engine: Engine) -> str:
    """Convenience wrapper for DAG usage."""
    try:
        return PerekrestokTableProcessor.process_perek_file(file_path, engine)
    except Exception as e:  # noqa: BLE001
        logger.error("Критическая ошибка при обработке файла %s: %s", file_path, e, exc_info=True)
        raise
