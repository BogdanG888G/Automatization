import os
import re
import logging
import time
from contextlib import closing
from io import StringIO
from typing import Optional, Tuple, Dict, List, Union

import pyodbc  # noqa: F401  # required for SQL Server ODBC connections behind SQLAlchemy raw_connection()
import pandas as pd
import numpy as np
from sqlalchemy import text, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class OkeyTableProcessor:
    """Raw‑layer loader for O'KEY retail files.

    Behaviour mirrors *PyaterochkaTableProcessor* so downstream stage code can be re‑used
    with minimal change. We standardise RU headers -> canonical raw column names (period,
    retail_chain, ...), read only the first MAX_ROWS rows per sheet, and create / append to
    a *raw* schema table (all NVARCHAR(255)) before later typed staging.
    """

    CHUNKSIZE = 100_000
    BATCH_SIZE = 10_000
    MAX_ROWS = 10_000  # safety cap

    # month tokens recognised in filenames (both rus & simple translit typos like "ceptember")
    _MONTH_MAP = {
        # full rus
        'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6,
        'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12,
        # short rus
        'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'май_': 5, 'май': 5, 'июн': 6,
        'июл': 7, 'авг': 8, 'сен': 9, 'сент': 9, 'окт': 10, 'ноя': 11, 'дек': 12,
        # crude latin translits frequently seen in filenames
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'sept': 9, 'september': 9, 'ceptember': 9,  # <<< typo in sample filename
        'oct': 10, 'nov': 11, 'dec': 12,
    }

    @classmethod
    def extract_okey_metadata(cls, source: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract (month, year) from filename.

        Accepts tokens of letters and 4‑digit years; tries month lookup in `_MONTH_MAP`.
        Returns (None, None) if nothing found.
        """
        src = str(source).lower()
        # split into alpha tokens & 4‑digit numeric tokens
        tokens = re.findall(r"[a-zа-я]+|\d{4}", src)
        year = next((int(t) for t in tokens if t.isdigit() and len(t) == 4), None)
        month = None
        for t in tokens:
            if t in cls._MONTH_MAP:
                month = cls._MONTH_MAP[t]
                break
        return month, year

    # ------------------------------------------------------------------
    # Column normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_headers(cols: List[str]) -> List[str]:
        """Lowercase, strip, collapse whitespace/punct to underscore."""
        normed = []
        for c in cols:
            c = str(c).strip().lower()
            c = re.sub(r"[\s\./,;]+", "_", c)
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
    def normalize_okey_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Map Russian O'KEY headers to raw canonical names used downstream."""
        df = df.copy()
        df.columns = cls._normalise_headers(df.columns.tolist())

        # mapping after normalisation
        column_mapping = {
            'период': 'period',
            'сеть': 'retail_chain',
            'категория': 'category',
            'категория_2': 'base_type',  # secondary category -> base_type analogue
            'поставщик': 'supplier',
            'бренд': 'brand',
            'наименование': 'product_name',
            'уни_наименование': 'unified_product_name',
            'уни_наименование_наименование': 'unified_product_name',  # guard weird merge errors
            'граммовка': 'weight',
            'вкус': 'flavor',
            'продажи_шт': 'sales_units',
            'продажи_руб': 'sales_rub',
            'продажи_тонн': 'sales_tonnes',
            'себест_руб': 'cost_rub',
            'себест__руб': 'cost_rub',
        }

        df.columns = [column_mapping.get(col, col) for col in df.columns]
        return df

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------
    @classmethod
    def _safe_read_okey_csv(cls, file_path: str) -> pd.DataFrame:
        """CSV reader (tab or semicolon). Reads only first MAX_ROWS rows."""
        # try semicolon first, fall back to tab
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
            if df.shape[1] == 1:  # probably wrong sep
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
            logger.warning("CSV parse error, attempting sanitised re‑read…")
            with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                lines = [f.readline() for _ in range(cls.MAX_ROWS + 1)]
            content = ''.join(lines)
            clean_content = re.sub(r'(?<!\\)"(?!\\)', '', content)
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
            logger.error("XLSB read error: %s", e)
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
            logger.error("Excel read error: %s", e)
            raise

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------
    @classmethod
    def process_okey_file(cls, file_path: str, engine: Engine) -> str:
        """Read O'KEY file -> normalise -> load to raw schema.

        Returns created table name in *raw*.
        """
        start_time = time.time()
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        table_name = re.sub(r"\W+", "_", base_name.lower())

        logger.info("[O'KEY] Начало обработки файла: %s (<=%s строк)", file_name, cls.MAX_ROWS)

        # --- read
        if file_path.endswith('.xlsx'):
            reader = cls._read_excel_file(file_path)
        elif file_path.endswith('.xlsb'):
            reader = cls._read_xlsb_file(file_path)
        elif file_path.endswith('.csv'):
            reader = {'data': cls._safe_read_okey_csv(file_path)}
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        processed_chunks: List[pd.DataFrame] = []
        for sheet_name, df in reader.items():
            if df.empty:
                logger.warning("[O'KEY] Лист %s пуст", sheet_name)
                continue
            logger.info("[O'KEY] Обработка листа %s, строк: %s", sheet_name, len(df))

            try:
                df = cls.normalize_okey_columns(df)

                # filename metadata
                month, year = cls.extract_okey_metadata(file_name)
                if month and year:
                    df = df.assign(sale_year=str(year), sale_month=str(month).zfill(2))

                # clean values (all strings in raw)
                df = df.replace([np.nan, None], '')
                for col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.replace('\u200b', '', regex=False)

                processed_chunks.append(df)
            except Exception as e:  # noqa: BLE001
                logger.error("[O'KEY] Ошибка обработки листа %s: %s", sheet_name, e, exc_info=True)
                continue

        if not processed_chunks:
            raise ValueError("File contains no valid data")

        final_df = pd.concat(processed_chunks, ignore_index=True)
        if final_df.empty:
            raise ValueError("No data after processing")

        # --- ensure raw table exists (all NVARCHAR(255))
        with engine.begin() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM information_schema.tables WHERE table_schema='raw' AND table_name=:table"),
                {"table": table_name},
            ).scalar()
            if not exists:
                cols_sql = []
                for col in final_df.columns:
                    clean_col = col.replace('"', '').replace("'", '')
                    cols_sql.append(f"[{clean_col}] NVARCHAR(255)")
                create_table_sql = f"CREATE TABLE raw.{table_name} ({', '.join(cols_sql)})"
                logger.debug("[O'KEY] SQL создания таблицы: %s", create_table_sql)
                conn.execute(text(create_table_sql))

        # --- insert
        cls.bulk_insert_okey(final_df, table_name, engine)

        duration = time.time() - start_time
        logger.info(
            "[O'KEY] Файл %s успешно загружен в raw.%s за %.2f сек (%s строк)",
            file_name, table_name, duration, len(final_df)
        )
        return table_name

    # ------------------------------------------------------------------
    # Bulk insert
    # ------------------------------------------------------------------
    @classmethod
    def bulk_insert_okey(cls, df: pd.DataFrame, table_name: str, engine: Engine):
        """Batch insert into raw.<table_name> (all values as strings)."""

        @event.listens_for(engine, 'before_cursor_execute')
        def _set_fast_executemany(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
            if executemany:
                cursor.fast_executemany = True

        data: List[tuple] = []
        for row in df.itertuples(index=False, name=None):
            clean_row = []
            for value in row:
                if pd.isna(value) or value in ('', None):
                    clean_row.append('')
                else:
                    s = str(value).strip()
                    # allow up to 1000 chars; DB col is 255 so driver will truncate if param longer; we pre‑truncate
                    if len(s) > 255:
                        s = s[:255]
                    clean_row.append(s)
            data.append(tuple(clean_row))

        cols = ', '.join(f'[{col}]' for col in df.columns)
        params = ', '.join(['?'] * len(df.columns))
        insert_sql = f"INSERT INTO raw.{table_name} ({cols}) VALUES ({params})"

        with closing(engine.raw_connection()) as conn:
            with conn.cursor() as cursor:
                logger.debug("[O'KEY] Пример данных для вставки: %s", data[0] if data else None)
                try:
                    cursor.executemany(insert_sql, data)
                    conn.commit()
                    logger.debug("[O'KEY] Успешно вставлено %s записей", len(data))
                except Exception as e:  # noqa: BLE001
                    conn.rollback()
                    logger.error(
                        "[O'KEY] Ошибка вставки. Первая строка: %s. Ошибка: %s", data[0] if data else None, e
                    )
                    raise


def create_okey_table_and_upload(file_path: str, engine: Engine) -> str:
    """Convenience wrapper for DAG usage."""
    try:
        return OkeyTableProcessor.process_okey_file(file_path, engine)
    except Exception as e:  # noqa: BLE001
        logger.error("Критическая ошибка при обработке файла %s: %s", file_path, e, exc_info=True)
        raise
