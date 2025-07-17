from __future__ import annotations

import os
import re
import logging
import time
from contextlib import closing
from io import StringIO
from typing import Dict, List, Tuple, Optional

import pyodbc
import numpy as np
import pandas as pd
from sqlalchemy import text, event, engine
from sqlalchemy.engine import Engine
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column configuration for Perekrestok
# ---------------------------------------------------------------------------
class ColumnConfig:
    """Column configuration / mapping for Perekrestok datasets.

    NOTE: Perekrestok source files vary. We union known layouts (summary &
    regional store detail). Add/adjust mappings as new columns appear.
    Keys in RENAME_MAP_* are *normalized RU* headers: lowercased, spaces→_,
    punctuation collapsed (match after basic normalization in code below).
    """

    # --- high-level business dimensions ---
    RENAME_MAP_DIM = {
        "период": "period_raw",  # e.g., "сен.24"; used to derive month/year
        "сеть": "retailer_name",  # e.g., "Перекрёсток", "Перекресток04"
        "регион": "region_name",
        "город": "city_name",
        "адрес": "store_address",  # may repeat; we truncate to 255 for raw
        "тт": "store_address_dup",  # some detailed files repeat address at end
        "рц": "dc_code",  # distribution center / warehouse code
        
    }

    # --- product hierarchy ---
    RENAME_MAP_PRODUCT = {
        "категория": "category_lvl1",
        "категория_2": "category_lvl2",  # second-level category in summary layout
        "тип_основы": "base_type",  # e.g., Прессованные рифленые
        "поставщик": "supplier_name",
        "бренд": "brand_name",
        "наименование": "product_name",  # display / final name
        "уни_наименование": "product_name_uni",  # unified/normalized name
        "граммовка": "pack_weight_g",  # integer grams (string source)
        # some files use Вкусы / Вкус
        "вкусы": "flavor_name",
        "вкус": "flavor_name",  # mapped to same canonical
        "себест__руб": "cost_rub",          # нормализация
        "себест__руб_": "cost_rub",
        "себест_руб": "cost_rub",
        "себест__руб": "cost_rub",
        "себест_ руб": "cost_rub",          # <- конкретно твой кейс: пробел после \
        "филиал": "branch_name",
    }

    # --- measures ---
    RENAME_MAP_MEAS = {
        "продажи,_шт": "sales_qty",
        "продажи_шт": "sales_qty",  # safety
        "продажи,_руб": "sales_rub",
        "продажи_руб": "sales_rub",
        "продажи,_тонн": "sales_tons",
        "продажи_тонн": "sales_tons",
        "себест.,_руб": "cost_rub",
        "себест__руб": "cost_rub",  # normalization artifact
        "себест._руб": "cost_rub",
        "себест._руб_": "cost_rub",
        "себест._руб_": "cost_rub",
        "себест._руб": "cost_rub",
        "себест__руб_": "cost_rub",
        "себест__руб.": "cost_rub",
    }

    # Combined full mapping
    RENAME_MAP: Dict[str, str] = {}
    RENAME_MAP.update(RENAME_MAP_DIM)
    RENAME_MAP.update(RENAME_MAP_PRODUCT)
    RENAME_MAP.update(RENAME_MAP_MEAS)

    # numeric columns (canonical EN names) we want as FLOAT in stage
    NUMERIC_COLS = {
        "sales_qty",
        "sales_rub",
        "sales_tons",
        "cost_rub",
        "pack_weight_g",  # leave as float; you can cast INT in downstream SQL
    }

    # allowed synonyms for month abbreviations found in Period field or filenames
    # keys lower-normalized; values month number 1..12
    MONTH_TOKENS = {
        # russian short (dot optional) & mis-spellings seen in files (ceptember)
        "янв": 1, "янв.": 1,
        "фев": 2, "фев.": 2,
        "мар": 3, "мар.": 3,
        "апр": 4, "апр.": 4,
        "май": 5, "мая": 5,
        "июн": 6, "июн.": 6,
        "июл": 7, "июл.": 7,
        "авг": 8, "авг.": 8,
        "сен": 9, "сен.": 9, "сеп": 9, "септ": 9, "цепт": 9, "сeп": 9,  # typos
        "окт": 10, "окт.": 10,
        "ноя": 11, "ноя.": 11,
        "дек": 12, "дек.": 12,
        # english
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9, "ceptember": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }


# ---------------------------------------------------------------------------
# Helpers: column normalization & uniqueness
# ---------------------------------------------------------------------------
_norm_ws_re = re.compile(r"\s+")
_non_alnum_re = re.compile(r"[^0-9a-zа-яё_]+", re.IGNORECASE)
_multi_us_re = re.compile(r"_{2,}")


def _base_normalize_header(name: object) -> str:
    """Basic RU header normalize to key used in mapping lookups.
    Lowercase, strip, collapse whitespace to _, replace punctuation with _,
    collapse repeats; *preserve* cyrillic for matching vs. RENAME_MAP keys.
    """
    if not isinstance(name, str):
        name = "" if name is None else str(name)
    s = name.strip().lower()
    s = _norm_ws_re.sub("_", s)
    s = _non_alnum_re.sub("_", s)
    s = _multi_us_re.sub("_", s)
    return s.strip("_")


def sanitize_and_make_unique_columns(columns: List[object]) -> List[str]:
    """Full sanitize → map RU→EN → ensure SQL-safe, unique.

    1. Normalize raw header to key.
    2. Lookup in ColumnConfig.RENAME_MAP; else transliterate-ish safe ascii.
    3. Guarantee not empty; supply col_i.
    4. Deduplicate with numeric suffixes _1, _2...
    """
    seen: Dict[str, int] = {}
    result: List[str] = []

    for i, raw in enumerate(columns):
        key = _base_normalize_header(raw)
        mapped = ColumnConfig.RENAME_MAP.get(key)
        if mapped:
            base = mapped
        else:
            # fallback ascii-ish: drop Cyrillic but keep digits/underscores
            # (SQL Server is fine with unicode, but we keep consistent ascii)
            tmp = re.sub(r"[^0-9a-z_]+", "_", key)  # key already lower ascii+cyr
            tmp = _multi_us_re.sub("_", tmp).strip("_")
            base = tmp or f"col_{i}"

        cnt = seen.get(base, 0)
        if cnt:
            col_name = f"{base}_{cnt}"
        else:
            col_name = base
        seen[base] = cnt + 1
        result.append(col_name)

    return result


# ---------------------------------------------------------------------------
# Numeric parse helper
# ---------------------------------------------------------------------------
_num_clean_re = re.compile(r"[^0-9,.-]")


def _coerce_num_series(s: pd.Series) -> pd.Series:
    """Coerce col of strings to float.
    Accept comma decimals, stray spaces, thousands junk. Empty→0.
    """
    if s.dtype.kind in ("i", "u", "f"):
        return s.astype(float)
    # stringfy
    s2 = s.astype(str).str.strip()
    # replace comma with dot *only if more commas than dots* or always? use always
    s2 = s2.str.replace(",", ".", regex=False)
    # drop non numeric tokens (keep - and .)
    s2 = s2.str.replace(_num_clean_re, "", regex=True)
    s2 = s2.replace({"": np.nan, ".": np.nan, "-": np.nan})
    out = pd.to_numeric(s2, errors="coerce")
    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# Converters used in raw→stage step
# ---------------------------------------------------------------------------

def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mapped numeric RU columns to canonical float columns.

    Works *in-place*: adds EN canonical columns; drops RU originals.
    """
    logger.info(f"Original columns: {df.columns.tolist()}")

    # make quick-lookup of RU->EN numeric subset
    numeric_map = {ru: en for ru, en in ColumnConfig.RENAME_MAP.items() if en in ColumnConfig.NUMERIC_COLS}

    # ensure we operate over normalized keys snapshot
    norm_cols = {_base_normalize_header(c): c for c in df.columns}

    for ru_norm, en_col in numeric_map.items():
        if ru_norm in norm_cols:
            src_col = norm_cols[ru_norm]
            try:
                df[en_col] = _coerce_num_series(df[src_col])
                if en_col != src_col:
                    df.drop(columns=[src_col], inplace=True)
            except Exception as e:  # pragma: no cover
                logger.error(f"Error converting numeric col {src_col}→{en_col}: {e}")
                raise

    logger.info(f"Columns after numeric conversion: {df.columns.tolist()}")
    return df


def _convert_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mapped RU dimension columns to canonical string columns."""
    norm_cols = {_base_normalize_header(c): c for c in df.columns}
    for ru_norm, en_col in ColumnConfig.RENAME_MAP.items():
        if ru_norm in norm_cols:
            src_col = norm_cols[ru_norm]
            try:
                df[en_col] = df[src_col].astype("string").fillna("")
                if en_col != src_col:
                    df.drop(columns=[src_col], inplace=True)
            except Exception as e:
                logger.error(f"Error converting string col {src_col}→{en_col}: {e}")
                raise
    return df


# ---------------------------------------------------------------------------
# Month / year extraction
# ---------------------------------------------------------------------------
_month_token_re = re.compile(r"([a-zа-яё]{3,}|\d{1,2})", re.IGNORECASE)
_year_re = re.compile(r"(20\d{2}|\d{2})")  # accept 2-digit year in period cell


def parse_month_year_from_tokens(tokens: List[str]) -> Tuple[Optional[int], Optional[int]]:
    m_num = None
    y_num = None
    for tok in tokens:
        t = tok.lower().strip().rstrip('.')
        if t in ColumnConfig.MONTH_TOKENS and m_num is None:
            m_num = ColumnConfig.MONTH_TOKENS[t]
        elif y_num is None:
            # try 4 or 2 digit year
            m = _year_re.fullmatch(t)
            if m:
                y = m.group(1)
                if len(y) == 2:
                    y = "20" + y  # naive pivot 2000s
                y_num = int(y)
    return m_num, y_num


def extract_month_year(source: str) -> Tuple[Optional[int], Optional[int]]:
    if not source:
        return None, None
    tokens = _month_token_re.findall(str(source))
    return parse_month_year_from_tokens(tokens)


# ---------------------------------------------------------------------------
# RAW load: read file → normalize columns → infer month/year → write raw schema
# ---------------------------------------------------------------------------
class TableProcessorPerek:
    CHUNKSIZE = 100_000  # not yet used; we load whole sheet then clip
    BATCH_SIZE = 10_000

    @classmethod
    def _safe_read_csv(cls, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(
                file_path,
                dtype="string",
                sep=";",
                quotechar='"',
                on_bad_lines="warn",
                encoding="utf-8",
                encoding_errors="replace",
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"CSV parse warn ({file_path}): {e}; attempting salvage")
            clean_lines = []
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                header = f.readline().strip()
                clean_lines.append(header)
                for line in f:
                    if line.count('"') % 2 == 0:
                        clean_lines.append(line.strip())
            return pd.read_csv(StringIO("\n".join(clean_lines)), dtype="string", sep=";", quotechar='"')

    @staticmethod
    def _read_excel_any(file_path: str) -> Dict[str, pd.DataFrame]:
        """Read xlsx/xlsb/xls; return dict sheet_name→DataFrame.
        We *do not* skip rows here because some files carry merged/multi header.
        We'll fix headers downstream.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".xlsx":
            return pd.read_excel(file_path, sheet_name=None, dtype="object", engine="openpyxl")
        if ext == ".xlsb":
            return pd.read_excel(file_path, sheet_name=None, dtype="object", engine="pyxlsb")
        # fall back generic engine (xlrd no xls by default in new pandas; openpyxl often works)
        return pd.read_excel(file_path, sheet_name=None, dtype="object")

    @classmethod
    def _repair_multiheader(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Detect & resolve 2-row headers (numbers+labels) into single row.

        Heuristic: If first row contains >=3 numeric tokens or 'перекр' is not in row0
        but row1 has many text labels, we combine row0+row1.
        """
        if df.empty:
            return df

        # if any column name is numeric-like AND first data row looks like header text -> treat as multiheader
        # We inspect first two *dataframe* rows, not df.columns (since pandas may auto 0..N).
        # When reading w/out header, pandas sets numeric RangeIndex; we need to catch that upstream.
        # So ensure df has proper columns: if default RangeIndex and row0 textual -> promote row0 as header.
        if isinstance(df.columns, pd.RangeIndex):
            # try to promote row0
            row0 = df.iloc[0].astype(str).tolist()
            # if >50% non-null textual -> use row0 as columns
            non_blank = [c for c in row0 if str(c).strip() and str(c).strip().lower() != 'nan']
            if non_blank and len(non_blank) >= len(row0) * 0.5:
                df = df.iloc[1:].reset_index(drop=True).copy()
                df.columns = row0

        # At this point we *might* still have multiindex if pandas recognized header=[0,1]; unify
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = [str(x) if x is not None else "" for x in df.columns.get_level_values(0)]
            lvl1 = [str(x) if x is not None else "" for x in df.columns.get_level_values(1)]
            combined = []
            for a, b in zip(lvl0, lvl1):
                a_s = a.strip()
                b_s = b.strip()
                if a_s and b_s and a_s != b_s:
                    combined.append(f"{a_s}_{b_s}")
                else:
                    combined.append(b_s or a_s)
            df.columns = combined
            return df

        # Single-level but maybe row0 numeric group row1 labels pattern? detect by scanning first data row
        # If >3 column names match r'^\d+$' we suspect hidden group row; check first actual row values for text labels
        cols = [str(c) for c in df.columns]
        numeric_like_cols = sum(bool(re.fullmatch(r"\d+", c)) for c in cols)
        if numeric_like_cols >= 3:
            # assume second row holds labels
            if len(df) >= 1:
                row_labels = df.iloc[0].astype(str).tolist()
                combined = []
                for g, lbl in zip(cols, row_labels):
                    g_s = g.strip()
                    lbl_s = lbl.strip()
                    if g_s and lbl_s and g_s != lbl_s:
                        combined.append(f"{g_s}_{lbl_s}")
                    else:
                        combined.append(lbl_s or g_s)
                df = df.iloc[1:].reset_index(drop=True).copy()
                df.columns = combined
        return df

    @classmethod
    def _normalize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Apply header repair + column normalize + dtype cleanup."""
        df = cls._repair_multiheader(df)
        # trim trailing full-null columns/rows (common after header merge)
        df = df.dropna(axis=0, how="all")
        df = df.dropna(axis=1, how="all")
        # base normalize names
        normed_cols = [_base_normalize_header(c) for c in df.columns]
        df.columns = normed_cols
        return df

    @classmethod
    def process_file(cls, file_path: str, engine: Engine) -> str:
        start_time = time.time()
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        table_name = re.sub(r"\W+", "_", base_name.lower())

        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".xlsx", ".xlsb", ".xls"}:
            sheets = cls._read_excel_any(file_path)
        elif ext == ".csv":
            sheets = {"data": cls._safe_read_csv(file_path)}
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        processed: List[pd.DataFrame] = []
        for sh_name, df in sheets.items():
            if df is None or df.empty:
                continue
            try:
                df = cls._normalize_dataframe(df)
                # limited sample safety: clip huge sheets in raw ingest (tune as needed)
                df = df.head(200000)

                # derive month/year from filename first, else Period column
                m, y = extract_month_year(file_name)
                if not m or not y:
                    if "период" in df.columns:
                        m2, y2 = extract_month_year(str(df["период"].iloc[0]))
                        m = m or m2
                        y = y or y2

                if m:
                    df["sale_month"] = int(m)
                if y:
                    df["sale_year"] = int(y)

                # ensure string dtype for all dims (object→string)
                for col in df.columns:
                    if col not in ColumnConfig.NUMERIC_COLS:
                        df[col] = df[col].astype("string")

                processed.append(df)
            except Exception as e:  # pragma: no cover
                logger.error(f"Error processing sheet {sh_name} in {file_path}: {e}", exc_info=True)
                continue

        if not processed:
            raise ValueError("File contains no valid data.")

        final_df = pd.concat(processed, ignore_index=True)
        del processed

        # create raw.<table>
        with engine.begin() as conn:
            table_exists = conn.execute(
                text(
                    """
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'raw' AND table_name = :tbl
                    """
                ),
                {"tbl": table_name},
            ).scalar()

            if not table_exists:
                # sanitize to safe names for raw table *once* to match metadata query later
                safe_cols = sanitize_and_make_unique_columns(list(final_df.columns))
                final_df.columns = safe_cols
                cols_sql = [f"[{c}] NVARCHAR(255)" for c in safe_cols]
                create_sql = f"CREATE TABLE raw.{table_name} ({', '.join(cols_sql)})"
                conn.execute(text(create_sql))
            else:
                # ensure columns sanitized even if table existed (assume prior run sanitized)
                final_df.columns = sanitize_and_make_unique_columns(list(final_df.columns))

        # insert
        TableProcessorPerek.bulk_insert(final_df, table_name, engine)

        dur = time.time() - start_time
        logger.info(
            f"File {file_name} loaded to raw.{table_name} in {dur:.2f}s ({len(final_df)} rows)."
        )
        return table_name

    @classmethod
    def bulk_insert(cls, df: pd.DataFrame, table_name: str, engine: Engine):
        """Fast insert to raw schema."""
        @event.listens_for(engine, 'before_cursor_execute')
        def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # pragma: no cover
            if executemany:
                cursor.fast_executemany = True

        # copy & clip to <=255 chars for NVARCHAR(255)
        df_str = df.where(pd.notnull(df), None).copy()
        for col in df_str.columns:
            df_str[col] = df_str[col].astype(str).str.slice(0, 255)

        rows = [tuple(None if (v is None or str(v).lower() == 'nan') else str(v) for v in r)
                for r in df_str.itertuples(index=False, name=None)]

        with closing(engine.raw_connection()) as raw_conn:
            with raw_conn.cursor() as cursor:
                cursor.fast_executemany = True
                cursor.setinputsizes([(pyodbc.SQL_WVARCHAR, 255, 0)] * len(df_str.columns))
                cols = ', '.join(f'[{c}]' for c in df_str.columns)
                params = ', '.join(['?'] * len(df_str.columns))
                sql = f"INSERT INTO raw.{table_name} ({cols}) VALUES ({params})"
                cursor.executemany(sql, rows)
                raw_conn.commit()


# Public wrapper -------------------------------------------------------------
def create_table_and_upload_perek(file_path: str, engine: Engine) -> str:
    return TableProcessorPerek.process_file(file_path, engine)


# ---------------------------------------------------------------------------
# RAW→STAGE transformation (use after raw ingest)
# ---------------------------------------------------------------------------

def _create_stage_table(conn: engine.Connection, table_name: str, df: pd.DataFrame, schema: str = 'perekrestok') -> None:
    logger.info(f"[Stage] DataFrame columns before table creation: {df.columns.tolist()}")

    new_columns = sanitize_and_make_unique_columns(list(df.columns))
    df.columns = new_columns

    dupes = [c for c in new_columns if new_columns.count(c) > 1]
    if dupes:
        raise ValueError(f"Duplicate column names after sanitization: {dupes}")

    col_defs = []
    for c in new_columns:
        if c in ColumnConfig.NUMERIC_COLS:
            col_defs.append(f"[{c}] FLOAT")
        else:
            col_defs.append(f"[{c}] NVARCHAR(255)")

    sql = f"""
    IF NOT EXISTS (
        SELECT * FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE s.name = '{schema}' AND t.name = '{table_name}'
    )
    BEGIN
        CREATE TABLE [{schema}].[{table_name}] (
            {', '.join(col_defs)}
        )
    END
    """

    trans = conn.begin()
    try:
        conn.execute(text(sql))
        trans.commit()
    except Exception as e:  # pragma: no cover
        trans.rollback()
        logger.error(f"Error creating stage table {schema}.{table_name}: {e}\nSQL: {sql}")
        raise


def _bulk_insert_stage(conn: engine.Connection, table_name: str, df: pd.DataFrame, schema: str = 'perekrestok') -> None:
    if df.empty:
        logger.warning("[Stage] Empty DataFrame, skipping insert")
        return

    cols = sanitize_and_make_unique_columns(list(df.columns))
    df = df.copy()
    df.columns = cols

    # only insert if table empty (same pattern as your other loaders)
    rc = conn.execute(text(f"SELECT COUNT(*) FROM [{schema}].[{table_name}]"))
    if rc.scalar() > 0:
        logger.info(f"[Stage] Table {schema}.{table_name} already has data; skipping insert.")
        return

    rows = []
    for tup in df.itertuples(index=False, name=None):
        row_vals = []
        for val, col in zip(tup, cols):
            if col in ColumnConfig.NUMERIC_COLS:
                if val is None or (isinstance(val, str) and not val.strip()):
                    row_vals.append(0.0)
                else:
                    try:
                        row_vals.append(float(val))
                    except Exception:
                        row_vals.append(0.0)
            else:
                row_vals.append(str(val) if val is not None else '')
        rows.append(tuple(row_vals))

    raw_conn = conn.connection
    cursor = raw_conn.cursor()
    try:
        cursor.fast_executemany = True
        col_sql = ', '.join(f'[{c}]' for c in cols)
        params = ', '.join(['?'] * len(cols))
        sql = f"INSERT INTO [{schema}].[{table_name}] ({col_sql}) VALUES ({params})"
        cursor.executemany(sql, rows)
        raw_conn.commit()
        logger.info(f"[Stage] Inserted {len(rows)} rows into {schema}.{table_name}.")
    except Exception as e:  # pragma: no cover
        raw_conn.rollback()
        logger.error(f"[Stage] Error inserting into {schema}.{table_name}: {e}\nSQL: {sql}")
        raise
    finally:
        cursor.close()


def convert_raw_to_stage(
    table_name: str,
    raw_engine: engine.Engine,
    stage_engine: engine.Engine,
    stage_schema: str = 'perekrestok',
    limit: int | None = None,
) -> None:
    """Read raw.<table_name>, normalize, map, and load to stage schema."""
    try:
        start_time = datetime.now()
        logger.info(f"[Stage] Starting processing of table {table_name}")

        # Gather metadata & size
        with raw_engine.connect() as conn:
            cols_res = conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = 'raw' AND TABLE_NAME = :tbl"
                ),
                {"tbl": table_name},
            )
            raw_cols = [r[0] for r in cols_res]
            tot_rows = conn.execute(text(f"SELECT COUNT(*) FROM raw.{table_name}")).scalar()
            logger.info(f"[Stage] Raw columns: {raw_cols}")
            logger.info(f"[Stage] Total rows to process: {tot_rows}")

        # stream read
        chunks: List[pd.DataFrame] = []
        sql_q = f"SELECT * FROM raw.{table_name}"
        if limit is not None:
            sql_q += f" ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"

        with raw_engine.connect().execution_options(stream_results=True) as conn:
            for chunk in pd.read_sql(text(sql_q), conn, chunksize=50000, dtype='object'):
                logger.info(f"[Stage] Received chunk rows={len(chunk)} cols={chunk.columns.tolist()}")
                # numeric then string conversions
                chunk = _convert_numeric_columns(chunk)
                chunk = _convert_string_columns(chunk)
                chunks.append(chunk)
                if limit is not None and sum(len(c) for c in chunks) >= limit:
                    break

        if not chunks:
            logger.warning("[Stage] No data extracted from raw table.")
            return

        df = pd.concat(chunks, ignore_index=True)
        if limit is not None:
            df = df.head(limit)
        logger.info(f"[Stage] Final columns after processing: {df.columns.tolist()}")


        if df.empty or len(df.columns) == 0:
            raise ValueError("[Stage] DataFrame empty after processing.")

        with stage_engine.connect() as conn:
            trans = conn.begin()
            try:
                _create_stage_table(conn, table_name, df, schema=stage_schema)
                _bulk_insert_stage(conn, table_name, df, schema=stage_schema)
                trans.commit()
                dur = (datetime.now() - start_time).total_seconds()
                logger.info(f"[Stage] Successfully loaded {len(df)} rows in {dur:.2f}s")
            except Exception:
                trans.rollback()
                raise
    except Exception as e:
        logger.error(f"[Stage ERROR] Error processing table {table_name}: {e}")
        raise
