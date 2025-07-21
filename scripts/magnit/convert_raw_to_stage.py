"""convert_raw_to_stage_optimized.py
====================================

High‑volume RAW→STAGE transformation + load pipeline for Magnit data.

Why this rewrite?
-----------------
The original implementation worked for modest data sizes but ran into trouble on
~million‑row tables due to:

* Creating the STAGE table from a *set* of column names (bug) instead of a DataFrame sample.
* Repeated lower/replace passes on columns per chunk.
* Passing wrong arg type into `_convert_string_columns` (expected `year`, given dict).
* Opening a new DB connection for every chunk insert (high overhead).
* Not reusing prepared INSERT statement; building strings repeatedly.
* Lack of deterministic column order -> INSERT column mismatch.
* Missing type coercion for many numeric columns; inconsistent decimal cleanup.
* Potentially expensive `SELECT *` with client‑side limit instead of server‑side TOP/ORDER.
* Stage existence check running against the raw engine by mistake.

This optimized version fixes those issues and adds:

* **Canonical STAGE schema** (ordered) built from union of 2024/2025 layouts.
* **Idempotent create / truncate / append policy** via `if_exists`.
* **Vectorized numeric cleanup** (space thousands, comma decimal, stray chars).
* **Efficient streaming read** via `pd.read_sql(..., chunksize=N)` (server‑side cursor).
* **Single fast_executemany pyodbc cursor** reused across all chunks.
* **Progress logging** (rows, rows/s) + optional callback.
* **Date harmonization** – 2024 month‑only -> date 1st‑of‑month; 2025 has sales_date.
* **Filler defaults** for missing columns.
* **Chunk memory trimming** – drop raw RU columns promptly.

Quick start
-----------

```python
from sqlalchemy import create_engine
from convert_raw_to_stage_optimized import convert_raw_to_stage

raw_engine = create_engine(RAW_CONN_STR)
stage_engine = create_engine(STAGE_CONN_STR)

convert_raw_to_stage(
    table_name="magnit_mart_2025",  # RAW table name (already loaded)
    raw_engine=raw_engine,
    stage_engine=stage_engine,
    stage_schema="magnit",
    if_exists="append",  # or 'replace_truncate' to wipe first
    chunk_size=100_000,   # tune
)
```

"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Column configuration & canonical target schema
# ------------------------------------------------------------------

class ColumnConfig:
    """Configuration: numeric targets, RU→EN renames (2024 vs 2025)."""

    NUMERIC_COLS = {
        'turnover_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'turnover_quantity': {'dtype': 'float64', 'default': 0.0},
        'incoming_price': {'dtype': 'float64', 'default': 0.0},
        'avg_sell_price': {'dtype': 'float64', 'default': 0.0},
        'writeoff_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'writeoff_quantity': {'dtype': 'float64', 'default': 0.0},
        'sales_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'sales_weight_kg': {'dtype': 'float64', 'default': 0.0},
        'sales_quantity': {'dtype': 'float64', 'default': 0.0},
        'avg_purchase_price': {'dtype': 'float64', 'default': 0.0},
        'margin_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'loss_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'loss_quantity': {'dtype': 'float64', 'default': 0.0},
        'promo_sales_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'stock_quantity': {'dtype': 'float64', 'default': 0.0},
        'stock_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'discount_amount_rub': {'dtype': 'float64', 'default': 0.0},
    }

    # Minimal 2024 mapping (month granularity; fewer metrics)
    RENAME_MAP_2024 = {
        'month': 'sales_month',
        'формат': 'store_format',
        'наименование_тт': 'store_name',
        'код_тт': 'store_code',
        'адрес_тт': 'store_address',
        'уровень_1': 'product_level_1',
        'уровень_2': 'product_level_2',
        'уровень_3': 'product_level_3',
        'уровень_4': 'product_level_4',
        'поставщик': 'supplier_name',
        'бренд': 'brand',
        'наименование_тп': 'product_name',
        'код_тп': 'product_code',
        'шк': 'barcode',
        'оборот_руб': 'turnover_amount_rub',
        'оборот_шт': 'turnover_quantity',
        'входящая_цена': 'incoming_price',
    }

    # Expanded 2025 mapping (day granularity; richer metrics)
    RENAME_MAP_2025 = {
        'month': 'sales_month',  # sometimes present in raw
        'дата': 'sales_date',
        'формат': 'store_format',
        'наименование_тт': 'store_name',
        'код_тт': 'store_code',
        'адрес_тт': 'store_address',
        'уровень_1': 'product_level_1',
        'уровень_2': 'product_level_2',
        'уровень_3': 'product_level_3',
        'уровень_4': 'product_level_4',
        'поставщик': 'supplier_name',
        'бренд': 'brand',
        'наименование_тп': 'product_name',
        'код_тп': 'product_code',
        'шк': 'barcode',
        'оборот_руб': 'turnover_amount_rub',
        'оборот_шт': 'turnover_quantity',
        'входящая_цена': 'incoming_price',
        'код_группы': 'product_group_code',
        'группа': 'product_group_name',
        'код_категории': 'product_category_code',
        'категория': 'product_category_name',
        'код_подкатегории': 'product_subcategory_code',
        'подкатегория': 'product_subcategory_name',
        'артикул': 'product_article',
        'код_поставщика': 'supplier_code',
        'регион': 'region',
        'город': 'city',
        'ср_цена_продажи': 'avg_sell_price',
        'списания_руб': 'writeoff_amount_rub',
        'списания_шт': 'writeoff_quantity',
        'продажи_руб': 'sales_amount_rub',
        'продажи_кг': 'sales_weight_kg',
        'продажи_шт': 'sales_quantity',
        'маржа_руб': 'margin_amount_rub',
        'потери_руб': 'loss_amount_rub',
        'потери_шт': 'loss_quantity',
        'промо_продажи_руб': 'promo_sales_amount_rub',
        'остаток_шт': 'stock_quantity',
        'остаток_руб': 'stock_amount_rub',
        'скидка_руб': 'discount_amount_rub',
    }

    # RU month to number (Titlecase & lowercase accepted)
    MONTH_MAP_RU = {
        'январь': '01', 'февраль': '02', 'март': '03', 'апрель': '04', 'май': '05', 'июнь': '06',
        'июль': '07', 'август': '08', 'сентябрь': '09', 'октябрь': '10', 'ноябрь': '11', 'декабрь': '12',
        'январь_': '01', 'февраль_': '02', # defensive extras if stray chars
    }

    @classmethod
    def canonical_stage_columns(cls) -> List[str]:
        """Return ordered list of *all* canonical stage columns."""
        base_dims = [
            'sales_date',      # daily if available
            'sales_month',     # YYYY-MM first day
            'store_format', 'store_name', 'store_code', 'store_address',
            'region', 'city',
            'supplier_code', 'supplier_name',
            'brand',
            'product_group_code', 'product_group_name',
            'product_category_code', 'product_category_name',
            'product_subcategory_code', 'product_subcategory_name',
            'product_level_1', 'product_level_2', 'product_level_3', 'product_level_4',
            'product_article', 'product_name', 'product_code', 'barcode',
        ]
        metrics = list(cls.NUMERIC_COLS.keys())
        return base_dims + metrics


# Precompute canonical order once
CANON_STAGE_COLS = ColumnConfig.canonical_stage_columns()
CANON_STAGE_SET = set(CANON_STAGE_COLS)

# ------------------------------------------------------------------
# Detection helpers
# ------------------------------------------------------------------

_RU_MONTH_RE = re.compile("[А-Яа-я]+")


def _detect_year(df: pd.DataFrame) -> int:
    """Rudimentary format detection.

    Heuristics:
    * If column 'дата' present (case insensitive) -> 2025 layout.
    * Else if column 'month' w/ RU month words in sample -> 2024 layout.
    * Default -> 2025.
    """
    low = [c.lower() for c in df.columns]
    if any(c == 'дата' for c in low):
        return 2025
    if 'month' in low:
        sample = str(df.iloc[0, low.index('month')]) if not df.empty else ''
        if _RU_MONTH_RE.search(sample):
            return 2024
    return 2025


def _active_rename_map(df: pd.DataFrame, year: int) -> Dict[str, str]:
    low_cols = {c.lower(): c for c in df.columns}
    src_map = ColumnConfig.RENAME_MAP_2024 if year == 2024 else ColumnConfig.RENAME_MAP_2025
    active: Dict[str, str] = {}
    for ru, en in src_map.items():
        if ru in low_cols:
            active[low_cols[ru]] = en  # preserve original case key
    return active


# ------------------------------------------------------------------
# Cleaning helpers
# ------------------------------------------------------------------

_CLEAN_CTRL = re.compile(r"[\x00-\x1F\x7F-\x9F]")
_NON_NUMERIC_KEEP = re.compile(r"[^0-9.\-]")


def _clean_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.normalize('NFKC')
        .str.replace(_CLEAN_CTRL, '', regex=True)
        .str.strip()
        .replace({'nan': ''})
    )


def _clean_numeric(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.replace(_NON_NUMERIC_KEEP, '', regex=True)
    )


def _parse_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(_clean_numeric(s), errors='coerce').astype('float64')


# ------------------------------------------------------------------
# Column conversions
# ------------------------------------------------------------------

def _convert_string_columns(df: pd.DataFrame, active_map: Dict[str, str]) -> pd.DataFrame:
    """Rename & clean string columns according to active RU→EN map."""
    if not active_map:
        return df
    for ru_col, en_col in active_map.items():
        if ru_col in df.columns:
            df[en_col] = _clean_text(df[ru_col])
            if en_col != ru_col:
                df.drop(columns=[ru_col], inplace=True)
    return df


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for tgt_col in ColumnConfig.NUMERIC_COLS:
        if tgt_col in df.columns:
            df[tgt_col] = _parse_float(df[tgt_col]).fillna(ColumnConfig.NUMERIC_COLS[tgt_col]['default'])
    return df


# ------------------------------------------------------------------
# Date harmonization
# ------------------------------------------------------------------

def _parse_sales_month(series: pd.Series, year_default: Optional[int] = None) -> pd.Series:
    """Convert RU month labels to first‑of‑month date."""
    if series.empty:
        return pd.to_datetime(pd.Series([], dtype='datetime64[ns]'))
    # Lowercase for map lookup
    s = series.astype(str).str.lower().str.strip()
    def _map_month(val: str) -> str:
        m = ColumnConfig.MONTH_MAP_RU.get(val, None)
        if m is None:
            # try prefix match (e.g., 'январь 2024')
            for k, v in ColumnConfig.MONTH_MAP_RU.items():
                if val.startswith(k):
                    return v
            return None
        return m
    mapped = s.map(_map_month)
    if year_default is None:
        # try to pull a 4‑digit year from original string; fallback current year
        yrs = s.str.extract(r"(20\d{2})", expand=False)
        year_series = yrs.fillna(datetime.utcnow().year).astype(int).astype(str)
    else:
        year_series = pd.Series([year_default] * len(s), index=s.index, dtype="object")
    out = '"' + year_series + '-' + mapped.fillna('01') + '-01"'
    # fallback -> parse; invalid -> NaT
    dt = pd.to_datetime(year_series + '-' + mapped.fillna('01') + '-01', errors='coerce')
    return dt


def _derive_date_fields(df: pd.DataFrame, year: int) -> pd.DataFrame:
    # If sales_date already there, parse -> date.
    if 'sales_date' in df.columns:
        df['sales_date'] = pd.to_datetime(df['sales_date'], errors='coerce').dt.date
        # also produce sales_month if missing
        if 'sales_month' not in df.columns:
            df['sales_month'] = pd.to_datetime(df['sales_date'], errors='coerce').dt.to_period('M').dt.to_timestamp().dt.date
    # Else if sales_month present but as RU string -> parse
    elif 'sales_month' in df.columns:
        dt = _parse_sales_month(df['sales_month'], year_default=year)
        df['sales_month'] = dt.dt.date
        df['sales_date'] = None  # unknown day
    return df


# ------------------------------------------------------------------
# Stage table DDL
# ------------------------------------------------------------------

def _stage_col_sql(col: str) -> str:
    if col in ColumnConfig.NUMERIC_COLS:
        return f'[{col}] DECIMAL(18, 2) NULL'
    if col in ('sales_date', 'sales_month'):
        return f'[{col}] DATE NULL'
    # strings
    return f'[{col}] NVARCHAR(255) NULL'


def _create_stage_table(stage_engine: Engine, table_name: str, if_exists: str = 'append', schema: str = 'magnit') -> None:
    """Idempotent create/truncate stage table using canonical schema."""
    insp = inspect(stage_engine)
    exists = insp.has_table(table_name, schema=schema)
    full_table = f"[{schema}].[{table_name}]"

    if exists:
        if if_exists == 'fail':
            raise RuntimeError(f"Stage table {full_table} already exists.")
        elif if_exists == 'replace_truncate':
            logger.info("Truncating existing stage table %s", full_table)
            with stage_engine.begin() as conn:
                conn.execute(text(f"TRUNCATE TABLE {full_table}"))
            return
        else:  # append
            logger.debug("Stage table %s exists; appending.", full_table)
            return

    cols_sql = ',\n                '.join(_stage_col_sql(c) for c in CANON_STAGE_COLS) + ",\n                load_dt DATETIME DEFAULT GETDATE()"
    create_sql = f"""
    CREATE TABLE {full_table} (
                {cols_sql}
    );
    """
    try:
        with stage_engine.begin() as conn:
            conn.execute(text(create_sql))
        logger.info("Created stage table %s", full_table)
    except SQLAlchemyError as exc:  # pragma: no cover - DDL errors
        logger.error("Stage table create failed %s: %s", full_table, exc)
        raise


# ------------------------------------------------------------------
# Insert (fast_executemany) helpers
# ------------------------------------------------------------------

def _prepare_insert_cursor(stage_engine: Engine, table_name: str, schema: str = 'magnit'):
    """Return (raw_conn, cursor, insert_sql) with fast_executemany enabled."""
    raw_conn = stage_engine.raw_connection()
    cursor = raw_conn.cursor()
    cursor.fast_executemany = True
    cols_clause = ', '.join(f'[{c}]' for c in CANON_STAGE_COLS)
    params_clause = ', '.join(['?'] * len(CANON_STAGE_COLS))
    insert_sql = f"INSERT INTO [{schema}].[{table_name}] ({cols_clause}) VALUES ({params_clause})"
    return raw_conn, cursor, insert_sql


def _insert_chunk(cursor, insert_sql: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    data = list(df.itertuples(index=False, name=None))
    cursor.executemany(insert_sql, data)
    return len(data)


# ------------------------------------------------------------------
# Main convert function
# ------------------------------------------------------------------

def convert_raw_to_stage(
    table_name: str,
    raw_engine: Engine,
    stage_engine: Engine,
    stage_schema: str = 'magnit',
    if_exists: str = 'append',
    chunk_size: int = 100_000,
    limit: Optional[int] = None,
    progress_cb: Optional[callable] = None,
) -> int:
    """Transform & load data from RAW.<table_name> into STAGE.<table_name>.

    Parameters
    ----------
    table_name : str
        Source RAW table name (no schema).
    raw_engine, stage_engine : Engine
        SQLAlchemy engines.
    stage_schema : str
        Destination schema name (default 'magnit').
    if_exists : {'append','replace_truncate','fail'}
        Stage table policy.
    chunk_size : int
        Rows per fetch from RAW.
    limit : int | None
        Optional row cap (debug/testing).
    progress_cb : callable(rows_loaded:int, seconds:float) | None

    Returns
    -------
    int
        Total rows loaded to STAGE.
    """
    start_ts = datetime.now().timestamp()
    logger.info("[Stage] Begin RAW→STAGE for %s", table_name)

    # ------------------------------------------------------------------
    # Sample RAW to detect layout
    # ------------------------------------------------------------------
    with raw_engine.connect() as raw_conn:
        sample_sql = text(f"SELECT TOP 100 * FROM raw.[{table_name}]")
        sample_df = pd.read_sql(sample_sql, raw_conn)
    year = _detect_year(sample_df)
    logger.info("Detected layout year=%s for table %s", year, table_name)
    act_map = _active_rename_map(sample_df, year)

    # Prepare STAGE target
    _create_stage_table(stage_engine, table_name, if_exists=if_exists, schema=stage_schema)

    # If STAGE already has rows and append==False we might skip; quick check
    if if_exists == 'append':
        with stage_engine.connect() as stage_conn:
            try:
                chk = stage_conn.execute(text(f"SELECT TOP 1 1 FROM [{stage_schema}].[{table_name}]")).scalar()
                if chk:
                    logger.info("[Stage] Data already present in %s.%s; appending new rows.", stage_schema, table_name)
            except SQLAlchemyError:
                pass

    # Build streaming query
    base_query = f"SELECT * FROM raw.[{table_name}]"
    if limit is not None:
        # Use server side TOP to avoid network transfer
        base_query = f"SELECT TOP {limit} * FROM raw.[{table_name}]"

    total_rows = 0

    # Prepare insert cursor once
    raw_stage_conn, stage_cursor, insert_sql = _prepare_insert_cursor(stage_engine, table_name, schema=stage_schema)

    try:
        with raw_engine.connect().execution_options(stream_results=True) as raw_conn:
            for chunk in pd.read_sql(text(base_query), raw_conn, chunksize=chunk_size):
                # Transform ---------------------------------------------------
                # Normalize column cases to exactly as in sample for map lookup
                cols_lower = {c.lower(): c for c in chunk.columns}
                # Rename RU→EN
                rename_dict = {src: tgt for src, tgt in act_map.items() if src in chunk.columns}
                if rename_dict:
                    chunk.rename(columns=rename_dict, inplace=True)
                # Standard text cleanup for new EN cols
                for c in list(rename_dict.values()):
                    if c in chunk.columns:
                        chunk[c] = _clean_text(chunk[c])
                # Numeric conversions (if RU leftover metrics also present)
                # First map RU metrics to EN if they didn't exist in sample (defensive)
                for ru, en in ColumnConfig.RENAME_MAP_2025.items():  # superset
                    if ru in chunk.columns and en not in chunk.columns:
                        chunk[en] = chunk[ru]
                # Now parse numerics
                for c in ColumnConfig.NUMERIC_COLS:
                    if c in chunk.columns:
                        chunk[c] = _parse_float(chunk[c]).fillna(ColumnConfig.NUMERIC_COLS[c]['default'])
                # Drop raw RU columns we don't need
                drop_cols = [c for c in chunk.columns if c not in CANON_STAGE_SET]
                if drop_cols:
                    chunk.drop(columns=drop_cols, inplace=True, errors='ignore')
                # Derive dates
                chunk = _derive_date_fields(chunk, year)
                # Ensure all stage cols present
                for c in CANON_STAGE_COLS:
                    if c not in chunk.columns:
                        if c in ColumnConfig.NUMERIC_COLS:
                            chunk[c] = ColumnConfig.NUMERIC_COLS[c]['default']
                        else:
                            chunk[c] = '' if c not in ('sales_date','sales_month') else None
                # Reorder & cast ------------------------------------------------
                chunk = chunk[CANON_STAGE_COLS]
                # Cast numerics to python float (avoid Decimal mismatch)
                for c in ColumnConfig.NUMERIC_COLS:
                    chunk[c] = chunk[c].astype('float64')
                # Insert -------------------------------------------------------
                inserted = _insert_chunk(stage_cursor, insert_sql, chunk)
                total_rows += inserted
                if total_rows and progress_cb:
                    try:
                        progress_cb(total_rows, datetime.now().timestamp() - start_ts)
                    except Exception:  # pragma: no cover
                        pass
                if total_rows % (chunk_size * 5) == 0:  # periodic commit
                    raw_stage_conn.commit()
        # final commit
        raw_stage_conn.commit()
    finally:
        try:
            stage_cursor.close()
        except Exception:  # pragma: no cover - close best effort
            pass
        try:
            raw_stage_conn.close()
        except Exception:  # pragma: no cover
            pass

    dur = datetime.now().timestamp() - start_ts
    logger.info("[Stage] Loaded %s rows from RAW.%s into %s.%s in %.1fs (%.0f rows/s)",
                total_rows, table_name, stage_schema, table_name, dur, total_rows / dur if dur else 0)
    return total_rows


# ------------------------------------------------------------------
# CLI test harness (optional)
# ------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse
    from sqlalchemy import create_engine

    parser = argparse.ArgumentParser(description="RAW→STAGE loader for Magnit")
    parser.add_argument("--raw", required=True, help="SQLAlchemy URL to RAW DB")
    parser.add_argument("--stage", required=True, help="SQLAlchemy URL to STAGE DB")
    parser.add_argument("table", help="RAW table name")
    parser.add_argument("--schema", default="magnit")
    parser.add_argument("--chunk", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--replace", action="store_true", help="TRUNCATE stage table before load")
    args = parser.parse_args()

    raw_eng = create_engine(args.raw)
    stage_eng = create_engine(args.stage)
    if_exists = 'replace_truncate' if args.replace else 'append'
    convert_raw_to_stage(
        table_name=args.table,
        raw_engine=raw_eng,
        stage_engine=stage_eng,
        stage_schema=args.schema,
        if_exists=if_exists,
        chunk_size=args.chunk,
        limit=args.limit,
    )
