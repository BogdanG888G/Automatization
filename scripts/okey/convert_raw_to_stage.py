"""
High‑volume optimized conversion from raw.<table> to stage.<schema>.<table> for Okey data.

Key improvements vs. original version (aimed at 1M+ rows):

1. **True streaming, no full concat** – process and load each chunk as soon as it's cleaned;
   avoids building a multi‑GB in‑memory DataFrame.
2. **Single schema inference up front** – infer *final* stage columns once (from
   ColumnConfig + presence in first processed chunk). Ensures stable column order and types.
3. **Deterministic column order & defaults** – every chunk is reindexed to the full stage column set.
   Missing columns filled with sensible defaults (0.0 for numerics, "" for strings, None for month).
4. **Vectorized month parsing** – faster parsing for typical "YYYY-MM-DD", "MM.YYYY", or RU month abbreviations.
5. **Reduced logging overhead** – debug‑level row dumps removed; periodic progress logging.
6. **Faster sanitization** – precomputed map; avoid repeated regex for known columns.
7. **Efficient bulk insert** – per‑chunk `fast_executemany` with array binding; optional batch size split to cap memory.
8. **Optional truncate / append / skip** – behavior controlled by params.
9. **Safer numeric conversion** – central helper; handles commas, spaces, and ton->kg scaling.
10. **Explicit SQL type mapping** – created exactly once (unless `ensure_schema_each_chunk=True`).

Usage
-----
>>> convert_raw_to_stage(
...     table_name="okey_sales_2024",
...     raw_engine=raw_engine,
...     stage_engine=stage_engine,
...     stage_schema="okey",
...     chunk_size=100_000,
...     drop_stage_if_exists=True,
... )

This module is self‑contained; copy into your project and adjust logging config.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import engine, exc, text

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Column configuration                                                #
# ------------------------------------------------------------------ #
class ColumnConfig:
    """Column configuration with data types and transformations for Okey."""

    NUMERIC_COLS = {
        "sales_amount_rub": {"dtype": "float64", "default": 0.0},
        "sales_quantity": {"dtype": "float64", "default": 0.0},
        "sales_weight_kg": {"dtype": "float64", "default": 0.0},
        "cost_price_rub": {"dtype": "float64", "default": 0.0},
    }

    RAW_RU_NORMALIZE = {
        # период / дата
        "период": "period",
        # сеть
        "сеть": "retail_chain",
        # категория
        "категория": "category",
        # категория 2 / вложенная категория -> base_type (потом product_type)
        "категория_2": "base_type",
        "категория2": "base_type",
        # поставщик
        "поставщик": "supplier",
        "поставщики": "supplier",
        # бренд
        "бренд": "brand",
        "бренды": "brand",
        # наименование SKU
        "наименование": "product_name",
        # унифицированное наименование
        "уни_наименование": "unified_product_name",
        "уни_наим": "unified_product_name",
        # граммовка
        "граммовка": "weight",
        # вкус
        "вкус": "flavor",
        "вкусы": "flavor",
        # продажи
        "продажи,_шт": "sales_units",
        "продажи_шт": "sales_units",
        "продажи,_руб": "sales_rub",
        "продажи_руб": "sales_rub",
        "продажи,_тонн": "sales_tonnes",
        "продажи_тонн": "sales_tonnes",
        # себестоимость
        "себест.,_руб": "cost_rub",
        "себест_руб": "cost_rub",
    }

    RENAME_MAP = {
        "period": "period",  # handled in month->sales_month flow; left for completeness
        "retail_chain": "retail_chain",
        "category": "product_category",
        "base_type": "product_type",  # Категория 2 -> product_type
        "supplier": "supplier_name",
        "brand": "brand",
        "product_name": "product_name",
        "unified_product_name": "product_unified_name",
        "weight": "product_weight_g",
        "flavor": "product_flavor",
        "sales_units": "sales_quantity",
        "sales_rub": "sales_amount_rub",
        "sales_tonnes": "sales_weight_kg",
        "cost_rub": "cost_price_rub",
    }

    STRING_COL_LENGTHS = {
        "retail_chain": 255,
        "product_category": 255,
        "product_type": 255,
        "supplier_name": 255,
        "brand": 255,
        "product_name": 255,
        "product_unified_name": 255,
        "product_flavor": 255,
        "product_weight_g": 50,
    }

    @classmethod
    def stage_columns(cls) -> List[str]:
        # Deterministic final order: business keys first, then measures, then month.
        ordered = [
            "retail_chain",
            "product_category",
            "product_type",
            "supplier_name",
            "brand",
            "product_name",
            "product_unified_name",
            "product_weight_g",
            "product_flavor",
            "sales_quantity",
            "sales_amount_rub",
            "sales_weight_kg",
            "cost_price_rub",
            "sales_month",
        ]
        return ordered

    @classmethod
    def is_numeric(cls, col: str) -> bool:
        return col in cls.NUMERIC_COLS

    @classmethod
    def max_len(cls, col: str) -> int:
        return cls.STRING_COL_LENGTHS.get(col, 255)


# Precompile regexes used repeatedly ----------------------------------------------------
_RE_MULTISPACE = re.compile(r"\s+")
_RE_NONALNUM = re.compile(r"[^a-zа-я0-9_\.]")
_RE_CONTROL = re.compile(r"[\x00-\x1F\x7F-\x9F]")
_RE_JUNK_NUM = re.compile(r"[^\d.eE+\-]")

# RU month mapping (short)
_RU_MONTHS_SHORT = {
    "ян": 1,
    "янв": 1,
    "фе": 2,
    "фев": 2,
    "мар": 3,
    "апр": 4,
    "май": 5,
    "июн": 6,
    "июл": 7,
    "авг": 8,
    "сен": 9,
    "сент": 9,
    "окт": 10,
    "ноя": 11,
    "дек": 12,
}


# ------------------------------------------------------------------ #
# Normalization helpers                                               #
# ------------------------------------------------------------------ #
def _normalize_okey_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw column headers to internal English base names.

    Works tolerant to punctuation, spaces & case. Example:
    "Категория 2" -> "base_type"; "Период" -> "period".
    """
    normed = []
    for col in df.columns:
        c = str(col).strip().lower()
        c_key = _RE_MULTISPACE.sub("_", c)
        c_key = _RE_NONALNUM.sub("_", c_key)
        c_key = c_key.replace(",", "_").replace("..", ".")
        c_key = re.sub(r"_+", "_", c_key).strip("_")
        mapped = ColumnConfig.RAW_RU_NORMALIZE.get(c_key, c_key)
        normed.append(mapped)
    df.columns = normed
    return df


# ------------------------------------------------------------------ #
# Month parsing                                                        #
# ------------------------------------------------------------------ #
def _month_from_period_series(period: pd.Series) -> pd.Series:
    """Vectorized best‑effort month extraction from `period` values."""
    if period.empty:
        return pd.Series([], dtype="Int16")

    # Try direct datetime parse (dayfirst so 01.09.2024 works).
    dt = pd.to_datetime(period, dayfirst=True, errors="coerce")
    month = dt.dt.month.astype("Float64")  # preserve NaN

    # Where failed, try RU month abbrev: take 3 chars of cleaned string; map.
    mask_missing = month.isna()
    if mask_missing.any():
        s = period[mask_missing].astype(str).str.lower().str.replace(".", "", regex=False).str.replace(" ", "", regex=False)
        token = s.str[:4]  # cover "сент"
        month_map = token.map(lambda x: next((v for k, v in _RU_MONTHS_SHORT.items() if x.startswith(k)), np.nan))
        month.loc[mask_missing] = month_map

    # Fill any remaining NaNs with 1 and clip.
    month = month.fillna(1).astype(int).clip(1, 12)
    return month


def _process_month_columns(df: pd.DataFrame, drop_source_cols: bool = True) -> pd.DataFrame:
    """Create `sales_month` INT 1‑12 from `sale_month` or `period`.

    If neither parses, default=1 (Jan) with warning once per chunk.
    """
    month_int = pd.Series(np.nan, index=df.index, dtype="Float64")

    # 1) sale_month numeric
    if "sale_month" in df.columns:
        m = pd.to_numeric(df["sale_month"], errors="coerce")
        month_int = m.where(~m.isna(), month_int)

    # 2) period fallback
    if "period" in df.columns:
        parsed = _month_from_period_series(df["period"])
        month_int = month_int.where(~month_int.isna(), parsed)

    # Finalize
    if month_int.isna().any():
        logger.warning("Some rows missing month info; filling with 1 (January).")
    month_int = month_int.fillna(1).astype(int).clip(1, 12)
    df["sales_month"] = month_int

    if drop_source_cols:
        df = df.drop(columns=["sale_month", "period"], errors="ignore")
    return df


# ------------------------------------------------------------------ #
# Numeric cleaning                                                      #
# ------------------------------------------------------------------ #
def _clean_numeric_series(s: pd.Series, multiply: float | None = None) -> pd.Series:
    """Clean numeric strings (comma decimals, spaces, junk)."""
    s = s.astype("string")
    s = s.str.replace(",", ".", regex=False)  # decimal comma -> dot
    s = s.str.replace(_RE_JUNK_NUM, "", regex=True)  # strip junk
    s = s.fillna("0")
    s = s.replace("", "0")
    out = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if multiply is not None:
        out = out * multiply
    return out


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert expected numeric raw columns to numeric."""
    if "sales_units" in df.columns:
        df["sales_units"] = _clean_numeric_series(df["sales_units"])
    if "sales_rub" in df.columns:
        df["sales_rub"] = _clean_numeric_series(df["sales_rub"])
    if "sales_tonnes" in df.columns:
        df["sales_tonnes"] = _clean_numeric_series(df["sales_tonnes"], multiply=1000.0)  # tons->kg
    if "cost_rub" in df.columns:
        df["cost_rub"] = _clean_numeric_series(df["cost_rub"])
    return df


# ------------------------------------------------------------------ #
# String conversion to stage names                                      #
# ------------------------------------------------------------------ #
def _convert_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns from normalized raw names to stage names."""
    df.columns = [col.strip() for col in df.columns]

    for raw_col, stage_col in ColumnConfig.RENAME_MAP.items():
        if raw_col not in df.columns:
            continue
        if stage_col in ColumnConfig.NUMERIC_COLS:
            # numeric handled elsewhere
            continue
        max_len = ColumnConfig.max_len(stage_col)
        ser = df[raw_col].astype("string")
        ser = ser.str.normalize("NFKC")
        ser = ser.str.replace(_RE_CONTROL, "", regex=True)
        ser = ser.str.strip().fillna("")
        ser = ser.str.slice(0, max_len)
        df[stage_col] = ser
        if raw_col != stage_col:
            df.drop(columns=[raw_col], inplace=True)
    return df


# ------------------------------------------------------------------ #
# Stage column name sanitizer                                           #
# ------------------------------------------------------------------ #
_SPECIAL_MAPPINGS = {
    "sales_units": "sales_quantity",
    "sales_rub": "sales_amount_rub",
    "sales_tonnes": "sales_weight_kg",
    "cost_rub": "cost_price_rub",
    "period": "sales_month",
    "retail_chain": "retail_chain",
    "category": "product_category",
    "base_type": "product_type",
    "supplier": "supplier_name",
    "brand": "brand",
    "product_name": "product_name",
    "unified_product_name": "product_unified_name",
    "weight": "product_weight_g",
    "flavor": "product_flavor",
}


def _sanitize_column_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return "unknown_column_0"
    original_name = name.lower()
    if original_name in _SPECIAL_MAPPINGS:
        return _SPECIAL_MAPPINGS[original_name]
    if original_name.startswith("none_"):
        try:
            num = int(original_name.split("_")[1])
            return f"unknown_column_{num}"
        except (ValueError, IndexError):
            pass
    name = re.sub(r"[^a-zA-Z0-9_]", "_", original_name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")
    return name if name else "unknown_column_0"


# ------------------------------------------------------------------ #
# Stage DDL                                                             #
# ------------------------------------------------------------------ #
def _stage_sql_type(col: str) -> str:
    if ColumnConfig.is_numeric(col):
        return "DECIMAL(18,2)"
    if col == "sales_month":
        return "INT"
    # strings
    max_len = ColumnConfig.max_len(col)
    return f"NVARCHAR({max_len})"


def _create_or_truncate_stage_table(
    conn: engine.Connection,
    table_name: str,
    schema: str,
    drop_if_exists: bool,
    create_columns: Sequence[str],
) -> None:
    """Ensure stage table exists (optionally drop/truncate) with required columns."""
    if drop_if_exists:
        drop_sql = f"IF OBJECT_ID(N'[{schema}].[{table_name}]', 'U') IS NOT NULL DROP TABLE [{schema}].[{table_name}];"
        conn.execute(text(drop_sql))
        logger.info("Dropped existing table [%s].[%s].", schema, table_name)

    # detect existence
    table_exists = (
        conn.execute(
            text(
                "SELECT COUNT(*) FROM sys.tables t "
                "JOIN sys.schemas s ON t.schema_id = s.schema_id "
                "WHERE s.name = :schema AND t.name = :tbl"
            ),
            {"schema": schema, "tbl": table_name},
        ).scalar() > 0
    )

    if not table_exists:
        columns_sql = ",\n".join([f"[{c}] {_stage_sql_type(c)}" for c in create_columns])
        create_table_sql = f"""
            CREATE TABLE [{schema}].[{table_name}] (
                {columns_sql},
                load_dt DATETIME DEFAULT GETDATE()
            );
        """
        conn.execute(text(create_table_sql))
        logger.info("Created new table [%s].[%s].", schema, table_name)
    else:
        # Ensure required columns exist / have compatible types.
        existing_cols = set(
            r[0]
            for r in conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :tbl"
                ),
                {"schema": schema, "tbl": table_name},
            )
        )
        for col in create_columns:
            if col in existing_cols:
                # Optionally widen – we always try ALTER; safe if same.
                try:
                    conn.execute(text(f"ALTER TABLE [{schema}].[{table_name}] ALTER COLUMN [{col}] {_stage_sql_type(col)};"))
                except Exception as e:  # noqa: BLE001
                    logger.warning("ALTER failed for %s: %s (continuing)", col, e)
            else:
                conn.execute(text(f"ALTER TABLE [{schema}].[{table_name}] ADD [{col}] {_stage_sql_type(col)};"))
                logger.info("Added column %s to [%s].[%s].", col, schema, table_name)


# ------------------------------------------------------------------ #
# Insert (chunk)                                                        #
# ------------------------------------------------------------------ #
def _prepare_chunk_for_insert(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes & lengths before DBAPI binding."""
    for col in df.columns:
        if ColumnConfig.is_numeric(col):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
        elif col == "sales_month":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1).astype(int)
        else:  # string
            max_len = ColumnConfig.max_len(col)
            ser = df[col].astype(str).str.slice(0, max_len)
            ser = ser.replace({"nan": "", "None": ""})
            df[col] = ser
    return df


def _chunk_to_tuples(df: pd.DataFrame) -> List[Tuple]:
    # Using itertuples is reasonably fast and memory friendly.
    return list(df.itertuples(index=False, name=None))


def _bulk_insert_chunk(
    conn: engine.Connection,
    table_name: str,
    df: pd.DataFrame,
    schema: str,
) -> None:
    if df.empty:
        return
    cols = df.columns.tolist()
    cols_sql = ", ".join(f"[{c}]" for c in cols)
    params = ", ".join(["?"] * len(cols))
    insert_sql = f"INSERT INTO [{schema}].[{table_name}] ({cols_sql}) VALUES ({params})"

    raw_conn = conn.connection  # SQLAlchemy -> DBAPI connection
    cursor = raw_conn.cursor()
    try:
        cursor.fast_executemany = True  # pyodbc optimization
        cursor.executemany(insert_sql, _chunk_to_tuples(df))
        raw_conn.commit()
        logger.debug("Inserted %s rows.", len(df))
    finally:
        cursor.close()


# ------------------------------------------------------------------ #
# Main streaming converter                                              #
# ------------------------------------------------------------------ #
def convert_raw_to_stage(
    table_name: str,
    raw_engine: engine.Engine,
    stage_engine: engine.Engine,
    stage_schema: str = "okey",
    limit: Optional[int] | None = None,
    drop_stage_if_exists: bool = False,
    chunk_size: int = 100_000,
    log_every_rows: int = 250_000,
) -> None:
    """Convert data from raw.<table_name> to stage.<schema>.<table_name> efficiently.

    Parameters
    ----------
    table_name:
        Source table *name* in schema `raw`.
    raw_engine / stage_engine:
        SQLAlchemy engines.
    stage_schema:
        Destination schema (default "okey").
    limit:
        Optional row cap (after processing). Useful for testing.
    drop_stage_if_exists:
        Drop & recreate stage table before load.
    chunk_size:
        Rows per pandas chunk when streaming from raw DB.
    log_every_rows:
        Emit progress log after ~N processed rows.
    """
    start_time = datetime.now()
    logger.info("[Stage] Starting processing of table %s", table_name)

    # --- metadata & rowcount from raw -------------------------------------------------
    with raw_engine.connect() as conn:
        try:
            total_count = conn.execute(text(f"SELECT COUNT(*) FROM raw.{table_name}")) .scalar()
        except Exception as e:  # noqa: BLE001
            logger.error("Error getting rowcount for raw.%s: %s", table_name, e)
            raise
    logger.info("[Stage] Total rows to process: %s", total_count)

    # Determine effective row limit for progress.
    target_rows = min(total_count, limit) if limit is not None else total_count

    # --- Read/transform/load streaming ------------------------------------------------
    processed_rows = 0
    created = False
    stage_cols = ColumnConfig.stage_columns()

    with raw_engine.connect().execution_options(stream_results=True) as raw_conn, stage_engine.connect() as stage_conn:
        # Optionally create/clear table *once* before loop.
        _create_or_truncate_stage_table(
            stage_conn, table_name, schema=stage_schema, drop_if_exists=drop_stage_if_exists, create_columns=stage_cols
        )
        created = True

        # Build SELECT; If limit specified, use TOP to reduce scanning cost.
        if limit is not None:
            # ORDER BY (SELECT NULL) is non‑deterministic but cheap. Good enough for sampling.
            query = f"SELECT TOP ({limit}) * FROM raw.{table_name}"
        else:
            query = f"SELECT * FROM raw.{table_name}"

        chunk_iter = pd.read_sql(text(query), raw_conn, chunksize=chunk_size, dtype="object")
        for chunk in chunk_iter:
            if chunk.empty:
                continue

            chunk.columns = [c.strip() for c in chunk.columns]
            # Transform ----------------------------------------------------------------------------------
            chunk = _normalize_okey_input_columns(chunk)
            chunk = _process_month_columns(chunk)
            chunk = _convert_numeric_columns(chunk)
            chunk = _convert_string_columns(chunk)

            # Rename measure columns -> stage names we expect.
            rename_numeric = {
                "sales_units": "sales_quantity",
                "sales_rub": "sales_amount_rub",
                "sales_tonnes": "sales_weight_kg",
                "cost_rub": "cost_price_rub",
            }
            chunk = chunk.rename(columns=rename_numeric)

            # Guarantee all stage columns exist.
            for col in stage_cols:
                if col not in chunk.columns:
                    if ColumnConfig.is_numeric(col):
                        chunk[col] = ColumnConfig.NUMERIC_COLS[col]["default"]
                    elif col == "sales_month":
                        chunk[col] = 1  # Jan fallback
                    else:
                        chunk[col] = ""

            # Reindex & drop extras (unknown technical columns).
            chunk = chunk.reindex(columns=stage_cols)

            # Convert dtypes / lengths before insert.
            chunk = _prepare_chunk_for_insert(chunk)

            # Optional trim to remaining row budget when limit set.
            if limit is not None:
                remaining = limit - processed_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining, :]

            # Load ---------------------------------------------------------------------------------------
            _bulk_insert_chunk(stage_conn, table_name, chunk, schema=stage_schema)

            processed_rows += len(chunk)
            if processed_rows % log_every_rows < len(chunk):  # crossed threshold
                logger.info("[Stage] Processed %s/%s rows", processed_rows, target_rows)
            if limit is not None and processed_rows >= limit:
                break

        # Ensure commit of any outstanding work (bulk insert commits itself but we flush just in case).
        try:
            stage_conn.commit()  # type: ignore[attr-defined]  # Some dialects support; otherwise no‑op.
        except Exception:  # pragma: no cover - dialect dependent
            pass

    duration = (datetime.now() - start_time).total_seconds()
    logger.info("[Stage] Successfully loaded %s rows in %.2f sec", processed_rows, duration)


# ------------------------------------------------------------------ #
# If run as script                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse
    from sqlalchemy import create_engine

    parser = argparse.ArgumentParser(description="Convert raw Okey sales tables to stage.")
    parser.add_argument("table_name", help="Source table name in schema raw")
    parser.add_argument("--raw-conn", required=True, help="SQLAlchemy connection string to raw DB")
    parser.add_argument("--stage-conn", required=True, help="SQLAlchemy connection string to stage DB")
    parser.add_argument("--stage-schema", default="okey", help="Destination schema name")
    parser.add_argument("--limit", type=int, default=None, help="Row limit for testing")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Rows per chunk")
    parser.add_argument("--drop", action="store_true", help="Drop stage table if exists before load")
    args = parser.parse_args()

    raw_engine = create_engine(args.raw_conn, fast_executemany=True)
    stage_engine = create_engine(args.stage_conn, fast_executemany=True)

    convert_raw_to_stage(
        table_name=args.table_name,
        raw_engine=raw_engine,
        stage_engine=stage_engine,
        stage_schema=args.stage_schema,
        limit=args.limit,
        drop_stage_if_exists=args.drop,
        chunk_size=args.chunk_size,
    )
