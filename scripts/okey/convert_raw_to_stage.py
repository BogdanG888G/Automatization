from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List

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

    # первичное RU->eng нормализационное отображение (до унификации между сетями)
    # сюда включаем разные варианты написания колонок из Excel/CSV
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

    # окончательное отображение в stage (то, чем будут называться колонки в DWH)
    RENAME_MAP = {
        "period": "period",
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

    # макс. длины строковых колонок в stage
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


# ------------------------------------------------------------------ #
# Normalization helpers                                               #
# ------------------------------------------------------------------ #
def _normalize_okey_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Okey column headers to internal English base names.

    Works tolerant to punctuation, spaces & case. Example:
    "Категория 2" -> "base_type"; "Период" -> "period".
    """
    normed = []
    for col in df.columns:
        c = str(col).strip().lower()
        # унифицируем: пробелы/знаки -> _
        c_key = re.sub(r"\s+", "_", c)
        c_key = re.sub(r"[^a-zа-я0-9_\.]", "_", c_key)
        # уберём двойные подчёркивания / запятые в коды
        c_key = c_key.replace(",", "_").replace("..", ".")
        c_key = re.sub(r"_+", "_", c_key).strip("_")

        mapped = ColumnConfig.RAW_RU_NORMALIZE.get(c_key, c_key)
        normed.append(mapped)

    df.columns = normed
    return df


# ------------------------------------------------------------------ #
# Month parsing                                                        #
# ------------------------------------------------------------------ #
_RU_MONTHS_SHORT = {
    "ян": 1, "янв": 1,
    "фе": 2, "фев": 2,
    "мар": 3,
    "апр": 4,
    "май": 5,
    "июн": 6,
    "июл": 7,
    "авг": 8,
    "сен": 9, "сент": 9,
    "окт": 10,
    "ноя": 11,
    "дек": 12,
}

def _process_month_columns(
    df: pd.DataFrame,
    *_,            # примем лишние позиционные аргументы, чтобы не падало
    drop_source_cols: bool = True,  # совместимость со старым вызовом; можно не передавать
) -> pd.DataFrame:
    """
    Сформировать колонку sales_month (INT 1-12) из:
      - sale_month (если есть числовой)
      - period (дата или 'сен.24' / '01.09.2024')
    Полностью убрать колонку sale_month (сырьё). period тоже убираем, если drop_source_cols=True.
    """
    month_int = pd.Series([np.nan] * len(df), dtype="float64")

    # 1. sale_month как есть
    if "sale_month" in df.columns:
        m = pd.to_numeric(df["sale_month"], errors="coerce")
        month_int = m.where(~m.isna(), month_int)

    # 2. period -> месяц
    if "period" in df.columns:
        parsed = []
        for v in df["period"]:
            if pd.isna(v):
                parsed.append(np.nan)
                continue
            # пробуем как дату
            try:
                dt = pd.to_datetime(v, dayfirst=True, errors="raise")
                parsed.append(dt.month)
                continue
            except Exception:
                pass
            # пробуем рус. аббрев.
            s = str(v).strip().lower()
            s_clean = s.replace(".", "").replace(" ", "")
            token = s_clean[:3]
            month_val = None
            for k, num in _RU_MONTHS_SHORT.items():
                if token.startswith(k):
                    month_val = num
                    break
            parsed.append(month_val if month_val is not None else np.nan)
        parsed = pd.to_numeric(pd.Series(parsed), errors="coerce")
        month_int = month_int.where(~month_int.isna(), parsed)

    # 3. fallback
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
    s = s.str.replace(r"[^\d.eE+\-]", "", regex=True)  # strip junk
    s = s.fillna("0")
    s = s.replace("", "0")
    out = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if multiply is not None:
        out = out * multiply
    return out


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert expected numeric raw columns to numeric."""
    logger.info("Original columns: %s", df.columns.tolist())

    if "sales_units" in df.columns:
        df["sales_units"] = _clean_numeric_series(df["sales_units"])
    if "sales_rub" in df.columns:
        df["sales_rub"] = _clean_numeric_series(df["sales_rub"])
    if "sales_tonnes" in df.columns:
        df["sales_tonnes"] = _clean_numeric_series(df["sales_tonnes"], multiply=1000.0)  # tons->kg
    if "cost_rub" in df.columns:
        df["cost_rub"] = _clean_numeric_series(df["cost_rub"])

    logger.info("Columns after numeric conversion: %s", df.columns.tolist())
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

        max_len = ColumnConfig.STRING_COL_LENGTHS.get(stage_col, 255)
        try:
            ser = df[raw_col].astype("string")
            ser = ser.str.normalize("NFKC")
            ser = ser.str.replace(r"[\x00-\x1F\x7F-\x9F]", "", regex=True)
            ser = ser.str.strip().fillna("")
            ser = ser.str.slice(0, max_len)
            df[stage_col] = ser
            if raw_col != stage_col:
                df.drop(columns=[raw_col], inplace=True)
            if df[stage_col].astype(str).str.len().gt(max_len).any():
                logger.warning("Some values in %s exceed max length %s after truncation", stage_col, max_len)
        except Exception as e:  # noqa: BLE001
            logger.error("Error converting string column %s: %s", raw_col, e)
            raise

    return df


# ------------------------------------------------------------------ #
# Stage column name sanitizer                                           #
# ------------------------------------------------------------------ #
def _sanitize_column_name(name: str) -> str:
    """Normalize column names to final stage naming."""
    if not name or not isinstance(name, str):
        return "unknown_column_0"

    original_name = name.lower()

    special_mappings = {
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

    if original_name in special_mappings:
        return special_mappings[original_name]

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
def _create_stage_table(
    conn: engine.Connection,
    table_name: str,
    df: pd.DataFrame,
    schema: str = "okey",
    drop_if_exists: bool = False,
) -> None:
    """Create (or optionally drop+recreate) stage table to match DataFrame."""

    if drop_if_exists:
        drop_sql = f"IF OBJECT_ID(N'[{schema}].[{table_name}]', 'U') IS NOT NULL DROP TABLE [{schema}].[{table_name}];"
        conn.execute(text(drop_sql))
        logger.info("Dropped existing table [%s].[%s].", schema, table_name)

    table_exists = (
        conn.execute(
            text(
                "SELECT COUNT(*) FROM sys.tables t "
                "JOIN sys.schemas s ON t.schema_id = s.schema_id "
                "WHERE s.name = :schema AND t.name = :tbl"
            ),
            {"schema": schema, "tbl": table_name},
        ).scalar()
        > 0
    )

    safe_columns = []
    logger.info("DF columns before stage load: %s", df.columns.tolist())
    logger.info("Max string lengths by column ->")

    for col in df.columns:
        if col in ColumnConfig.NUMERIC_COLS:
            col_type = "DECIMAL(18,2)"
        elif col == "sales_month":
            col_type = "INT"
        elif "date" in col or "month" in col:
            col_type = "DATE"
        else:
            max_len = ColumnConfig.STRING_COL_LENGTHS.get(col, 255)
            col_type = f"NVARCHAR({max_len})"
            max_val_len = int(df[col].astype(str).str.len().max()) if col in df.columns else 0
            logger.info("  %s: data_max=%s -> using %s", col, max_val_len, col_type)
        safe_columns.append((col, col_type))

    if not safe_columns:
        raise ValueError("No valid columns to create table")

    if not table_exists:
        columns_sql = ",\n".join([f"[{col}] {col_type}" for col, col_type in safe_columns])
        create_table_sql = f"""
            CREATE TABLE [{schema}].[{table_name}] (
                {columns_sql},
                load_dt DATETIME DEFAULT GETDATE()
            );
        """
        try:
            conn.execute(text(create_table_sql))
            logger.info("Created new table [%s].[%s].", schema, table_name)
        except Exception as e:  # noqa: BLE001
            logger.error("Error creating table: %s\nSQL:\n%s", e, create_table_sql)
            raise
    else:
        for col, col_type in safe_columns:
            try:
                col_exists = (
                    conn.execute(
                        text(
                            "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
                            "WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :tbl AND COLUMN_NAME = :col"
                        ),
                        {"schema": schema, "tbl": table_name, "col": col},
                    ).scalar()
                    > 0
                )

                if col_exists:
                    alter_sql = f"ALTER TABLE [{schema}].[{table_name}] ALTER COLUMN [{col}] {col_type};"
                    conn.execute(text(alter_sql))
                    logger.info("Altered column %s in [%s].[%s].", col, schema, table_name)
                else:
                    add_sql = f"ALTER TABLE [{schema}].[{table_name}] ADD [{col}] {col_type};"
                    conn.execute(text(add_sql))
                    logger.info("Added column %s to [%s].[%s].", col, schema, table_name)
            except Exception as e:  # noqa: BLE001
                logger.error("Error altering column %s: %s", col, e)
                raise


# ------------------------------------------------------------------ #
# Insert                                                                #
# ------------------------------------------------------------------ #
def _bulk_insert_data(
    conn: engine.Connection,
    table_name: str,
    df: pd.DataFrame,
    schema: str = "okey",
    debug_fallback: bool = True,
) -> None:
    """Bulk insert data once table is ready."""
    if df.empty:
        logger.warning("Empty DataFrame, skipping insert.")
        return

    safe_columns = [_sanitize_column_name(col) for col in df.columns if col]
    if not safe_columns:
        raise ValueError("No valid columns for insertion")

    try:
        row_count_result = conn.execute(text(f"SELECT COUNT(*) FROM [{schema}].[{table_name}]"))
        row_count = row_count_result.scalar()
    except exc.DBAPIError:  # table may not exist yet
        row_count = 0

    if row_count > 0:
        logger.info("Table [%s].[%s] already contains data (%s rows), skipping insert.", schema, table_name, row_count)
        return

    data = []
    for row in df.itertuples(index=False):
        processed_row = []
        for val, col in zip(row, df.columns):
            if col in ColumnConfig.NUMERIC_COLS:
                processed_val = float(val) if pd.notna(val) else 0.0
            elif col == "sales_month":
                processed_val = int(val) if pd.notna(val) else None
            elif "date" in col or "month" in col:
                if pd.notna(val):
                    dt = pd.to_datetime(val, errors="coerce")
                    processed_val = dt.date() if not pd.isna(dt) else None
                else:
                    processed_val = None
            else:
                max_len = ColumnConfig.STRING_COL_LENGTHS.get(col, 255)
                processed_val = "" if pd.isna(val) else str(val)[:max_len]
            processed_row.append(processed_val)
        data.append(tuple(processed_row))

    cols = ", ".join([f"[{col}]" for col in safe_columns])
    params = ", ".join(["?"] * len(safe_columns))

    raw_conn = conn.connection  # SQLAlchemy -> DBAPI
    cursor = raw_conn.cursor()
    insert_sql = f"INSERT INTO [{schema}].[{table_name}] ({cols}) VALUES ({params})"

    try:
        cursor.fast_executemany = True
        cursor.executemany(insert_sql, data)
        raw_conn.commit()
        logger.info("Inserted %s rows into [%s].[%s].", len(df), schema, table_name)
    except Exception as e:  # noqa: BLE001
        raw_conn.rollback()
        logger.error("Bulk insert failed: %s\nSQL: %s", e, insert_sql)
        if debug_fallback:
            logger.error("Falling back to row-wise debug insert to isolate bad row...")
            for i, row_data in enumerate(data):
                try:
                    cursor.execute(insert_sql, row_data)
                except Exception as row_e:  # noqa: BLE001
                    logger.error("Row %s failed: %s; data=%s", i, row_e, row_data)
                    raise
            raw_conn.commit()
            logger.info("Row-wise fallback insert completed.")
        else:
            raise
    finally:
        cursor.close()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #
def convert_raw_to_stage(
    table_name: str,
    raw_engine: engine.Engine,
    stage_engine: engine.Engine,
    stage_schema: str = "okey",
    limit: int | None = None,
    drop_stage_if_exists: bool = False,
) -> None:
    """Convert data from raw.<table_name> to stage.<schema>.<table_name> for Okey."""

    try:
        start_time = datetime.now()
        logger.info("[Stage] Starting processing of table %s", table_name)

        # --- grab metadata from raw
        with raw_engine.connect() as conn:
            try:
                result = conn.execute(
                    text(
                        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                        "WHERE TABLE_SCHEMA = 'raw' AND TABLE_NAME = :table_name"
                    ),
                    {"table_name": table_name},
                )
                actual_columns = [row[0] for row in result]
                logger.info("Actual columns in raw table: %s", actual_columns)

                total_count = conn.execute(text(f"SELECT COUNT(*) FROM raw.{table_name}")).scalar()
                logger.info("[Stage] Total rows to process: %s", total_count)
            except Exception as e:  # noqa: BLE001
                logger.error("Error getting table metadata: %s", e)
                raise

        # --- read data in chunks
        chunks: List[pd.DataFrame] = []
        query = f"SELECT * FROM raw.{table_name}"
        if limit is not None:
            query += f" ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"

        with raw_engine.connect().execution_options(stream_results=True) as conn:
            try:
                for chunk in pd.read_sql(text(query), conn, chunksize=100000, dtype="object"):
                    chunk.columns = [col.strip() for col in chunk.columns]
                    logger.info("Processing chunk with %s rows", len(chunk))

                    try:
                        chunk = _normalize_okey_input_columns(chunk)
                        chunk = _process_month_columns(chunk)
                        chunk = _convert_numeric_columns(chunk)
                        chunk = _convert_string_columns(chunk)
                        chunks.append(chunk)
                    except Exception as e:  # noqa: BLE001
                        logger.error("Error processing chunk: %s", e)
                        raise

                    processed_count = sum(len(c) for c in chunks)
                    logger.info(
                        "[Stage] Processed %s/%s rows", processed_count, limit if limit is not None else total_count
                    )
                    if limit is not None and processed_count >= limit:
                        break
            except Exception as e:  # noqa: BLE001
                logger.error("Error reading data: %s", e)
                raise

        if not chunks:
            logger.warning("[Stage] No data to process.")
            return

        df = pd.concat(chunks, ignore_index=True)
        if limit is not None:
            df = df.head(limit)

        # drop technical/unknown columns if any slipped through
        columns_to_drop = [c for c in df.columns if c.startswith("none_") or c.startswith("unknown_column_")]
        df = df.drop(columns=columns_to_drop, errors="ignore")

        logger.info("Final DataFrame shape after dropping unused columns: %s", df.shape)

        if df.empty or len(df.columns) == 0:
            raise ValueError("DataFrame contains no data or columns after processing.")

        # sanitize to final stage names (idempotent)
        df.columns = [_sanitize_column_name(col) for col in df.columns]

        # --- load to stage
        with stage_engine.connect() as conn:
            trans = conn.begin()
            try:
                _create_stage_table(conn, table_name, df, schema=stage_schema, drop_if_exists=drop_stage_if_exists)
                _bulk_insert_data(conn, table_name, df, schema=stage_schema, debug_fallback=True)
                trans.commit()
                duration = (datetime.now() - start_time).total_seconds()
                logger.info("[Stage] Successfully loaded %s rows in %.2f sec", len(df), duration)
            except Exception as e:  # noqa: BLE001
                trans.rollback()
                logger.error("Error loading to stage: %s", e)
                raise

    except Exception as e:  # noqa: BLE001
        logger.error("[Stage ERROR] Error processing table %s: %s", table_name, e)
        raise
