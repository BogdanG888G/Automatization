import logging
import re
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy import engine, exc, text

logger = logging.getLogger(__name__)


class ColumnConfig:
    """Column configuration with data types and transformations for Pyaterochka."""
    NUMERIC_COLS = {
        "sales_amount_rub": {"dtype": "float64", "default": 0.0},
        "sales_quantity": {"dtype": "float64", "default": 0.0},
        "sales_weight_kg": {"dtype": "float64", "default": 0.0},
        "cost_price_rub": {"dtype": "float64", "default": 0.0},
    }

    RENAME_MAP = {
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
# Helpers                                                            #
# ------------------------------------------------------------------ #
def _process_month_column(df: pd.DataFrame) -> pd.DataFrame:
    """Превращаем 'сен.24' в колонку sales_month (тип datetime64[ns])."""
    if "period" not in df.columns and "sales_month" not in df.columns:
        raise ValueError("Neither 'period' nor 'sales_month' column found in DataFrame")

    month_col = "period" if "period" in df.columns else "sales_month"

    month_map = {
        "янв": "01", "фев": "02", "мар": "03", "апр": "04",
        "май": "05", "июн": "06", "июл": "07", "авг": "08",
        "сен": "09", "окт": "10", "ноя": "11", "дек": "12",
    }

    def parse_month(month_str):
        if pd.isna(month_str):
            return "2024-01-01"
        try:
            s = str(month_str).strip().lower()
            month_part = s[:3]
            year_part = "20" + s[-2:] if len(s) > 3 else "2024"
            return f"{year_part}-{month_map.get(month_part, '01')}-01"
        except Exception:  # noqa: BLE001
            return "2024-01-01"

    df["sales_month"] = df[month_col].apply(parse_month)
    df["sales_month"] = pd.to_datetime(df["sales_month"], errors="coerce").fillna(pd.to_datetime("2024-01-01"))

    if month_col != "sales_month":
        df = df.drop(columns=[month_col], errors="ignore")

    return df


def _clean_numeric_series(s: pd.Series, multiply: float | None = None) -> pd.Series:
    """Очистка числовых значений (запятые, пробелы, мусор), с опциональным умножением."""
    # работаем в строках, но сохраняем NaN
    s = s.astype("string")
    # заменяем запятую на точку (десятичный разделитель)
    s = s.str.replace(",", ".", regex=False)
    # оставляем цифры, точку, знак, экспоненту
    s = s.str.replace(r"[^\d.eE+\-]", "", regex=True)
    # пустые -> 0
    s = s.fillna("0")
    s = s.replace("", "0")
    out = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if multiply is not None:
        out = out * multiply
    return out


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Конвертируем числовые колонки исходного формата (raw)."""
    logger.info("Original columns: %s", df.columns.tolist())

    if "sales_units" in df.columns:
        df["sales_units"] = _clean_numeric_series(df["sales_units"])
    if "sales_rub" in df.columns:
        df["sales_rub"] = _clean_numeric_series(df["sales_rub"])
    if "sales_tonnes" in df.columns:
        # тонны -> кг
        df["sales_tonnes"] = _clean_numeric_series(df["sales_tonnes"], multiply=1000.0)
    if "cost_rub" in df.columns:
        df["cost_rub"] = _clean_numeric_series(df["cost_rub"])

    logger.info("Columns after numeric conversion: %s", df.columns.tolist())
    return df


def _convert_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Конвертируем строковые RU-колонки в EN-названия с обрезкой и очисткой.
    Числовые RU-колонки не трогаем (они переименуются позже через sanitize).
    """
    df.columns = [col.strip() for col in df.columns]

    for ru_col, en_col in ColumnConfig.RENAME_MAP.items():
        if ru_col not in df.columns:
            continue

        # пропускаем числовые целевые колонки — их обработали отдельно
        if en_col in ColumnConfig.NUMERIC_COLS:
            continue

        max_len = ColumnConfig.STRING_COL_LENGTHS.get(en_col, 255)

        try:
            ser = df[ru_col].astype("string")  # pandas StringDtype
            ser = ser.str.normalize("NFKC")
            ser = ser.str.replace(r"[\x00-\x1F\x7F-\x9F]", "", regex=True)
            ser = ser.str.strip()
            ser = ser.fillna("")
            ser = ser.str.slice(0, max_len)
            df[en_col] = ser

            if ru_col != en_col:
                df.drop(columns=[ru_col], inplace=True)

            too_long_mask = df[en_col].astype(str).str.len() > max_len
            if too_long_mask.any():
                logger.warning("Some values in %s exceed max length %s after truncation", en_col, max_len)
        except Exception as e:  # noqa: BLE001
            logger.error("Error converting string column %s: %s", ru_col, e)
            raise

    return df


def _sanitize_column_name(name: str) -> str:
    """Normalize column names to stage naming."""
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
# Stage DDL                                                          #
# ------------------------------------------------------------------ #
def _create_stage_table(
    conn: engine.Connection,
    table_name: str,
    df: pd.DataFrame,
    schema: str = "pyaterochka",
    drop_if_exists: bool = False,  # <<< опционально жёстко пересоздаём
) -> None:
    """Создаём или (опционально) пересоздаём stage-таблицу под DataFrame."""

    if drop_if_exists:
        drop_sql = f"IF OBJECT_ID(N'[{schema}].[{table_name}]', 'U') IS NOT NULL DROP TABLE [{schema}].[{table_name}];"
        conn.execute(text(drop_sql))
        logger.info("Dropped existing table [%s].[%s].", schema, table_name)

    # есть ли таблица
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

    # определить типы колонок
    safe_columns = []
    logger.info("DF columns before stage load: %s", df.columns.tolist())
    logger.info("Max string lengths by column ->")

    for col in df.columns:
        if col in ColumnConfig.NUMERIC_COLS:
            col_type = "DECIMAL(18,2)"
        elif "date" in col or "month" in col:
            col_type = "DATE"
        else:
            max_len = ColumnConfig.STRING_COL_LENGTHS.get(col, 255)
            col_type = f"NVARCHAR({max_len})"
            max_val_len = int(df[col].astype(str).str.len().max())
            logger.info("  %s: data_max=%s -> using %s", col, max_val_len, col_type)
        safe_columns.append((col, col_type))

    if not safe_columns:
        raise ValueError("No valid columns to create table")

    if not table_exists:
        # создать новую
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
        # синхронизируем схему
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
# Insert                                                             #
# ------------------------------------------------------------------ #
def _bulk_insert_data(
    conn: engine.Connection,
    table_name: str,
    df: pd.DataFrame,
    schema: str = "pyaterochka",
    debug_fallback: bool = True,  # <<< включаем построчный fallback при ошибке
) -> None:
    """Вставляем данные (однократно) в stage-таблицу."""

    if df.empty:
        logger.warning("Empty DataFrame, skipping insert.")
        return

    # столбцы в нужном порядке
    safe_columns = [_sanitize_column_name(col) for col in df.columns if col]
    if not safe_columns:
        raise ValueError("No valid columns for insertion")

    # если таблицы нет (на всякий случай) -> 0
    try:
        row_count_result = conn.execute(
            text(f"SELECT COUNT(*) FROM [{schema}].[{table_name}]")
        )
        row_count = row_count_result.scalar()
    except exc.DBAPIError:
        row_count = 0

    if row_count > 0:
        logger.info(
            "Table [%s].[%s] already contains data (%s rows), skipping insert.",
            schema, table_name, row_count,
        )
        return

    # подготовка данных
    data = []
    for row in df.itertuples(index=False):
        processed_row = []
        for val, col in zip(row, df.columns):
            if col in ColumnConfig.NUMERIC_COLS:
                processed_val = float(val) if pd.notna(val) else 0.0
            elif "date" in col or "month" in col:
                if pd.notna(val):
                    dt = pd.to_datetime(val, errors="coerce")
                    processed_val = dt.date() if not pd.isna(dt) else None
                else:
                    processed_val = None
            else:
                max_len = ColumnConfig.STRING_COL_LENGTHS.get(col, 255)
                if pd.isna(val):
                    processed_val = ""
                else:
                    processed_val = str(val)[:max_len]
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
# Main                                                               #
# ------------------------------------------------------------------ #
def convert_raw_to_stage(
    table_name: str,
    raw_engine: engine.Engine,
    stage_engine: engine.Engine,
    stage_schema: str = "pyaterochka",
    limit: int | None = None,
    drop_stage_if_exists: bool = False,  # <<< удобно когда нет таблицы / хотим пересоздать
) -> None:
    """
    Конвертируем данные из raw.<table_name> в stage.<schema>.<table_name> (Pyaterochka).
    """

    try:
        start_time = datetime.now()
        logger.info("[Stage] Starting processing of table %s", table_name)

        # --- метаданные raw
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

                total_count = conn.execute(
                    text(f"SELECT COUNT(*) FROM raw.{table_name}")
                ).scalar()
                logger.info("[Stage] Total rows to process: %s", total_count)
            except Exception as e:  # noqa: BLE001
                logger.error("Error getting table metadata: %s", e)
                raise

        # --- читаем данные чанками
        chunks: List[pd.DataFrame] = []
        query = f"SELECT * FROM raw.{table_name}"
        if limit is not None:
            # ORDER BY (SELECT NULL) -> репродуцируем случайный? безопасный порядок
            query += f" ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"

        with raw_engine.connect().execution_options(stream_results=True) as conn:
            try:
                for chunk in pd.read_sql(
                    text(query),
                    conn,
                    chunksize=100000,
                    dtype="object",
                ):
                    chunk.columns = [col.strip() for col in chunk.columns]
                    logger.info("Processing chunk with %s rows", len(chunk))

                    try:
                        chunk = _process_month_column(chunk)
                        chunk = _convert_numeric_columns(chunk)
                        chunk = _convert_string_columns(chunk)
                        chunks.append(chunk)
                    except Exception as e:  # noqa: BLE001
                        logger.error("Error processing chunk: %s", e)
                        raise

                    processed_count = sum(len(c) for c in chunks)
                    logger.info(
                        "[Stage] Processed %s/%s rows",
                        processed_count,
                        limit if limit is not None else total_count,
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

        # дроп технические неизвестные
        columns_to_drop = [
            col
            for col in df.columns
            if col.startswith("none_") or col.startswith("unknown_column_")
        ]
        df = df.drop(columns=columns_to_drop, errors="ignore")

        logger.info("Final DataFrame shape after dropping unused columns: %s", df.shape)

        if df.empty or len(df.columns) == 0:
            raise ValueError("DataFrame contains no data or columns after processing.")

        # нормализуем имена для stage
        df.columns = [_sanitize_column_name(col) for col in df.columns]

        # --- загрузка в stage
        with stage_engine.connect() as conn:
            trans = conn.begin()
            try:
                _create_stage_table(
                    conn,
                    table_name,
                    df,
                    schema=stage_schema,
                    drop_if_exists=drop_stage_if_exists,
                )
                _bulk_insert_data(conn, table_name, df, schema=stage_schema, debug_fallback=True)
                trans.commit()
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    "[Stage] Successfully loaded %s rows in %.2f sec",
                    len(df),
                    duration,
                )
            except Exception as e:  # noqa: BLE001
                trans.rollback()
                logger.error("Error loading to stage: %s", e)
                raise

    except Exception as e:  # noqa: BLE001
        logger.error("[Stage ERROR] Error processing table %s: %s", table_name, e)
        raise
