import re
import pandas as pd
import numpy as np
from sqlalchemy import text, engine
from datetime import datetime
from typing import List
import logging

logger = logging.getLogger(__name__)

class ColumnConfig:
    RENAME_MAP = {
        'списания,_руб.': 'writeoff_amount_rub',
        'списания,_шт.': 'writeoff_quantity',
        'продажи,_c_ндс': 'sales_amount_with_vat',
        'потери,_руб.': 'loss_amount_rub',
        'потери,шт': 'loss_quantity',
        'промо_продажи,_c_ндс': 'promo_sales_amount_with_vat',
        'маржа,_руб.': 'margin_amount_rub',
        'ср.цена_продажи': 'avg_sell_price',
        'ср.цена_покупки': 'avg_purchase_price',
        'уровень_3': 'level_3',
        'уровень_4': 'level_4',
        'уровень_5': 'level_5',
        'втм': 'vtm',
        'товар': 'product',
        'адрес': 'address',
        'код_товара': 'product_code',
        'магазины': 'stores',
        'количество': 'quantity_first_week',
        'себестоимость_с_ндс': 'cost_with_vat_first_week',
        'сумма_с_ндс': 'amount_with_vat_first_week',
        'количество.1': 'quantity_second_week',
        'себестоимость_с_ндс.1': 'cost_with_vat_second_week',
        'сумма_с_ндс.1': 'amount_with_vat_second_week',
        'количество.2': 'quantity_third_week',
        'себестоимость_с_ндс.2': 'cost_with_vat_third_week',
        'сумма_с_ндс.2': 'amount_with_vat_third_week',
        'количество.3': 'quantity_fourth_week',
        'себестоимость_с_ндс.3': 'cost_with_vat_fourth_week',
        'сумма_с_ндс.3': 'amount_with_vat_fourth_week',
        'количество.4': 'quantity_fifth_week',
        'себестоимость_с_ндс.4': 'cost_with_vat_fifth_week',
        'сумма_с_ндс.4': 'amount_with_vat_fifth_week',
        'количество.5': 'quantity_summary',
        'себестоимость_с_ндс.5': 'cost_with_vat_summary',
        'сумма_с_ндс.5': 'amount_with_vat_summary',
        'sale_year': 'sale_year',
        'sale_month': 'sale_month'
    }
    NUMERIC_COLS = {
        'writeoff_amount_rub',
        'writeoff_quantity',
        'sales_amount_with_vat',
        'loss_amount_rub',
        'loss_quantity',
        'promo_sales_amount_with_vat',
        'margin_amount_rub',
        'avg_sell_price',
        'avg_purchase_price',
    }

def sanitize_and_make_unique_columns(columns):
    seen = {}
    sanitized_columns = []

    def _sanitize_column_name(name: str) -> str:
        if not name or not isinstance(name, str):
            return ''
        original_name = name.lower()
        special_mappings = ColumnConfig.RENAME_MAP
        if original_name in special_mappings:
            return special_mappings[original_name]
        # Заменяем всё кроме латиницы, цифр, и _
        name = re.sub(r'[^a-z0-9_]', '_', original_name)
        name = re.sub(r'_{2,}', '_', name)
        return name.strip('_')

    for i, col in enumerate(columns):
        sanitized = _sanitize_column_name(col)
        if not sanitized:
            sanitized = f"col_{i}"
        base_name = sanitized
        count = seen.get(base_name, 0)
        if count > 0:
            sanitized = f"{base_name}_{count}"
        seen[base_name] = count + 1
        sanitized_columns.append(sanitized)

    return sanitized_columns

def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Приводим к lower и с заменой пробелов
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Ищем числовые колонки по переименованию и конвертим
    for ru_col, en_col in ColumnConfig.RENAME_MAP.items():
        if en_col in ColumnConfig.NUMERIC_COLS and ru_col in df.columns:
            df[en_col] = (
                df[ru_col]
                .astype(str)
                .str.replace(',', '.')
                .str.replace(r'[^\d.]', '', regex=True)
                .replace('', '0')
                .astype(float)
                .fillna(0)
            )
            df.drop(columns=[ru_col], inplace=True)
    return df

def _convert_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    for ru_col, en_col in ColumnConfig.RENAME_MAP.items():
        if ru_col in df.columns and en_col not in ColumnConfig.NUMERIC_COLS:
            df[en_col] = df[ru_col].astype('string').fillna('')
            if ru_col != en_col:
                df.drop(columns=[ru_col], inplace=True)
    return df

def _create_stage_table(conn: engine.Connection, table_name: str, df: pd.DataFrame, schema: str = 'diksi') -> None:
    # Обработка имен колонок и их типов
    new_columns = sanitize_and_make_unique_columns(list(df.columns))
    df.columns = new_columns

    if len(df.columns) != len(set(df.columns)):
        duplicates = [col for col in df.columns if df.columns.tolist().count(col) > 1]
        raise ValueError(f"Duplicate column names after sanitization: {duplicates}")

    col_defs = []
    for col in df.columns:
        col_type = 'FLOAT' if col in ColumnConfig.NUMERIC_COLS else 'NVARCHAR(255)'
        col_defs.append(f'[{col}] {col_type}')

    create_sql = f"""
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

    try:
        trans = conn.begin()
        conn.execute(text(create_sql))
        trans.commit()
        logger.info(f"Table [{schema}].[{table_name}] created or already exists")
    except Exception as e:
        if 'trans' in locals():
            trans.rollback()
        logger.error(f"Error creating table: {e}")
        raise

def _bulk_insert_data(conn: engine.Connection, table_name: str, df: pd.DataFrame, schema: str = 'diksi') -> None:
    if df.empty:
        logger.warning("Empty DataFrame, skipping insert")
        return
    
    safe_columns = sanitize_and_make_unique_columns(list(df.columns))
    df.columns = safe_columns

    cols_str = ', '.join([f'[{col}]' for col in safe_columns])
    params_str = ', '.join(['?'] * len(safe_columns))
    
    raw_conn = conn.connection
    cursor = raw_conn.cursor()
    try:
        cursor.fast_executemany = True
        insert_sql = f"INSERT INTO [{schema}].[{table_name}] ({cols_str}) VALUES ({params_str})"
        data = []
        for row in df.itertuples(index=False):
            processed_row = []
            for val, col in zip(row, safe_columns):
                if col in ColumnConfig.NUMERIC_COLS:
                    processed_val = float(val) if pd.notna(val) else 0.0
                else:
                    processed_val = str(val) if pd.notna(val) else ''
                processed_row.append(processed_val)
            data.append(tuple(processed_row))
        cursor.executemany(insert_sql, data)
        raw_conn.commit()
        logger.info(f"Inserted {len(df)} rows into [{schema}].[{table_name}]")
    except Exception as e:
        raw_conn.rollback()
        logger.error(f"Error inserting data: {e}")
        raise
    finally:
        cursor.close()

def convert_raw_to_stage(
    table_name: str,
    raw_engine: engine.Engine,
    stage_engine: engine.Engine,
    stage_schema: str = 'diksi',
    limit: int = None,
) -> None:
    try:
        start_time = datetime.now()
        logger.info(f"[Stage] Starting processing of table {table_name}")

        # Получаем колонки из raw таблицы
        with raw_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = 'raw' AND TABLE_NAME = :table_name"
                ),
                {'table_name': table_name}
            )
            actual_columns = [row[0].lower().replace(' ', '_') for row in result]
            logger.info(f"Actual columns in raw table: {actual_columns}")

            total_count = conn.execute(text(f"SELECT COUNT(*) FROM raw.{table_name}")).scalar()
            logger.info(f"[Stage] Total rows to process: {total_count}")

        query = f"SELECT * FROM raw.{table_name}"
        if limit is not None:
            query += f" ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"

        with stage_engine.connect() as stage_conn:
            # Создадим пустую таблицу по структуре из первого чанка
            first_chunk_processed = False

            with raw_engine.connect().execution_options(stream_results=True) as raw_conn:
                for chunk in pd.read_sql(text(query), raw_conn, chunksize=50000, dtype='object'):
                    chunk.columns = [col.lower().replace(' ', '_') for col in chunk.columns]
                    logger.info(f"Chunk received with columns: {chunk.columns.tolist()} and rows: {len(chunk)}")

                    chunk = _convert_numeric_columns(chunk)
                    chunk = _convert_string_columns(chunk)

                    chunk.columns = sanitize_and_make_unique_columns(list(chunk.columns))

                    if not first_chunk_processed:
                        _create_stage_table(stage_conn, table_name, chunk, stage_schema)
                        first_chunk_processed = True

                    try:
                        trans = stage_conn.begin()
                        _bulk_insert_data(stage_conn, table_name, chunk, stage_schema)
                        trans.commit()
                    except Exception as e:
                        trans.rollback()
                        logger.error(f"Error inserting chunk: {e}")
                        raise

                    logger.info(f"[Stage] Processed chunk with {len(chunk)} rows")

        logger.info(f"[Stage] Finished processing table {table_name} in {datetime.now() - start_time}")

    except Exception as e:
        logger.error(f"Failed to convert table {table_name}: {e}")
        raise
