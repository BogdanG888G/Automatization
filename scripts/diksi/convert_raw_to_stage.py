import re
import pandas as pd
import numpy as np
from sqlalchemy import text, engine
from datetime import datetime
from typing import List
import logging

logger = logging.getLogger(__name__)

# Пример настроек ColumnConfig — нужно заменить на твои реальные данные
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
        # Добавь остальные переименования
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
        # Добавь остальные числовые колонки
    }


def sanitize_and_make_unique_columns(columns):
    seen = {}
    sanitized_columns = []

    def _sanitize_column_name(name: str) -> str:
        if not name or not isinstance(name, str):
            return ''

        original_name = name.lower()

        special_mappings = {
            'списания,_руб.': 'writeoff_amount_rub',
            'списания,_шт.': 'writeoff_quantity',
            'продажи,_c_ндс': 'sales_amount_with_vat',
            'потери,_руб.': 'loss_amount_rub',
            'потери,шт': 'loss_quantity',
            'промо_продажи,_c_ндс': 'promo_sales_amount_with_vat',
            'маржа,_руб.': 'margin_amount_rub',
            'ср.цена_продажи': 'avg_sell_price',
            'ср.цена_покупки': 'avg_purchase_price'
        }

        if original_name in special_mappings:
            return special_mappings[original_name]

        # Заменяем все символы кроме латиницы, цифр и _ на _
        name = re.sub(r'[^a-z0-9_]', '_', original_name)
        name = re.sub(r'_{2,}', '_', name)  # Сокращаем несколько _ подряд до одного
        name = name.strip('_')

        return name

    for i, col in enumerate(columns):
        sanitized = _sanitize_column_name(col)
        if not sanitized:
            sanitized = f"col_{i}"  # Для пустых имен даем уникальное имя

        base_name = sanitized
        count = seen.get(base_name, 0)
        if count > 0:
            sanitized = f"{base_name}_{count}"
        seen[base_name] = count + 1

        sanitized_columns.append(sanitized)

    return sanitized_columns


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Original columns: {df.columns.tolist()}")
    
    # Нормализуем имена колонок
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Сопоставляем колонки с русскими именами, которые есть в DataFrame, с английскими для конвертации
    numeric_cols = {ru: en for ru, en in ColumnConfig.RENAME_MAP.items() if en in ColumnConfig.NUMERIC_COLS}
    
    for ru_col, en_col in numeric_cols.items():
        if ru_col in df.columns:
            try:
                df[en_col] = (
                    df[ru_col]
                    .astype(str)
                    .str.replace(',', '.')  # заменяем запятую на точку
                    .str.replace(r'[^\d.]', '', regex=True)  # удаляем все, кроме цифр и точки
                    .replace('', '0')  # пустые строки заменяем на '0'
                    .astype(np.float64)
                    .fillna(0)
                )
                df.drop(ru_col, axis=1, inplace=True)
            except Exception as e:
                logger.error(f"Error converting column {ru_col}: {e}")
                raise
    
    logger.info(f"Columns after numeric conversion: {df.columns.tolist()}")
    return df


def _convert_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    for ru_col, en_col in ColumnConfig.RENAME_MAP.items():
        if ru_col in df.columns:
            try:
                df[en_col] = df[ru_col].astype('string').fillna('')
                if ru_col != en_col:
                    df.drop(ru_col, axis=1, inplace=True)
            except Exception as e:
                logger.error(f"Error converting string column {ru_col}: {e}")
                raise
    
    return df


def _create_stage_table(conn: engine.Connection, table_name: str, df: pd.DataFrame, schema: str = 'diksi') -> None:
    logger.info(f"DataFrame columns before table creation: {df.columns.tolist()}")
    
    # Сначала корректно обработать названия колонок (sanitize_and_make_unique_columns принимает список, тут df.columns — Index)
    new_columns = sanitize_and_make_unique_columns(list(df.columns))
    df.columns = new_columns
    
    if len(df.columns) != len(set(df.columns)):
        duplicates = [col for col in df.columns if df.columns.tolist().count(col) > 1]
        raise ValueError(f"Duplicate column names after sanitization: {duplicates}")
    
    safe_columns = []
    for col in df.columns:
        if not col:
            continue
        col_type = 'FLOAT' if col in ColumnConfig.NUMERIC_COLS else 'NVARCHAR(255)'
        safe_columns.append(f'[{col}] {col_type}')
    
    if not safe_columns:
        raise ValueError("No valid columns to create table")
    
    create_table_sql = f"""
    IF NOT EXISTS (
        SELECT * FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE s.name = '{schema}' AND t.name = '{table_name}'
    )
    BEGIN
        CREATE TABLE [{schema}].[{table_name}] (
            {', '.join(safe_columns)}
        )
    END
    """
    
    try:
        trans = conn.begin()
        conn.execute(text(create_table_sql))
        trans.commit()
    except Exception as e:
        if 'trans' in locals():
            trans.rollback()
        logger.error(f"Error creating table: {e}\nSQL: {create_table_sql}")
        raise


def _bulk_insert_data(conn: engine.Connection, table_name: str, df: pd.DataFrame, schema: str = 'diksi') -> None:
    if df.empty:
        logger.warning("Empty DataFrame, skipping insert")
        return
    
    safe_columns = sanitize_and_make_unique_columns(list(df.columns))
    if not safe_columns:
        raise ValueError("No valid columns for insertion")
    
    # Проверяем наличие данных
    row_count_result = conn.execute(text(f"SELECT COUNT(*) FROM [{schema}].[{table_name}]"))
    row_count = row_count_result.scalar()
    if row_count > 0:
        logger.info(f"Table [{schema}].[{table_name}] already contains data ({row_count} rows), skipping insert")
        return
    
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
    
    cols = ', '.join([f'[{col}]' for col in safe_columns])
    params = ', '.join(['?'] * len(safe_columns))
    
    raw_conn = conn.connection
    cursor = raw_conn.cursor()
    try:
        cursor.fast_executemany = True
        insert_sql = f"INSERT INTO [{schema}].[{table_name}] ({cols}) VALUES ({params})"
        cursor.executemany(insert_sql, data)
        raw_conn.commit()
        logger.info(f"Inserted {len(df)} rows into [{schema}].[{table_name}]")
    except Exception as e:
        raw_conn.rollback()
        logger.error(f"Error inserting data: {e}\nSQL: {insert_sql}")
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
        
        with raw_engine.connect() as conn:
            try:
                result = conn.execute(
                    text(
                        f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                        f"WHERE TABLE_SCHEMA = 'raw' AND TABLE_NAME = :table_name"
                    ),
                    {'table_name': table_name},
                )
                actual_columns = [row[0].lower().replace(' ', '_') for row in result]
                logger.info(f"Actual columns in raw table: {actual_columns}")

                total_count = conn.execute(text(f"SELECT COUNT(*) FROM raw.{table_name}")).scalar()
                logger.info(f"[Stage] Total rows to process: {total_count}")
            except Exception as e:
                logger.error(f"Error getting table metadata: {e}")
                raise
        
        valid_rename_map = {
            ru: en for ru, en in ColumnConfig.RENAME_MAP.items() if ru in actual_columns
        }
        logger.info(f"Active RENAME_MAP: {valid_rename_map}")
        
        chunks: List[pd.DataFrame] = []
        query = f"SELECT * FROM raw.{table_name}"
        if limit is not None:
            query += f" ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
        
        with raw_engine.connect().execution_options(stream_results=True) as conn:
            try:
                for chunk in pd.read_sql(text(query), conn, chunksize=50000, dtype='object'):
                    chunk.columns = [col.lower().replace(' ', '_') for col in chunk.columns]
                    logger.info(f"Received chunk with columns: {chunk.columns.tolist()}")
                    
                    chunk = _convert_numeric_columns(chunk)
                    chunk = _convert_string_columns(chunk)
                    chunks.append(chunk)
                    
                    processed_count = sum(len(c) for c in chunks)
                    logger.info(f"[Stage] Processed {processed_count}/{limit if limit else total_count} rows")
                    
                    if limit is not None and processed_count >= limit:
                        break
            except Exception as e:
                logger.error(f"Error reading data: {e}")
                raise
        
        if not chunks:
            logger.warning("[Stage] No data to process")
            return
        
        df = pd.concat(chunks, ignore_index=True)
        if limit is not None:
            df = df.head(limit)
        logger.info(f"Final columns after processing: {df.columns.tolist()}")
        del chunks
        
        if df.empty or len(df.columns) == 0:
            raise ValueError("DataFrame contains no data or columns after processing")
        
        with stage_engine.connect() as conn:
            trans = None
            try:
                trans = conn.begin()
                df.columns = sanitize_and_make_unique_columns(list(df.columns))
                _create_stage_table(conn, table_name, df, stage_schema)
                _bulk_insert_data(conn, table_name, df, stage_schema)
                trans.commit()
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"[Stage] Successfully loaded {len(df)} rows in {duration:.2f} sec")
            except Exception as e:
                if trans:
                    trans.rollback()
                logger.error(f"Error loading to stage: {e}")
                raise
    except Exception as e:
        logger.error(f"[Stage ERROR] Error processing table {table_name}: {str(e)}")
        raise
