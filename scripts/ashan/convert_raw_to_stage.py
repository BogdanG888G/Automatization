import pandas as pd
import logging
from sqlalchemy import text, exc, engine
from contextlib import closing
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class ColumnConfig:
    """Column configuration with data types and transformations."""
    NUMERIC_COLS = {
        'average_sell_price': {'dtype': 'float64', 'default': 0.0},
        'writeoff_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'writeoff_quantity': {'dtype': 'float64', 'default': 0.0},
        'sales_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'sales_weight_kg': {'dtype': 'float64', 'default': 0.0},    
        'sales_quantity': {'dtype': 'float64', 'default': 0.0},
        'average_cost_price': {'dtype': 'float64', 'default': 0.0},
        'margin_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'loss_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'loss_quantity': {'dtype': 'float64', 'default': 0.0},
        'promo_sales_amount_rub': {'dtype': 'float64', 'default': 0.0}
    }
    
    RENAME_MAP = {
    'дата': 'sales_date',
    'сегмент': 'product_segment',
    'семья': 'product_family_code',
    'название_семьи': 'product_family_name',
    'артикул': 'product_article',
    'наименование': 'product_name',
    'поставщик': 'supplier_code',
    'наименование_поставщика': 'supplier_name',
    'магазин': 'store_code',
    'город': 'city',
    'адрес': 'store_address',
    'формат': 'store_format',
    'месяц': 'month_name',
    'ср.цена_продажи': 'avg_sell_price',
    'списания,_руб.': 'writeoff_amount_rub',
    'списания,_шт.': 'writeoff_quantity',
    'продажи,_c_ндс': 'sales_amount_with_vat',
    'продажи,_кг': 'sales_weight_kg',
    'продажи,_шт': 'sales_quantity',
    'ср.цена_покупки': 'avg_purchase_price',
    'маржа,_руб.': 'margin_amount_rub',
    'потери,_руб.': 'loss_amount_rub',
    'потери,шт': 'loss_quantity',
    'промо_продажи,_c_ндс': 'promo_sales_amount_with_vat'
}

def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns with specific format handling."""
    logger.info(f"Original columns: {df.columns.tolist()}")
    
    # First normalize column names to match our RENAME_MAP
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    numeric_cols = {ru: en for ru, en in ColumnConfig.RENAME_MAP.items() 
                   if en in ColumnConfig.NUMERIC_COLS}
    
    for ru_col, en_col in numeric_cols.items():
        if ru_col in df.columns:
            try:
                # Convert to string first to handle various formats
                df[en_col] = (
                    df[ru_col]
                    .astype(str)
                    .str.replace(',', '.')  # Replace comma decimal separator
                    .str.replace(r'[^\d.]', '', regex=True)  # Remove non-numeric chars
                    .replace('', '0')
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
    """Convert string columns to English names."""
    # First normalize column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Process all columns that are in our RENAME_MAP
    for ru_col, en_col in ColumnConfig.RENAME_MAP.items():
        if ru_col in df.columns:
            try:
                df[en_col] = df[ru_col].astype('string').fillna('')
                if ru_col != en_col:  # Only drop if the names are different
                    df.drop(ru_col, axis=1, inplace=True)
            except Exception as e:
                logger.error(f"Error converting string column {ru_col}: {e}")
                raise
    
    return df

def _sanitize_column_name(name: str) -> str:
    """Sanitize column name for SQL Server while preserving meaningful names"""
    if not name or not isinstance(name, str):
        return ''
    
    # Сначала сохраняем оригинальное имя для проверки специальных случаев
    original_name = name.lower()
    
    # Специальные обработки для конкретных столбцов, чтобы избежать дублирования
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
    
    # Проверяем специальные случаи
    if original_name in special_mappings:
        return special_mappings[original_name]
    
    # Общая обработка для остальных столбцов
    # Удаляем все не-буквенно-цифровые символы, кроме подчеркивания
    name = re.sub(r'[^a-zA-Z0-9_]', '_', original_name)
    # Удаляем последовательные подчеркивания
    name = re.sub(r'_{2,}', '_', name)
    # Удаляем подчеркивания в начале и конце
    name = name.strip('_')
    
    return name


def _create_stage_table(conn: engine.Connection, table_name: str, 
                       df: pd.DataFrame, schema: str = 'ashan') -> None:
    """Create table in stage schema with proper data types"""
    logger.info(f"DataFrame columns before table creation: {df.columns.tolist()}")
    
    # Ensure all columns are properly named first
    df.columns = [_sanitize_column_name(col) for col in df.columns]
    
    # Verify no duplicate columns after sanitization
    if len(df.columns) != len(set(df.columns)):
        duplicates = [col for col in df.columns if list(df.columns).count(col) > 1]
        raise ValueError(f"Duplicate column names after sanitization: {duplicates}")
    
    safe_columns = []
    for col in df.columns:
        if not col:
            continue
            
        # Determine column type
        col_type = 'FLOAT' if col in ColumnConfig.NUMERIC_COLS else 'NVARCHAR(255)'
        safe_columns.append(f'[{col}] {col_type}')
    
    if not safe_columns:
        raise ValueError("No valid columns to create table")
    
    create_table_sql = f"""
        IF NOT EXISTS (SELECT * FROM sys.tables t 
                      JOIN sys.schemas s ON t.schema_id = s.schema_id 
                      WHERE s.name = '{schema}' AND t.name = '{table_name}')
        BEGIN
            CREATE TABLE [{schema}].[{table_name}] (
                {', '.join(safe_columns)}
            )
        END
    """
    
    # Use proper transaction management
    try:
        # Start a transaction explicitly
        trans = conn.begin()
        conn.execute(text(create_table_sql))
        trans.commit()
    except Exception as e:
        if 'trans' in locals():
            trans.rollback()
        logger.error(f"Error creating table: {e}\nSQL: {create_table_sql}")
        raise


def _bulk_insert_data(conn: engine.Connection, table_name: str, 
                     df: pd.DataFrame, schema: str = 'ashan') -> None:
    """Insert data only if table is empty, no truncate."""
    if df.empty:
        logger.warning("Empty DataFrame, skipping insert")
        return

    safe_columns = [_sanitize_column_name(col) for col in df.columns if col]
    if not safe_columns:
        raise ValueError("No valid columns for insertion")

    # Проверяем, есть ли уже данные в таблице
    row_count_result = conn.execute(
        text(f"SELECT COUNT(*) FROM [{schema}].[{table_name}]")
    )
    row_count = row_count_result.scalar()
    if row_count > 0:
        logger.info(f"Table [{schema}].[{table_name}] already contains data ({row_count} rows), skipping insert")
        return

    # Подготавливаем данные для вставки
    data = []
    for row in df.itertuples(index=False):
        processed_row = []
        for val, col in zip(row, df.columns):
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


def convert_raw_to_stage(table_name: str, raw_engine: engine.Engine, 
                        stage_engine: engine.Engine, stage_schema: str = 'ashan', 
                        limit: int = None) -> None:
    """
    Convert data from raw to stage schema.
    
    Args:
        table_name: Name of the table in raw schema
        raw_engine: SQLAlchemy engine for raw database
        stage_engine: SQLAlchemy engine for stage database
        stage_schema: Name of stage schema
        limit: Row limit for processing
    """
    try:
        start_time = datetime.now()
        logger.info(f"[Stage] Starting processing of table {table_name}")
        
        # 1. Get actual column names
        with raw_engine.connect() as conn:
            try:
                result = conn.execute(text(
                    f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    f"WHERE TABLE_SCHEMA = 'raw' AND TABLE_NAME = :table_name"
                ), {'table_name': table_name})
                actual_columns = [row[0].lower().replace(' ', '_') for row in result]
                
                logger.info(f"Actual columns in raw table: {actual_columns}")
                
                total_count = conn.execute(
                    text(f"SELECT COUNT(*) FROM raw.{table_name}")
                ).scalar()
                logger.info(f"[Stage] Total rows to process: {total_count}")
            except Exception as e:
                logger.error(f"Error getting table metadata: {e}")
                raise

        # 2. Filter RENAME_MAP for existing columns
        valid_rename_map = {
            ru: en for ru, en in ColumnConfig.RENAME_MAP.items() 
            if ru in actual_columns
        }
        logger.info(f"Active RENAME_MAP: {valid_rename_map}")

        # 3. Read data with chunking and row limit
        chunks: List[pd.DataFrame] = []
        query = f"SELECT * FROM raw.{table_name}"
        if limit is not None:
            query += f" ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
        
        with raw_engine.connect().execution_options(stream_results=True) as conn:
            try:
                for chunk in pd.read_sql(
                    text(query),
                    conn,
                    chunksize=50000,
                    dtype='object'  # Read all as object initially
                ):
                    # Normalize column names first
                    chunk.columns = [col.lower().replace(' ', '_') for col in chunk.columns]
                    logger.info(f"Received chunk with columns: {chunk.columns.tolist()}")
                    
                    # Process chunk
                    try:
                        chunk = _convert_numeric_columns(chunk)
                        chunk = _convert_string_columns(chunk)
                        chunks.append(chunk)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                        raise
                    
                    processed_count = sum(len(c) for c in chunks)
                    logger.info(f"[Stage] Processed {processed_count}/{limit if limit is not None else total_count} rows")
                    
                    if limit is not None and processed_count >= limit:
                        break
            except Exception as e:
                logger.error(f"Error reading data: {e}")
                raise

        if not chunks:
            logger.warning("[Stage] No data to process")
            return

        # 4. Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        if limit is not None:
            df = df.head(limit)
        logger.info(f"Final columns after processing: {df.columns.tolist()}")
        del chunks
        
        if df.empty or len(df.columns) == 0:
            raise ValueError("DataFrame contains no data or columns after processing")
        
        # 5. Load to stage with proper connection management
        with stage_engine.connect() as conn:
            trans = None
            try:
                # Start transaction
                trans = conn.begin()
                
                # Process data
                df.columns = [_sanitize_column_name(col) for col in df.columns]
                _create_stage_table(conn, table_name, df, stage_schema)
                
                # Bulk insert
                _bulk_insert_data(conn, table_name, df, stage_schema)
                
                # Commit if all succeeded
                trans.commit()
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"[Stage] Successfully loaded {len(df)} rows in {duration:.2f} sec"
                )
            except Exception as e:
                if trans:
                    trans.rollback()
                logger.error(f"Error loading to stage: {e}")
                raise
                
    except Exception as e:
        logger.error(f"[Stage ERROR] Error processing table {table_name}: {str(e)}")
        raise