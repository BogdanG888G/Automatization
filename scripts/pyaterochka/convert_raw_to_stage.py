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
    """Column configuration with data types and transformations for Pyaterochka."""
    NUMERIC_COLS = {
        'sales_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'sales_quantity': {'dtype': 'float64', 'default': 0.0},
        'sales_weight_kg': {'dtype': 'float64', 'default': 0.0},
        'cost_price_rub': {'dtype': 'float64', 'default': 0.0}
    }
    
    RENAME_MAP = {
        'period': 'sales_month',
        'retail_chain': 'retail_chain',
        'category': 'product_category',
        'base_type': 'product_type',
        'supplier': 'supplier_name',
        'brand': 'brand',
        'product_name': 'product_name',
        'unified_product_name': 'product_unified_name',
        'weight': 'product_weight_g',
        'flavor': 'product_flavor',
        'sales_units': 'sales_quantity',
        'sales_rub': 'sales_amount_rub',
        'sales_tonnes': 'sales_weight_kg',
        'cost_rub': 'cost_price_rub'
    }

    # Добавляем максимальные длины для строковых колонок
    STRING_COL_LENGTHS = {
        'retail_chain': 255,
        'product_category': 255,
        'product_type': 255,
        'supplier_name': 255,
        'brand': 255,
        'product_name': 255,
        'product_unified_name': 255,
        'product_flavor': 255,
        'product_weight_g': 50
    }

def _process_month_column(df: pd.DataFrame) -> pd.DataFrame:
    """Process month column from format like 'сен.24' to proper date."""
    if 'period' not in df.columns and 'sales_month' not in df.columns:
        raise ValueError("Neither 'period' nor 'sales_month' column found in DataFrame")
    
    month_col = 'period' if 'period' in df.columns else 'sales_month'
    
    month_map = {
        'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04',
        'май': '05', 'июн': '06', 'июл': '07', 'авг': '08',
        'сен': '09', 'окт': '10', 'ноя': '11', 'дек': '12'
    }
    
    def parse_month(month_str):
        if pd.isna(month_str):
            return '2024-01-01'
            
        try:
            month_str = str(month_str).strip().lower()
            month_part = month_str[:3]
            year_part = '20' + month_str[-2:] if len(month_str) > 3 else '2024'
            return f"{year_part}-{month_map.get(month_part, '01')}-01"
        except Exception:
            return '2024-01-01'
    
    df['sales_month'] = df[month_col].apply(parse_month)
    df['sales_month'] = pd.to_datetime(df['sales_month'], errors='coerce').fillna(pd.to_datetime('2024-01-01'))
    
    if month_col != 'sales_month':
        df = df.drop(columns=[month_col], errors='ignore')
    
    return df

def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns with specific format handling."""
    logger.info(f"Original columns: {df.columns.tolist()}")
    
    for col in df.columns:
        if col in ['sales_units', 'sales_rub', 'sales_tonnes', 'cost_rub']:
            try:
                # Улучшенное преобразование научной нотации
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(',', '.', regex=False)
                    .str.replace(r'[^\d.eE+-]', '', regex=True)
                    .replace('', '0')
                    .apply(lambda x: float(x) if x.strip() else 0.0)
                )
                # Convert tons to kg for weight
                if col == 'sales_tonnes':
                    df[col] = df[col] * 1000
            except Exception as e:
                logger.error(f"Error converting column {col}: {e}")
                raise
    
    logger.info(f"Columns after numeric conversion: {df.columns.tolist()}")
    return df

def _convert_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns to English names."""
    df.columns = [col.strip() for col in df.columns]
    
    for ru_col, en_col in ColumnConfig.RENAME_MAP.items():
        if ru_col in df.columns:
            try:
                if en_col in ColumnConfig.NUMERIC_COLS:
                    continue
                    
                max_len = ColumnConfig.STRING_COL_LENGTHS.get(en_col, 255)
                # Add more robust string truncation
                df[en_col] = (
                    df[ru_col]
                    .astype(str)
                    .str.normalize('NFKC')
                    .str.replace(r'[\x00-\x1F\x7F-\x9F]', '', regex=True)
                    .str.strip()
                    .str.slice(0, max_len)
                    .fillna('')
                )
                
                # Add validation to ensure no values exceed max length
                if (df[en_col].str.len() > max_len).any():
                    logger.warning(f"Some values in {en_col} exceed max length {max_len} after truncation")
                
                if ru_col != en_col:
                    df.drop(ru_col, axis=1, inplace=True)
            except Exception as e:
                logger.error(f"Error converting string column {ru_col}: {e}")
                raise
    
    return df

def _sanitize_column_name(name: str) -> str:
    """Sanitize column name for SQL Server."""
    if not name or not isinstance(name, str):
        return 'unknown_column_0'
    
    original_name = name.lower()
    
    special_mappings = {
        'sales_units': 'sales_quantity',
        'sales_rub': 'sales_amount_rub',
        'sales_tonnes': 'sales_weight_kg',
        'cost_rub': 'cost_price_rub',
        'period': 'sales_month',
        'retail_chain': 'retail_chain',
        'category': 'product_category',
        'base_type': 'product_type',
        'supplier': 'supplier_name',
        'brand': 'brand',
        'product_name': 'product_name',
        'unified_product_name': 'product_unified_name',
        'weight': 'product_weight_g',
        'flavor': 'product_flavor'
    }
    
    if original_name in special_mappings:
        return special_mappings[original_name]
    
    if original_name.startswith('none_'):
        try:
            num = int(original_name.split('_')[1])
            return f'unknown_column_{num}'
        except (ValueError, IndexError):
            pass

    name = re.sub(r'[^a-zA-Z0-9_]', '_', original_name)
    name = re.sub(r'_{2,}', '_', name)
    name = name.strip('_')
    
    return name if name else 'unknown_column_0'

def _create_stage_table(conn: engine.Connection, table_name: str, 
                       df: pd.DataFrame, schema: str = 'pyaterochka') -> None:
    """Create or alter table in stage schema with proper data types."""
    # Check if table exists
    table_exists = conn.execute(text(
        f"SELECT COUNT(*) FROM sys.tables t "
        f"JOIN sys.schemas s ON t.schema_id = s.schema_id "
        f"WHERE s.name = '{schema}' AND t.name = '{table_name}'"
    )).scalar() > 0
    
    # Prepare column definitions
    safe_columns = []
    for col in df.columns:
        if not col:
            continue
            
        if col in ColumnConfig.NUMERIC_COLS:
            col_type = 'DECIMAL(18, 2)'
        elif 'date' in col or 'month' in col:
            col_type = 'DATE'
        else:
            max_len = ColumnConfig.STRING_COL_LENGTHS.get(col, 255)
            col_type = f'NVARCHAR({max_len})'
    
        safe_columns.append((col, col_type))
    
    if not safe_columns:
        raise ValueError("No valid columns to create table")
    
    if not table_exists:
        # Create new table
        columns_sql = ',\n'.join([f'[{col}] {col_type}' for col, col_type in safe_columns])
        create_table_sql = f"""
            CREATE TABLE [{schema}].[{table_name}] (
                {columns_sql},
                load_dt DATETIME DEFAULT GETDATE()
            )
        """
        try:
            conn.execute(text(create_table_sql))
            logger.info(f"Created new table [{schema}].[{table_name}]")
        except Exception as e:
            logger.error(f"Error creating table: {e}\nSQL: {create_table_sql}")
            raise
    else:
        # Alter existing table to match new schema
        for col, col_type in safe_columns:
            try:
                # Check if column exists
                col_exists = conn.execute(text(
                    f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
                    f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}' "
                    f"AND COLUMN_NAME = '{col}'"
                )).scalar() > 0
                
                if col_exists:
                    # Alter existing column
                    alter_sql = f"""
                        ALTER TABLE [{schema}].[{table_name}] 
                        ALTER COLUMN [{col}] {col_type}
                    """
                    conn.execute(text(alter_sql))
                    logger.info(f"Altered column {col} in [{schema}].[{table_name}]")
                else:
                    # Add new column
                    add_sql = f"""
                        ALTER TABLE [{schema}].[{table_name}] 
                        ADD [{col}] {col_type}
                    """
                    conn.execute(text(add_sql))
                    logger.info(f"Added column {col} to [{schema}].[{table_name}]")
            except Exception as e:
                logger.error(f"Error altering column {col}: {e}")
                continue

def _bulk_insert_data(conn: engine.Connection, table_name: str, 
                     df: pd.DataFrame, schema: str = 'pyaterochka') -> None:
    """Insert data only if table is empty."""
    if df.empty:
        logger.warning("Empty DataFrame, skipping insert")
        return

    safe_columns = [_sanitize_column_name(col) for col in df.columns if col]
    if not safe_columns:
        raise ValueError("No valid columns for insertion")

    row_count_result = conn.execute(
        text(f"SELECT COUNT(*) FROM [{schema}].[{table_name}]")
    )
    row_count = row_count_result.scalar()
    if row_count > 0:
        logger.info(f"Table [{schema}].[{table_name}] already contains data ({row_count} rows), skipping insert")
        return

    data = []
    for row in df.itertuples(index=False):
        processed_row = []
        for val, col in zip(row, df.columns):
            if col in ColumnConfig.NUMERIC_COLS:
                processed_val = float(val) if pd.notna(val) else 0.0
            elif 'date' in col or 'month' in col:
                processed_val = str(val) if pd.notna(val) else None
            else:
                # Обрезаем строки перед вставкой
                max_len = ColumnConfig.STRING_COL_LENGTHS.get(col, 255)
                processed_val = str(val)[:max_len] if pd.notna(val) else ''
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
                        stage_engine: engine.Engine, stage_schema: str = 'pyaterochka', 
                        limit: int = None) -> None:
    """
    Convert data from raw to stage schema for Pyaterochka.
    """
    try:
        start_time = datetime.now()
        logger.info(f"[Stage] Starting processing of table {table_name}")
        
        with raw_engine.connect() as conn:
            try:
                result = conn.execute(text(
                    f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    f"WHERE TABLE_SCHEMA = 'raw' AND TABLE_NAME = :table_name"
                ), {'table_name': table_name})
                actual_columns = [row[0] for row in result]
                
                logger.info(f"Actual columns in raw table: {actual_columns}")
                
                total_count = conn.execute(
                    text(f"SELECT COUNT(*) FROM raw.{table_name}")
                ).scalar()
                logger.info(f"[Stage] Total rows to process: {total_count}")
            except Exception as e:
                logger.error(f"Error getting table metadata: {e}")
                raise

        with raw_engine.connect() as conn:
            sample_df = pd.read_sql(
                text(f"SELECT TOP 100 * FROM raw.{table_name}"),
                conn
            )
        
        chunks: List[pd.DataFrame] = []
        query = f"SELECT * FROM raw.{table_name}"
        if limit is not None:
            query += f" ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
        
        with raw_engine.connect().execution_options(stream_results=True) as conn:
            try:
                for chunk in pd.read_sql(
                    text(query),
                    conn,
                    chunksize=100000,
                    dtype='object'
                ):
                    chunk.columns = [col.strip() for col in chunk.columns]
                    logger.info(f"Processing chunk with {len(chunk)} rows")
                    
                    try:
                        chunk = _process_month_column(chunk)
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

        df = pd.concat(chunks, ignore_index=True)
        if limit is not None:
            df = df.head(limit)
            
        columns_to_drop = [col for col in df.columns 
                         if col.startswith('none_') or 
                         col.startswith('unknown_column_')]
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        logger.info(f"Final DataFrame shape after dropping unused columns: {df.shape}")

        del chunks
        
        if df.empty or len(df.columns) == 0:
            raise ValueError("DataFrame contains no data or columns after processing")
        
        with stage_engine.connect() as conn:
            trans = None
            try:
                trans = conn.begin()
                
                df.columns = [_sanitize_column_name(col) for col in df.columns]
                _create_stage_table(conn, table_name, df, stage_schema)
                _bulk_insert_data(conn, table_name, df, stage_schema)
                
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