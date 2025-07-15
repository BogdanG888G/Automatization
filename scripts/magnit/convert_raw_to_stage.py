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
    """Column configuration with data types and transformations for Magnit."""
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
        'discount_amount_rub': {'dtype': 'float64', 'default': 0.0}
    }
    
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
        'входящая_цена': 'incoming_price'
    }
    
    RENAME_MAP_2025 = {
        'month': 'sales_month',
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
        'скидка_руб': 'discount_amount_rub'
    }

def _detect_year(df: pd.DataFrame) -> int:
    """Detect year based on data format."""
    if 'month' in df.columns:
        sample_month = df['month'].iloc[0] if len(df) > 0 else ''
        if isinstance(sample_month, str) and any(m in sample_month for m in ['Январь', 'Февраль', 'Декабрь']):
            return 2024
    return 2025

def _process_2024_data(df: pd.DataFrame) -> pd.DataFrame:
    """Special processing for 2024 year data format."""
    logger.info("Applying 2024 year data format processing")
    
    # Process month column
    month_map = {
        'Январь': '01', 'Февраль': '02', 'Март': '03', 'Апрель': '04',
        'Май': '05', 'Июнь': '06', 'Июль': '07', 'Август': '08',
        'Сентябрь': '09', 'Октябрь': '10', 'Ноябрь': '11', 'Декабрь': '12'
    }
    
    df['month'] = df['month'].apply(lambda x: f"2024-{month_map[x]}-01")
    df['month'] = pd.to_datetime(df['month'])
    
    # Process numeric columns with comma as decimal separator
    numeric_cols = ['оборот_руб', 'оборот_шт', 'входящая_цена']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '.', regex=False)
                .str.replace(r'[^\d.]', '', regex=True)
                .replace('', '0')
                .astype(np.float64)
                .fillna(0)
            )
    
    return df

def _process_2025_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processing for 2025 year data format."""
    logger.info("Applying 2025 year data format processing")
    return df

def _convert_numeric_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Convert numeric columns with specific format handling."""
    logger.info(f"Original columns: {df.columns.tolist()}")
    
    # Normalize column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Add specific numeric columns to process
    numeric_cols = {
        'оборот_руб': 'turnover_amount_rub',
        'оборот_шт': 'turnover_quantity',
        'входящая_цена': 'incoming_price',
        'ср_цена_продажи': 'avg_sell_price',
        'списания_руб': 'writeoff_amount_rub',
        'списания_шт': 'writeoff_quantity',
        'продажи_руб': 'sales_amount_rub',
        'продажи_кг': 'sales_weight_kg',
        'продажи_шт': 'sales_quantity',
        'ср_цена_закупки': 'avg_purchase_price',
        'маржа_руб': 'margin_amount_rub',
        'потери_руб': 'loss_amount_rub',
        'потери_шт': 'loss_quantity',
        'промо_продажи_руб': 'promo_sales_amount_rub',
        'остаток_шт': 'stock_quantity',
        'остаток_руб': 'stock_amount_rub',
        'скидка_руб': 'discount_amount_rub'
    }
    
    for ru_col, en_col in numeric_cols.items():
        if ru_col in df.columns:
            try:
                # Handle different decimal separators based on year
                if year == 2024:
                    df[en_col] = (
                        df[ru_col]
                        .astype(str)
                        .str.replace(',', '.', regex=False)
                        .str.replace(r'[^\d.]', '', regex=True)
                        .replace('', '0')
                        .astype(np.float64)
                        .fillna(0)
                    )
                else:
                    df[en_col] = (
                        df[ru_col]
                        .astype(str)
                        .str.replace(',', '.', regex=False)
                        .str.replace(r'[^\d.]', '', regex=True)
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

def _convert_string_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Convert string columns to English names."""
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    rename_map = ColumnConfig.RENAME_MAP_2024 if year == 2024 else ColumnConfig.RENAME_MAP_2025
    
    for ru_col, en_col in rename_map.items():
        if ru_col in df.columns:
            try:
                df[en_col] = (
                    df[ru_col]
                    .astype(str)
                    .str.normalize('NFKC')  # Normalize unicode
                    .str.replace(r'[\x00-\x1F\x7F-\x9F]', '', regex=True)  # Remove control chars
                    .str.strip()
                    .fillna('')
                )
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
    
    # Special mappings for known columns
    special_mappings = {
        'списания_руб': 'writeoff_amount_rub',
        'списания_шт': 'writeoff_quantity',
        'продажи_руб': 'sales_amount_rub',
        'потери_руб': 'loss_amount_rub',
        'потери_шт': 'loss_quantity',
        'промо_продажи_руб': 'promo_sales_amount_rub',
        'маржа_руб': 'margin_amount_rub',
        'ср_цена_продажи': 'avg_sell_price',
        'ср_цена_закупки': 'avg_purchase_price',
        'остаток_шт': 'stock_quantity',
        'остаток_руб': 'stock_amount_rub',
        'оборот_руб': 'turnover_amount_rub',
        'оборот_шт': 'turnover_quantity',
        'входящая_цена': 'incoming_price',
        'наименование_тт': 'store_name',
        'код_тт': 'store_code',
        'адрес_тт': 'store_address',
        'уровень_1': 'product_level_1',
        'уровень_2': 'product_level_2',
        'уровень_3': 'product_level_3',
        'уровень_4': 'product_level_4',
        'наименование_тп': 'product_name',
        'код_тп': 'product_code',
        'шк': 'barcode',
        'себестоимсть_в_руб.': 'cost_price_rub',
        'quantity_sold': 'sales_quantity',
        'sales_amount': 'sales_amount_rub',
        'код': 'store_code',
        'адрес': 'store_address',
        'уровень': 'product_level',
        'наименование': 'product_name'
    }
    
    if original_name in special_mappings:
        return special_mappings[original_name]
    
    # Handle numbered none_ columns
    if original_name.startswith('none_'):
        try:
            num = int(original_name.split('_')[1])
            return f'unknown_column_{num}'
        except (ValueError, IndexError):
            pass

    # Preserve some original info in unknown columns
    name = re.sub(r'[^a-zA-Z0-9_]', '_', original_name)
    name = re.sub(r'_{2,}', '_', name)
    name = name.strip('_')
    
    if not name:
        return 'unknown_column_0'
    
    return name if name else 'unknown_column_0'

def _create_stage_table(conn: engine.Connection, table_name: str, 
                       df: pd.DataFrame, schema: str = 'magnit') -> None:
    """Create table in stage schema with proper data types."""
    df = df[[col for col in df.columns 
            if not (col.startswith('none_') or 
                   col.startswith('unknown_column_'))]]
    logger.info(f"DataFrame columns before table creation: {df.columns.tolist()}")
    
    df.columns = [_sanitize_column_name(col) for col in df.columns]
    
    # Remove any empty column names that might have been created
    df.columns = [col if col else f'unknown_column_{i}' for i, col in enumerate(df.columns)]
    
    if len(df.columns) != len(set(df.columns)):
        duplicates = [col for col in df.columns if list(df.columns).count(col) > 1]
        raise ValueError(f"Duplicate column names after sanitization: {duplicates}")
    
    safe_columns = []
    for col in df.columns:
        if not col:
            continue
            
        # Type handling
        if col in ColumnConfig.NUMERIC_COLS:
            col_type = 'DECIMAL(18, 2)'
        elif 'date' in col or 'month' in col:
            col_type = 'DATE'
        else:
            col_type = 'NVARCHAR(255)'
    
        safe_columns.append(f'[{col}] {col_type}')
    
    if not safe_columns:
        raise ValueError("No valid columns to create table")
    
    create_table_sql = f"""
        IF NOT EXISTS (SELECT * FROM sys.tables t 
                      JOIN sys.schemas s ON t.schema_id = s.schema_id 
                      WHERE s.name = '{schema}' AND t.name = '{table_name}')
        BEGIN
            CREATE TABLE [{schema}].[{table_name}] (
                {', '.join(safe_columns)},
                load_dt DATETIME DEFAULT GETDATE()
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

def _bulk_insert_data(conn: engine.Connection, table_name: str, 
                     df: pd.DataFrame, schema: str = 'magnit') -> None:
    """Insert data only if table is empty."""
    if df.empty:
        logger.warning("Empty DataFrame, skipping insert")
        return

    safe_columns = [_sanitize_column_name(col) for col in df.columns if col]
    if not safe_columns:
        raise ValueError("No valid columns for insertion")

    # Check if table already has data
    row_count_result = conn.execute(
        text(f"SELECT COUNT(*) FROM [{schema}].[{table_name}]")
    )
    row_count = row_count_result.scalar()
    if row_count > 0:
        logger.info(f"Table [{schema}].[{table_name}] already contains data ({row_count} rows), skipping insert")
        return

    # Prepare data with proper handling
    data = []
    for row in df.itertuples(index=False):
        processed_row = []
        for val, col in zip(row, df.columns):
            if col in ColumnConfig.NUMERIC_COLS:
                processed_val = float(val) if pd.notna(val) else 0.0
            elif 'date' in col or 'month' in col:
                processed_val = str(val) if pd.notna(val) else None
            else:
                processed_val = str(val)[:500] if pd.notna(val) else ''
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
                        stage_engine: engine.Engine, stage_schema: str = 'magnit', 
                        limit: int = None) -> None:
    """
    Convert data from raw to stage schema.
    
    Args:
        table_name: Name of the table in raw schema
        raw_engine: SQLAlchemy engine for raw database
        stage_engine: SQLAlchemy engine for stage database
        stage_schema: Name of stage schema (default 'magnit')
        limit: Row limit for processing
    """
    try:
        start_time = datetime.now()
        logger.info(f"[Stage] Starting processing of table {table_name}")
        
        # 1. Get actual column names from raw
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

        # 2. Read sample data to detect year
        with raw_engine.connect() as conn:
            sample_df = pd.read_sql(
                text(f"SELECT TOP 100 * FROM raw.{table_name}"),
                conn
            )
        
        year = _detect_year(sample_df)
        logger.info(f"Detected data format year: {year}")
        
        # 3. Select appropriate rename map
        rename_map = ColumnConfig.RENAME_MAP_2024 if year == 2024 else ColumnConfig.RENAME_MAP_2025
        valid_rename_map = {
            ru: en for ru, en in rename_map.items() 
            if ru in actual_columns
        }
        logger.info(f"Active RENAME_MAP: {valid_rename_map}")

        # 4. Read data with chunking
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
                    chunk.columns = [col.lower().replace(' ', '_') for col in chunk.columns]
                    logger.info(f"Processing chunk with {len(chunk)} rows")
                    
                    try:
                        # Apply year-specific processing
                        if year == 2024:
                            chunk = _process_2024_data(chunk)
                        else:
                            chunk = _process_2025_data(chunk)
                            
                        chunk = _convert_numeric_columns(chunk, year)
                        chunk = _convert_string_columns(chunk, year)
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

        # 5. Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        if limit is not None:
            df = df.head(limit)
            
        # Remove unused columns
        columns_to_drop = [col for col in df.columns 
                         if col.startswith('none_') or 
                         col.startswith('unknown_column_')]
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        logger.info(f"Final DataFrame shape after dropping unused columns: {df.shape}")

        del chunks
        
        if df.empty or len(df.columns) == 0:
            raise ValueError("DataFrame contains no data or columns after processing")
        
        # 6. Load to stage
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