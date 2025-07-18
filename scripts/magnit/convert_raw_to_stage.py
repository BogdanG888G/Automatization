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

def _bulk_insert_chunk(conn: engine.Connection, table_name: str, 
                      chunk_df: pd.DataFrame, schema: str = 'magnit') -> None:
    """Быстрая вставка чанка данных"""
    if chunk_df.empty:
        return

    # Преобразование данных в список кортежей
    data = [tuple(x) for x in chunk_df.to_numpy()]
    cols = ', '.join([f'[{col}]' for col in chunk_df.columns])
    params = ', '.join(['?'] * len(chunk_df.columns))
    
    raw_conn = conn.connection
    cursor = raw_conn.cursor()
    
    try:
        insert_sql = f"INSERT INTO [{schema}].[{table_name}] ({cols}) VALUES ({params})"
        cursor.fast_executemany = True
        cursor.executemany(insert_sql, data)
        raw_conn.commit()
    except Exception as e:
        raw_conn.rollback()
        logger.error(f"Error inserting data: {e}")
        raise
    finally:
        cursor.close()

def convert_raw_to_stage(table_name: str, raw_engine: engine.Engine, 
                        stage_engine: engine.Engine, stage_schema: str = 'magnit', 
                        limit: int = None) -> None:
    """Оптимизированная конвертация данных из RAW в STAGE"""
    try:
        start_time = datetime.now()
        logger.info(f"[Stage] Starting optimized processing for {table_name}")
        
        # 1. Получение метаданных
        with raw_engine.connect() as conn:
            # Проверка существования данных в STAGE
            try:
                check_sql = f"""
                    SELECT 1 
                    FROM {stage_schema}.{table_name} 
                    TABLESAMPLE (1 ROWS)
                """
                if conn.execute(text(check_sql)).scalar():
                    logger.info(f"Data already exists in {stage_schema}.{table_name}, skipping")
                    return
            except:
                pass
            
            # Определение структуры таблицы
            result = conn.execute(text(
                f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_SCHEMA = 'raw' AND TABLE_NAME = :table_name"
            ), {'table_name': table_name})
            actual_columns = {row[0].lower().replace(' ', '_') for row in result}
            
            # Определение года
            sample_df = pd.read_sql(
                text(f"SELECT TOP 100 * FROM raw.{table_name}"),
                conn
            )
            year = _detect_year(sample_df)
            rename_map = ColumnConfig.RENAME_MAP_2024 if year == 2024 else ColumnConfig.RENAME_MAP_2025
            active_rename_map = {k: v for k, v in rename_map.items() if k in actual_columns}
            
            # Определение ожидаемых колонок
            expected_columns = (
                set(active_rename_map.values()) | 
                set(ColumnConfig.NUMERIC_COLS.keys())
            )
            
            # Создание таблицы в STAGE
            with stage_engine.connect() as stage_conn:
                _create_stage_table(stage_conn, table_name, expected_columns, stage_schema)

        # 2. Потоковая обработка данных
        query = f"SELECT * FROM raw.{table_name}"
        if limit:
            query += f" ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
        
        processed_rows = 0
        chunk_size = 50000  # Оптимальный размер чанка для fast_executemany
        
        with raw_engine.connect().execution_options(stream_results=True) as conn:
            for chunk in pd.read_sql(
                text(query), 
                conn, 
                chunksize=chunk_size
            ):
                # Нормализация колонок
                chunk.columns = [col.lower().replace(' ', '_') for col in chunk.columns]
                
                # Обработка данных
                if year == 2024:
                    chunk = _process_2024_data(chunk)
                else:
                    chunk = _process_2025_data(chunk)
                
                chunk = _convert_numeric_columns(chunk, year)
                chunk = _convert_string_columns(chunk, active_rename_map)
                
                # Добавление отсутствующих колонок
                for col in expected_columns:
                    if col not in chunk.columns:
                        if col in ColumnConfig.NUMERIC_COLS:
                            chunk[col] = ColumnConfig.NUMERIC_COLS[col]['default']
                        else:
                            chunk[col] = ''
                
                # Вставка данных
                with stage_engine.connect() as stage_conn:
                    _bulk_insert_chunk(
                        stage_conn, 
                        table_name, 
                        chunk[list(expected_columns)], 
                        stage_schema
                    )
                
                processed_rows += len(chunk)
                logger.info(f"Processed {processed_rows} rows")
        
        logger.info(f"[Stage] Finished loading {processed_rows} rows in "
                   f"{(datetime.now() - start_time).total_seconds():.2f} seconds")
                
    except Exception as e:
        logger.error(f"[Stage ERROR] Processing failed: {str(e)}")
        raise