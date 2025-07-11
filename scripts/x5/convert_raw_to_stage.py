import pandas as pd
import logging
import re
import numpy as np
from sqlalchemy import text
from contextlib import closing
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

class X5ColumnConfig:
    """Конфигурация колонок для X5 с оптимизированным доступом"""
    
    # Числовые колонки с типами данных
    NUMERIC_COLS: Dict[str, str] = {
        'quantity': 'float32',
        'gross_turnover': 'float32',
        'gross_cost': 'float32',
        'avg_cost_price': 'float32',
        'avg_sell_price': 'float32',
        'sale_year': 'int16',
        'sale_month': 'int8'
    }
    
    # Оптимизированный маппинг колонок (рус → англ)
    RENAME_MAP: Dict[str, str] = {
        'Сеть': 'retailer',
        'Филиал': 'branch',
        'Регион': 'region',
        'Город': 'city',
        'Адрес': 'address',
        'Завод': 'factory',
        'Завод1': 'factory',
        'Завод2': 'factory2',
        'Тов.иер.ур.2': 'prod_level_2',
        'Тов.иер.ур.3': 'prod_level_3',
        'Тов.иер.ур.4': 'prod_level_4',
        'Материал': 'material',
        'Материал1': 'material',
        'Материал2': 'material2',
        'Бренд': 'brand',
        'Вендор': 'vendor',
        'Основной поставщик': 'main_supplier',
        'Поставщик склада РЦ': 'warehouse_supplier',
        'Поставщик склада': 'warehouse_supplier',
        'Количество': 'quantity',
        'Количество без ед. изм.': 'quantity',
        'Оборот с НДС': 'gross_turnover',
        'Оборот с НДС без ед.изм.': 'gross_turnover',
        'Общая себестоимость': 'gross_cost',
        'Общая себестоимость с НДС без ед. изм.': 'gross_cost',
        'Средняя цена по себестоимости': 'avg_cost_price',
        'Средняя цена по себестоимости с НДС': 'avg_cost_price',
        'Средняя цена продажи': 'avg_sell_price',
        'Средняя цена продажи с НДС': 'avg_sell_price'
    }
    
    # Регулярные выражения для очистки колонок
    CLEAN_PATTERNS = [
        (r'\s*\(.*?\)', ''),      # Удалить всё в скобках
        (r'[\n\r]', ' '),          # Заменить переносы на пробелы
        (r'\s+', ' '),             # Множественные пробелы → один
        (r'[^\w\s]', ''),          # Удалить спецсимволы
    ]

class X5DataProcessor:
    """Оптимизированный процессор для данных X5"""
    
    BATCH_SIZE = 50000  # Размер батча для вставки
    
    @staticmethod
    def clean_column_name(col: str) -> str:
        """Оптимизированная очистка названий колонок"""
        for pattern, repl in X5ColumnConfig.CLEAN_PATTERNS:
            col = re.sub(pattern, repl, col)
        return col.strip().lower()
    
    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Нормализация колонок с обработкой дубликатов"""
        # Очистка и приведение к нижнему регистру
        df.columns = [X5DataProcessor.clean_column_name(col) for col in df.columns]
        
        # Обработка дубликатов
        seen = {}
        new_columns = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        
        df.columns = new_columns
        return df
    
    @staticmethod
    def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизированная конвертация числовых колонок"""
        numeric_cols = {ru: en for ru, en in X5ColumnConfig.RENAME_MAP.items() 
                       if en in X5ColumnConfig.NUMERIC_COLS}
        
        for ru_col, en_col in numeric_cols.items():
            if ru_col in df.columns:
                # Векторизованные операции для быстрой конвертации
                df[en_col] = (
                    df[ru_col]
                    .astype(str)
                    .str.replace(',', '.', regex=False)
                    .replace('', '0')
                    .astype(X5ColumnConfig.NUMERIC_COLS[en_col]))
                df.drop(ru_col, axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def convert_string_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизация строковых колонок"""
        str_cols = [ru for ru in X5ColumnConfig.RENAME_MAP.keys() 
                   if ru in df.columns and X5ColumnConfig.RENAME_MAP[ru] not in X5ColumnConfig.NUMERIC_COLS]
        
        for ru_col in str_cols:
            en_col = X5ColumnConfig.RENAME_MAP[ru_col]
            df[en_col] = df[ru_col].astype('string').fillna('')
            df.drop(ru_col, axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def bulk_insert(df: pd.DataFrame, table_name: str, engine, schema: str):
        """Оптимизированная массовая вставка данных"""
        try:
            with closing(engine.connect()) as conn:
                # Создание таблицы с оптимальными типами
                create_sql = f"""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables t 
                        JOIN sys.schemas s ON t.schema_id = s.schema_id 
                        WHERE s.name = '{schema}' AND t.name = '{table_name}'
                    )
                    BEGIN
                        CREATE TABLE {schema}.{table_name} (
                            {', '.join([
                                f'{col} {X5ColumnConfig.NUMERIC_COLS.get(col, "NVARCHAR(255)")}' 
                                for col in df.columns
                            ])}
                        )
                    END
                """
                conn.execute(text(create_sql))
                conn.commit()
                
                # Быстрая вставка через cursor
                with conn.connection.cursor() as cursor:
                    cursor.fast_executemany = True
                    
                    # Подготовка данных
                    data = [tuple(x) for x in df.itertuples(index=False, name=None)]
                    cols = ', '.join([f'[{col}]' for col in df.columns])
                    params = ', '.join(['?'] * len(df.columns))
                    
                    # Чанкованная вставка
                    for i in range(0, len(data), X5DataProcessor.BATCH_SIZE):
                        batch = data[i:i + X5DataProcessor.BATCH_SIZE]
                        insert_sql = f"INSERT INTO {schema}.{table_name} ({cols}) VALUES ({params})"
                        cursor.executemany(insert_sql, batch)
                        conn.commit()
                        
        except Exception as e:
            logger.error(f"Ошибка при вставке данных: {str(e)}")
            raise

def convert_raw_to_stage(table_name: str, raw_engine, stage_engine, stage_schema='x5'):
    """Оптимизированная загрузка данных из raw в stage"""
    try:
        logger.info(f"[X5 Stage] Начало обработки таблицы {table_name}")
        
        # 1. Чтение данных с прогресс-баром
        chunks = []
        with raw_engine.connect() as conn:
            total_rows = conn.execute(
                text(f"SELECT COUNT(*) FROM raw.{table_name}")
            ).scalar()
            logger.info(f"[X5 Stage] Всего строк для обработки: {total_rows}")
        
        # 2. Чтение и обработка по частям
        processed_rows = 0
        for chunk in pd.read_sql_table(
            table_name,
            raw_engine,
            schema='raw',
            chunksize=100000,
            dtype='object'  # Чтение как строки для экономии памяти
        ):
            # Нормализация и преобразование
            chunk = X5DataProcessor.normalize_columns(chunk)
            chunk = X5DataProcessor.convert_numeric_columns(chunk)
            chunk = X5DataProcessor.convert_string_columns(chunk)
            
            chunks.append(chunk)
            processed_rows += len(chunk)
            logger.info(f"[X5 Stage] Обработано {processed_rows}/{total_rows} строк ({processed_rows/total_rows:.1%})")
        
        # 3. Объединение и загрузка
        if chunks:
            final_df = pd.concat(chunks, ignore_index=True)
            X5DataProcessor.bulk_insert(final_df, table_name, stage_engine, stage_schema)
            logger.info(f"[X5 Stage] Успешно загружено {len(final_df)} строк в {stage_schema}.{table_name}")
        else:
            logger.warning("[X5 Stage] Нет данных для обработки")
            
    except Exception as e:
        logger.error(f"[X5 Stage ERROR] Ошибка при обработке таблицы {table_name}: {str(e)}", exc_info=True)
        raise