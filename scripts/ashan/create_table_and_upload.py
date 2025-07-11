import os
import re
import logging
import time
from datetime import datetime
from contextlib import closing
import pandas as pd
import numpy as np
from sqlalchemy import text, event, Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class TableProcessor:
    """Класс для обработки и загрузки таблиц с оптимизацией памяти и производительности."""
    
    # Оптимизированные константы
    CHUNKSIZE = 100000  # Размер чанка для обработки
    BATCH_SIZE = 50000  # Размер батча для вставки в БД
    
    # Словарь месяцев с оптимизированным доступом
    MONTHS = {
        **{m.lower(): i+1 for i, m in enumerate([
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        ])},
        **{m.lower(): i+1 for i, m in enumerate([
            'январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
            'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'
        ])}
    }

    @staticmethod
    def extract_metadata(filename: str) -> tuple:
        """Оптимизированное извлечение метаданных из имени файла."""
        name = os.path.splitext(os.path.basename(filename))[0].lower()
        tokens = re.findall(r'[a-zа-я]+|\d{4}', name)
        
        year = next((int(t) for t in tokens if t.isdigit() and len(t) == 4), None)
        month = next((TableProcessor.MONTHS.get(t) for t in tokens if t in TableProcessor.MONTHS), None)
        
        return month, year

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизированная нормализация названий столбцов."""
        df.columns = [re.sub(r'\s+', '_', col.strip().lower()) for col in df.columns]
        return df

    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизация типов данных в DataFrame."""
        # Конвертация строковых колонок
        str_cols = df.select_dtypes(include=['object']).columns
        for col in str_cols:
            df[col] = df[col].astype('string')
        
        # Оптимизация числовых колонок
        num_cols = df.select_dtypes(include=['float']).columns
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df

    @classmethod
    def process_file(cls, file_path: str, engine) -> str:
        """Основной метод обработки файла с оптимизацией памяти."""
        try:
            start_time = time.time()
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            table_name = re.sub(r'\W+', '_', base_name)
            
            # 1. Определение типа файла и чтение
            if file_path.endswith('.xlsx'):
                reader = pd.read_excel(file_path, sheet_name=None, dtype='string', engine='openpyxl')
            else:
                reader = {'data': pd.read_csv(file_path, dtype='string', sep=';', quotechar='"')}
            
            # 2. Параллельная обработка листов
            processed_chunks = []
            for sheet_name, df in reader.items():
                try:
                    # Быстрая проверка на пустоту
                    if df.empty:
                        continue
                        
                    df = cls.normalize_columns(df)
                    
                    # Извлечение метаданных
                    month, year = cls.extract_metadata(file_name)
                    if not month and 'month' in df.columns:
                        month, year = cls.extract_metadata(df['month'].iloc[0])
                    
                    # Добавление метаданных
                    if month and year:
                        df['sale_year'] = str(year)
                        df['sale_month'] = str(month)
                    
                    processed_chunks.append(cls.optimize_dataframe(df))
                except Exception as e:
                    logger.error(f"Ошибка обработки листа {sheet_name}: {str(e)}")
                    continue
            
            if not processed_chunks:
                raise ValueError("Все листы пусты или содержат ошибки")
            
            # 3. Объединение с оптимизацией памяти
            final_df = pd.concat(processed_chunks, ignore_index=True)
            del processed_chunks
            
            # 4. Создание таблицы в БД
            with engine.connect() as conn:
                # Проверка существования таблицы
                table_exists = conn.execute(
                    text("""
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'raw' AND table_name = :table
                    """),
                    {"table": table_name}
                ).scalar()
                
                if not table_exists:
                    # Оптимизированное создание таблицы
                    columns_sql = []
                    for col, dtype in zip(final_df.columns, final_df.dtypes):
                        sql_type = 'NVARCHAR(255)' if dtype == 'object' else 'FLOAT'
                        columns_sql.append(f'[{col}] {sql_type}')
                    
                    create_sql = f"CREATE TABLE raw.{table_name} ({', '.join(columns_sql)})"
                    conn.execute(text(create_sql))
                    conn.commit()
            
            # 5. Оптимизированная загрузка данных
            cls.bulk_insert(final_df, table_name, engine)
            
            logger.info(f"Файл {file_name} обработан за {time.time()-start_time:.2f} сек. "
                       f"Загружено {len(final_df)} строк в raw.{table_name}")
            
            return table_name
            
        except Exception as e:
            logger.error(f"Критическая ошибка при обработке файла {file_path}: {str(e)}", exc_info=True)
            raise

    @classmethod
    def bulk_insert(cls, df: pd.DataFrame, table_name: str, engine):
        """Оптимизированная массовая вставка данных."""
        @event.listens_for(engine, 'before_cursor_execute')
        def set_fast_executemany(conn, cursor, stmt, params, context, executemany):
            if executemany:
                cursor.fast_executemany = True
        
        try:
            with closing(engine.raw_connection()) as conn:
                with conn.cursor() as cursor:
                    # Подготовка данных для вставки
                    data = [tuple(x) for x in df.itertuples(index=False, name=None)]
                    cols = ', '.join([f'[{col}]' for col in df.columns])
                    params = ', '.join(['?'] * len(df.columns))
                    
                    # Чанкованная вставка
                    for i in range(0, len(data), cls.BATCH_SIZE):
                        batch = data[i:i + cls.BATCH_SIZE]
                        insert_sql = f"INSERT INTO raw.{table_name} ({cols}) VALUES ({params})"
                        cursor.executemany(insert_sql, batch)
                        conn.commit()
                        
        except SQLAlchemyError as e:
            logger.error(f"Ошибка при вставке данных в raw.{table_name}: {str(e)}")
            raise

def create_table_and_upload(file_path: str, engine):
    """Точка входа для обработки файла."""
    return TableProcessor.process_file(file_path, engine)