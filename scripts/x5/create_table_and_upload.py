import os
import re
import logging
import time
from contextlib import closing
import pandas as pd
from sqlalchemy import text, event
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class TableProcessor:
    """Класс для обработки и загрузки таблиц с оптимизацией памяти и производительности."""
    
    # Оптимизированные константы
    CHUNKSIZE = 10000  # Размер чанка для обработки
    BATCH_SIZE = 5000  # Размер батча для вставки в БД
    
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

    # Стандартизация нестандартных названий колонок для X5
    X5_COLUMN_RENAME_MAP = {
        'завод.1': 'factory2',
        'материал.1': 'material2'
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
    def clean_column_name(col: str) -> str:
        """Очистка и нормализация имени столбца."""
        col = re.sub(r'\s+', ' ', col.strip())  # Удаление лишних пробелов
        col = re.sub(r'\s*\(.*?\)', '', col)    # Удаление скобок и их содержимого
        col = re.sub(r'[^\w]', '_', col)        # Замена спецсимволов на подчеркивания
        col = col.lower()                       # Приведение к нижнему регистру
        return col[:100]                        # Ограничение длины имени

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизированная нормализация названий столбцов."""
        # Очистка имен колонок
        df.columns = [TableProcessor.clean_column_name(col) for col in df.columns]
        
        # Обработка дубликатов
        seen = {}
        new_columns = []
        for col in df.columns:
            new_col = col
            count = 1
            while new_col in seen:
                new_col = f"{col}_{count}"
                count += 1
            seen[new_col] = True
            new_columns.append(new_col)
        
        df.columns = new_columns
        return df

    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизация типов данных в DataFrame."""
        # Конвертация всех данных в строки с ограничением длины
        for col in df.columns:
            df[col] = df[col].astype(str).str[:250]
        return df

    @classmethod
    def process_file(cls, file_path: str, engine) -> str:
        """Основной метод обработки файла с оптимизацией памяти."""
        try:
            start_time = time.time()
            file_name = os.path.basename(file_path)
            logger.info(f"[START] Обработка файла: {file_name}")
            
            # 1. Чтение файла
            df = pd.read_csv(file_path, dtype=str, sep=';', quotechar='"', skiprows=1, nrows=cls.CHUNKSIZE)
            
            if df.empty:
                raise ValueError(f"Файл {file_name} не содержит данных")
            
            # 2. Нормализация колонок
            df = cls.normalize_columns(df)
            
            # 3. Специальная обработка для X5
            if 'x5' in file_name.lower():
                rename_map = {col: cls.X5_COLUMN_RENAME_MAP[col] 
                            for col in df.columns if col in cls.X5_COLUMN_RENAME_MAP}
                if rename_map:
                    logger.info(f"[INFO] Переименование колонок для X5: {rename_map}")
                    df.rename(columns=rename_map, inplace=True)
            
            # 4. Добавление метаданных из имени файла
            month, year = cls.extract_metadata(file_name)
            if month and year:
                df['sale_year'] = str(year)
                df['sale_month'] = str(month)
                logger.info(f"[INFO] Установлены sale_year={year}, sale_month={month}")
            else:
                logger.warning(f"[WARN] Не удалось определить месяц и год из имени файла")
            
            # 5. Оптимизация DataFrame
            df = cls.optimize_dataframe(df)
            
            # 6. Создание имени таблицы
            table_name = re.sub(r'\W+', '_', os.path.splitext(file_name)[0].lower())
            table_name = table_name[:128]
            
            # 7. Проверка существования таблицы
            with engine.connect() as conn:
                table_exists = conn.execute(
                    text("SELECT 1 FROM information_schema.tables "
                         "WHERE table_schema = 'raw' AND table_name = :table"),
                    {"table": table_name}
                ).scalar()
                
                if not table_exists:
                    # Создание таблицы с оптимальными типами данных
                    columns_sql = [f'[{col}] NVARCHAR(250)' for col in df.columns]
                    create_sql = f"CREATE TABLE raw.{table_name} ({', '.join(columns_sql)})"
                    conn.execute(text(create_sql))
                    conn.commit()
                    logger.info(f"[INFO] Таблица raw.{table_name} создана")
                else:
                    logger.info(f"[INFO] Таблица raw.{table_name} уже существует")
                
                # Проверка на существующие данные
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM raw.{table_name}")).scalar()
                if count_result > 0:
                    logger.warning(f"[SKIP] В таблице уже есть {count_result} строк. Загрузка пропущена.")
                    return table_name
            
            # 8. Оптимизированная загрузка данных
            cls.bulk_insert(df, table_name, engine)
            
            elapsed = time.time() - start_time
            logger.info(f"[SUCCESS] Загружено {len(df)} строк в raw.{table_name} за {elapsed:.2f} сек.")
            
            return table_name
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка при обработке файла {file_path}: {str(e)}", exc_info=True)
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
            logger.error(f"[ERROR] Ошибка при вставке данных в raw.{table_name}: {str(e)}")
            raise

def create_table_and_upload(file_path: str, engine) -> str:
    """Точка входа для обработки файла."""
    return TableProcessor.process_file(file_path, engine)