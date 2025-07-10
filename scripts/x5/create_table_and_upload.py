import pandas as pd
from sqlalchemy import text, event
from sqlalchemy.engine import Engine
import os
import re
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4,
    'май': 5, 'июнь': 6, 'июль': 7, 'август': 8,
    'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12,
}

def extract_year_month_from_filename(file_name):
    """
    Извлекает год и месяц из названия файла, например: x5_january_2025.csv
    Возвращает (месяц, год) или (None, None), если не найдено.
    """
    name = os.path.splitext(os.path.basename(file_name))[0].lower()
    tokens = re.split(r'[\W_]+', name)
    
    year = next((int(t) for t in tokens if t.isdigit() and len(t) == 4), None)
    month = next((MONTHS[t] for t in tokens if t in MONTHS), None)
    
    if year and month:
        return month, year
    return None, None   

def create_table_and_upload(file_path, engine):
    try:
        file_name = os.path.basename(file_path)
        df = pd.read_csv(file_path, dtype=str, sep=';', quotechar='"', skiprows=1)
        df = df.head(10000)
        df = df.fillna('').astype(str)

        # Добавление даты из имени файла
        month, year = extract_year_month_from_filename(file_name)
        if month and year:
            df['sale_year'] = year
            df['sale_month'] = month
            logger.info(f"[INFO] Установлены sale_year = {year}, sale_month = {month} из имени файла {file_name}")
        else:
            logger.warning(f"[WARN] Не удалось определить месяц и год из имени файла {file_name}")

        logger.info(f">>> Заголовки колонок: {df.columns.tolist()}")

        table_name = re.sub(r'\W+', '_', os.path.splitext(file_name)[0].lower())

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT 1 
                    FROM information_schema.tables 
                    WHERE table_schema = 'raw' AND table_name = :table
                """),
                {"table": table_name}
            ).first()

            if not result:
                columns = ', '.join([f'[{col}] NVARCHAR(250)' for col in df.columns])
                create_stmt = f"CREATE TABLE raw.{table_name} ({columns})"
                logger.info(f"[INFO] Создаём таблицу raw.{table_name}")
                with conn.begin():
                    conn.execute(text(create_stmt))
                logger.info(f"[SUCCESS] Таблица raw.{table_name} создана")
            else:
                logger.info(f"[INFO] Таблица raw.{table_name} уже существует")

            count_result = conn.execute(text(f"SELECT COUNT(*) FROM raw.{table_name}")).scalar()
            if count_result > 0:
                logger.warning(f"[SKIP] В таблице raw.{table_name} уже есть данные ({count_result} строк). Загрузка пропущена.")
                return table_name

        logger.info(f"[INFO] Подготовка к загрузке {len(df)} строк в таблицу raw.{table_name}")

        # Настройка fast_executemany
        raw_conn = engine.raw_connection()
        cursor = raw_conn.cursor()
        cursor.fast_executemany = True

        start_time = time.time()

        @event.listens_for(Engine, "before_cursor_execute")
        def enable_fast_executemany(conn, cursor, statement, parameters, context, executemany):
            if executemany:
                cursor.fast_executemany = True

        df.to_sql(
            name=table_name,
            schema='raw',
            con=engine,
            if_exists='append',
            index=False,
            chunksize=1000
        )

        elapsed = time.time() - start_time
        logger.info(f"[SUCCESS] Загружено {len(df)} строк в raw.{table_name} за {elapsed:.2f} сек.")

        cursor.close()
        raw_conn.close()

        return table_name

    except Exception as e:
        logger.error(f"[ERROR] Ошибка при обработке файла {file_path}: {e}", exc_info=True)
        raise
