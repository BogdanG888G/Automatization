import pandas as pd
from sqlalchemy import text, event
import os
import re
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_table_and_upload(file_path, engine):
    try:
        file_name = os.path.basename(file_path).lower()
        df = pd.read_csv(file_path, dtype=str, sep=';', quotechar='"')
        #df = df.head(20000)  # для отладки
        df['sale_year'] = 2025
        df['sale_month'] = 1
        df = df.fillna('').astype(str)
        logger.info(f">>> Заголовки колонок: {df.columns.tolist()}")

        name = os.path.basename(file_path).replace('.csv', '')
        table_name = re.sub(r'\W+', '_', name.lower())

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
                columns = ', '.join([f'[{col}] NVARCHAR(MAX)' for col in df.columns])
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

        from sqlalchemy.engine import Engine
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
            chunksize=1000  # Можно увеличить
        )

        elapsed = time.time() - start_time
        logger.info(f"[SUCCESS] Загружено {len(df)} строк в raw.{table_name} за {elapsed:.2f} сек.")

        cursor.close()
        raw_conn.close()

        return table_name

    except Exception as e:
        logger.error(f"[ERROR] Ошибка при обработке файла {file_path}: {e}")
        raise
