import pandas as pd
from sqlalchemy import text, event
from sqlalchemy.engine import Engine
import os
import re
import logging
import time

logger = logging.getLogger(__name__)

def create_table_and_upload(file_path, engine):
    try:
        file_name = os.path.basename(file_path).lower()

        # Читаем CSV — Ашан использует ; и кавычки
        df = pd.read_csv(file_path, dtype=str, sep=';', quotechar='"')
        df['sale_year'] = 2025
        df['sale_month'] = 1

        df = df.fillna('').astype(str)

        logger.info(f"[INFO] Колонки в файле: {df.columns.tolist()}")

        # Генерация имени таблицы
        name = os.path.splitext(file_name)[0]
        table_name = re.sub(r'\W+', '_', name)

        with engine.connect() as conn:
            # Проверка существования таблицы
            result = conn.execute(
                text("""
                    SELECT 1 
                    FROM information_schema.tables 
                    WHERE table_schema = 'raw' AND table_name = :table
                """),
                {"table": table_name}
            ).first()

            if not result:
                columns_sql = ', '.join([f'[{col}] NVARCHAR(MAX)' for col in df.columns])
                create_sql = f"CREATE TABLE raw.{table_name} ({columns_sql})"
                logger.info(f"[INFO] Создаём таблицу raw.{table_name}")
                with conn.begin():
                    conn.execute(text(create_sql))
                logger.info(f"[SUCCESS] Таблица raw.{table_name} создана")
            else:
                logger.info(f"[INFO] Таблица raw.{table_name} уже существует")

            # Проверка наличия данных
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM raw.{table_name}")).scalar()
            if row_count > 0:
                logger.warning(f"[SKIP] Таблица raw.{table_name} уже содержит данные ({row_count} строк). Загрузка пропущена.")
                return table_name

        # Настройка fast_executemany
        raw_conn = engine.raw_connection()
        cursor = raw_conn.cursor()
        cursor.fast_executemany = True

        @event.listens_for(Engine, "before_cursor_execute")
        def _enable_fast_executemany(conn, cursor, statement, parameters, context, executemany):
            if executemany:
                cursor.fast_executemany = True

        logger.info(f"[INFO] Начинаем загрузку {len(df)} строк в raw.{table_name}")
        start = time.time()

        df.to_sql(
            name=table_name,
            schema='raw',
            con=engine,
            if_exists='append',
            index=False,
            chunksize=1000
        )

        duration = time.time() - start
        logger.info(f"[SUCCESS] Загружено {len(df)} строк в raw.{table_name} за {duration:.2f} сек.")

        cursor.close()
        raw_conn.close()

        return table_name

    except Exception as e:
        logger.error(f"[ERROR] Ошибка при загрузке файла {file_path}: {e}")
        raise
