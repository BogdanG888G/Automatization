import pandas as pd
from sqlalchemy import text
import os
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_table_and_upload(file_path, engine):
    try:
        file_name = os.path.basename(file_path).lower()
        df = pd.read_csv(file_path, dtype=str, sep=';', quotechar='"')
        df = df.head(100)
        df['sale_year'] = 2025
        df['sale_month'] = 1
        df = df.astype(str)
        logger.info(f">>> Заголовки колонок: {df.columns.tolist()}")

        name = os.path.basename(file_path).replace('.csv', '')
        table_name = re.sub(r'\W+', '_', name.lower())

        with engine.connect() as conn:
            # Проверяем существует ли таблица
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
                
                logger.info(f"[INFO] Создаём таблицу raw.{table_name} с колонками: {df.columns.tolist()}")

                with conn.begin():
                    conn.execute(text(create_stmt))
                logger.info(f"[SUCCESS] Таблица raw.{table_name} создана")
            else:
                logger.info(f"[INFO] Таблица raw.{table_name} уже существует")

            logger.info(f"Загружаем {len(df)} строк с {len(df.columns)} колонками в таблицу {table_name}")

            df = df.fillna('').astype(str)
            
            # Выводим пользователя для отладки
            user_result = conn.execute(text("SELECT SYSTEM_USER"))
            logger.info(f"Используемый пользователь: {user_result.scalar()}")

            # Вставка данных с коммитом в транзакции
            with conn.begin():
                df.to_sql(
                    name=table_name,
                    schema='raw',
                    con=conn,
                    if_exists='append',
                    index=False,
                    chunksize=500
                )
            logger.info(f"[SUCCESS] Данные загружены в raw.{table_name}")

        return table_name

    except Exception as e:
        logger.error(f"[ERROR] Ошибка при обработке файла {file_path}: {e}")
        raise
