import os
import re
import logging
import time
from contextlib import closing
from io import StringIO

import pyodbc
import pandas as pd
import numpy as np
from sqlalchemy import text, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class TableProcessor:
    CHUNKSIZE = 100_000
    BATCH_SIZE = 10_000

    MONTHS = {
        **{m.lower(): i + 1 for i, m in enumerate([
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        ])},
        **{m.lower(): i + 1 for i, m in enumerate([
            'январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
            'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'
        ])}
    }

    @staticmethod
    def extract_metadata(source: str) -> tuple[int | None, int | None]:
        tokens = re.findall(r'[a-zа-я]+|\d{4}', str(source).lower())
        year = next((int(t) for t in tokens if t.isdigit() and len(t) == 4), None)
        month = next((TableProcessor.MONTHS.get(t) for t in tokens if t in TableProcessor.MONTHS), None)
        return month, year

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [re.sub(r'\s+', '_', str(col).strip().lower()) for col in df.columns]
        return df

    @classmethod
    def _safe_read_csv(cls, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(
                file_path,
                dtype='string',
                sep=';',
                quotechar='"',
                on_bad_lines='warn',
                encoding_errors='replace'
            )
        except pd.errors.ParserError:
            logger.warning("Ошибка парсинга CSV, попытка ручной очистки.")
            clean_lines = []
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                header = f.readline().strip()
                clean_lines.append(header)

                for line in f:
                    if line.count('"') % 2 == 0:
                        clean_lines.append(line.strip())

            clean_csv = StringIO('\n'.join(clean_lines))
            return pd.read_csv(clean_csv, dtype='string', sep=';', quotechar='"')

    @classmethod
    def process_file(cls, file_path: str, engine: Engine) -> str:
        start_time = time.time()
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        table_name = re.sub(r'\W+', '_', base_name.lower())

        # Чтение файла
        if file_path.endswith('.xlsx'):
            try:
                reader = pd.read_excel(file_path, sheet_name=None, dtype='string', engine='openpyxl')
            except Exception as e:
                logger.error(f"Ошибка чтения Excel: {e}")
                raise
        elif file_path.endswith('.csv'):
            reader = {'data': cls._safe_read_csv(file_path)}
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path}")

        processed_chunks = []

        for sheet_name, df in reader.items():
            try:
                if df.empty:
                    continue

                df = cls.normalize_columns(df)
                df = df.head(20000)  # ограничение для отладки / скорости

                month, year = cls.extract_metadata(file_name)
                if not month and 'month' in df.columns:
                    month, year = cls.extract_metadata(df['month'].iloc[0])

                if month and year:
                    df = df.assign(sale_year=str(year), sale_month=str(month))

                # Преобразуем все данные к строкам (тексту) без исключений!
                df = df.where(pd.notnull(df), None)  # None для NaN
                for col in df.columns:
                    df[col] = df[col].astype('string')

                processed_chunks.append(df)
            except Exception as e:
                logger.error(f"Ошибка обработки листа {sheet_name}: {e}", exc_info=True)
                continue

        if not processed_chunks:
            raise ValueError("Файл не содержит валидных данных.")

        final_df = pd.concat(processed_chunks, ignore_index=True)
        del processed_chunks

        with engine.begin() as conn:
            table_exists = conn.execute(
                text("""
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'raw' AND table_name = :table
                """),
                {"table": table_name}
            ).scalar()

            if not table_exists:
                columns_sql = [f'[{col}] NVARCHAR(255)' for col in final_df.columns]
                create_sql = f"CREATE TABLE raw.{table_name} ({', '.join(columns_sql)})"
                conn.execute(text(create_sql))

        cls.bulk_insert(final_df, table_name, engine)

        duration = time.time() - start_time
        logger.info(f"Файл {file_name} загружен в raw.{table_name} за {duration:.2f} сек ({len(final_df)} строк).")
        return table_name

    @classmethod
    def bulk_insert(cls, df: pd.DataFrame, table_name: str, engine: Engine):
        @event.listens_for(engine, 'before_cursor_execute')
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            if executemany:
                cursor.fast_executemany = True

        try:
            # Конвертируем все данные в строки и обрабатываем NULL
            df_str = df.where(pd.notnull(df), None)
            for col in df_str.columns:
                df_str[col] = df_str[col].astype(str).str.slice(0, 255)
            
            # Подготовка данных - гарантируем, что все значения либо строки, либо None
            data = []
            for row in df_str.itertuples(index=False, name=None):
                clean_row = []
                for value in row:
                    if value is None or pd.isna(value):
                        clean_row.append(None)
                    else:
                        # Явное преобразование в строку и обрезка
                        clean_row.append(str(value)[:255] if value is not None else None)
                data.append(tuple(clean_row))

            with closing(engine.raw_connection()) as conn:
                with conn.cursor() as cursor:
                    cursor.fast_executemany = True
                    
                    # Явно указываем тип параметров как NVARCHAR для всех столбцов
                    input_sizes = [(pyodbc.SQL_WVARCHAR, 255, 0)] * len(df_str.columns)
                    cursor.setinputsizes(input_sizes)

                    # SQL-запрос
                    cols = ', '.join(f'[{col}]' for col in df_str.columns)
                    params = ', '.join(['?'] * len(df_str.columns))
                    insert_sql = f"INSERT INTO raw.{table_name} ({cols}) VALUES ({params})"

                    # Вставка небольшими батчами с обработкой ошибок
                    for i in range(0, len(data), cls.BATCH_SIZE):
                        batch = data[i:i + cls.BATCH_SIZE]
                        try:
                            cursor.executemany(insert_sql, batch)
                            conn.commit()
                        except Exception as e:
                            logger.error(f"Ошибка вставки батча [{i}:{i+len(batch)}]: {e}")
                            
                            # Попробуем вставить построчно для диагностики
                            for j, row_data in enumerate(batch):
                                try:
                                    cursor.execute(insert_sql, row_data)
                                    conn.commit()
                                except Exception as row_error:
                                    logger.error(f"ОШИБКА В СТРОКЕ {i+j}: {row_error}")
                                    logger.error(f"Проблемные данные: {row_data}")
                                    conn.rollback()
                                    raise
                            raise

        except Exception as e:
            logger.error(f"Ошибка вставки в таблицу raw.{table_name}: {e}", exc_info=True)
            raise

def create_table_and_upload(file_path: str, engine: Engine) -> str:
    return TableProcessor.process_file(file_path, engine)
