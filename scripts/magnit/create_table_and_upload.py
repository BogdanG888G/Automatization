import os
import re
import logging
import time
from io import StringIO
from contextlib import closing
import pandas as pd
from sqlalchemy import text, event
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class TableProcessor:
    CHUNKSIZE = 100000
    BATCH_SIZE = 50000

    MONTHS = {
        **{m.lower(): i+1 for i, m in enumerate([
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'])},
        **{m.lower(): i+1 for i, m in enumerate([
            'январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
            'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'])}
    }

    @staticmethod
    def extract_metadata(filename: str) -> tuple:
        name = os.path.splitext(os.path.basename(filename))[0].lower()
        tokens = re.findall(r'[a-zа-я]+|\d{4}', name)
        year = next((int(t) for t in tokens if t.isdigit() and len(t) == 4), None)
        month = next((TableProcessor.MONTHS.get(t) for t in tokens if t in TableProcessor.MONTHS), None)
        return month, year

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [re.sub(r'\s+', '_', col.strip().lower()) for col in df.columns]
        return df

    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('string')
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

    @classmethod
    def detect_format(cls, df: pd.DataFrame, year: int) -> str:
        cols = set(df.columns.str.lower())
        if year == 2023:
            return 'magnit_2023'
        elif year == 2024:
            if {'sku', 'shop', 'sales'}.intersection(cols):
                return 'magnit_2024_v1'
            elif {'код_товара', 'магазин', 'отгрузка'}.intersection(cols):
                return 'magnit_2024_v2'
        raise ValueError("Неизвестный формат Магнита")

    @classmethod
    def parse_magnit_2023(cls, df):
        df = df.rename(columns={'товар': 'sku', 'магазин': 'shop', 'кол-во': 'qty'})
        return df

    @classmethod
    def parse_magnit_2024_v1(cls, df):
        df = df.rename(columns={'sku': 'sku', 'shop': 'shop', 'sales': 'qty'})
        return df

    @classmethod
    def parse_magnit_2024_v2(cls, df):
        df = df.rename(columns={'код_товара': 'sku', 'магазин': 'shop', 'отгрузка': 'qty'})
        return df

    @classmethod
    def process_sheet(cls, df: pd.DataFrame, file_name: str):
        if df.empty:
            return None
        df = cls.normalize_columns(df)
        month, year = cls.extract_metadata(file_name)
        df_format = cls.detect_format(df, year)

        if df_format == 'magnit_2023':
            df = cls.parse_magnit_2023(df)
        elif df_format == 'magnit_2024_v1':
            df = cls.parse_magnit_2024_v1(df)
        elif df_format == 'magnit_2024_v2':
            df = cls.parse_magnit_2024_v2(df)

        df['sale_year'] = str(year)
        df['sale_month'] = str(month)
        return cls.optimize_dataframe(df)

    @classmethod
    def process_file(cls, file_path: str, engine):
        try:
            start_time = time.time()
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            table_name = re.sub(r'\W+', '_', base_name)

            reader = pd.read_excel(file_path, sheet_name=None, dtype='string', engine='openpyxl')

            processed_chunks = []
            for sheet_name, df in reader.items():
                try:
                    parsed = cls.process_sheet(df, file_name)
                    if parsed is not None:
                        processed_chunks.append(parsed)
                except Exception as e:
                    logger.warning(f"Ошибка в листе {sheet_name}: {e}")
                    continue

            if not processed_chunks:
                raise ValueError("Все листы пусты или с ошибками")

            final_df = pd.concat(processed_chunks, ignore_index=True)

            with engine.connect() as conn:
                table_exists = conn.execute(
                    text("""
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'raw' AND table_name = :table
                    """),
                    {"table": table_name}
                ).scalar()

                if not table_exists:
                    columns_sql = []
                    for col, dtype in zip(final_df.columns, final_df.dtypes):
                        sql_type = 'NVARCHAR(255)' if dtype == 'string' else 'FLOAT'
                        columns_sql.append(f'[{col}] {sql_type}')
                    create_sql = f"CREATE TABLE raw.{table_name} ({', '.join(columns_sql)})"
                    conn.execute(text(create_sql))
                    conn.commit()

            cls.bulk_insert(final_df, table_name, engine)

            logger.info(f"Файл {file_name} обработан за {time.time()-start_time:.2f} сек. Загружено {len(final_df)} строк в raw.{table_name}")
            return table_name

        except Exception as e:
            logger.error(f"Ошибка при обработке {file_path}: {e}", exc_info=True)
            raise

    @classmethod
    def bulk_insert(cls, df: pd.DataFrame, table_name: str, engine):
        @event.listens_for(engine, 'before_cursor_execute')
        def set_fast_executemany(conn, cursor, stmt, params, context, executemany):
            if executemany:
                cursor.fast_executemany = True
        try:
            with closing(engine.raw_connection()) as conn:
                with conn.cursor() as cursor:
                    data = [tuple(x) for x in df.itertuples(index=False, name=None)]
                    cols = ', '.join([f'[{col}]' for col in df.columns])
                    params = ', '.join(['?'] * len(df.columns))
                    for i in range(0, len(data), cls.BATCH_SIZE):
                        batch = data[i:i + cls.BATCH_SIZE]
                        insert_sql = f"INSERT INTO raw.{table_name} ({cols}) VALUES ({params})"
                        cursor.executemany(insert_sql, batch)
                        conn.commit()
        except SQLAlchemyError as e:
            logger.error(f"Ошибка вставки в {table_name}: {e}")
            raise

def create_table_and_upload(file_path: str, engine):
    return TableProcessor.process_file(file_path, engine)
