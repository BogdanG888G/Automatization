import os
import re
import logging
import time
from contextlib import closing
from io import StringIO
import pickle
from sklearn.base import BaseEstimator
import pyodbc
import pandas as pd
import numpy as np
from sqlalchemy import text, event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class TableProcessorDiksi:
    CHUNKSIZE = 100_000
    CHUNK_SIZE = 50_000
    BATCH_SIZE = 50_000

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
    def enrich_with_ml_models(df: pd.DataFrame) -> pd.DataFrame:
        """
        Обогащает DataFrame колонками: brand, flavor, weight, product_type, city, region, district, branch.
        """
        name_col_candidates = [c for c in df.columns if 'наимен' in c.lower() or 'тов' in c.lower()]
        if not name_col_candidates:
            logger.warning("[Diksi] Не найдена колонка с названием продукта — enrichment пропущен.")
            return df

        product_col = name_col_candidates[0]
        df['товар'] = df[product_col]

        model_product_dir = "ml_models/product_enrichment"
        model_address_dir = "ml_models/address_enrichment"
        model_paths = {
            'predicted_flavor': ("flavor_model.pkl", "flavor_vectorizer.pkl"),
            'predicted_brand': ("brand_model.pkl", "brand_vectorizer.pkl"),
            'predicted_weight': ("weight_model.pkl", "weight_vectorizer.pkl"),
            'predicted_product_type': ("type_model.pkl", "type_vectorizer.pkl"),
        }

        def load_pickle(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        # Обогащение по продукту
        for col_name, (model_file, vec_file) in model_paths.items():
            try:
                model = load_pickle(os.path.join(model_product_dir, model_file))
                vectorizer = load_pickle(os.path.join(model_product_dir, vec_file))
                vec = vectorizer.transform(df['товар'].astype(str))
                df[col_name] = model.predict(vec)
            except Exception as e:
                logger.warning(f"[Diksi] Не удалось обогатить '{col_name}': {e}")

        # =============================
        # Обогащение по адресу
        # =============================
        address_col_candidates = [c for c in df.columns if 'адрес' in c.lower()]
        if not address_col_candidates:
            logger.warning("[Diksi] Не найдена колонка с адресом — enrichment по адресу пропущен.")
            return df

        address_col = address_col_candidates[0]
        df['адрес'] = df[address_col]

        address_model_paths = {
            'predicted_city': ("city_model.pkl", "city_vectorizer.pkl"),
            'predicted_region': ("region_model.pkl", "region_vectorizer.pkl"),
            'predicted_branch': ("branch_model.pkl", "branch_vectorizer.pkl"),
        }

        for col_name, (model_file, vec_file) in address_model_paths.items():
            try:
                model = load_pickle(os.path.join(model_address_dir, model_file))
                vectorizer = load_pickle(os.path.join(model_address_dir, vec_file))
                vec = vectorizer.transform(df['адрес'].astype(str))
                df[col_name] = model.predict(vec)
            except Exception as e:
                logger.warning(f"[Diksi] Не удалось обогатить '{col_name}': {e}")

        return df


    @staticmethod
    def extract_metadata(source: str) -> tuple[int | None, int | None]:
        tokens = re.findall(r'[a-zа-я]+|\d{4}', str(source).lower())
        year = next((int(t) for t in tokens if t.isdigit() and len(t) == 4), None)
        month = next((TableProcessorDiksi.MONTHS.get(t) for t in tokens if t in TableProcessorDiksi.MONTHS), None)
        return month, year

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        new_cols = []
        seen = {}

        for col in df.columns:
            base_name = re.sub(r'\s+', '_', str(col).strip().lower())
            if base_name in seen:
                seen[base_name] += 1
                new_name = f"{base_name}_{seen[base_name]}"
            else:
                seen[base_name] = 0
                new_name = base_name
            new_cols.append(new_name)

        df.columns = new_cols
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
                encoding='utf-8',
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
        import math
        start_time = time.time()
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        table_name = re.sub(r'\W+', '_', base_name.lower())
        enriched = False  # enrichment только один раз

        with engine.begin() as conn:
            table_exists = conn.execute(
                text("""
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'raw' AND table_name = :table
                """),
                {"table": table_name}
            ).scalar()

        if file_path.endswith('.xlsx') or file_path.endswith('.xlsb'):
            engine_name = 'openpyxl' if file_path.endswith('.xlsx') else 'pyxlsb'
            reader = pd.read_excel(file_path, sheet_name=None, dtype='string', engine=engine_name, skiprows=1)

            for sheet_name, df in reader.items():
                if df.empty:
                    continue

                df = cls.normalize_columns(df)

                # enrichment
                if not enriched:
                    df = cls.enrich_with_ml_models(df)
                    enriched = True

                month, year = cls.extract_metadata(file_name)
                if not month and 'month' in df.columns:
                    month, year = cls.extract_metadata(df['month'].iloc[0])

                if month and year:
                    df['sale_year'] = str(year)
                    df['sale_month'] = str(month)

                if not table_exists:
                    with engine.begin() as conn:
                        columns_sql = [f'[{col}] NVARCHAR(255)' for col in df.columns]
                        create_sql = f"CREATE TABLE raw.{table_name} ({', '.join(columns_sql)})"
                        conn.execute(text(create_sql))
                    table_exists = True

                total_rows = len(df)
                chunks_count = math.ceil(total_rows / cls.CHUNK_SIZE)
                for i in range(chunks_count):
                    chunk = df.iloc[i*cls.CHUNK_SIZE : (i+1)*cls.CHUNK_SIZE]
                    chunk = chunk.where(pd.notnull(chunk), None)
                    for col in chunk.columns:
                        chunk[col] = chunk[col].astype('string')
                    cls.bulk_insert(chunk, table_name, engine)

        elif file_path.endswith('.csv'):
            csv_iter = pd.read_csv(file_path, dtype='string', sep=';', chunksize=cls.CHUNK_SIZE)
            for chunk in csv_iter:
                chunk = cls.normalize_columns(chunk)

                # enrichment
                if not enriched:
                    chunk = cls.enrich_with_ml_models(chunk)
                    enriched = True

                month, year = cls.extract_metadata(file_name)
                if not month and 'month' in chunk.columns:
                    month, year = cls.extract_metadata(chunk['month'].iloc[0])

                if month and year:
                    chunk['sale_year'] = str(year)
                    chunk['sale_month'] = str(month)

                chunk = chunk.where(pd.notnull(chunk), None)
                for col in chunk.columns:
                    chunk[col] = chunk[col].astype('string')

                if not table_exists:
                    with engine.begin() as conn:
                        columns_sql = [f'[{col}] NVARCHAR(255)' for col in chunk.columns]
                        create_sql = f"CREATE TABLE raw.{table_name} ({', '.join(columns_sql)})"
                        conn.execute(text(create_sql))
                    table_exists = True

                cls.bulk_insert(chunk, table_name, engine)

        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path}")

        duration = time.time() - start_time
        logger.info(f"Файл {file_name} загружен в raw.{table_name} за {duration:.2f} сек.")
        return table_name


    @classmethod
    def bulk_insert(cls, df: pd.DataFrame, table_name: str, engine: Engine):
        @event.listens_for(engine, 'before_cursor_execute')
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            if executemany:
                cursor.fast_executemany = True

        try:
            df_str = df.where(pd.notnull(df), None)
            for col in df_str.columns:
                df_str[col] = df_str[col].astype(str).str.slice(0, 255)

            with closing(engine.raw_connection()) as conn:
                with conn.cursor() as cursor:
                    cursor.fast_executemany = True
                    input_sizes = [(pyodbc.SQL_WVARCHAR, 255, 0)] * len(df_str.columns)
                    cursor.setinputsizes(input_sizes)

                    cols = ', '.join(f'[{col}]' for col in df_str.columns)
                    params = ', '.join(['?'] * len(df_str.columns))
                    insert_sql = f"INSERT INTO raw.{table_name} ({cols}) VALUES ({params})"

                    for i in range(0, len(df_str), cls.BATCH_SIZE):
                        batch_df = df_str.iloc[i:i + cls.BATCH_SIZE]
                        data = []
                        for row in batch_df.itertuples(index=False, name=None):
                            clean_row = []
                            for value in row:
                                clean_row.append(None if value is None or pd.isna(value) else str(value)[:255])
                            data.append(tuple(clean_row))
                        try:
                            cursor.executemany(insert_sql, data)
                            conn.commit()
                        except Exception as e:
                            logger.error(f"Ошибка вставки батча [{i}:{i+len(data)}]: {e}")
                            for j, row_data in enumerate(data):
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


def create_table_and_upload_diksi(file_path: str, engine: Engine) -> str:
    return TableProcessorDiksi.process_file(file_path, engine)
