import os
import re
import logging
import time
from contextlib import closing
from io import StringIO
from typing import Optional, Tuple, Dict, List, Union

import pyodbc  # noqa: F401  # required for SQL Server ODBC connections behind SQLAlchemy raw_connection()
import pandas as pd
import numpy as np
from sqlalchemy import text, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import pickle 

logger = logging.getLogger(__name__)


class OkeyTableProcessor:


    CHUNKSIZE = 100_000
    BATCH_SIZE = 10_000
    MAX_ROWS = 10_000  # safety cap
    enrichment_models_loaded = False

    # month tokens recognised in filenames (both rus & simple translit typos like "ceptember")
    _MONTH_MAP = {
        # full rus
        'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6,
        'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12,
        # short rus
        'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'май_': 5, 'май': 5, 'июн': 6,
        'июл': 7, 'авг': 8, 'сен': 9, 'сент': 9, 'окт': 10, 'ноя': 11, 'дек': 12,
        # crude latin translits frequently seen in filenames
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'sept': 9, 'september': 9, 'ceptember': 9,  # <<< typo in sample filename
        'oct': 10, 'nov': 11, 'dec': 12,
    }

    def __init__(self):
        self.enrichment_models_loaded = False
        self._load_enrichment_models()

    @classmethod
    def extract_okey_metadata(cls, source: str) -> Tuple[Optional[int], Optional[int]]:
        src = str(source).lower()
        # split into alpha tokens & 4‑digit numeric tokens
        tokens = re.findall(r"[a-zа-я]+|\d{4}", src)
        year = next((int(t) for t in tokens if t.isdigit() and len(t) == 4), None)
        month = None
        for t in tokens:
            if t in cls._MONTH_MAP:
                month = cls._MONTH_MAP[t]
                break
        return month, year


    def _load_enrichment_models(self):
        def load_model_and_vectorizer(model_name: str, folder: str):
            with open(f"{folder}/{model_name}_model.pkl", "rb") as f_model:
                model = pickle.load(f_model)
            with open(f"{folder}/{model_name}_vectorizer.pkl", "rb") as f_vec:
                vectorizer = pickle.load(f_vec)
            return model, vectorizer

        # Папки с моделями
        product_dir = "ml_models/product_enrichment"
        address_dir = "ml_models/address_enrichment"

        # Модели обогащения по названию продукта (product_name)
        self.brand_model, self.brand_vectorizer = load_model_and_vectorizer("brand", product_dir)
        self.flavor_model, self.flavor_vectorizer = load_model_and_vectorizer("flavor", product_dir)
        self.weight_model, self.weight_vectorizer = load_model_and_vectorizer("weight", product_dir)
        self.type_model, self.type_vectorizer = load_model_and_vectorizer("type", product_dir)

        # Модели обогащения по адресу
        self.city_model, self.city_vectorizer = load_model_and_vectorizer("city", address_dir)
        self.region_model, self.region_vectorizer = load_model_and_vectorizer("region", address_dir)
        self.branch_model, self.branch_vectorizer = load_model_and_vectorizer("branch", address_dir)

        self.enrichment_models_loaded = True

    def _enrich_product_data(self, df: pd.DataFrame) -> pd.DataFrame:
            if not self.enrichment_models_loaded:
                self._load_enrichment_models()

            if 'product_name' in df.columns:
                product_names = df['product_name'].fillna("")

                def predict(model, vectorizer):
                    try:
                        X_vec = vectorizer.transform(product_names)
                        return model.predict(X_vec)
                    except Exception:
                        return [""] * len(product_names)

                df['brand_predicted'] = predict(self.brand_model, self.brand_vectorizer)
                df['flavor_predicted'] = predict(self.flavor_model, self.flavor_vectorizer)
                df['weight_predicted'] = predict(self.weight_model, self.weight_vectorizer)
                df['type_predicted'] = predict(self.type_model, self.type_vectorizer)

            address_col_candidates = [c for c in df.columns if any(k in c.lower() for k in ['адрес', 'address'])]
            if address_col_candidates:
                address_col = address_col_candidates[0]
                addresses = df[address_col].fillna("")

                def predict_address(model, vectorizer):
                    try:
                        X_vec = vectorizer.transform(addresses)
                        return model.predict(X_vec)
                    except Exception:
                        return [""] * len(addresses)

                df['city_predicted'] = predict_address(self.city_model, self.city_vectorizer)
                df['region_predicted'] = predict_address(self.region_model, self.region_vectorizer)
                df['branch_predicted'] = predict_address(self.branch_model, self.branch_vectorizer)

            return df



    # ------------------------------------------------------------------
    # Column normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_headers(cols: List[str]) -> List[str]:
        """Lowercase, strip, collapse whitespace/punct to underscore."""
        normed = []
        for c in cols:
            c = str(c).strip().lower()
            c = re.sub(r"[\s\./,;]+", "_", c)
            c = re.sub(r"_+", "_", c)
            c = c.strip("_")
            normed.append(c)
        # resolve duplicates
        seen: Dict[str, int] = {}
        out = []
        for c in normed:
            if c in seen:
                seen[c] += 1
                out.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                out.append(c)
        return out

    @classmethod
    def normalize_okey_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Map Russian O'KEY headers to raw canonical names used downstream."""
        df = df.copy()
        df.columns = cls._normalise_headers(df.columns.tolist())

        # mapping after normalisation
        column_mapping = {
            'период': 'period',
            'сеть': 'retail_chain',
            'категория': 'category',
            'категория_2': 'base_type',  # secondary category -> base_type analogue
            'поставщик': 'supplier',
            'бренд': 'brand',
            'наименование': 'product_name',
            'уни_наименование': 'unified_product_name',
            'уни_наименование_наименование': 'unified_product_name',  # guard weird merge errors
            'граммовка': 'weight',
            'вкус': 'flavor',
            'продажи_шт': 'sales_units',
            'продажи_руб': 'sales_rub',
            'продажи_тонн': 'sales_tonnes',
            'себест_руб': 'cost_rub',
            'себест__руб': 'cost_rub',
        }

        df.columns = [column_mapping.get(col, col) for col in df.columns]
        return df

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------
    @classmethod
    def _safe_read_okey_csv(cls, file_path: str) -> pd.DataFrame:
        """CSV reader (tab or semicolon). Reads only first MAX_ROWS rows."""
        # try semicolon first, fall back to tab
        try:
            df = pd.read_csv(
                file_path,
                dtype='string',
                sep=';',
                quotechar='"',
                on_bad_lines='warn',
                encoding='utf-8-sig',
                decimal=',',
                thousands=' ',
                nrows=cls.MAX_ROWS,
            )
            if df.shape[1] == 1:  # probably wrong sep
                df = pd.read_csv(
                    file_path,
                    dtype='string',
                    sep='\t',
                    quotechar='"',
                    on_bad_lines='warn',
                    encoding='utf-8-sig',
                    decimal=',',
                    thousands=' ',
                    nrows=cls.MAX_ROWS,
                )
            return df
        except pd.errors.ParserError:
            logger.warning("CSV parse error, attempting sanitised re‑read…")
            with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                lines = [f.readline() for _ in range(cls.MAX_ROWS + 1)]
            content = ''.join(lines)
            clean_content = re.sub(r'(?<!\\)"(?!\\)', '', content)
            return pd.read_csv(
                StringIO(clean_content),
                dtype='string',
                sep=';',
                quotechar='"',
                decimal=',',
                thousands=' ',
            )

    @classmethod
    def _read_xlsb_file(cls, file_path: str) -> Dict[str, pd.DataFrame]:
        """Read XLSB (subset MAX_ROWS)."""
        try:
            import pyxlsb  # local import
            out: Dict[str, pd.DataFrame] = {}
            with pyxlsb.open_workbook(file_path) as wb:
                for sheet_name in wb.sheets:
                    with wb.get_sheet(sheet_name) as sheet:
                        data = []
                        headers = None
                        for i, row in enumerate(sheet.rows()):
                            if i > cls.MAX_ROWS:
                                break
                            if i == 0:
                                headers = [str(c.v) if c.v is not None else f"none_{idx}" for idx, c in enumerate(row)]
                            else:
                                data.append([str(c.v) if c.v is not None else '' for c in row])
                        if headers and data:
                            out[sheet_name] = pd.DataFrame(data, columns=headers)
            if not out:
                raise ValueError("XLSB file contains no data")
            return out
        except Exception as e:  # noqa: BLE001
            logger.error("XLSB read error: %s", e)
            raise

    @classmethod
    def _read_excel_file(cls, file_path: str) -> Dict[str, pd.DataFrame]:
        """Read XLSX (subset MAX_ROWS)."""
        try:
            return pd.read_excel(
                file_path,
                sheet_name=None,
                dtype='string',
                engine='openpyxl',
                decimal=',',
                thousands=' ',
                nrows=cls.MAX_ROWS,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Excel read error: %s", e)
            raise


    def _process_and_insert_chunk(self, df, table_name, fname_month, fname_year, engine, create_table: bool):
            df = self.normalize_okey_columns(df)

            # Метаданные
            if (fname_month is None or fname_year is None) and 'period' in df.columns and not df['period'].isna().all():
                m2, y2 = self.extract_okey_metadata(str(df['period'].iloc[0]))
                fname_month = fname_month or m2
                fname_year = fname_year or y2

            if fname_month and fname_year:
                df['sale_year'] = str(fname_year)
                df['sale_month'] = str(fname_month).zfill(2)

            df = df.fillna('')
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = (
                        df[col].str.strip()
                            .str.replace('\u200b', '', regex=False)
                            .str.replace('\r\n', ' ', regex=False)
                            .str.replace('\n', ' ', regex=False)
                    )

            try:
                df = self._enrich_product_data(df)
            except Exception as e:
                print(f"[WARN] Обогащение не выполнено: {e}")

            if create_table:
                with engine.begin() as conn:
                    exists = conn.execute(
                        text("SELECT 1 FROM information_schema.tables WHERE table_schema = 'okey' AND table_name = :table"),
                        {"table": table_name}
                    ).scalar()
                    if not exists:
                        cols_sql = [f"[{col}] NVARCHAR(255)" for col in df.columns]
                        create_sql = f"CREATE TABLE okey.{table_name} ({', '.join(cols_sql)})"
                        conn.execute(text(create_sql))

            self.bulk_insert_okey(df, table_name, engine)



    def bulk_insert_okey(self, df, table_name, engine):
            conn = engine.raw_connection()
            cursor = conn.cursor()
            cursor.fast_executemany = True

            cols = df.columns.tolist()
            insert_sql = f"INSERT INTO okey.{table_name} ({', '.join([f'[{c}]' for c in cols])}) VALUES ({', '.join(['?' for _ in cols])})"

            data = df.values.tolist()
            for i in range(0, len(data), self.BATCH_SIZE):
                batch = data[i:i + self.BATCH_SIZE]
                cursor.executemany(insert_sql, batch)

            conn.commit()
            cursor.close()
            conn.close()


    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------
    def process_okey_file(self, file_path: str, engine: Engine) -> str:
            import time
            start_time = time.time()

            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            table_name = re.sub(r"\W+", "_", base_name.lower())

            if file_path.endswith('.csv'):
                reader = pd.read_csv(
                    file_path,
                    dtype='string',
                    sep=';',
                    quotechar='"',
                    decimal=',',
                    thousands=' ',
                    chunksize=self.CHUNKSIZE,
                    on_bad_lines='warn',
                    encoding='utf-8-sig',
                )
            elif file_path.endswith('.xlsx'):
                reader = pd.read_excel(file_path, sheet_name=None)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            fname_month, fname_year = self.extract_okey_metadata(file_name)
            first_chunk = True

            for chunk in (reader if isinstance(reader, pd.io.parsers.TextFileReader) else [reader]):
                if isinstance(chunk, dict):  # multiple sheets
                    for _, df in chunk.items():
                        self._process_and_insert_chunk(df, table_name, fname_month, fname_year, engine, first_chunk)
                        first_chunk = False
                else:
                    self._process_and_insert_chunk(chunk, table_name, fname_month, fname_year, engine, first_chunk)
                    first_chunk = False

            duration = time.time() - start_time
            print(f"[Okey] Загружено в okey.{table_name} за {duration:.2f} сек")
            return table_name


def create_okey_table_and_upload(file_path: str, engine: Engine) -> str:
    """Convenience wrapper for DAG usage."""
    try:
        processor = OkeyTableProcessor()
        return processor.process_okey_file(file_path, engine)
    except Exception as e:  # noqa: BLE001
        logger.error("Критическая ошибка при обработке файла %s: %s", file_path, e, exc_info=True)
        raise
