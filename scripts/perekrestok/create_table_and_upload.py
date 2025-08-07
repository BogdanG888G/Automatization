import os
import re
import logging
import time
from contextlib import closing
from io import StringIO
from typing import Optional, Tuple, Dict, List

import pyodbc  # noqa: F401  # required for SQL Server ODBC connections via raw_connection()
import pandas as pd
import numpy as np
from sqlalchemy import text, event
from sqlalchemy.engine import Engine
import pickle 
logger = logging.getLogger(__name__)


class PerekrestokTableProcessor:

    CHUNKSIZE = 100_000
    BATCH_SIZE = 50_000
    MAX_ROWS = None  # safety cap
    enrichment_models_loaded = False

    # Month map: full, short, dotted, digit mix, latin translit, common typos
    _MONTH_MAP: Dict[str, int] = {
        # full rus
        'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6,
        'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12,
        # short rus (various endings)
        'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'июн': 6, 'июл': 7, 'авг': 8,
        'сен': 9, 'сент': 9, 'окт': 10, 'ноя': 11, 'дек': 12,
        # dotted short tokens often seen in period column (сен.24)
        'янв.': 1, 'фев.': 2, 'мар.': 3, 'апр.': 4, 'май.': 5, 'июн.': 6,
        'июл.': 7, 'авг.': 8, 'сен.': 9, 'окт.': 10, 'ноя.': 11, 'дек.': 12,
        # latin translit & typos
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'sept': 9, 'september': 9, 'ceptember': 9,
        'oct': 10, 'nov': 11, 'dec': 12,
    }

    # ------------------------------------------------------------------ #
    # Period / metadata extraction                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def extract_perek_metadata(cls, source: str) -> Tuple[Optional[int], Optional[int]]:

        src = str(source).lower().strip()

        # quick try: period tokens like 'сен.24'
        m = re.match(r'([a-zа-я\.]+)[\s\-_]?(?:\'?)(\d{2,4})$', src)
        if m:
            m_tok, y_tok = m.groups()
            m_tok = m_tok.strip()
            if m_tok in cls._MONTH_MAP:
                month = cls._MONTH_MAP[m_tok]
                year = cls._coerce_year(y_tok)
                return month, year

        tokens = re.findall(r"[a-zа-я]+|\d{2,4}", src)
        year = None
        month = None
        for t in tokens:
            if t in cls._MONTH_MAP and month is None:
                month = cls._MONTH_MAP[t]
            elif t.isdigit():
                if len(t) == 4 and year is None:
                    year = int(t)
                elif len(t) == 2 and year is None:
                    year = cls._coerce_year(t)
        return month, year

    @staticmethod
    def _coerce_year(y_tok: str) -> Optional[int]:
        try:
            y = int(y_tok)
        except Exception:  # noqa: BLE001
            return None
        if y < 100:  # 2‑digit year heuristic
            return 2000 + y if y < 80 else 1900 + y
        return y

    # ------------------------------------------------------------------ #
    # Header normalisation & RU→EN mapping                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_headers(cols: List[str]) -> List[str]:
        """Lowercase, strip, collapse whitespace/punct to underscore, resolve duplicates."""
        normed = []
        for c in cols:
            c = str(c).strip().lower()
            c = re.sub(r"[\\s\\./,;]+", "_", c)
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
    def normalize_perek_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Map Perekrestok headers to canonical raw names."""
        df = df.copy()
        df.columns = cls._normalise_headers(df.columns.tolist())

        # mapping after normalisation
        column_mapping = {
            # period / chain
            'период': 'period',
            'сеть': 'retail_chain',
            'перекрёсток': 'retail_chain',   # guard rare header row containing value instead?
            'перекресток04': 'retail_chain', # sometimes header broken; fallback

            # regional
            'регион': 'region',
            'город': 'city',
            'адрес': 'address',
            'рц': 'rc_code',
            'тт': 'tt_code',  # store / trading point code

            # product hierarchy
            'категория': 'category',
            'категория_2': 'category_2',
            'тип_основы': 'base_type',

            # commercial
            'поставщик': 'supplier',
            'бренд': 'brand',

            # product names
            'наименование': 'product_name',
            'уни_наименование': 'unified_product_name',
            'уни_наименование_наименование': 'unified_product_name',  # guard weird merges

            # pack / taste
            'граммовка': 'weight_g',
            'вкусы': 'flavor',
            'вкус': 'flavor',

            # metrics
            'продажи_шт': 'sales_units',
            'продажи_шт_': 'sales_units',
            'продажи_руб': 'sales_rub',
            'продажи_тонн': 'sales_tonnes',
            'себест_руб': 'cost_rub',
            'себест__руб': 'cost_rub',
            'себест_руб_': 'cost_rub',
            'себест_руб__': 'cost_rub',

            # alt separators from sample (commas, spaces converted in _normalise_headers)
            'продажи_шт_': 'sales_units',
            'продажи_руб_': 'sales_rub',
            'продажи_тонн_': 'sales_tonnes',
            'себест_руб': 'cost_rub',
        }

        df.columns = [column_mapping.get(col, col) for col in df.columns]
        return df

    # ------------------------------------------------------------------ #
    # Readers                                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def _safe_read_perek_csv(cls, file_path: str) -> pd.DataFrame:
        """CSV reader (try ; then tab). Reads only first MAX_ROWS rows."""
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
            if df.shape[1] == 1:  # wrong sep?
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
            logger.warning("[Perekrestok] CSV parse error, attempting sanitised re‑read…")
            with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                lines = [f.readline() for _ in range(cls.MAX_ROWS + 1)]
            content = ''.join(lines)
            clean_content = re.sub(r'(?<!\\)\"(?!\\)', '', content)
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
            logger.error("[Perekrestok] XLSB read error: %s", e)
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
            logger.error("[Perekrestok] Excel read error: %s", e)
            raise

    @classmethod
    def _enrich_product_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Загружаем модели один раз
        if not cls.enrichment_models_loaded:
            cls._load_enrichment_models()

        # Функция для извлечения веса из строки product_name
        weight_pattern = re.compile(
            r'(\d+(?:[.,]\d+)?)[ ]?(г(?:р|рамм)?|кг|кг\.|гр|грамм)',
            flags=re.IGNORECASE
        )

        def extract_weight(text):
            match = weight_pattern.search(text)
            if match:
                value = match.group(1).replace(',', '.')
                unit = match.group(2).lower()
                try:
                    value_float = float(value)
                    if 'кг' in unit:
                        return value_float * 1000  # перевод в граммы
                    else:
                        return value_float
                except:
                    return None
            return None

        # Обогащение по product_name
        if 'product_name' in df.columns:
            product_names = df['product_name'].fillna("")

            def predict_product_attr(model, vectorizer):
                try:
                    X_vec = vectorizer.transform(product_names)
                    return model.predict(X_vec)
                except Exception:
                    return [""] * len(product_names)

            df['brand_predicted'] = predict_product_attr(cls.brand_model, cls.brand_vectorizer)
            df['flavor_predicted'] = predict_product_attr(cls.flavor_model, cls.flavor_vectorizer)
            df['weight_predicted'] = predict_product_attr(cls.weight_model, cls.weight_vectorizer)
            df['type_predicted'] = predict_product_attr(cls.type_model, cls.type_vectorizer)

            # Добавляем столбец с извлечённым из текста весом
            df['weight_extracted'] = product_names.apply(extract_weight)

        # Обогащение по адресу → город → регион и филиал
        if 'address' in df.columns:
            addresses = df['address'].fillna("")

            try:
                X_city = cls.city_vectorizer.transform(addresses)
                df['city_predicted'] = cls.city_model.predict(X_city)
            except Exception:
                df['city_predicted'] = [""] * len(addresses)

            city_inputs = df['city_predicted'].fillna("").astype(str)

            try:
                X_region = cls.region_vectorizer.transform(city_inputs)
                df['region_predicted'] = cls.region_model.predict(X_region)
            except Exception:
                df['region_predicted'] = [""] * len(city_inputs)

            try:
                X_branch = cls.branch_vectorizer.transform(city_inputs)
                df['branch_predicted'] = cls.branch_model.predict(X_branch)
            except Exception:
                df['branch_predicted'] = [""] * len(city_inputs)

        return df



    @classmethod
    def _process_and_insert_chunk(cls, df, table_name, fname_month, fname_year, engine, create_table: bool):
        # Нормализация и очистка
        df = cls.normalize_perek_columns(df)

        month, year = fname_month, fname_year
        if (month is None or year is None) and 'period' in df.columns and not df['period'].isna().all():
            m2, y2 = cls.extract_perek_metadata(str(df['period'].iloc[0]))
            month = month or m2
            year = year or y2
        if month and year:
            df['sale_year'] = str(year)
            df['sale_month'] = str(month).zfill(2)

        df = df.fillna('')

        # Векторная очистка строковых столбцов
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = (
                    df[col].str.strip()
                        .str.replace('\u200b', '', regex=False)
                        .str.replace('\r\n', ' ', regex=False)
                        .str.replace('\n', ' ', regex=False)
                )

        # --------------------------- #
        # 🔥 Обогащение товара и адреса
        # --------------------------- #
        try:
            df = cls._enrich_product_data(df)
        except Exception as e:
            print(f"[WARN] Не удалось обогатить данные: {e}")

        # --------------------------- #
        # 🧱 Создание таблицы при необходимости
        # --------------------------- #
        if create_table:
            with engine.begin() as conn:
                exists = conn.execute(
                    text("SELECT 1 FROM information_schema.tables WHERE table_schema = 'raw' AND table_name = :table"),
                    {"table": table_name}
                ).scalar()
                if not exists:
                    cols_sql = [f"[{col}] NVARCHAR(255)" for col in df.columns]
                    create_sql = f"CREATE TABLE raw.{table_name} ({', '.join(cols_sql)})"
                    conn.execute(text(create_sql))

        # --------------------------- #
        # 🚀 Вставка
        # --------------------------- #
        cls.bulk_insert_perek(df, table_name, engine)

    # ------------------------------------------------------------------ #
    # Main processing                                                     #
    # ------------------------------------------------------------------ #
    @classmethod
    def process_perek_file(cls, file_path: str, engine):
        import time
        start_time = time.time()
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        table_name = re.sub(r"\W+", "_", base_name.lower())

        # Read file
        if file_path.endswith('.csv'):
            # читаем CSV чанками для памяти
            reader = pd.read_csv(
                file_path,
                dtype='string',
                sep=';',
                quotechar='"',
                decimal=',',
                thousands=' ',
                chunksize=cls.CHUNKSIZE,
                on_bad_lines='warn',
                encoding='utf-8-sig',
            )
        elif file_path.endswith('.xlsx') or file_path.endswith('.xlsb'):
            # для Excel - пока полный прочтём (лучше конвертировать в CSV вне скрипта)
            reader = cls._read_excel_file(file_path) if file_path.endswith('.xlsx') else cls._read_xlsb_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Метаданные
        fname_month, fname_year = cls.extract_perek_metadata(file_name)

        # Создаем таблицу при необходимости
        first_chunk = True
        for chunk in (reader if isinstance(reader, pd.io.parsers.TextFileReader) else [reader]):
            if isinstance(chunk, dict):
                # excel case: dict(sheet_name -> df)
                for sheet_name, df in chunk.items():
                    cls._process_and_insert_chunk(df, table_name, fname_month, fname_year, engine, first_chunk)
                    first_chunk = False
            else:
                # csv chunk case
                cls._process_and_insert_chunk(chunk, table_name, fname_month, fname_year, engine, first_chunk)
                first_chunk = False

        duration = time.time() - start_time
        print(f"[Perekrestok] Файл {file_name} загружен в raw.{table_name} за {duration:.2f} сек")
        return table_name


    # ------------------------------------------------------------------ #
    # Bulk insert                                                        #
    # ------------------------------------------------------------------ #
    @classmethod
    def bulk_insert_perek(cls, df, table_name, engine):
        conn = engine.raw_connection()
        cursor = conn.cursor()
        cursor.fast_executemany = True

        cols = df.columns.tolist()
        insert_sql = f"INSERT INTO raw.{table_name} ({', '.join([f'[{c}]' for c in cols])}) VALUES ({', '.join(['?' for _ in cols])})"

        data = df.values.tolist()
        batch_size = cls.BATCH_SIZE

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            cursor.executemany(insert_sql, batch)
        conn.commit()
        cursor.close()
        conn.close()


def create_perek_table_and_upload(file_path: str, engine: Engine) -> str:
    """Convenience wrapper for DAG usage."""
    try:
        return PerekrestokTableProcessor.process_perek_file(file_path, engine)
    except Exception as e:  # noqa: BLE001
        logger.error("Критическая ошибка при обработке файла %s: %s", file_path, e, exc_info=True)
        raise
