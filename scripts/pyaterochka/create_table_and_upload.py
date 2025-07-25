import os
import re
import logging
import time
from contextlib import closing
from io import StringIO
from typing import Optional, Tuple, Dict, List, Union

import pyodbc
import pandas as pd
import numpy as np
from sqlalchemy import text, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class PyaterochkaTableProcessor:
    CHUNKSIZE = 100_000
    BATCH_SIZE = 10_000
    MAX_ROWS = 10_000  # Максимальное количество строк для чтения

    # Специфичные для Пятёрочки форматы данных и метаданных
    PYATEROCHKA_MONTHS = {
        **{m.lower(): i + 1 for i, m in enumerate([
            'январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
            'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'
        ])},
        **{m[:3].lower(): i + 1 for i, m in enumerate([
            'январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
            'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'
        ])},
        'сен': 9, 'окт': 10, 'ноя': 11, 'дек': 12,
        'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'май': 5, 'июн': 6
    }

    @staticmethod
    def extract_pyaterochka_metadata(source: str) -> Tuple[Optional[int], Optional[int]]:
        """Извлекает месяц и год из названия файла Пятёрочки"""
        tokens = re.findall(r'[a-zа-я]+|\d{4}', str(source).lower())
        year = next((int(t) for t in tokens if t.isdigit() and len(t) == 4), None)
        
        month = None
        for t in tokens:
            if t in PyaterochkaTableProcessor.PYATEROCHKA_MONTHS:
                month = PyaterochkaTableProcessor.PYATEROCHKA_MONTHS[t]
                break
        
        return month, year

    @staticmethod
    def normalize_pyaterochka_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Нормализация названий столбцов для данных Пятёрочки"""
        # Первоначальная нормализация
        df.columns = [re.sub(r'\s+', '_', str(col).strip().lower()) for col in df.columns]
        
        # Обработка дубликатов
        seen = {}
        new_columns = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_col = f"{col}_{seen[col]}"
                new_columns.append(new_col)
            else:
                seen[col] = 0
                new_columns.append(col)
        
        df.columns = new_columns
        
        # Логирование переименованных колонок
        if len(seen) != len(new_columns):
            renamed = [f"{col} -> {new_col}" 
                      for col, new_col in zip(df.columns, new_columns) 
                      if col != new_col]
            logger.info(f"Переименованы дублирующиеся колонки: {', '.join(renamed)}")
        
        # Стандартное преобразование имен для Пятёрочки
        column_mapping = {
            'период': 'period',
            'сеть': 'retail_chain',
            'категория': 'category',
            'тип_основы': 'base_type',
            'поставщики': 'supplier',
            'бренды': 'brand',
            'наименование': 'product_name',
            'уни_наименование': 'unified_product_name',
            'граммовка': 'weight',
            'вкусы': 'flavor',
            'продажи,_шт': 'sales_units',
            'продажи,_руб': 'sales_rub',
            'продажи,_тонн': 'sales_tonnes',
            'себест.,_руб': 'cost_rub'
        }
        
        df.columns = [column_mapping.get(col, col) for col in df.columns]
        return df

    @classmethod
    def _safe_read_pyaterochka_csv(cls, file_path: str) -> pd.DataFrame:
        """Безопасное чтение CSV с обработкой ошибок (только первые 10k строк)"""
        try:
            return pd.read_csv(
                file_path,
                dtype='string',
                sep='\t',
                quotechar='"',
                on_bad_lines='warn',
                encoding='utf-8-sig',
                decimal=',',
                thousands=' ',
                nrows=cls.MAX_ROWS  # Читаем только первые 10k строк
            )
        except pd.errors.ParserError:
            logger.warning("Ошибка парсинга CSV, попытка ручной очистки...")
            with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                content = ''.join([f.readline() for _ in range(cls.MAX_ROWS + 1)])  # Читаем только первые 10k строк
                clean_content = re.sub(r'(?<!\\)"(?!\\)', '', content)  # Удаляем одиночные кавычки
                return pd.read_csv(
                    StringIO(clean_content),
                    dtype='string',
                    sep='\t',
                    quotechar='"',
                    decimal=',',
                    thousands=' '
                )

    @classmethod
    def _read_xlsb_file(cls, file_path: str) -> Dict[str, pd.DataFrame]:
        """Чтение XLSB файлов с улучшенной обработкой (только первые 10k строк)"""
        try:
            import pyxlsb
            with pyxlsb.open_workbook(file_path) as wb:
                sheet_names = wb.sheets
                if not sheet_names:
                    raise ValueError("XLSB файл не содержит листов")
                
                reader = {}
                for sheet_name in sheet_names:
                    with wb.get_sheet(sheet_name) as sheet:
                        data = []
                        headers = None
                        for i, row in enumerate(sheet.rows()):
                            if i >= cls.MAX_ROWS + 1:  # Ограничиваем количество строк
                                break
                            if i == 0:
                                headers = [str(item.v) if item.v is not None else f"none_{idx}" 
                                         for idx, item in enumerate(row)]
                            else:
                                data.append([str(item.v) if item.v is not None else '' for item in row])
                        
                        if headers and data:
                            df = pd.DataFrame(data, columns=headers)
                            reader[sheet_name] = df
                
                if not reader:
                    raise ValueError("XLSB файл не содержит данных")
                
                return reader
        except Exception as e:
            logger.error(f"Ошибка чтения XLSB: {str(e)}")
            raise

    @classmethod
    def _read_excel_file(cls, file_path: str) -> Dict[str, pd.DataFrame]:
        """Чтение Excel файлов с обработкой ошибок (только первые 10k строк)"""
        try:
            return pd.read_excel(
                file_path, 
                sheet_name=None, 
                dtype='string', 
                engine='openpyxl',
                decimal=',',
                thousands=' ',
                nrows=cls.MAX_ROWS  # Читаем только первые 10k строк
            )
        except Exception as e:
            logger.error(f"Ошибка чтения Excel: {e}")
            raise

    @classmethod
    def process_pyaterochka_file(cls, file_path: str, engine: Engine) -> str:
        """Основной метод обработки файла Пятёрочки"""
        start_time = time.time()
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        table_name = re.sub(r'\W+', '_', base_name.lower())

        logger.info(f"Начало обработки файла: {file_name} (будет прочитано максимум {cls.MAX_ROWS} строк)")
        
        # Чтение файла в зависимости от формата
        if file_path.endswith('.xlsx'):
            reader = cls._read_excel_file(file_path)
        elif file_path.endswith('.xlsb'):
            reader = cls._read_xlsb_file(file_path)
        elif file_path.endswith('.csv'):
            reader = {'data': cls._safe_read_pyaterochka_csv(file_path)}
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path}")

        processed_chunks = []
        for sheet_name, df in reader.items():
            try:
                if df.empty:
                    logger.warning(f"Лист {sheet_name} пуст")
                    continue

                logger.info(f"Обработка листа {sheet_name}, строк: {len(df)}")
                
                # Нормализация и обработка данных
                df = cls.normalize_pyaterochka_columns(df)
                
                # Извлечение метаданных
                month, year = cls.extract_pyaterochka_metadata(file_name)
                if month and year:
                    df = df.assign(sale_year=str(year), sale_month=str(month).zfill(2))

                # Очистка данных - все значения как строки
                df = df.replace([np.nan, None], '')
                for col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.replace('\u200b', '')

                processed_chunks.append(df)
            except Exception as e:
                logger.error(f"Ошибка обработки листа {sheet_name}: {e}", exc_info=True)
                continue

        if not processed_chunks:
            raise ValueError("Файл не содержит валидных данных")

        final_df = pd.concat(processed_chunks, ignore_index=True)
        
        # Проверка на пустые данные после обработки
        if final_df.empty:
            raise ValueError("Файл не содержит данных после обработки")

        # Создание таблицы в БД - все колонки как NVARCHAR
        with engine.begin() as conn:
            if not conn.execute(
                text("SELECT 1 FROM information_schema.tables WHERE table_schema='raw' AND table_name=:table"),
                {"table": table_name}
            ).scalar():
                columns_sql = []
                for col in final_df.columns:
                    clean_col = col.replace('"', '').replace("'", "")
                    columns_sql.append(f'[{clean_col}] NVARCHAR(255)')
                
                create_table_sql = f"CREATE TABLE raw.{table_name} ({', '.join(columns_sql)})"
                logger.debug(f"SQL создания таблицы: {create_table_sql}")
                conn.execute(text(create_table_sql))

        # Вставка данных
        cls.bulk_insert_pyaterochka(final_df, table_name, engine)

        duration = time.time() - start_time
        logger.info(f"Файл {file_name} успешно загружен в raw.{table_name} за {duration:.2f} сек ({len(final_df)} строк)")
        return table_name

    @classmethod
    def bulk_insert_pyaterochka(cls, df: pd.DataFrame, table_name: str, engine: Engine):
        """Массовая вставка данных (оптимизировано для 1 млн строк)"""
        @event.listens_for(engine, 'before_cursor_execute')
        def set_fast_executemany(conn, cursor, statement, parameters, context, executemany):
            if executemany:
                cursor.fast_executemany = True

        # Подготовка SQL
        cols = ', '.join(f'[{col}]' for col in df.columns)
        params = ', '.join(['?'] * len(df.columns))
        insert_sql = f"INSERT INTO raw.{table_name} ({cols}) VALUES ({params})"

        def row_generator():
            """Генератор строк без сохранения в памяти"""
            for row in df.itertuples(index=False, name=None):
                clean_row = [
                    '' if pd.isna(value) or value in ('', None)
                    else str(value).strip()[:1000]
                    for value in row
                ]
                yield tuple(clean_row)

        # Вставка батчами
        with closing(engine.raw_connection()) as conn:
            cursor = conn.cursor()
            batch = []
            count = 0
            for row in row_generator():
                batch.append(row)
                if len(batch) >= cls.BATCH_SIZE:
                    cursor.executemany(insert_sql, batch)
                    conn.commit()
                    count += len(batch)
                    logger.debug(f"Вставлено {count} записей...")
                    batch.clear()
            if batch:
                cursor.executemany(insert_sql, batch)
                conn.commit()
                logger.debug(f"Финальные {len(batch)} записей вставлены.")

def create_pyaterochka_table_and_upload(file_path: str, engine: Engine) -> str:
    """Интерфейсная функция для вызова из DAG"""
    try:
        return PyaterochkaTableProcessor.process_pyaterochka_file(file_path, engine)
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке файла {file_path}: {e}", exc_info=True)
        raise