import pandas as pd
import logging
from sqlalchemy import text, exc, engine
import numpy as np
from typing import List
from datetime import datetime
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class ColumnConfig:
    """Configuration для колонок: типы данных и переименование."""
    NUMERIC_COLS = {
        'average_sell_price',
        'writeoff_amount_rub',
        'writeoff_quantity',
        'sales_amount_rub',
        'sales_weight_kg',
        'sales_quantity',
        'average_cost_price',
        'margin_amount_rub',
        'loss_amount_rub',
        'loss_quantity',
        'promo_sales_amount_rub',
        'avg_sell_price',
        'writeoff_amount_rub',
        'writeoff_quantity',
        'sales_amount_with_vat',
        'sales_weight_kg',
        'sales_quantity',
        'avg_purchase_price',
        'margin_amount_rub',
        'loss_amount_rub',
        'loss_quantity',
        'promo_sales_amount_with_vat'
    }

    RENAME_MAP = {
        'дата': 'sale_date',
        'месяц': 'month_name',
        'месяц_сырое': 'month_raw',
        'сегмент': 'product_segment',
        'семья': 'product_family_code',
        'название_семьи': 'product_family_name',
        'артикул': 'product_article',
        'наименование': 'product_name',
        'наименование_товара': 'product_name',
        'ср_цена_продажи': 'avg_sell_price',
        'ср.цена_продажи': 'avg_sell_price',
        'ср_цена_покупки': 'avg_purchase_price',
        'ср.цена_покупки': 'avg_purchase_price',
        
        'поставщик': 'supplier_code',
        'код_поставщика': 'supplier_code',
        'наименование_поставщика': 'supplier_name',

        'магазин': 'store_code',
        'город': 'city',
        'адрес': 'store_address',
        'формат': 'store_format',

        'списания_руб': 'writeoff_amount_rub',
        'списания,_руб.': 'writeoff_amount_rub',
        'списания_шт': 'writeoff_quantity',
        'списания,_шт.': 'writeoff_quantity',

        'продажи_c_ндс': 'sales_amount_with_vat',
        'продажи,_c_ндс': 'sales_amount_with_vat',
        'продажи_шт': 'sales_quantity',
        'продажи,_шт': 'sales_quantity',
        'продажи_кг': 'sales_weight_kg',
        'продажи,_кг': 'sales_weight_kg',

        'маржа_руб': 'margin_amount_rub',
        'маржа,_руб.': 'margin_amount_rub',

        'потери_руб': 'loss_amount_rub',
        'потери,_руб.': 'loss_amount_rub',
        'потери_шт': 'loss_quantity',
        'потери,шт': 'loss_quantity',

        'промо_продажи_c_ндс': 'promo_sales_amount_with_vat',
        'промо_продажи,_c_ндс': 'promo_sales_amount_with_vat',

        'sale_month': 'sale_month',
        'sale_year': 'sale_year',
        'sale_date': 'sale_date',
        'month_raw': 'month_raw',
        'date_raw': 'date_raw'
    }



def _sanitize_column_name(name: str) -> str:
    """Очищаем имя колонки, разрешая латиницу, кириллицу, цифры и подчеркивания."""
    if not name or not isinstance(name, str):
        return ''

    special_mappings = {
        'списания,_руб.': 'writeoff_amount_rub',
        'списания,_шт.': 'writeoff_quantity',
        'продажи,_c_ндс': 'sales_amount_with_vat',
        'потери,_руб.': 'loss_amount_rub',
        'потери,шт': 'loss_quantity',
        'промо_продажи,_c_ндс': 'promo_sales_amount_with_vat',
        'маржа,_руб.': 'margin_amount_rub',
        'ср.цена_продажи': 'avg_sell_price',
        'ср.цена_покупки': 'avg_purchase_price'
    }

    name_lower = name.lower()
    if name_lower in special_mappings:
        return special_mappings[name_lower]

    # Разрешаем латинские, кириллические буквы, цифры и _
    cleaned = re.sub(r'[^\wа-яё]', '_', name_lower, flags=re.IGNORECASE)

    cleaned = re.sub(r'_{2,}', '_', cleaned)
    cleaned = cleaned.strip('_')
    return cleaned


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразуем числовые колонки, учитывая запятые и мусор."""
    # Переименование столбцов для дальнейшей работы
    rename_lower = {k.lower(): v for k, v in ColumnConfig.RENAME_MAP.items()}
    df.columns = [col.lower() for col in df.columns]

    for ru_col, en_col in rename_lower.items():
        if ru_col in df.columns and en_col in ColumnConfig.NUMERIC_COLS:
            try:
                # Преобразование к строке, замена запятой на точку и удаление мусора
                df[en_col] = (
                    df[ru_col]
                    .astype(str)
                    .str.replace(',', '.')
                    .str.replace(r'[^\d\.]', '', regex=True)
                    .replace('', '0')
                    .astype(float)
                    .fillna(0.0)
                )
                if ru_col != en_col:
                    df.drop(columns=ru_col, inplace=True)
            except Exception as e:
                logger.error(f"Ошибка конвертации числовой колонки '{ru_col}': {e}")
                raise
    return df


def _convert_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Переименовываем и приводим к строковому типу все остальные колонки."""
    rename_lower = {k.lower(): v for k, v in ColumnConfig.RENAME_MAP.items()}
    df.columns = [col.lower() for col in df.columns]

    for ru_col, en_col in rename_lower.items():
        if ru_col in df.columns and en_col not in ColumnConfig.NUMERIC_COLS:
            try:
                df[en_col] = df[ru_col].astype(str).fillna('')
                if ru_col != en_col:
                    df.drop(columns=ru_col, inplace=True)
            except Exception as e:
                logger.error(f"Ошибка конвертации строковой колонки '{ru_col}': {e}")
                raise
    return df


def _create_stage_table(conn: engine.Connection, table_name: str, df: pd.DataFrame, schema: str = 'ashan') -> None:
    logger.info(f"Создаём таблицу [{schema}].[{table_name}] с колонками: {df.columns.tolist()}")

    # Чистим имена колонок
    cleaned_columns = [_sanitize_column_name(col) for col in df.columns]

    # Заменяем пустые имена на уникальные
    for i, col in enumerate(cleaned_columns):
        if not col:
            cleaned_columns[i] = f"col_{i+1}"

    # Устранение дубликатов, добавляя суффикс с индексом
    seen = {}
    for i, col in enumerate(cleaned_columns):
        if col in seen:
            seen[col] += 1
            cleaned_columns[i] = f"{col}_{seen[col]}"
        else:
            seen[col] = 0

    df.columns = cleaned_columns

    duplicates = [col for col in df.columns if df.columns.tolist().count(col) > 1]
    if duplicates:
        raise ValueError(f"Дублирующиеся имена колонок после очистки: {duplicates}")

    columns_sql = []
    for col in df.columns:
        if not col:
            continue
        col_type = 'FLOAT' if col in ColumnConfig.NUMERIC_COLS else 'NVARCHAR(255)'
        columns_sql.append(f'[{col}] {col_type}')

    if not columns_sql:
        raise ValueError("Нет валидных колонок для создания таблицы")

    create_sql = f"""
    IF NOT EXISTS (
        SELECT * FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE s.name = '{schema}' AND t.name = '{table_name}'
    )
    BEGIN
        CREATE TABLE [{schema}].[{table_name}] (
            {', '.join(columns_sql)}
        )
    END
    """

    try:
        trans = conn.begin()
        conn.execute(text(create_sql))
        trans.commit()
    except Exception as e:
        trans.rollback()
        logger.error(f"Ошибка создания таблицы: {e}\nSQL:\n{create_sql}")
        raise



def _bulk_insert_data(conn: engine.Connection, table_name: str, df: pd.DataFrame, schema: str = 'ashan', batch_size: int = 50000) -> None:
    """Вставляем данные батчами с помощью fast_executemany. Вставляются все данные, даже если таблица не пустая."""
    if df.empty:
        logger.warning("Пустой DataFrame, пропускаем вставку")
        return

    # Очищаем и нормализуем имена колонок
    cleaned_columns = [_sanitize_column_name(col) for col in df.columns]

    # Замена пустых имён
    for i, col in enumerate(cleaned_columns):
        if not col:
            cleaned_columns[i] = f"col_{i+1}"

    # Устранение дубликатов, добавляя суффикс с индексом
    seen = {}
    for i, col in enumerate(cleaned_columns):
        if col in seen:
            seen[col] += 1
            cleaned_columns[i] = f"{col}_{seen[col]}"
        else:
            seen[col] = 0

    df.columns = cleaned_columns

    # Проверяем размер df для вставки
    total_rows = len(df)
    logger.info(f"Начинаем вставку в таблицу [{schema}].[{table_name}] {total_rows} строк батчами по {batch_size}")

    cols = ', '.join(f'[{col}]' for col in df.columns)
    params = ', '.join('?' for _ in df.columns)

    raw_conn = conn.connection
    cursor = raw_conn.cursor()
    cursor.fast_executemany = True
    insert_sql = f"INSERT INTO [{schema}].[{table_name}] ({cols}) VALUES ({params})"

    try:
        for start in range(0, total_rows, batch_size):
            chunk = df.iloc[start:start + batch_size]
            data = []
            for row in chunk.itertuples(index=False):
                processed_row = []
                for val, col in zip(row, df.columns):
                    if col in ColumnConfig.NUMERIC_COLS:
                        processed_row.append(float(val) if pd.notna(val) else 0.0)
                    else:
                        processed_row.append(str(val) if pd.notna(val) else '')
                data.append(tuple(processed_row))

            cursor.executemany(insert_sql, data)
            raw_conn.commit()
            logger.info(f"Вставлено строк: {start}–{start + len(chunk)} в таблицу [{schema}].[{table_name}]")

    except Exception as e:
        raw_conn.rollback()
        logger.error(f"Ошибка вставки данных: {e}\nSQL:\n{insert_sql}")
        raise

    finally:
        cursor.close()



def convert_raw_to_stage(table_name: str, raw_engine: engine.Engine, stage_engine: engine.Engine,
                         stage_schema: str = 'ashan', batch_size: int = 100000, limit: int = None) -> None:
    try:
        start_time = datetime.now()
        logger.info(f"[Stage] Начинаем обработку таблицы {table_name}")

        with raw_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'raw' AND TABLE_NAME = :table_name"
            ), {'table_name': table_name}).fetchall()
            actual_columns = [row[0].lower() for row in result]

        with raw_engine.connect() as conn:
            sql_first = text(f"""
                SELECT * FROM raw.{table_name}
                ORDER BY (SELECT NULL)
                OFFSET 0 ROWS FETCH NEXT :batch_size ROWS ONLY
            """)
            df_sample = pd.read_sql(sql_first, conn, params={'batch_size': batch_size})

        if df_sample.empty:
            logger.info(f"[Stage] Нет данных для загрузки в таблицу {table_name}")
            return

        df_sample.columns = [col.lower() for col in df_sample.columns]
        df_sample = _convert_numeric_columns(df_sample)
        df_sample = _convert_string_columns(df_sample)
        rename_map_norm = {k.lower(): v for k, v in ColumnConfig.RENAME_MAP.items()}
        valid_rename_map = {k: v for k, v in rename_map_norm.items() if k in df_sample.columns}
        df_sample.rename(columns=valid_rename_map, inplace=True)

        with stage_engine.connect() as stage_conn:
            _create_stage_table(stage_conn, table_name, df_sample, schema=stage_schema)
            # !!! Добавляем очистку таблицы перед загрузкой:
            stage_conn.execute(text(f"TRUNCATE TABLE [{stage_schema}].[{table_name}]"))

        offset = 0
        total_rows = 0

        while True:
            with raw_engine.connect() as conn:
                sql_query = text(f"""
                    SELECT * FROM raw.{table_name}
                    ORDER BY (SELECT NULL)
                    OFFSET :offset ROWS FETCH NEXT :batch_size ROWS ONLY
                """)
                df = pd.read_sql(sql_query, conn, params={'offset': offset, 'batch_size': batch_size})

            if df.empty:
                break

            df.columns = [col.lower() for col in df.columns]
            df = _convert_numeric_columns(df)
            df = _convert_string_columns(df)
            df.rename(columns=valid_rename_map, inplace=True)

            with stage_engine.connect() as stage_conn:
                _bulk_insert_data(stage_conn, table_name, df, schema=stage_schema)

            batch_rows = len(df)
            total_rows += batch_rows
            logger.info(f"[Stage] Вставлено строк: {offset}–{offset + batch_rows} в таблицу {stage_schema}.{table_name}")

            offset += batch_size

        logger.info(f"[Stage] Успешно загружено в {stage_schema}.{table_name} — {total_rows} строк, время: {datetime.now() - start_time}")
    except Exception as ex:
        logger.error(f"Ошибка конвертации raw->{stage_schema} для {table_name}: {ex}")
        raise


if __name__ == "__main__":
    from sqlalchemy import create_engine

    # Настройки соединения
    RAW_CONNECTION_STRING = 'mssql+pyodbc://user:pass@server/raw?driver=ODBC+Driver+17+for+SQL+Server'
    STAGE_CONNECTION_STRING = 'mssql+pyodbc://user:pass@server/ashan?driver=ODBC+Driver+17+for+SQL+Server'

    raw_engine = create_engine(RAW_CONNECTION_STRING)
    stage_engine = create_engine(STAGE_CONNECTION_STRING)

    # Пример вызова
    convert_raw_to_stage('sales_table_name', raw_engine, stage_engine, stage_schema='ashan', limit=None)
