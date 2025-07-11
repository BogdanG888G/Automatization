import pandas as pd
import logging
from sqlalchemy import text
from contextlib import closing
import numpy as np

logger = logging.getLogger(__name__)

# Константы вынесены в отдельный класс для удобства
class ColumnConfig:
    NUMERIC_COLS = {
        'average_sell_price': {'dtype': 'float64', 'default': 0.0},
        'writeoff_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'writeoff_quantity': {'dtype': 'float64', 'default': 0.0},
        'sales_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'sales_weight_kg': {'dtype': 'float64', 'default': 0.0},
        'sales_quantity': {'dtype': 'float64', 'default': 0.0},
        'average_cost_price': {'dtype': 'float64', 'default': 0.0},
        'margin_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'loss_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'loss_quantity': {'dtype': 'float64', 'default': 0.0},
        'promo_sales_amount_rub': {'dtype': 'float64', 'default': 0.0}
    }
    
    RENAME_MAP = {
        'Дата': 'sales_date',
        'Сегмент': 'product_segment',
        'СЕМЬЯ': 'product_family_code',
        'НАЗВАНИЕ СЕМЬИ': 'product_family_name',
        'АРТИКУЛ': 'product_article',
        'НАИМЕНОВАНИЕ': 'product_name',
        'ПОСТАВЩИК': 'supplier_code',
        'НАИМЕНОВАНИЕ ПОСТАВЩИКА': 'supplier_name',
        'Магазин': 'store_code',
        'Город': 'city',
        'Адрес': 'store_address',
        'Формат': 'store_format',
        'Месяц': 'month_name',
        'Ср.цена продажи': 'average_sell_price',
        'Списания, руб.': 'writeoff_amount_rub',
        'Списания, шт.': 'writeoff_quantity',
        'Продажи, c НДС': 'sales_amount_rub',
        'Продажи, кг': 'sales_weight_kg',
        'Продажи, шт': 'sales_quantity',
        'Ср.цена покупки': 'average_cost_price',
        'Маржа, руб.': 'margin_amount_rub',
        'Потери, руб.': 'loss_amount_rub',
        'Потери,шт': 'loss_quantity',
        'Промо Продажи, c НДС': 'promo_sales_amount_rub'
    }

def convert_raw_to_stage(table_name: str, raw_engine, stage_engine, stage_schema='ashan'):
    """Оптимизированная загрузка данных из raw в stage с преобразованием."""
    try:
        # 1. Чтение данных с прогресс-баром и chunking
        total_rows = 0
        chunks = []
        
        # Получаем общее количество строк для прогресс-логирования
        with raw_engine.connect() as conn:
            total_count = conn.execute(
                text(f"SELECT COUNT(*) FROM raw.{table_name}")
            ).scalar()
            logger.info(f"[Stage] Начинаем загрузку {total_count} строк из raw.{table_name}")

        # Чтение данных частями
        for chunk in pd.read_sql_table(
            table_name,
            raw_engine,
            schema='raw',
            chunksize=50000,
            dtype={col: 'object' for col in ColumnConfig.RENAME_MAP.keys()}
        ):
            chunks.append(chunk)
            total_rows += len(chunk)
            logger.info(f"[Stage] Загружено {total_rows}/{total_count} строк ({total_rows/total_count:.1%})")

        if not chunks:
            logger.warning("[Stage] Нет данных для обработки")
            return

        df = pd.concat(chunks, ignore_index=True)
        del chunks  # Освобождаем память

        # 2. Оптимизированное преобразование числовых колонок
        numeric_cols = {ru: en for ru, en in ColumnConfig.RENAME_MAP.items() 
                       if en in ColumnConfig.NUMERIC_COLS}
        
        for ru_col, en_col in numeric_cols.items():
            if ru_col in df.columns:
                # Векторизованные операции вместо apply
                df[en_col] = (
                    df[ru_col]
                    .astype(str)
                    .str.replace(r'[,\s]', '.', regex=True)
                    .replace('', '0')
                    .astype(np.float64)
                    .fillna(0)
                )
                df.drop(ru_col, axis=1, inplace=True)

        # 3. Оптимизация строковых колонок
        str_cols = [ru for ru in ColumnConfig.RENAME_MAP.keys() 
                   if ru in df.columns and ru not in numeric_cols]
        
        for ru_col in str_cols:
            en_col = ColumnConfig.RENAME_MAP[ru_col]
            df[en_col] = df[ru_col].astype('string').fillna('')
            df.drop(ru_col, axis=1, inplace=True)

        # 4. Оптимизированная загрузка в БД
        with closing(stage_engine.connect()) as conn:
            # Создание таблицы с оптимальными типами данных
            create_table_sql = f"""
                IF NOT EXISTS (SELECT * FROM sys.tables t 
                              JOIN sys.schemas s ON t.schema_id = s.schema_id 
                              WHERE s.name = '{stage_schema}' AND t.name = '{table_name}')
                BEGIN
                    CREATE TABLE {stage_schema}.{table_name} (
                        {', '.join([
                            f'{col} FLOAT' if col in ColumnConfig.NUMERIC_COLS 
                            else f'{col} NVARCHAR(255)' 
                            for col in df.columns
                        ])}
                    )
                END
            """
            conn.execute(text(create_table_sql))
            conn.commit()

            # Быстрая загрузка через bulk insert
            with conn.connection.cursor() as cursor:
                cursor.fast_executemany = True
                
                # Подготовка данных
                data = [tuple(x) for x in df.to_records(index=False)]
                cols = ', '.join(df.columns)
                params = ', '.join(['?'] * len(df.columns))
                
                # Очистка таблицы перед вставкой
                cursor.execute(f"TRUNCATE TABLE {stage_schema}.{table_name}")
                
                # Пакетная вставка
                insert_sql = f"INSERT INTO {stage_schema}.{table_name} ({cols}) VALUES ({params})"
                cursor.executemany(insert_sql, data)
                
            conn.commit()

        logger.info(f"[Stage] Успешно загружено {len(df)} строк в {stage_schema}.{table_name}")

    except Exception as e:
        logger.error(f"[Stage ERROR] Ошибка при обработке таблицы {table_name}: {str(e)}", exc_info=True)
        raise