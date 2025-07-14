import pandas as pd
import logging
from sqlalchemy import text
from contextlib import closing
import numpy as np

logger = logging.getLogger(__name__)

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

def convert_raw_to_stage(table_name: str, raw_engine, stage_engine, stage_schema='magnit'):
    """Загрузка из raw в stage с преобразованием данных."""
    try:
        # 1. Получаем количество строк для прогресс-индикации
        with raw_engine.connect() as conn:
            total_count = conn.execute(text(f"SELECT COUNT(*) FROM raw.{table_name}")).scalar()
            logger.info(f"[Stage] Начинаем загрузку {total_count} строк из raw.{table_name}")

        # 2. Читаем данные из raw по частям
        chunks = []
        total_rows = 0
        for chunk in pd.read_sql_table(table_name, raw_engine, schema='raw', chunksize=50000):
            chunks.append(chunk)
            total_rows += len(chunk)
            logger.info(f"[Stage] Считано {total_rows}/{total_count} строк ({total_rows/total_count:.1%})")

        if not chunks:
            logger.warning("[Stage] Нет данных для обработки")
            return

        df = pd.concat(chunks, ignore_index=True)
        del chunks

        # 3. Переименование колонок (если в raw русские названия)
        df.rename(columns=ColumnConfig.RENAME_MAP, inplace=True)

        # 4. Преобразование числовых колонок (векторно)
        for col in ColumnConfig.NUMERIC_COLS.keys():
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r'[,\s]', '.', regex=True)
                    .replace({'': '0', None: '0'})
                    .astype(np.float64)
                    .fillna(0)
                )

        # 5. Преобразование остальных колонок в строки
        for col in df.columns:
            if col not in ColumnConfig.NUMERIC_COLS:
                df[col] = df[col].astype(str).fillna('')

        # 6. Создание таблицы в stage, если не существует
        with closing(stage_engine.connect()) as conn:
            create_sql = f"""
            IF NOT EXISTS (SELECT * FROM sys.tables t
                JOIN sys.schemas s ON t.schema_id = s.schema_id
                WHERE s.name = '{stage_schema}' AND t.name = '{table_name}')
            BEGIN
                CREATE TABLE {stage_schema}.{table_name} (
                    {', '.join([
                        f"{col} FLOAT" if col in ColumnConfig.NUMERIC_COLS else f"{col} NVARCHAR(255)"
                        for col in df.columns
                    ])}
                )
            END
            """
            conn.execute(text(create_sql))
            conn.commit()

            # 7. Очищаем таблицу перед загрузкой
            conn.execute(text(f"TRUNCATE TABLE {stage_schema}.{table_name}"))
            conn.commit()

            # 8. Быстрая пакетная вставка через pyodbc (fast_executemany)
            with conn.connection.cursor() as cursor:
                cursor.fast_executemany = True
                cols = ', '.join(df.columns)
                params = ', '.join(['?'] * len(df.columns))
                insert_sql = f"INSERT INTO {stage_schema}.{table_name} ({cols}) VALUES ({params})"
                data = [tuple(x) for x in df.to_records(index=False)]
                cursor.executemany(insert_sql, data)
            conn.commit()

        logger.info(f"[Stage] Успешно загружено {len(df)} строк в {stage_schema}.{table_name}")

    except Exception as e:
        logger.error(f"[Stage ERROR] Ошибка при обработке таблицы {table_name}: {e}", exc_info=True)
        raise
