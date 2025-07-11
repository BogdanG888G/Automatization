import pandas as pd
import logging

logger = logging.getLogger(__name__)

NUMERIC_COLUMNS = [
    'avg_sell_price',
    'writeoff_rub',
    'writeoff_qty',
    'sales_rub',
    'sales_kg',
    'sales_qty',
    'avg_cost_price',
    'margin_rub',
    'loss_rub',
    'loss_qty',
    'promo_sales_rub'
]

COLUMN_RENAME_MAPPING = {
    'Дата': 'period',
    'Сегмент': 'segment',
    'СЕМЬЯ': 'family_code',
    'НАЗВАНИЕ СЕМЬИ': 'family_name',
    'АРТИКУЛ': 'article',
    'НАИМЕНОВАНИЕ': 'product_name',
    'ПОСТАВЩИК': 'supplier_code',
    'НАИМЕНОВАНИЕ ПОСТАВЩИКА': 'supplier_name',
    'Магазин': 'store_code',
    'Город': 'city',
    'Адрес': 'address',
    'Формат': 'format',
    'Месяц': 'month_name',
    'Ср.цена продажи': 'avg_sell_price',
    'Списания, руб.': 'writeoff_rub',
    'Списания, шт.': 'writeoff_qty',
    'Продажи, c НДС': 'sales_rub',
    'Продажи, кг': 'sales_kg',
    'Продажи, шт': 'sales_qty',
    'Ср.цена покупки': 'avg_cost_price',
    'Маржа, руб.': 'margin_rub',
    'Потери, руб.': 'loss_rub',
    'Потери,шт': 'loss_qty',
    'Промо Продажи, c НДС': 'promo_sales_rub'
}

def convert_raw_to_stage(table_name: str, raw_engine, stage_engine, stage_schema='ashan'):
    try:
        df = pd.read_sql(f"SELECT * FROM raw.{table_name}", raw_engine)
        logger.info(f"[Stage] Загружено {len(df)} строк из raw.{table_name}")

        # Преобразование числовых колонок
        for ru_col, en_col in COLUMN_RENAME_MAPPING.items():
            if en_col in NUMERIC_COLUMNS and ru_col in df.columns:
                df[ru_col] = (
                    df[ru_col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.replace(" ", "", regex=False)
                )
                df[ru_col] = pd.to_numeric(df[ru_col], errors='coerce').fillna(0)

        # Остальные колонки в строки
        for col in df.columns:
            if col not in NUMERIC_COLUMNS:
                df[col] = df[col].astype(str).fillna('')

        # Переименование
        df.rename(columns=COLUMN_RENAME_MAPPING, inplace=True)

        # Загрузка в Stage
        df.to_sql(
            name=table_name,
            schema=stage_schema,
            con=stage_engine,
            if_exists='replace',
            index=False,
            chunksize=1000
        )

        logger.info(f"[Stage] Данные успешно загружены в {stage_schema}.{table_name} с английскими колонками")

    except Exception as e:
        logger.error(f"[Stage ERROR] Ошибка при загрузке в {stage_schema}.{table_name}: {e}", exc_info=True)
        raise
