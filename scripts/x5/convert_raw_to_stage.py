import pandas as pd
import logging

logger = logging.getLogger(__name__)

NUMERIC_COLUMNS = [
    'Количество (без ед. изм.)',
    'Оборот с НДС (без ед.изм.)',
    'Общая себестоимость (с НДС) (без ед. изм.)',
    'Средняя цена по себестоимости (с НДС)',
    'Средняя цена продажи (с НДС)',
    'sale_year',
    'sale_month'
]

COLUMN_RENAME_MAPPING = {
    'Сеть': 'retailer',
    'Филиал': 'branch',
    'Регион': 'region',
    'Город': 'city',
    'Адрес': 'address',
    'Завод': 'factory',
    'Завод2': 'factory2',
    'Тов.иер.ур.2': 'prod_level_2',
    'Тов.иер.ур.3': 'prod_level_3',
    'Тов.иер.ур.4': 'prod_level_4',
    'Материал': 'material',
    'Материал2': 'material2',
    'Бренд': 'brand',
    'Вендор': 'vendor',
    'Основной поставщик': 'main_supplier',
    'Поставщик склада (РЦ)': 'warehouse_supplier',
    'Количество (без ед. изм.)': 'quantity',
    'Оборот с НДС (без ед.изм.)': 'gross_turnover',
    'Общая себестоимость (с НДС) (без ед. изм.)': 'gross_cost',
    'Средняя цена по себестоимости (с НДС)': 'avg_cost_price',
    'Средняя цена продажи (с НДС)': 'avg_sell_price',
}


def convert_raw_to_stage(table_name: str, raw_engine, stage_engine, stage_schema='x5'):
    try:
        df = pd.read_sql(f"SELECT * FROM raw.{table_name}", raw_engine)
        logger.info(f"[Stage] Загружено {len(df)} строк из raw.{table_name}")

        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.'), errors='coerce'
                ).fillna(0)

        for col in df.columns:
            if col not in NUMERIC_COLUMNS:
                df[col] = df[col].astype(str).fillna('')

        # Переименование колонок на английский
        df.rename(columns=COLUMN_RENAME_MAPPING, inplace=True)

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
