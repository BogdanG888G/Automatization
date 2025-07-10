import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)

# Итоговые названия после clean_column_names и переименования
NUMERIC_COLUMNS = [
    'quantity',
    'gross_turnover',
    'gross_cost',
    'avg_cost_price',
    'avg_sell_price',
    'sale_year',
    'sale_month'
]

<<<<<<< HEAD
# Упрощённая мапа уже после нормализации (скобки, пробелы и т.д. удалены)
=======
>>>>>>> 66c7e5c (Дополнил данные и пометил дальнейший план)
COLUMN_RENAME_MAPPING = {
    'Сеть': 'retailer',
    'Филиал': 'branch',
    'Регион': 'region',
    'Город': 'city',
    'Адрес': 'address',
    'Завод': 'factory',
<<<<<<< HEAD
    'Завод1': 'factory',
=======
>>>>>>> 66c7e5c (Дополнил данные и пометил дальнейший план)
    'Завод2': 'factory2',
    'Тов.иер.ур.2': 'prod_level_2',
    'Тов.иер.ур.3': 'prod_level_3',
    'Тов.иер.ур.4': 'prod_level_4',
    'Материал': 'material',
<<<<<<< HEAD
    'Материал1': 'material',
=======
>>>>>>> 66c7e5c (Дополнил данные и пометил дальнейший план)
    'Материал2': 'material2',
    'Бренд': 'brand',
    'Вендор': 'vendor',
    'Основной поставщик': 'main_supplier',
<<<<<<< HEAD
    'Поставщик склада РЦ': 'warehouse_supplier',
    'Поставщик склада': 'warehouse_supplier',
    'Количество': 'quantity',
    'Количество без ед. изм.': 'quantity',
    'Оборот с НДС': 'gross_turnover',
    'Оборот с НДС без ед.изм.': 'gross_turnover',
    'Общая себестоимость': 'gross_cost',
    'Общая себестоимость с НДС без ед. изм.': 'gross_cost',
    'Средняя цена по себестоимости': 'avg_cost_price',
    'Средняя цена по себестоимости с НДС': 'avg_cost_price',
    'Средняя цена продажи': 'avg_sell_price',
    'Средняя цена продажи с НДС': 'avg_sell_price',
}

def clean_column_names(columns):
    """
    Удаляет переносы строк, скобки, двойные пробелы и дублирующиеся колонки.
    """
    cleaned = []
    seen = set()
    for col in columns:
        c = col.replace('\n', ' ').replace('\r', ' ').strip()
        c = re.sub(r'\s+', ' ', c)  # двойные пробелы → один
        c = re.sub(r'\s*\(.*?\)', '', c)  # удалить всё в скобках
        if c in seen:
            count = sum(1 for x in cleaned if x.startswith(c))
            c = f"{c}{count+1}"
        cleaned.append(c)
        seen.add(c)
    return cleaned

=======
    'Поставщик склада (РЦ)': 'warehouse_supplier',
    'Количество (без ед. изм.)': 'quantity',
    'Оборот с НДС (без ед.изм.)': 'gross_turnover',
    'Общая себестоимость (с НДС) (без ед. изм.)': 'gross_cost',
    'Средняя цена по себестоимости (с НДС)': 'avg_cost_price',
    'Средняя цена продажи (с НДС)': 'avg_sell_price',
}

>>>>>>> 66c7e5c (Дополнил данные и пометил дальнейший план)

def convert_raw_to_stage(table_name: str, raw_engine, stage_engine, stage_schema='x5'):
    try:
        df = pd.read_sql(f"SELECT * FROM raw.{table_name}", raw_engine)
        logger.info(f"[Stage] Загружено {len(df)} строк из raw.{table_name}")

        # Очистка и нормализация колонок
        df.columns = clean_column_names(df.columns)
        logger.info(f"[Stage] Нормализованные колонки: {df.columns.tolist()}")

        # Приведение числовых колонок
        for ru_col, en_col in COLUMN_RENAME_MAPPING.items():
            if en_col in NUMERIC_COLUMNS and ru_col in df.columns:
                df[ru_col] = pd.to_numeric(
                    df[ru_col].astype(str).str.replace(',', '.'), errors='coerce'
                ).fillna(0)

        # Приведение остальных колонок к строкам
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
