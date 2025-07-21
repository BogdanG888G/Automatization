# convert_raw_to_stage.py
# -*- coding: utf-8 -*-
import logging
import re
from contextlib import closing
from typing import Dict

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Конфигурация                                                        #
# ------------------------------------------------------------------ #
class X5ColumnConfig:
    # Целевые SQL-типы для stage (SQL Server)
    NUMERIC_COLS: Dict[str, str] = {
        'quantity': 'FLOAT',
        'gross_turnover': 'FLOAT',
        'gross_cost': 'FLOAT',
        'avg_cost_price': 'FLOAT',
        'avg_sell_price': 'FLOAT',
        'sale_year': 'INT',
        'sale_month': 'INT',
    }

    # Нормализованные имена: русские текстовые → англ
    # (в raw вы могли иметь уже нормализованные; здесь — для подстраховки)
    RENAME_MAP: Dict[str, str] = {
        # орг
        'сеть': 'retail_chain',
        'филиал': 'branch',
        'регион': 'region',
        'город': 'city',
        'адрес': 'address',
        'завод': 'factory',
        'завод1': 'factory',
        'завод2': 'factory2',
        # иерархия
        'тов.иер.ур.2': 'prod_level_2',
        'тов.иер.ур.3': 'prod_level_3',
        'тов.иер.ур.4': 'prod_level_4',
        # материал
        'материал': 'material',
        'материал1': 'material',
        'материал2': 'material2',
        # прочие
        'бренд': 'brand',
        'вендор': 'vendor',
        'основной поставщик': 'main_supplier',
        'поставщик склада рц': 'warehouse_supplier',
        'поставщик склада': 'warehouse_supplier',
        # метрики
        'количество': 'quantity',
        'количество без ед. изм.': 'quantity',
        'оборот с ндс': 'gross_turnover',
        'оборот с ндс без ед.изм.': 'gross_turnover',
        'общая себестоимость': 'gross_cost',
        'общая себестоимость с ндс без ед. изм.': 'gross_cost',
        'средняя цена по себестоимости': 'avg_cost_price',
        'средняя цена по себестоимости с ндс': 'avg_cost_price',
        'средняя цена продажи': 'avg_sell_price',
        'средняя цена продажи с ндс': 'avg_sell_price',
    }

    # Грязные токены, которые считаем отсутствием числового значения
    INVALID_NUMERIC_TOKENS = {
        '', 'nan', 'none', 'null', '-', '–', '—', 'н/д', 'nd', 'na', 'n/a',
        'итого', 'итог', 'total', 'sum', 'subtotal'
    }


# ------------------------------------------------------------------ #
# Хелперы очистки                                                     #
# ------------------------------------------------------------------ #
NBSP_CHARS = ''.join(chr(c) for c in (0x00A0, 0x202F, 0x2007))  # non-breaking spaces family
SPACE_CLEAN_RE = re.compile(f'[{NBSP_CHARS}\\s]+', flags=re.UNICODE)
NON_NUMERIC_KEEP_SIGNS_RE = re.compile(r'[^0-9eE\+\-\.]+')


def _clean_column_name(col: str) -> str:
    if not isinstance(col, str):
        col = '' if col is None else str(col)
    col = col.strip().lower()
    col = re.sub(r'\s+', ' ', col)
    col = col.replace('.', ' ').replace(',', ' ')
    col = re.sub(r'\s+', ' ', col).strip()
    return col


def _make_unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        base = c or 'col'
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f'{base}_{seen[base]}')
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_column_name(c) for c in df.columns]
    # применим rename map (к ключам тоже нормализация)
    cleaned_map = {_clean_column_name(k): v for k, v in X5ColumnConfig.RENAME_MAP.items()}
    new_cols = [cleaned_map.get(c, c) for c in df.columns]
    new_cols = _make_unique_columns(new_cols)
    df.columns = new_cols
    return df


def _coerce_numeric_series(series: pd.Series, is_int: bool = False, col_log_name: str = '') -> pd.Series:
    """
    Очистка и конвертация в числовой тип. Возвращает float (или Int64, если is_int).
    """
    # в строку
    s = series.astype(str).str.lower()

    # все варианты пробелов, NBSP → ничего
    s = s.apply(lambda x: SPACE_CLEAN_RE.sub('', x))

    # заменить запятые на точку
    s = s.str.replace(',', '.', regex=False)

    # drop грязные предопределённые токены → NaN
    s = s.apply(lambda x: pd.NA if x in X5ColumnConfig.INVALID_NUMERIC_TOKENS else x)

    # убрать всё, кроме цифр/точки/знаков/экспоненты
    s = s.apply(lambda x: NON_NUMERIC_KEEP_SIGNS_RE.sub('', x) if isinstance(x, str) else x)

    # пустые строки → NaN
    s = s.replace('', pd.NA)

    # to numeric
    num = pd.to_numeric(s, errors='coerce')

    # отладка: покажем до 5 грязных образцов
    bad_mask = num.isna() & s.notna()
    if bad_mask.any():
        samples = s[bad_mask].unique()[:5]
        logger.warning(f"[X5 Stage] Неконвертируемые значения в числовой колонке {col_log_name}: {samples}")

    # тип
    if is_int:
        # nullable Int64 (SQL сами приведём)
        num = num.round().astype('Int64')
    else:
        num = num.astype(float)

    return num


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    На входе df уже с нормализованными именами.
    Пробуем привести все колонки, чьи имена есть в NUMERIC_COLS.
    Остальные остаются текстовыми.
    """
    df = df.copy()
    for col, sql_type in X5ColumnConfig.NUMERIC_COLS.items():
        if col not in df.columns:
            continue
        is_int = sql_type.upper() == 'INT'
        df[col] = _coerce_numeric_series(df[col], is_int=is_int, col_log_name=col)
    return df


def _convert_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Всё, что не в NUMERIC_COLS, переводим в строковый dtype (pandas StringDtype).
    """
    df = df.copy()
    for c in df.columns:
        if c in X5ColumnConfig.NUMERIC_COLS:
            continue
        df[c] = df[c].astype('string').fillna('')
    return df


def _df_to_param_rows(df: pd.DataFrame):
    """
    Преобразуем DataFrame в список кортежей Python с заменой NaN/pd.NA на None.
    """
    arr = df.to_dict(orient='records')
    out_rows = []
    for rec in arr:
        row = []
        for v in rec.values():
            if v is pd.NA or (isinstance(v, float) and np.isnan(v)):
                row.append(None)
            else:
                row.append(v)
        out_rows.append(tuple(row))
    return out_rows


# ------------------------------------------------------------------ #
# bulk_insert                                                        #
# ------------------------------------------------------------------ #
def _bulk_insert(df: pd.DataFrame, table_name: str, engine, schema: str):
    """
    Создаём таблицу (если нет) + массовая вставка.
    """
    logger.info(f"[X5 Stage] Начало bulk_insert: {schema}.{table_name}, количество строк: {len(df)}")

    # подготовка типов
    cols_ddl = []
    for col in df.columns:
        sql_type = X5ColumnConfig.NUMERIC_COLS.get(col, 'NVARCHAR(255)')
        cols_ddl.append(f'[{col}] {sql_type}')
    logger.debug(f"[X5 Stage] DDL для {schema}.{table_name}: {', '.join(cols_ddl)}")

    create_sql = f"""
    IF NOT EXISTS (
        SELECT * FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE s.name = '{schema}' AND t.name = '{table_name}'
    )
    BEGIN
        CREATE TABLE [{schema}].[{table_name}] (
            {', '.join(cols_ddl)}
        )
    END
    """

    with closing(engine.connect()) as conn:
        with conn.begin():
            logger.info(f"[X5 Stage] Проверка/создание таблицы {schema}.{table_name}")
            conn.execute(text(create_sql))
            logger.info(f"[X5 Stage] Таблица {schema}.{table_name} готова к вставке данных.")

        # pyodbc cursor
        with conn.connection.cursor() as cursor:  # type: ignore[attr-defined]
            cursor.fast_executemany = True

            cols = ', '.join(f'[{c}]' for c in df.columns)
            params = ', '.join(['?'] * len(df.columns))
            insert_sql = f"INSERT INTO [{schema}].[{table_name}] ({cols}) VALUES ({params})"

            rows = _df_to_param_rows(df)
            if not rows:
                logger.warning(f"[X5 Stage] Нет данных для вставки в {schema}.{table_name}")
                return

            BATCH_SIZE = 50_000
            logger.info(f"[X5 Stage] Начало вставки данных батчами по {BATCH_SIZE} строк.")
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]
                try:
                    cursor.executemany(insert_sql, batch)
                    logger.debug(f"[X5 Stage] Успешно вставлен батч {i}-{i+len(batch)} строк.")
                except Exception as e:  # pragma: no cover
                    logger.error(f"[X5 Stage] Ошибка executemany батча {i}-{i+len(batch)}: {e}", exc_info=True)
                    raise

            logger.info(f"[X5 Stage] Вставка завершена. Всего вставлено {len(rows)} строк в {schema}.{table_name}.")


# ------------------------------------------------------------------ #
# Public API                                                         #
# ------------------------------------------------------------------ #
def convert_raw_to_stage(table_name: str, raw_engine, stage_engine, stage_schema='x5', limit=None):
    """
    Читаем raw.raw_table → нормализуем имена → приводим числовые → строки →
    создаём stage.<schema>.<table> → bulk insert.
    """
    try:
        logger.info(f"[X5 Stage] Начало обработки таблицы {table_name}")

        # сколько строк в raw
        with raw_engine.connect() as conn:
            total_rows = conn.execute(text(f"SELECT COUNT(*) FROM raw.{table_name}")).scalar()
        logger.info(f"[X5 Stage] Всего строк для обработки: {total_rows}")

        # читаем
        sql_query = f"SELECT TOP {limit} * FROM raw.{table_name}" if limit else f"SELECT * FROM raw.{table_name}"
        chunksize = 100_000

        processed_rows = 0
        chunks = []
        for chunk in pd.read_sql_query(sql_query, raw_engine, chunksize=chunksize):
            chunk = _normalize_columns(chunk)
            chunk = _convert_numeric_columns(chunk)
            chunk = _convert_string_columns(chunk)

            chunks.append(chunk)
            processed_rows += len(chunk)
            logger.info(f"[X5 Stage] Обработано {processed_rows}/{total_rows} строк ({processed_rows/total_rows:.1%})")

            if limit and processed_rows >= limit:
                break

        if not chunks:
            logger.warning("[X5 Stage] Нет данных для обработки.")
            return

        final_df = pd.concat(chunks, ignore_index=True)

        # вставка
        _bulk_insert(final_df, table_name, stage_engine, stage_schema)
        logger.info(f"[X5 Stage] Успешно загружено {len(final_df)} строк в {stage_schema}.{table_name}")

    except Exception as e:
        logger.error(f"[X5 Stage ERROR] Ошибка при обработке таблицы {table_name}: {e}", exc_info=True)
        raise

def schedule_stage_conversion(table_name: str, raw_engine, stage_engine, stage_schema='x5', limit=None):
    return convert_raw_to_stage(table_name, raw_engine, stage_engine, stage_schema, limit)
