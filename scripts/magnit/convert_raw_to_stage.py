from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Iterable, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# =====================================================================================
# Canonical Stage schema config
# =====================================================================================
class ColumnConfig:
    """Configuration: numeric targets, RU→EN renames (2024 vs 2025), RAW_TURBO aliases."""

    NUMERIC_COLS = {
        'turnover_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'turnover_quantity': {'dtype': 'float64', 'default': 0.0},
        'incoming_price': {'dtype': 'float64', 'default': 0.0},
        'avg_sell_price': {'dtype': 'float64', 'default': 0.0},
        'writeoff_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'writeoff_quantity': {'dtype': 'float64', 'default': 0.0},
        'sales_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'sales_weight_kg': {'dtype': 'float64', 'default': 0.0},
        'sales_quantity': {'dtype': 'float64', 'default': 0.0},
        'avg_purchase_price': {'dtype': 'float64', 'default': 0.0},
        'margin_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'loss_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'loss_quantity': {'dtype': 'float64', 'default': 0.0},
        'promo_sales_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'stock_quantity': {'dtype': 'float64', 'default': 0.0},
        'stock_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'discount_amount_rub': {'dtype': 'float64', 'default': 0.0},
    }

    # Minimal 2024 mapping (month granularity; fewer metrics)
    RENAME_MAP_2024 = {
        'month': 'month',
        'формат': 'формат',
        'наименование_тт': 'наименование_тт',
        'код_тт': 'код_тт',
        'адрес_тт': 'адрес_тт',
        'уровень_1': 'уровень_1',
        'уровень_2': 'уровень_2',
        'уровень_3': 'уровень_3',
        'уровень_4': 'уровень_4',
        'поставщик': 'поставщик',
        'бренд': 'бренд',
        'наименование_тп': 'наименование_тп',
        'код_тп': 'код_тп',
        'шк': 'шк',
        'оборот_руб': 'оборот_руб',
        'оборот_шт': 'оборот_шт',
        'входящая_цена': 'входящая_цена',
    }


    # Expanded 2025 mapping (day granularity; richer metrics)
    RENAME_MAP_2025 = {
    'month': 'sales_month',
    'дата': 'sales_date',
    'формат': 'store_format',
    'наименование_тт': 'store_name',
    'код_тт': 'store_code',
    'адрес_тт': 'store_address',
    'уровень_1': 'product_level_1',
    'уровень_2': 'product_level_2',
    'уровень_3': 'product_level_3',
    'уровень_4': 'product_level_4',
    'поставщик': 'supplier_name',
    'бренд': 'brand',
    'наименование_тп': 'product_name',
    'код_тп': 'product_code',
    'шк': 'barcode',
    'оборот_руб': 'turnover_amount_rub',
    'оборот_шт': 'turnover_quantity',
    'входящая_цена': 'incoming_price',
    'код_группы': 'product_group_code',
    'группа': 'product_group_name',
    'код_категории': 'product_category_code',
    'категория': 'product_category_name',
    'код_подкатегории': 'product_subcategory_code',
    'подкатегория': 'product_subcategory_name',
    'артикул': 'product_article',
    'код_поставщика': 'supplier_code',
    'регион': 'region',
    'город': 'city',
    'ср_цена_продажи': 'avg_sell_price',
    'списания_руб': 'writeoff_amount_rub',
    'списания_шт': 'writeoff_quantity',
    'продажи_руб': 'sales_amount_rub',
    'продажи_кг': 'sales_weight_kg',
    'продажи_шт': 'sales_quantity',
    'маржа_руб': 'margin_amount_rub',
    'потери_руб': 'loss_amount_rub',
    'потери_шт': 'loss_quantity',
    'промо_продажи_руб': 'promo_sales_amount_rub',
    'остаток_шт': 'stock_quantity',
    'остаток_руб': 'stock_amount_rub',
    'скидка_руб': 'discount_amount_rub',
}


    RENAME_MAP_RAW_TURBO = {
        'year': 'tmp_year',                 # intermediate → sales_month
        'код': 'store_code',
        'store_name': 'store_name',         # already EN
        'адрес': 'store_address',
        'уровень_1': 'product_level_1',
        'уровень_2': 'product_level_2',
        'уровень_3': 'product_level_3',
        'уровень_4': 'product_level_4',
        'position_code': 'product_article', # alt SKU id
        'наименование': 'product_name',
        'бренд': 'brand',
        'поставщик': 'supplier_name',
        'barcode': 'barcode',
        'quantity_sold': 'sales_quantity',
        'себестоимсть_в_руб.': 'incoming_price',  # total cost per line
        'sales_amount': 'sales_amount_rub',       # line revenue
        'оборот_шт': 'sales_quantity',
        'входящая_цена': 'incoming_price',
        'наименование_тт': 'store_name',
        'код_тт': 'store_code',
        'адрес_тт': 'store_address',
        'формат': 'store_format',
        'наименование_тп': 'product_name',
        'код_тп': 'product_code',
        'шк': 'barcode',
        'month': 'sales_month',
        'поставщик': 'supplier_name',
    }


    # RU month to number
    MONTH_MAP_RU = {
        'январь': '01', 'февраль': '02', 'март': '03', 'апрель': '04', 'май': '05', 'июнь': '06',
        'июль': '07', 'август': '08', 'сентябрь': '09', 'октябрь': '10', 'ноябрь': '11', 'декабрь': '12',
    }

    # English month map (for filename parsing)
    MONTH_MAP_EN = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }

    @classmethod
    def canonical_stage_columns(cls) -> List[str]:
        base_dims = [
            'sales_date', 'sales_month',
            'store_format', 'store_name', 'store_code', 'store_address',
            'region', 'city',
            'supplier_code', 'supplier_name',
            'brand',
            'product_group_code', 'product_group_name',
            'product_category_code', 'product_category_name',
            'product_subcategory_code', 'product_subcategory_name',
            'product_level_1', 'product_level_2', 'product_level_3', 'product_level_4',
            'product_article', 'product_name', 'product_code', 'barcode',
        ]
        metrics = list(cls.NUMERIC_COLS.keys())
        return base_dims + metrics


# Precompute canonical order
CANON_STAGE_COLS = ColumnConfig.canonical_stage_columns()
CANON_STAGE_SET = set(CANON_STAGE_COLS)


RU_STAGE_EN_MAP = {
'month': 'month',
'формат': 'store_format',
'наименование_тт': 'store_name',
'код_тт': 'store_code',
'адрес_тт': 'store_address',
'уровень_1': 'level_1',
'уровень_2': 'level_2',
'уровень_3': 'level_3',
'уровень_4': 'level_4',
'поставщик': 'supplier_name',
'бренд': 'brand',
'наименование_тп': 'product_name',
'код_тп': 'product_code',
'шк': 'barcode',
'оборот_руб': 'turnover_rub',
'оборот_шт': 'turnover_qty',
'входящая_цена': 'purchase_price',
# load_dt handled separately (added in DDL)
}
# columns we TREAT AS NUMERIC metrics (DECIMAL + numeric coercion)
STAGE_NUMERIC_METRICS = {'turnover_rub', 'turnover_qty', 'purchase_price'}


NUMERIC_COLS = {
        'turnover_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'turnover_quantity': {'dtype': 'float64', 'default': 0.0},
        'incoming_price': {'dtype': 'float64', 'default': 0.0},
        'avg_sell_price': {'dtype': 'float64', 'default': 0.0},
        'writeoff_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'writeoff_quantity': {'dtype': 'float64', 'default': 0.0},
        'sales_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'sales_weight_kg': {'dtype': 'float64', 'default': 0.0},
        'sales_quantity': {'dtype': 'float64', 'default': 0.0},
        'avg_purchase_price': {'dtype': 'float64', 'default': 0.0},
        'margin_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'loss_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'loss_quantity': {'dtype': 'float64', 'default': 0.0},
        'promo_sales_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'stock_quantity': {'dtype': 'float64', 'default': 0.0},
        'stock_amount_rub': {'dtype': 'float64', 'default': 0.0},
        'discount_amount_rub': {'dtype': 'float64', 'default': 0.0},
    }

def _convert_numeric_cols(chunk: pd.DataFrame) -> pd.DataFrame:
    for col, props in NUMERIC_COLS.items():
        if col in chunk.columns:
            # Пытаемся конвертировать к float64, ошибки в NaN
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            # Заполняем NaN дефолтным значением
            chunk[col].fillna(props['default'], inplace=True)
    return chunk


# columns that look like ID / codes but лучше хранить текстом (NVARCHAR)
STAGE_ID_LIKE_TEXT = {'store_code', 'product_code', 'barcode'}

# regex: drop these RAW cols
DROP_COL_PAT = re.compile(r"^none_\d+$", re.I)

_CTRL_RE = re.compile(r"[\x00-\x1F\x7F-\x9F]")
_NON_NUM_RE = re.compile(r"[^0-9.\-]")

def _norm_text_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(_CTRL_RE, '', regex=True)
         .str.strip()
         .replace({'nan': None, 'None': None, '': None})
    )

def _norm_numeric_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(' ', '', regex=False)
         .str.replace(',', '.', regex=False)
         .str.replace(_NON_NUM_RE, '', regex=True)
         .replace({'': None})
         .astype(float)
    )

_FLOATISH_RE = re.compile(r'^-?\d+(\.\d+)?$')

def _text_from_floatish(s: pd.Series) -> pd.Series:
    """'260309.000000' -> '260309'; '1000235032.000000' -> '1000235032'."""
    x = s.astype(str).str.strip()
    x = x.where(~x.str.fullmatch(_FLOATISH_RE), x.str.replace(r'\.0+$', '', regex=True))
    return x.replace({'nan': None, 'None': None, '': None})

RAW_TO_STAGE_MAP = {
    'month': 'month',
    'формат': 'format',
    'наименование_тт': 'store_name',
    'код_тт': 'store_id',
    'адрес_тт': 'store_address',
    'уровень_1': 'level_1',
    'уровень_2': 'level_2',
    'уровень_3': 'level_3',
    'уровень_4': 'level_4',
    'поставщик': 'supplier',
    'бренд': 'brand',
    'наименование_тп': 'product_name',
    'код_тп': 'product_id',
    'шк': 'barcode',
    'оборот_руб': 'revenue_rub',
    'оборот_шт': 'revenue_qty',
    'входящая_цена': 'purchase_price'
}

def _map_raw_columns(raw_df: pd.DataFrame) -> List[str]:
    """Преобразует колонки RAW → Stage на английский"""
    mapped = []
    for col in raw_df.columns:
        col_lower = col.lower()
        if col_lower in RAW_TO_STAGE_MAP:
            mapped.append(RAW_TO_STAGE_MAP[col_lower])
        else:
            mapped.append(col_lower)  # fallback
    return mapped

def _infer_sql_type(series: pd.Series) -> str:
    """Определяет SQL тип колонки"""
    if pd.api.types.is_numeric_dtype(series):
        return "DECIMAL(18,3)"
    return "NVARCHAR(255)"

def _stage_column_list_from_raw(sample_df: pd.DataFrame) -> list[str]:
    """
    Возвращает:
      stage_cols_en – список английских имен колонок для Stage (RU→EN_MAP или identity)
    Порядок сохраняется.
    """
    stage_cols_en = []
    seen = set()

    for c in sample_df.columns:
        if DROP_COL_PAT.match(str(c)):
            continue
        en = RU_STAGE_EN_MAP.get(c.lower(), c)
        base = en
        idx = 1
        while en in seen:
            idx += 1
            en = f"{base}_{idx}"
        seen.add(en)
        stage_cols_en.append(en)

    return stage_cols_en




def _mirror_col_sql_en(col: str) -> str:
    cl = col.lower()
    if cl in STAGE_NUMERIC_METRICS:
        return f'[{col}] DECIMAL(38, 6) NULL'
    # codes as NVARCHAR – длинные, чтобы не отрезать адреса
    if cl in STAGE_ID_LIKE_TEXT or cl.endswith('_code'):
        return f'[{col}] NVARCHAR(255) NULL'
    # адреса и произвольный текст
    return f'[{col}] NVARCHAR(255) NULL'

def _create_stage_table_mirror(stage_engine, table_name, stage_cols, sample_df=None, schema='magnit'):
    """
    Создает Stage таблицу. Если sample_df не задан, используется FLOAT/NVARCHAR(255).
    """
    columns_sql = []
    if sample_df is not None:
        for c, raw_c in zip(stage_cols, sample_df.columns):
            sql_type = _infer_sql_type(sample_df[raw_c])
            columns_sql.append(f"[{c}] {sql_type} NOT NULL")
    else:
        for col in stage_cols:
            sql_type = "FLOAT" if col in NUMERIC_COLS else "NVARCHAR(255)"
            columns_sql.append(f"[{col}] {sql_type} NOT NULL")

    create_sql = f"""
    IF OBJECT_ID('{schema}.{table_name}', 'U') IS NULL
    CREATE TABLE [{schema}].[{table_name}] (
        {', '.join(columns_sql)}
    )
    """
    with stage_engine.begin() as conn:
        conn.execute(text(create_sql))


def _fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет NaN для всех типов"""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('')
    return df


def _apply_basic_cleaning_stage(chunk: pd.DataFrame, raw_cols: list[str], stage_cols_en: list[str]) -> pd.DataFrame:
    """
    chunk: DataFrame, ещё в raw именах (подмножество raw_cols)
    raw_cols: порядок исходных колонок
    stage_cols_en: целевые имена того же размера
    Returns new DF с переименованными колонками и базовой очисткой.
    """
    # Отбираем и выравниваем колонки
    chunk = chunk[raw_cols].copy()
    rename_map = dict(zip(raw_cols, stage_cols_en))
    chunk.rename(columns=rename_map, inplace=True)

    # Очистка
    for c in chunk.columns:
        cl = c.lower()
        if cl in STAGE_NUMERIC_METRICS:
            try:
                chunk[c] = _norm_numeric_series(chunk[c])
            except Exception:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        elif cl in STAGE_ID_LIKE_TEXT:
            chunk[c] = _text_from_floatish(chunk[c])
        else:
            chunk[c] = _norm_text_series(chunk[c])
    return chunk


# =====================================================================================
# Detection helpers
# =====================================================================================
_RU_MONTH_RE = re.compile("[А-Яа-я]+")
_EN_MONTH_RE = re.compile("(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*", re.I)
_YEAR_RE = re.compile(r"(20\d{2})")


def _detect_layout_year_from_sample(df: pd.DataFrame) -> int:
    low = [c.lower() for c in df.columns]
    if any(c == 'дата' for c in low):
        return 2025
    if 'month' in low:
        sample = str(df.iloc[0, low.index('month')]) if not df.empty else ''
        if _RU_MONTH_RE.search(sample):
            return 2024
    # if we saw 'year' numeric in raw_turbo
    if 'year' in low:
        try:
            y = pd.to_numeric(df.iloc[0, low.index('year')], errors='coerce')
            if 2000 <= y <= 2100:
                return int(y)
        except Exception:  # pragma: no cover
            pass
    return 2025


def _infer_month_year_from_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract (month_int, year_int) from table_name or filename."""
    low = name.lower()
    # month
    m = None
    m_match = _EN_MONTH_RE.search(low)
    if m_match:
        m = ColumnConfig.MONTH_MAP_EN.get(m_match.group(1)[:3]) or ColumnConfig.MONTH_MAP_EN.get(m_match.group(1))
    # year
    y = None
    y_match = _YEAR_RE.search(low)
    if y_match:
        y = int(y_match.group(1))
    return m, y


def _active_rename_map(df: pd.DataFrame, year: int) -> Dict[str, str]:
    """Select best rename map based on sample columns."""
    low_cols = {c.lower(): c for c in df.columns}

    # direct 2025 match?
    score_2025 = sum(1 for k in ColumnConfig.RENAME_MAP_2025 if k in low_cols)
    score_2024 = sum(1 for k in ColumnConfig.RENAME_MAP_2024 if k in low_cols)
    score_turbo = sum(1 for k in ColumnConfig.RENAME_MAP_RAW_TURBO if k in low_cols)

    if score_turbo >= max(score_2024, score_2025):
        src_map = ColumnConfig.RENAME_MAP_RAW_TURBO
    elif year == 2024 and score_2024 >= score_2025:
        src_map = ColumnConfig.RENAME_MAP_2024
    else:
        src_map = ColumnConfig.RENAME_MAP_2025

    active: Dict[str, str] = {}
    for ru_low, en in src_map.items():
        if ru_low in low_cols:
            active[low_cols[ru_low]] = en
    return active


# =====================================================================================
# Cleaning helpers
# =====================================================================================
import unicodedata
_CLEAN_CTRL = re.compile(r"[\x00-\x1F\x7F-\x9F]")
_NON_NUMERIC_KEEP = re.compile(r"[^0-9.\-]")


def _clean_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.normalize('NFKC')
         .str.replace(_CLEAN_CTRL, '', regex=True)
         .str.strip()
         .replace({'nan': ''})
    )


def _clean_numeric(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(' ', '', regex=False)
         .str.replace(',', '.', regex=False)
         .str.replace(_NON_NUMERIC_KEEP, '', regex=True)
    )


def _parse_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(_clean_numeric(s), errors='coerce').astype('float64')


# =====================================================================================
# Date helpers
# =====================================================================================

def _parse_sales_month(series: pd.Series, year_default: Optional[int] = None) -> pd.Series:
    if series.empty:
        return pd.to_datetime(pd.Series([], dtype='datetime64[ns]'))
    s = series.astype(str).str.lower().str.strip()
    mapped = s.map(ColumnConfig.MONTH_MAP_RU).fillna('01')
    if year_default is None:
        yrs = s.str.extract(r"(20\d{2})", expand=False)
        year_series = yrs.fillna(datetime.utcnow().year).astype(int).astype(str)
    else:
        year_series = pd.Series([year_default] * len(s), index=s.index, dtype="object")
    dt = pd.to_datetime(year_series + '-' + mapped + '-01', errors='coerce')
    return dt


def _derive_date_fields(df: pd.DataFrame, year_guess: int, table_name: str) -> pd.DataFrame:
    # 1) if sales_date present → parse
    if 'sales_date' in df.columns:
        df['sales_date'] = pd.to_datetime(df['sales_date'], errors='coerce').dt.date
        if 'sales_month' not in df.columns:
            df['sales_month'] = pd.to_datetime(df['sales_date']).dt.to_period('M').dt.to_timestamp().dt.date
        return df

    # 2) if tmp_year + month from filename
    month_fn, year_fn = _infer_month_year_from_name(table_name)

    # try 'tmp_year' numeric col from RAW_TURBO mapping
    tmp_year_series = None
    if 'tmp_year' in df.columns:
        tmp_year_series = pd.to_numeric(df['tmp_year'], errors='coerce').astype('Int64')

    # Year priority: tmp_year col > year_fn > year_guess
    used_year = None
    if tmp_year_series is not None and tmp_year_series.notna().any():
        used_year = int(tmp_year_series.dropna().iloc[0])
    elif year_fn is not None:
        used_year = year_fn
    else:
        used_year = year_guess

    if month_fn is not None and used_year is not None:
        df['sales_month'] = pd.to_datetime({
            'year': [used_year] * len(df),
            'month': [month_fn] * len(df),
            'day': [1] * len(df)
        })
        df['sales_date'] = pd.NaT  # unknown day
        return df

    # 3) fallback: sales_month from RU text if present
    if 'sales_month' in df.columns:
        dt = _parse_sales_month(df['sales_month'], year_default=used_year)
        df['sales_month'] = dt.dt.date
        df['sales_date'] = pd.NaT
        return df

    # final fallback
    df['sales_month'] = pd.NaT
    df['sales_date'] = pd.NaT
    return df


# =====================================================================================
# Stage table DDL
# =====================================================================================

def _stage_col_sql(col: str) -> str:
    if col in ColumnConfig.NUMERIC_COLS:
        return f'[{col}] DECIMAL(18, 2) NULL'
    if col in ('sales_date', 'sales_month'):
        return f'[{col}] DATE NULL'
    return f'[{col}] NVARCHAR(255) NULL'


def _create_stage_table(stage_engine: Engine, table_name: str, if_exists: str = 'append', schema: str = 'magnit') -> None:
    insp = inspect(stage_engine)
    exists = insp.has_table(table_name, schema=schema)
    full_table = f"[{schema}].[{table_name}]"

    if exists:
        if if_exists == 'fail':
            raise RuntimeError(f"Stage table {full_table} already exists.")
        elif if_exists == 'replace_truncate':
            logger.info("Truncating existing stage table %s", full_table)
            with stage_engine.begin() as conn:
                conn.execute(text(f"TRUNCATE TABLE {full_table}"))
            return
        else:
            logger.debug("Stage table %s exists; appending.", full_table)
            return

    cols_sql = ',\n                '.join(_stage_col_sql(c) for c in CANON_STAGE_COLS) + ",\n                load_dt DATETIME DEFAULT GETDATE()"
    create_sql = f"""
    CREATE TABLE {full_table} (
                {cols_sql}
    );
    """
    with stage_engine.begin() as conn:
        conn.execute(text(create_sql))
    logger.info("Created stage table %s", full_table)


# =====================================================================================
# Insert helpers (fast_executemany)
# =====================================================================================

def _prepare_insert_cursor(stage_engine: Engine, table_name: str, schema: str = 'magnit'):
    raw_conn = stage_engine.raw_connection()
    cursor = raw_conn.cursor()
    cursor.fast_executemany = True
    cols_clause = ', '.join(f'[{c}]' for c in CANON_STAGE_COLS)
    params_clause = ', '.join(['?'] * len(CANON_STAGE_COLS))
    insert_sql = f"INSERT INTO [{schema}].[{table_name}] ({cols_clause}) VALUES ({params_clause})"
    return raw_conn, cursor, insert_sql


def _insert_chunk(cursor, insert_sql: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    data = list(df.itertuples(index=False, name=None))
    cursor.executemany(insert_sql, data)
    return len(data)

DROP_COL_PAT = re.compile(r"^none_\\d+$", re.I)

def _stage_column_list_from_raw(sample_df: pd.DataFrame) -> List[str]:
    """Return ordered list of columns to materialize in Stage:
    - preserve raw order
    - drop technical none_* cols
    """
    cols = []
    for c in sample_df.columns:
        if DROP_COL_PAT.match(str(c)):
            continue
        cols.append(c)
    return cols


_NUMERIC_NAME_HINTS = {
    'оборот_руб', 'оборот_шт', 'входящая_цена',
    'quantity_sold', 'sales_amount', 'себестоимсть_в_руб.',
    'year', 'код', 'код_тт', 'код_тп', 'position_code',
}

def _mirror_col_sql(col: str) -> str:
    cl = col.lower()
    if cl in _NUMERIC_NAME_HINTS:
        return f'[{col}] DECIMAL(38, 6) NULL'
    return f'[{col}] NVARCHAR(4000) NULL'


def _norm_numeric_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(' ', '', regex=False)
         .str.replace(',', '.', regex=False)
         .str.replace(r'[^0-9.\\-]', '', regex=True)
         .replace({'': np.nan})
         .astype(float)
    )

def _norm_text_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r'[\\x00-\\x1F\\x7F-\\x9F]', '', regex=True)
         .str.strip()
         .replace({'nan': None, 'None': None, '': None})
    )


def _apply_basic_cleaning(chunk: pd.DataFrame) -> pd.DataFrame:
    for col in chunk.columns:
        cl = col.lower()
        if cl in _NUMERIC_NAME_HINTS:
            try:
                chunk[col] = _norm_numeric_series(chunk[col])
            except Exception:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        else:
            chunk[col] = _norm_text_series(chunk[col])
    return chunk


def _prepare_insert_cursor_mirror(stage_engine: Engine, table_name: str, stage_cols: List[str], schema: str = 'magnit'):
    raw_conn = stage_engine.raw_connection()
    cursor = raw_conn.cursor()
    cursor.fast_executemany = True
    col_list = ', '.join(f'[{c}]' for c in stage_cols)
    qmarks = ', '.join(['?'] * len(stage_cols))
    insert_sql = f"INSERT INTO [{schema}].[{table_name}] ({col_list}) VALUES ({qmarks})"
    return raw_conn, cursor, insert_sql

COLUMN_RENAME_MAP = {
    'month': 'month',
    'формат': 'format',
    'наименование_тт': 'store_name',
    'код_тт': 'store_code',
    'адрес_тт': 'store_address',
    'уровень_1': 'level_1',
    'уровень_2': 'level_2',
    'уровень_3': 'level_3',
    'уровень_4': 'level_4',
    'поставщик': 'supplier',
    'бренд': 'brand',
    'наименование_тп': 'product_name',
    'код_тп': 'product_code',
    'шк': 'barcode',
    'оборот_руб': 'turnover_rub',
    'оборот_шт': 'turnover_qty',
    'входящая_цена': 'purchase_price',
    'load_dt': 'load_dt'
}



# =====================================================================================
# Main convert function
# =====================================================================================
import time
import os
import tempfile

import time
import os
from typing import Optional, Callable
import pandas as pd
import tempfile
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

def _sanitize_batch(batch):
    fixed_batch = []
    for row in batch:
        fixed_row = []
        for val in row:
            if isinstance(val, str):
                v = val.strip()
                # Пытаемся конвертнуть в число
                if v.replace('.', '', 1).isdigit():
                    fixed_row.append(float(v) if '.' in v else int(v))
                else:
                    fixed_row.append(None)  # или 0, если нужно
            else:
                fixed_row.append(val)
        fixed_batch.append(tuple(fixed_row))
    return fixed_batch

def convert_raw_to_stage(
    table_name: str,
    raw_engine: Engine,
    stage_engine: Engine,
    stage_schema: str = 'magnit',
    chunk_size: int = 50_000,
    batch_size: int = 10_000,
    limit: Optional[int] = None,
    progress_cb: Optional[Callable[[int, float], None]] = None,
) -> int:
    start_ts = time.time()
    logger.info("[Stage/Mirror] Begin RAW→STAGE for %s", table_name)

    with raw_engine.connect() as raw_conn:
        sample_sql = text(f"SELECT TOP 100 * FROM raw.[{table_name}]")
        sample_df = pd.read_sql(sample_sql, raw_conn)

    if sample_df.empty:
        logger.warning("[Stage/Mirror] RAW.%s empty; nothing to load.", table_name)
        return 0

    # Маппинг колонок
    stage_cols_en = _map_raw_columns(sample_df)
    logger.info("[Stage/Mirror] RAW.%s -> Stage cols: %s", table_name, ', '.join(stage_cols_en))

    # Создаём Stage таблицу
    _create_stage_table_mirror(stage_engine, table_name, stage_cols_en, sample_df, schema=stage_schema)

    placeholders = ', '.join(['?'] * len(stage_cols_en))
    cols_escaped = ', '.join(f'[{c}]' for c in stage_cols_en)
    insert_sql = f"INSERT INTO [{stage_schema}].[{table_name}] ({cols_escaped}) VALUES ({placeholders})"

    total_rows = 0
    with raw_engine.connect().execution_options(stream_results=True) as raw_conn:
        raw_stage_conn = stage_engine.raw_connection()
        cursor = raw_stage_conn.cursor()
        cursor.fast_executemany = True

        base_query = f"SELECT * FROM raw.[{table_name}]"
        if limit is not None:
            base_query = f"SELECT TOP {limit} * FROM raw.[{table_name}]"

        for chunk in pd.read_sql(text(base_query), raw_conn, chunksize=chunk_size):
            if chunk.empty:
                continue

            chunk.columns = stage_cols_en
            chunk = _fill_nulls(chunk)
            chunk = _convert_numeric_cols(chunk)  # оставить, если есть кастомная логика
            data_tuples = list(chunk.itertuples(index=False, name=None))

            for i in range(0, len(data_tuples), batch_size):
                batch = data_tuples[i:i + batch_size]
                cursor.executemany(insert_sql, batch)
                raw_stage_conn.commit()

            total_rows += len(chunk)
            logger.info("[Stage/Mirror] Inserted %d rows (cumulative %d).", len(chunk), total_rows)

            if progress_cb:
                try:
                    progress_cb(total_rows, time.time() - start_ts)
                except Exception:
                    pass

        cursor.close()
        raw_stage_conn.close()

    dur = time.time() - start_ts
    logger.info("[Stage/Mirror] Loaded %s rows RAW.%s -> %s.%s in %.1fs", total_rows, table_name, stage_schema, table_name, dur)
    return total_rows


# =====================================================================================
# CLI harness
# =====================================================================================
if __name__ == "__main__":  # pragma: no cover
    import argparse
    from sqlalchemy import create_engine

    parser = argparse.ArgumentParser(description="RAW→STAGE loader for Magnit (optimized)")
    parser.add_argument("--raw", required=True, help="SQLAlchemy URL to RAW DB")
    parser.add_argument("--stage", required=True, help="SQLAlchemy URL to STAGE DB")
    parser.add_argument("table", help="RAW table name")
    parser.add_argument("--schema", default="magnit")
    parser.add_argument("--chunk", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--replace", action="store_true", help="TRUNCATE stage table before load")
    args = parser.parse_args()

    raw_eng = create_engine(args.raw)
    stage_eng = create_engine(args.stage)
    if_exists = 'replace_truncate' if args.replace else 'append'
    convert_raw_to_stage(
        table_name=args.table,
        raw_engine=raw_eng,
        stage_engine=stage_eng,
        stage_schema=args.schema,
        if_exists=if_exists,
        chunk_size=args.chunk,
        limit=args.limit,
    )
