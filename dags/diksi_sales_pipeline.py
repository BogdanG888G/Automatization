"""
Airflow DAG: Diksi Sales Pipeline (optimized)
--------------------------------------------

Задачи:
- Сканировать входную директорию на файлы Дикси.
- Читать файлы любого формата (.csv, .xlsx, .xls, .xlsb) с учётом того, что
  *первая строка* содержит служебные маркеры ("1 1 1 2 2" / "Итоги"),
  а *вторая строка* — настоящие заголовки.
- Делать нормализованные и уникальные имена столбцов (дописывать _1, _2 ... при дубликатах).
- Грузить в Test.raw через внешнюю функцию `create_table_and_upload_diksi`.
- Конвертировать в Stage.<schema> через `convert_raw_to_stage`.
- Архивировать обработанные файлы.

Примечание:
Этот DAG предполагает, что модуль `diksi.create_table_and_upload` уже содержит
обновлённую логику чтения файлов со `skiprows=1` или аналогичной обработкой «вторая строка — шапка».
Если ещё нет — обнови модуль по ранее присланному мной коду.

Версия: 2025-07-17 (Europe/Madrid)
"""

from __future__ import annotations

import os
import re
import shutil
import logging
import unicodedata
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import chardet
import pyxlsb  # XLSB support

from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from sqlalchemy import create_engine, exc

# ------------------------------------------------------------------
# Database connection strings (update host / creds as needed)
# ------------------------------------------------------------------
DEFAULT_CONN_TEST = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Test"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)
DEFAULT_CONN_STAGE = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
class DiksiConfig:
    MAX_CONCURRENT_TASKS = 4
    MAX_FILES_PER_RUN = 8
    TASK_TIMEOUT = timedelta(minutes=60)
    MAX_FILE_SIZE = 3 * 1024 * 1024 * 1024  # 3GB
    max_active_tis_per_dag=2  # Ограничьте параллельные выполнения

    DATA_DIR = "/opt/airflow/data"
    ARCHIVE_DIR = "/opt/airflow/archive"
    ALLOWED_EXT = {".csv", ".xlsx", ".xls", ".xlsb"}
    FILE_PREFIX = "diksi_"
    STAGE_SCHEMA = "diksi"

    @staticmethod
    def get_processing_pool():
        try:
            pools = Variable.get("airflow_pools", default_var="default_pool")
            return "diksi_file_pool" if "diksi_file_pool" in pools.split(",") else "default_pool"
        except Exception:  # Variable not set, etc.
            return "default_pool"

    PROCESSING_POOL = get_processing_pool()
    POOL_SLOTS = 6

    CONN_SETTINGS = {
        "pool_size": 2,
        "max_overflow": 3,
        "pool_timeout": 30,
        "pool_recycle": 3600,
        "connect_args": {
            "timeout": 900,
            "application_name": "airflow_diksi_loader",
        },
    }


# ------------------------------------------------------------------
# Engine cache
# ------------------------------------------------------------------
_engine_cache = {}
_last_cache_update = datetime.min


def get_engine(db_type: str):
    """Get (cached) SQLAlchemy engine."""
    global _last_cache_update

    if (datetime.now() - _last_cache_update) > timedelta(hours=6):
        _engine_cache.clear()
        _last_cache_update = datetime.now()

    if db_type not in _engine_cache:
        try:
            if db_type.lower() == "test":
                conn_str = DEFAULT_CONN_TEST
            elif db_type.lower() == "stage":
                conn_str = DEFAULT_CONN_STAGE
            else:
                raise ValueError(f"Unknown DB type: {db_type}")

            _engine_cache[db_type] = create_engine(conn_str, **DiksiConfig.CONN_SETTINGS)
        except Exception as e:  # noqa: BLE001
            logging.error(f"Error creating engine for {db_type}: {str(e)}")
            raise

    return _engine_cache[db_type]


# ------------------------------------------------------------------
# File encoding detection
# ------------------------------------------------------------------

def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding using chardet (sample first ~10KB)."""
    with open(file_path, "rb") as f:  # noqa: PTH123
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result["encoding"] if result["confidence"] > 0.7 else "utf-8"


# ------------------------------------------------------------------
# Preprocess text-ish files (.csv, possibly .xls mis-saved)
# ------------------------------------------------------------------

def preprocess_diksi_file(file_path: str) -> str:
    """Light preprocessing: strip NULs, quotes, unify decimal sep, re-encode UTF-8.

    Returns path to temp processed file; caller responsible for cleanup.
    For binary (.xlsb) just return original path.
    """
    if file_path.endswith(".xlsb"):
        return file_path  # binary; skip

    temp_path = file_path + ".processed"
    encoding = detect_file_encoding(file_path)

    try:
        with open(file_path, "r", encoding=encoding) as infile, open(  # noqa: PTH123
            temp_path, "w", encoding="utf-8"
        ) as outfile:  # noqa: PTH123
            for i, line in enumerate(infile):
                line = line.replace("\x00", "").replace('"', "").strip()
                if not line:
                    continue

                if file_path.endswith(".csv"):
                    parts = line.split(";")
                    if len(parts) > 1:
                        parts = [
                            p.replace(",", ".") if p.replace(",", "").replace(".", "").isdigit() else p
                            for p in parts
                        ]
                        line = ";".join(parts)

                outfile.write(line + "\n")

        return temp_path
    except Exception as e:  # noqa: BLE001
        logging.error(f"File preprocessing error: {str(e)}")
        return file_path


# ------------------------------------------------------------------
# Column normalization utilities
# ------------------------------------------------------------------

# Частичный словарь транслитерации / глоссарий повторяющихся колонок
_TRANSLIT_MAP = {
    "уровень": "uroven",
    "втм": "vtm",
    "товар": "tovar",
    "адрес": "adres",
    "код": "kod",
    "магазины": "magaziny",
    "количество": "kolichestvo",
    "себестоимость_с_ндс": "sebestoimost_s_nds",
    "себестоимость": "sebestoimost",  # fallback
    "с_ндс": "s_nds",
    "сумма_с_ндс": "summa_s_nds",
    "сумма": "summa",  # fallback
    "итоги": "itogi",
}


def _normalize_str(s: str) -> str:
    """Base normalization: lower, strip, spaces->_, collapse repeats."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)  # normalize unicode
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s


def make_unique_columns(cols) -> list[str]:
    """Normalize + transliterate recurring Russian column names; ensure uniqueness.

    Example: ['Количество','Количество','Сумма с НДС'] ->
    ['kolichestvo','kolichestvo_1','summa_s_nds']
    """
    normed: list[str] = []
    seen: dict[str, int] = {}

    for raw in cols:
        base = _normalize_str(raw)
        # Apply key-word replacements greedily
        for ru, lat in _TRANSLIT_MAP.items():
            base = base.replace(ru, lat)
        # Drop any residual non-word chars
        base = re.sub(r"[^0-9a-zA-Z_]+", "", base)
        base = base.strip("_") or "col"

        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            name = base

        normed.append(name)

    return normed


def apply_second_row_header(df: pd.DataFrame) -> pd.DataFrame:
    """Detect & use 2nd row as header when 1st row is service/meta.

    Works when df read with header=None (pandas numeric columns). If df already
    has string column names (pandas used some row as header), just normalize.
    """
    if df is None or df.empty:
        return df

    # If first col name already str and not integer-ish -> assume header already applied
    if not isinstance(df.columns[0], (int, float)):
        df = df.copy()
        df.columns = make_unique_columns(df.columns)
        return df

    # We'll look at row0 and row1 strings
    row0 = df.iloc[0].astype(str).str.strip().tolist()
    row1 = df.iloc[1].astype(str).str.strip().tolist() if len(df) > 1 else []

    def _is_meta(val: str) -> bool:
        v = str(val).strip().lower()
        return v.isdigit() or v in {"итоги", "", "nan", "none"}

    meta_ratio = sum(_is_meta(v) for v in row0) / max(len(row0), 1)

    if meta_ratio >= 0.5 and row1:  # treat row1 as header, drop row0+row1
        header = row1
        body = df.iloc[2:].reset_index(drop=True)
    else:  # treat row0 as header, drop it
        header = row0
        body = df.iloc[1:].reset_index(drop=True)

    body.columns = make_unique_columns(header)
    return body


# ------------------------------------------------------------------
# File reader (main)
# ------------------------------------------------------------------

def read_diksi_file(file_path: str, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Read Diksi file with optional row limit.

    Strategy:
    - Always read raw with *header=None*.
    - Apply `apply_second_row_header` to pick the true header row (2nd row in our case).
    - Supports CSV, XLSX/XLS, XLSB.
    """
    if not os.path.exists(file_path):  # noqa: PTH110
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    # ---------------- XLSB ----------------
    if ext == ".xlsb":
        try:
            with pyxlsb.open_workbook(file_path) as wb:  # noqa: PTH123
                # pyxlsb sheet index starts at 1
                with wb.get_sheet(1) as sheet:
                    data = []
                    for i, row in enumerate(sheet.rows()):
                        if max_rows is not None and i >= max_rows:
                            break
                        data.append([item.v for item in row])
            if not data:
                return None
            df = pd.DataFrame(data)
            df = apply_second_row_header(df)
            return df
        except Exception as e:  # noqa: BLE001
            logging.error(f"Failed to read XLSB file: {str(e)}")
            return None

    # ---------------- CSV ----------------
    if ext == ".csv":
        encoding = detect_file_encoding(file_path)
        try:
            df = pd.read_csv(
                file_path,
                delimiter=";",
                decimal=".",
                thousands=" ",
                encoding=encoding,
                header=None,  # key!
                nrows=max_rows,
            )
            df = apply_second_row_header(df)
            return df
        except Exception as e:  # noqa: BLE001
            logging.warning(f"CSV read failed ({file_path}): {str(e)}")
            # fallback through preprocess
            processed_path = preprocess_diksi_file(file_path)
            try:
                df = pd.read_csv(processed_path, delimiter=";", decimal=".", thousands=" ", header=None)
                df = apply_second_row_header(df)
                return df
            finally:
                if processed_path != file_path and os.path.exists(processed_path):  # noqa: PTH110
                    os.remove(processed_path)  # noqa: PTH107

    # ---------------- Excel (.xlsx / .xls) ----------------
    # Try openpyxl first (works for .xlsx; for .xls pandas may choose xlrd or engine auto)
    try:
        df = pd.read_excel(file_path, engine="openpyxl", header=None, nrows=max_rows)
        df = apply_second_row_header(df)
        return df
    except Exception:  # noqa: BLE001
        logging.warning(f"openpyxl read failed, trying default engine: {file_path}")

    # Try pandas default engine fallback
    try:
        df = pd.read_excel(file_path, header=None, nrows=max_rows)
        df = apply_second_row_header(df)
        return df
    except Exception as e:  # noqa: BLE001
        logging.warning(f"Excel read fallback failed ({file_path}): {str(e)}")

    # Final fallback: preprocess (in case actually CSV renamed or damaged Excel)
    processed_path = preprocess_diksi_file(file_path)
    try:
        if processed_path.lower().endswith(".csv"):
            df = pd.read_csv(processed_path, delimiter=";", decimal=".", thousands=" ", header=None)
        else:
            df = pd.read_excel(processed_path, engine="openpyxl", header=None)
        df = apply_second_row_header(df)
        return df
    finally:
        if processed_path != file_path and os.path.exists(processed_path):  # noqa: PTH110
            os.remove(processed_path)  # noqa: PTH107

    return None


# ------------------------------------------------------------------
# Archiving helper
# ------------------------------------------------------------------

def archive_empty_file(file_path: str, empty: bool = False, error: bool = False) -> None:  # noqa: FBT002, FBT003
    """Archive file (preserve original name).

    Flags `empty` / `error` зарезервированы — при желании можно добавлять суффиксы.
    Сейчас оставляем исходное имя, чтобы не ломать трассировку.
    """
    os.makedirs(DiksiConfig.ARCHIVE_DIR, exist_ok=True)  # noqa: PTH103

    archive_name = os.path.basename(file_path)  # noqa: PTH119
    archive_path = os.path.join(DiksiConfig.ARCHIVE_DIR, archive_name)  # noqa: PTH118

    try:
        shutil.move(file_path, archive_path)  # noqa: PTH118
        logging.info(f"File archived: {archive_path}")
    except Exception as e:  # noqa: BLE001
        logging.error(f"Failed to archive file {file_path}: {str(e)}")


# ------------------------------------------------------------------
# Scan task
# ------------------------------------------------------------------
@task(
    task_id="scan_diksi_files",
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=DiksiConfig.TASK_TIMEOUT,
    pool=DiksiConfig.PROCESSING_POOL,
    pool_slots=1,
)
def scan_diksi_files() -> List[str]:
    """Scan for incoming Diksi files matching criteria."""
    try:
        valid_files: list[str] = []
        total_size = 0

        logging.info(f"Scanning directory: {DiksiConfig.DATA_DIR}")
        if not os.path.exists(DiksiConfig.DATA_DIR):  # noqa: PTH110
            raise FileNotFoundError(f"Directory not found: {DiksiConfig.DATA_DIR}")

        for root, _, files in os.walk(DiksiConfig.DATA_DIR):  # noqa: PTH112
            for f in files:
                f_lower = f.lower()
                ext = os.path.splitext(f)[1].lower()
                if ("diksi" in f_lower) and (ext in DiksiConfig.ALLOWED_EXT):
                    file_path = os.path.join(root, f)  # noqa: PTH118
                    file_size = os.path.getsize(file_path)  # noqa: PTH122

                    if file_size == 0:
                        logging.warning(f"Skipping empty file: {file_path}")
                        continue

                    if total_size + file_size > DiksiConfig.MAX_FILE_SIZE:
                        logging.warning(f"File size limit exceeded. Skipping {file_path}")
                        continue

                    valid_files.append(file_path)
                    total_size += file_size

                    if len(valid_files) >= DiksiConfig.MAX_FILES_PER_RUN:
                        break

            if len(valid_files) >= DiksiConfig.MAX_FILES_PER_RUN:
                break

        logging.info(
            "Found %s Diksi files (total size: %.2f MB)",
            len(valid_files),
            total_size / 1024 / 1024,
        )
        return valid_files[: DiksiConfig.MAX_FILES_PER_RUN]

    except Exception as e:  # noqa: BLE001
        logging.error(f"File scanning error: {str(e)}", exc_info=True)
        raise


# ------------------------------------------------------------------
# Process single file task
# ------------------------------------------------------------------
@task(
    task_id="process_diksi_file",
    retries=3,
    retry_delay=timedelta(minutes=5),
    execution_timeout=DiksiConfig.TASK_TIMEOUT,
    pool=DiksiConfig.PROCESSING_POOL,
    pool_slots=1,
)
def process_diksi_file(file_path: str) -> None:
    """Process a single Diksi file: read, load raw, convert to stage, archive."""
    try:
        if not os.path.exists(file_path):  # noqa: PTH110
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check empty
        if os.path.getsize(file_path) == 0:  # noqa: PTH122
            logging.warning(f"Skipping empty file: {file_path}")
            archive_empty_file(file_path, empty=True)
            return

        # Quick read (sample) primarily for logging & sanity; actual load done in create_table_and_upload_diksi
        df_sample = read_diksi_file(file_path, max_rows=None)
        if df_sample is None or df_sample.empty:
            logging.error(f"Failed to read data from file: {file_path}")
            archive_empty_file(file_path, error=True)
            return

        logging.info("Columns after header-fix: %s", list(df_sample.columns))
        logging.info("Sample rows read: %s", len(df_sample))

        # ------------------------------------------------------------------
        # RAW load
        # ------------------------------------------------------------------
        engine_test = get_engine("test")
        try:
            from diksi.create_table_and_upload import create_table_and_upload_diksi  # local import on purpose

            table_name = create_table_and_upload_diksi(file_path, engine=engine_test)
            logging.info(f"Diksi data loaded to raw.{table_name}")
        except exc.SQLAlchemyError as e:  # pragma: no cover - database error path
            engine_test.dispose()
            logging.error(f"Raw load error: {str(e)}")
            archive_empty_file(file_path, error=True)
            raise RuntimeError(f"Raw load error: {str(e)}")

        # ------------------------------------------------------------------
        # STAGE load
        # ------------------------------------------------------------------
        engine_stage = get_engine("stage")
        try:
            from diksi.convert_raw_to_stage import convert_raw_to_stage  # local import on purpose

            convert_raw_to_stage(
                table_name=table_name,
                raw_engine=engine_test,
                stage_engine=engine_stage,
                stage_schema=DiksiConfig.STAGE_SCHEMA,
                limit=None,
            )
            logging.info("Diksi data loaded to stage")
        except exc.SQLAlchemyError as e:  # pragma: no cover
            engine_stage.dispose()
            logging.error(f"Stage load error: {str(e)}")
            archive_empty_file(file_path, error=True)
            raise RuntimeError(f"Stage load error: {str(e)}")
        finally:
            # dispose both
            engine_test.dispose()
            engine_stage.dispose()

        # ------------------------------------------------------------------
        # Archive success
        # ------------------------------------------------------------------
        archive_empty_file(file_path)

    except Exception as e:  # noqa: BLE001
        logging.error(f"Diksi file processing error: {file_path} - {str(e)}", exc_info=True)
        archive_empty_file(file_path, error=True)
        raise


# ------------------------------------------------------------------
# DAG definition
# ------------------------------------------------------------------

def _default_args():
    return {
        "owner": "airflow",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "depends_on_past": False,
        "max_active_tis_per_dag": DiksiConfig.MAX_CONCURRENT_TASKS,
    }


with DAG(
    dag_id="diksi_sales_pipeline_optimized",
    default_args=_default_args(),
    start_date=datetime(2025, 7, 1),
    schedule_interval="0 20 * * *",  # каждый день в 15:00,  # triggered manually / externally
    catchup=False,
    max_active_tasks=DiksiConfig.MAX_CONCURRENT_TASKS,
    concurrency=DiksiConfig.MAX_CONCURRENT_TASKS,
    tags=["diksi", "optimized"],
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    scanned_files = scan_diksi_files()

    with TaskGroup("diksi_file_processing_group") as processing_group:
        process_diksi_file.expand(file_path=scanned_files)

    start >> scanned_files >> processing_group >> end
