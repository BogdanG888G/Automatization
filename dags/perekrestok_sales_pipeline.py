from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import os
import shutil
import logging
from sqlalchemy import create_engine, exc
import pandas as pd
from typing import List, Optional
from airflow.utils.task_group import TaskGroup
import chardet
import pyxlsb  # For XLSB file support


# ------------------------------------------------------------------ #
# DB connection strings                                              #
# ------------------------------------------------------------------ #
DEFAULT_CONN_TEST = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Test"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)
DEFAULT_CONN_STAGE = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)


# ------------------------------------------------------------------ #
# Config                                                             #
# ------------------------------------------------------------------ #
class PerekConfig:
    MAX_CONCURRENT_TASKS = 4
    MAX_FILES_PER_RUN = 8
    TASK_TIMEOUT = timedelta(minutes=45)
    MAX_FILE_SIZE = 3 * 1024 * 1024 * 1024  # 3GB

    DATA_DIR = "/opt/airflow/data"
    ARCHIVE_DIR = "/opt/airflow/archive"
    ALLOWED_EXT = {'.csv', '.xlsx', '.xls', '.xlsb'}
    FILE_PREFIX = "perekrestok"  # we'll match more variants in scan
    STAGE_SCHEMA = "perekrestok"  # update if actual schema name differs (e.g., perekrestok)

    @staticmethod
    def get_processing_pool():
        try:
            pools = Variable.get("airflow_pools", default_var="default_pool")
            return "perek_file_pool" if "perek_file_pool" in pools.split(",") else "default_pool"
        except Exception:  # noqa: BLE001
            return "default_pool"

    PROCESSING_POOL = get_processing_pool()
    POOL_SLOTS = 6

    CONN_SETTINGS = {
        'pool_size': 2,
        'max_overflow': 3,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'connect_args': {
            'timeout': 900,
            'application_name': 'airflow_perek_loader'
        }
    }


# ------------------------------------------------------------------ #
# Engine cache                                                       #
# ------------------------------------------------------------------ #
_engine_cache = {}
_last_cache_update = datetime.min


def get_engine(db_type: str):
    """Get SQLAlchemy engine with caching."""
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

            _engine_cache[db_type] = create_engine(conn_str, **PerekConfig.CONN_SETTINGS)
        except Exception as e:  # noqa: BLE001
            logging.error(f"Error creating engine for {db_type}: {str(e)}")
            raise

    return _engine_cache[db_type]


# ------------------------------------------------------------------ #
# File helpers                                                       #
# ------------------------------------------------------------------ #
def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding using chardet."""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'


def preprocess_perek_file(file_path: str) -> str:
    """Pre-process Perek files that might have parsing issues (CSV cleanup)."""
    if file_path.endswith('.xlsb'):
        return file_path  # Skip preprocessing for binary files

    temp_path = file_path + ".processed"
    encoding = detect_file_encoding(file_path)

    try:
        with open(file_path, 'r', encoding=encoding) as infile, \
             open(temp_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                line = line.replace('\x00', '').replace('"', '').strip()
                if not line:
                    continue

                if file_path.endswith('.csv'):
                    parts = line.split(';')
                    if len(parts) > 1:
                        # convert comma decimals to dot if purely numeric-ish
                        parts = [
                            p.replace(',', '.') if p.replace(',', '').replace('.', '').isdigit() else p
                            for p in parts
                        ]
                        line = ';'.join(parts)

                outfile.write(line + '\n')

        return temp_path
    except Exception as e:  # noqa: BLE001
        logging.error(f"File preprocessing error: {str(e)}")
        return file_path


def read_perek_file(file_path: str, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Read Perek file with optional row limit."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # XLSB
    if file_path.endswith('.xlsb'):
        try:
            with pyxlsb.open_workbook(file_path) as wb:
                # take first sheet
                first_sheet = wb.sheets[0] if wb.sheets else None
                if first_sheet is None:
                    logging.error(f"No sheets in XLSB: {file_path}")
                    return None
                with wb.get_sheet(first_sheet) as sheet:
                    data = []
                    for i, row in enumerate(sheet.rows()):
                        if max_rows is not None and i >= max_rows:
                            break
                        data.append([item.v for item in row])
                    if not data:
                        return None
                    # First row as header
                    cols = [str(c) if c is not None else f"col_{idx}" for idx, c in enumerate(data[0])]
                    df = pd.DataFrame(data[1:], columns=cols)
                    return df
        except Exception as e:  # noqa: BLE001
            logging.error(f"Failed to read XLSB file: {str(e)}")
            return None

    # Other formats
    encoding = detect_file_encoding(file_path)

    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(
                file_path,
                delimiter=';',
                decimal='.',
                thousands=' ',
                encoding=encoding,
                nrows=max_rows,
            )
        else:  # Excel
            return pd.read_excel(
                file_path,
                engine='openpyxl',
                nrows=max_rows,
            )
    except Exception as e:  # noqa: BLE001
        logging.warning(f"Standard read failed, trying fallbacks: {str(e)}")

    # CSV fallbacks
    if file_path.endswith('.csv'):
        strategies = [
            {'sep': ';', 'decimal': '.', 'thousands': ' ', 'encoding': encoding},
            {'engine': 'python', 'encoding': encoding},
            {'encoding': 'cp1251'},
            {'encoding': 'latin1'},
        ]
        for strategy in strategies:
            try:
                return pd.read_csv(file_path, **strategy)
            except Exception:  # noqa: BLE001
                continue

    processed_path = preprocess_perek_file(file_path)
    try:
        if processed_path.endswith('.csv'):
            return pd.read_csv(processed_path, delimiter=';', decimal='.', thousands=' ')
        else:
            return pd.read_excel(processed_path, engine='openpyxl')
    finally:
        if processed_path != file_path and os.path.exists(processed_path):
            os.remove(processed_path)

    return None


# ------------------------------------------------------------------ #
# Scan task                                                          #
# ------------------------------------------------------------------ #
@task(
    task_id="scan_perek_files",
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=PerekConfig.TASK_TIMEOUT,
    pool=PerekConfig.PROCESSING_POOL,
    pool_slots=1,
)
def scan_perek_files() -> List[str]:
    """Scan for Perekrestok files matching criteria."""
    try:
        valid_files = []
        total_size = 0

        logging.info(f"[Perek] Scanning directory: {PerekConfig.DATA_DIR}")
        if not os.path.exists(PerekConfig.DATA_DIR):
            raise FileNotFoundError(f"Directory not found: {PerekConfig.DATA_DIR}")

        # variants we accept in filename
        name_tokens = ("perek", "perekrestok", "перекр", "перекресток")

        for root, _, files in os.walk(PerekConfig.DATA_DIR):
            for f in files:
                ext_ok = os.path.splitext(f)[1].lower() in PerekConfig.ALLOWED_EXT
                name_ok = any(tok in f.lower() for tok in name_tokens)
                if not (ext_ok and name_ok):
                    continue

                file_path = os.path.join(root, f)
                file_size = os.path.getsize(file_path)

                if file_size == 0:
                    logging.warning(f"[Perek] Skipping empty file: {file_path}")
                    continue

                if total_size + file_size > PerekConfig.MAX_FILE_SIZE:
                    logging.warning(f"[Perek] File size limit exceeded. Skipping {file_path}")
                    continue

                valid_files.append(file_path)
                total_size += file_size

                if len(valid_files) >= PerekConfig.MAX_FILES_PER_RUN:
                    break

            if len(valid_files) >= PerekConfig.MAX_FILES_PER_RUN:
                break

        logging.info(
            f"[Perek] Found {len(valid_files)} Perek files "
            f"(total size: {total_size / 1024 / 1024:.2f} MB)"
        )
        return valid_files[:PerekConfig.MAX_FILES_PER_RUN]

    except Exception as e:  # noqa: BLE001
        logging.error(f"[Perek] File scanning error: {str(e)}", exc_info=True)
        raise


# ------------------------------------------------------------------ #
# Process single file                                                #
# ------------------------------------------------------------------ #
@task(
    task_id="process_perek_file",
    retries=3,
    retry_delay=timedelta(minutes=5),
    execution_timeout=PerekConfig.TASK_TIMEOUT,
    pool=PerekConfig.PROCESSING_POOL,
    pool_slots=1,
)
def process_perek_file(file_path: str):
    """Process a single Perekrestok file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Empty?
        if os.path.getsize(file_path) == 0:
            logging.warning(f"[Perek] Skipping empty file: {file_path}")
            archive_perek_file(file_path, empty=True)
            return

        df = read_perek_file(file_path, max_rows=10000)
        if df is None or df.empty:
            logging.error(f"[Perek] Failed to read data from file: {file_path}")
            archive_perek_file(file_path, error=True)
            return

        # --- DB raw load
        engine_test = get_engine('test')
        try:
            from perekrestok.create_table_and_upload import create_perek_table_and_upload
            table_name = create_perek_table_and_upload(file_path, engine=engine_test)
            logging.info(f"[Perek] Data loaded to raw.{table_name}")
        except exc.SQLAlchemyError as e:
            engine_test.dispose()
            logging.error(f"[Perek] Raw load error: {str(e)}")
            archive_perek_file(file_path, error=True)
            raise RuntimeError(f"Raw load error: {str(e)}")

        # --- Stage load
        engine_stage = get_engine('stage')
        try:
            from perekrestok.convert_raw_to_stage import convert_raw_to_stage
            convert_raw_to_stage(
                table_name=table_name,
                raw_engine=engine_test,
                stage_engine=engine_stage,
                stage_schema=PerekConfig.STAGE_SCHEMA,
                limit=10000,
            )
            logging.info("[Perek] Data loaded to stage.")
        except exc.SQLAlchemyError as e:
            engine_stage.dispose()
            logging.error(f"[Perek] Stage load error: {str(e)}")
            archive_perek_file(file_path, error=True)
            raise RuntimeError(f"Stage load error: {str(e)}")
        finally:
            engine_test.dispose()
            engine_stage.dispose()

        # Archive successfully processed file
        archive_perek_file(file_path)

    except Exception as e:  # noqa: BLE001
        logging.error(f"[Perek] File processing error: {file_path} - {str(e)}", exc_info=True)
        archive_perek_file(file_path, error=True)
        raise


# ------------------------------------------------------------------ #
# Archive                                                            #
# ------------------------------------------------------------------ #
def archive_perek_file(file_path: str, empty: bool = False, error: bool = False):
    """Archive file (preserve original name/extension)."""
    os.makedirs(PerekConfig.ARCHIVE_DIR, exist_ok=True)

    archive_name = os.path.basename(file_path)
    archive_path = os.path.join(PerekConfig.ARCHIVE_DIR, archive_name)

    try:
        shutil.move(file_path, archive_path)
        logging.info(f"[Perek] File archived: {archive_path}")
    except Exception as e:  # noqa: BLE001
        logging.error(f"[Perek] Failed to archive file {file_path}: {str(e)}")


# ------------------------------------------------------------------ #
# DAG                                                                #
# ------------------------------------------------------------------ #
default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'depends_on_past': False,
    'max_active_tis_per_dag': PerekConfig.MAX_CONCURRENT_TASKS,
}

with DAG(
    dag_id="perek_sales_pipeline_optimized",
    default_args=default_args,
    start_date=datetime(2025, 7, 1),
    schedule_interval=None,
    catchup=False,
    max_active_tasks=PerekConfig.MAX_CONCURRENT_TASKS,
    concurrency=PerekConfig.MAX_CONCURRENT_TASKS,
    tags=["perek", "perekrestok", "optimized"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    scanned_files = scan_perek_files()

    with TaskGroup("perek_file_processing_group") as processing_group:
        process_perek_file.expand(file_path=scanned_files)

    start >> scanned_files >> processing_group >> end
