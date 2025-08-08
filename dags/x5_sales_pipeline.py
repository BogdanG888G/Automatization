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
import pyxlsb


# ----------------------------------- #
# DB connection strings for X5         #
# ----------------------------------- #
DEFAULT_CONN_TEST = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Test"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)
DEFAULT_CONN_STAGE = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)


# ----------------------------------- #
# Config                              #
# ----------------------------------- #
class X5Config:
    MAX_CONCURRENT_TASKS = 2
    MAX_FILES_PER_RUN = 6
    TASK_TIMEOUT = timedelta(minutes=1200)
    MAX_FILE_SIZE = 4 * 1024 * 1024 * 1024  # 3GB
    POOL_SLOTS = 4  # Сбалансировать с pool_size БД

    DATA_DIR = "/opt/airflow/data"
    ARCHIVE_DIR = "/opt/airflow/archive"
    ALLOWED_EXT = {'.csv', '.xlsx', '.xls', '.xlsb'}
    FILE_PREFIXES = ("x5", "x5retail")  # Можно добавить вариации
    STAGE_SCHEMA = "x5"  # Имя схемы для X5

    @staticmethod
    def get_processing_pool():
        try:
            pools = Variable.get("airflow_pools", default_var="default_pool")
            return "x5_file_pool" if "x5_file_pool" in pools.split(",") else "default_pool"
        except Exception:
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
            'application_name': 'airflow_x5_loader'
        }
    }


# ----------------------------------- #
# Engine cache                       #
# ----------------------------------- #
_engine_cache = {}
_last_cache_update = datetime.min


def get_engine(db_type: str):
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

            _engine_cache[db_type] = create_engine(conn_str, **X5Config.CONN_SETTINGS)
        except Exception as e:
            logging.error(f"Error creating engine for {db_type}: {str(e)}")
            raise

    return _engine_cache[db_type]


# ----------------------------------- #
# File helpers                       #
# ----------------------------------- #
def detect_file_encoding(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'


def preprocess_x5_file(file_path: str) -> str:
    if file_path.endswith('.xlsb'):
        return file_path

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
                        parts = [
                            p.replace(',', '.') if p.replace(',', '').replace('.', '').isdigit() else p
                            for p in parts
                        ]
                        line = ';'.join(parts)

                outfile.write(line + '\n')

        return temp_path
    except Exception as e:
        logging.error(f"File preprocessing error: {str(e)}")
        return file_path

HEADER_CANDIDATES_X5 = (
    "сеть", "филиал", "регион", "город", "адрес",
    "тов.иер", "материал", "количество", "оборот", "средняя", "поставщик"
)

def _detect_header_row_x5(file_path: str, encoding: str = "utf-8", max_scan: int = 10) -> int:
    """
    Возвращает индекс строки (0-based), которую стоит использовать как header при чтении CSV X5.
    Если ничего подходящего не нашли — вернём 0.
    """
    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            for idx in range(max_scan):
                line = f.readline()
                if not line:  # EOF
                    break
                low = line.lower()
                hit_count = sum(tok in low for tok in HEADER_CANDIDATES_X5)
                if hit_count >= 2:
                    return idx
    except Exception:
        pass
    return 0



def read_x5_file(file_path: str, max_rows: Optional[int] = None, skip_first_row: bool = False) -> Optional[pd.DataFrame]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    skiprows = 1 if skip_first_row else 0

    if file_path.endswith('.xlsb'):
        try:
            with pyxlsb.open_workbook(file_path) as wb:
                first_sheet = wb.sheets[0] if wb.sheets else None
                if first_sheet is None:
                    logging.error(f"No sheets in XLSB: {file_path}")
                    return None
                with wb.get_sheet(first_sheet) as sheet:
                    data = []
                    for i, row in enumerate(sheet.rows()):
                        if max_rows is not None and i >= max_rows + skiprows:
                            break
                        data.append([item.v for item in row])
                    if not data:
                        return None
                    # Пропускаем первую строку, если надо
                    data = data[skiprows:]
                    if not data:
                        return None
                    cols = [str(c) if c is not None else f"col_{idx}" for idx, c in enumerate(data[0])]
                    df = pd.DataFrame(data[1:], columns=cols)
                    return df
        except Exception as e:
            logging.error(f"Failed to read XLSB file: {str(e)}")
            return None

    encoding = detect_file_encoding(file_path)

    try:
        if file_path.endswith('.csv'):
            hdr_idx = _detect_header_row_x5(file_path, encoding=encoding)
            df = pd.read_csv(
                file_path,
                delimiter=';',
                decimal='.',
                thousands=' ',
                encoding=encoding,
                header=hdr_idx,
                nrows=max_rows,
            )
            return df
        else:
            return pd.read_excel(
                file_path,
                engine='openpyxl',
                nrows=max_rows,
                header=0,   # Excel обычно чистый
            )
    except Exception as e:
        logging.warning(f"Standard read failed, trying fallbacks: {str(e)}")

    # --- CSV fallbacks ---
    if file_path.endswith('.csv'):
        hdr_idx = _detect_header_row_x5(file_path, encoding=encoding)
        strategies = [
            {'sep': ';', 'decimal': '.', 'thousands': ' ', 'encoding': encoding, 'header': hdr_idx},
            {'engine': 'python', 'encoding': encoding, 'header': hdr_idx},
            {'encoding': 'cp1251', 'header': hdr_idx},
            {'encoding': 'latin1', 'header': hdr_idx},
        ]
        for strategy in strategies:
            try:
                return pd.read_csv(file_path, **strategy)
            except Exception:
                continue

    processed_path = preprocess_x5_file(file_path)
    try:
        if processed_path.endswith('.csv'):
            hdr_idx = _detect_header_row_x5(processed_path, encoding='utf-8')
            return pd.read_csv(processed_path, delimiter=';', decimal='.', thousands=' ', header=hdr_idx)
        else:
            return pd.read_excel(processed_path, engine='openpyxl', header=0)
    finally:
        if processed_path != file_path and os.path.exists(processed_path):
            os.remove(processed_path)

    return None



# ----------------------------------- #
# Scan task                         #
# ----------------------------------- #
@task(
    task_id="scan_x5_files",
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=X5Config.TASK_TIMEOUT,
    pool=X5Config.PROCESSING_POOL,
    pool_slots=1,
)
def scan_x5_files() -> List[str]:
    try:
        valid_files = []
        total_size = 0

        logging.info(f"[X5] Scanning directory: {X5Config.DATA_DIR}")
        if not os.path.exists(X5Config.DATA_DIR):
            raise FileNotFoundError(f"Directory not found: {X5Config.DATA_DIR}")

        for root, _, files in os.walk(X5Config.DATA_DIR):
            for f in files:
                ext_ok = os.path.splitext(f)[1].lower() in X5Config.ALLOWED_EXT
                name_ok = any(tok in f.lower() for tok in X5Config.FILE_PREFIXES)
                if not (ext_ok and name_ok):
                    continue

                file_path = os.path.join(root, f)
                file_size = os.path.getsize(file_path)

                if file_size == 0:
                    logging.warning(f"[X5] Skipping empty file: {file_path}")
                    continue

                if total_size + file_size > X5Config.MAX_FILE_SIZE:
                    logging.warning(f"[X5] File size limit exceeded. Skipping {file_path}")
                    continue

                valid_files.append(file_path)
                total_size += file_size

                if len(valid_files) >= X5Config.MAX_FILES_PER_RUN:
                    break

            if len(valid_files) >= X5Config.MAX_FILES_PER_RUN:
                break

        logging.info(f"[X5] Found {len(valid_files)} X5 files (total size: {total_size / 1024 / 1024:.2f} MB)")
        return valid_files[:X5Config.MAX_FILES_PER_RUN]

    except Exception as e:
        logging.error(f"[X5] File scanning error: {str(e)}", exc_info=True)
        raise


# ----------------------------------- #
# Process single file                 #
# ----------------------------------- #
@task(
    task_id="process_x5_file",
    retries=3,
    retry_delay=timedelta(minutes=5),
    execution_timeout=timedelta(minutes=120),  # Увеличить таймаут для больших файлов
    pool=X5Config.PROCESSING_POOL,
    pool_slots=2,  # Увеличить слоты для ресурсоемких задач
)
def process_x5_file(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        logging.info(f"Processing file: {file_path} ({file_size/1024/1024:.2f} MB)")

        if file_size == 0:
            logging.warning(f"[X5] Skipping empty file: {file_path}")
            archive_x5_file(file_path, empty=True)
            return

        # Прямая передача файла без чтения в память
        engine_test = get_engine('test')
        try:
            from x5.create_table_and_upload import create_x5_table_and_upload
            table_name = create_x5_table_and_upload(file_path, engine=engine_test)
            logging.info(f"[X5] Data loaded to raw.{table_name}")
        except exc.SQLAlchemyError as e:
            logging.error(f"[X5] Raw load error: {str(e)}")
            archive_x5_file(file_path)
            raise RuntimeError(f"Raw load error: {str(e)}")

        # Для больших файлов - отложенная обработка в Stage
        if file_size > 100 * 1024 * 1024:  # >100MB
            logging.info("Scheduling async stage conversion for large file")
            from x5.convert_raw_to_stage import schedule_stage_conversion
            schedule_stage_conversion(
                table_name=table_name,
                raw_engine=engine_test,
                stage_engine=get_engine('stage'),
                stage_schema=X5Config.STAGE_SCHEMA
            )
        else:
            logging.info("Immediate stage conversion")
            from x5.convert_raw_to_stage import convert_raw_to_stage
            convert_raw_to_stage(
                table_name=table_name,
                raw_engine=engine_test,
                stage_engine=get_engine('stage'),
                stage_schema=X5Config.STAGE_SCHEMA
            )

        archive_x5_file(file_path)

    except Exception as e:
        logging.error(f"[X5] Error processing file {file_path}: {str(e)}", exc_info=True)
        archive_x5_file(file_path)

def archive_x5_file(file_path: str):
    try:
        base_name = os.path.basename(file_path)
        archive_path = X5Config.ARCHIVE_DIR  # просто одна папка архива
        os.makedirs(archive_path, exist_ok=True)
        dest_path = os.path.join(archive_path, base_name)
        shutil.move(file_path, dest_path)
        logging.info(f"[X5] Archived file {file_path} to {dest_path}")
    except Exception as e:
        logging.error(f"[X5] Archiving failed for {file_path}: {str(e)}")

# ----------------------------------- #
# DAG definition                    #
# ----------------------------------- #
with DAG(
    dag_id="x5_sales_data_pipeline",
    start_date=datetime(2025, 7, 15),
    schedule_interval="0 8 * * *",  # каждый день в 15:00
    catchup=False,
    max_active_runs=1,
    tags=["x5", "sales", "data"],
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    files = scan_x5_files()

    with TaskGroup("process_x5_files") as process_group:
        process_tasks = process_x5_file.expand(file_path=files)

    start >> files >> process_group >> end
