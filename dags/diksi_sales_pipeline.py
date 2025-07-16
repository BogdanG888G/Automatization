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

# Database connection strings
DEFAULT_CONN_TEST = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Test"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)
DEFAULT_CONN_STAGE = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)

class DiksiConfig:
    MAX_CONCURRENT_TASKS = 4
    MAX_FILES_PER_RUN = 8
    TASK_TIMEOUT = timedelta(minutes=45)
    MAX_FILE_SIZE = 3 * 1024 * 1024 * 1024  # 3GB

    DATA_DIR = "/opt/airflow/data"
    ARCHIVE_DIR = "/opt/airflow/archive"
    ALLOWED_EXT = {'.csv', '.xlsx', '.xls', '.xlsb'}  # Added .xlsb
    FILE_PREFIX = "diksi_"
    STAGE_SCHEMA = "diksi"

    @staticmethod
    def get_processing_pool():
        try:
            pools = Variable.get("airflow_pools", default_var="default_pool")
            return "diksi_file_pool" if "diksi_file_pool" in pools.split(",") else "default_pool"
        except:
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
            'application_name': 'airflow_diksi_loader'
        }
    }

_engine_cache = {}
_last_cache_update = datetime.min

def get_engine(db_type: str):
    """Get SQLAlchemy engine with caching"""
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
        except Exception as e:
            logging.error(f"Error creating engine for {db_type}: {str(e)}")
            raise

    return _engine_cache[db_type]

def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding using chardet"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'

def preprocess_diksi_file(file_path: str) -> str:
    """Pre-process Diksi files that might have parsing issues"""
    if file_path.endswith('.xlsb'):
        return file_path  # Skip preprocessing for binary files
        
    temp_path = file_path + ".processed"
    encoding = detect_file_encoding(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding) as infile, \
             open(temp_path, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile):
                line = line.replace('\x00', '').replace('"', '').strip()
                if not line:
                    continue
                    
                if file_path.endswith('.csv'):
                    parts = line.split(';')
                    if len(parts) > 1:
                        parts = [p.replace(',', '.') if p.replace(',', '').replace('.', '').isdigit() else p 
                               for p in parts]
                        line = ';'.join(parts)
                
                outfile.write(line + '\n')
                
        return temp_path
    except Exception as e:
        logging.error(f"File preprocessing error: {str(e)}")
        return file_path

def read_diksi_file(file_path: str, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Read Diksi file with optional row limit"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Special handling for XLSB files
    if file_path.endswith('.xlsb'):
        try:
            with pyxlsb.open_workbook(file_path) as wb:
                with wb.get_sheet(1) as sheet:
                    data = []
                    for i, row in enumerate(sheet.rows()):
                        if max_rows is not None and i >= max_rows:
                            break
                        data.append([item.v for item in row])
                    return pd.DataFrame(data)
        except Exception as e:
            logging.error(f"Failed to read XLSB file: {str(e)}")
            return None
    
    # For other file types
    encoding = detect_file_encoding(file_path)
    
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(
                file_path, 
                delimiter=';', 
                decimal='.', 
                thousands=' ', 
                encoding=encoding,
                nrows=max_rows  # Ограничение строк для CSV
            )
        else:  # Excel
            return pd.read_excel(
                file_path, 
                engine='openpyxl',
                nrows=max_rows  # Ограничение строк для Excel
            )
    except Exception as e:
        logging.warning(f"Standard read failed, trying fallbacks: {str(e)}")
    
    if file_path.endswith('.csv'):
        strategies = [
            {'sep': ';', 'decimal': '.', 'thousands': ' ', 'encoding': encoding},
            {'engine': 'python', 'encoding': encoding},
            {'error_bad_lines': False, 'warn_bad_lines': True},
            {'encoding': 'cp1251'},
            {'encoding': 'latin1'}
        ]
        
        for strategy in strategies:
            try:
                return pd.read_csv(file_path, **strategy)
            except:
                continue
    
    processed_path = preprocess_diksi_file(file_path)
    try:
        if processed_path.endswith('.csv'):
            return pd.read_csv(processed_path, delimiter=';', decimal='.', thousands=' ')
        else:
            return pd.read_excel(processed_path, engine='openpyxl')
    finally:
        if processed_path != file_path and os.path.exists(processed_path):
            os.remove(processed_path)
    
    return None

@task(
    task_id="scan_diksi_files",
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=DiksiConfig.TASK_TIMEOUT,
    pool=DiksiConfig.PROCESSING_POOL,
    pool_slots=1
)
def scan_diksi_files() -> List[str]:
    """Scan for Diksi files matching criteria"""
    try:
        valid_files = []
        total_size = 0

        logging.info(f"Scanning directory: {DiksiConfig.DATA_DIR}")
        if not os.path.exists(DiksiConfig.DATA_DIR):
            raise FileNotFoundError(f"Directory not found: {DiksiConfig.DATA_DIR}")

        for root, _, files in os.walk(DiksiConfig.DATA_DIR):
            for f in files:
                if ('diksi' in f.lower() and 
                    os.path.splitext(f)[1].lower() in DiksiConfig.ALLOWED_EXT):
                    
                    file_path = os.path.join(root, f)
                    file_size = os.path.getsize(file_path)
                    
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

        logging.info(f"Found {len(valid_files)} Diski files (total size: {total_size / 1024 / 1024:.2f} MB)")
        return valid_files[:DiksiConfig.MAX_FILES_PER_RUN]

    except Exception as e:
        logging.error(f"File scanning error: {str(e)}", exc_info=True)
        raise

@task(
    task_id="process_diksi_file",
    retries=3,
    retry_delay=timedelta(minutes=5),
    execution_timeout=DiksiConfig.TASK_TIMEOUT,
    pool=DiksiConfig.PROCESSING_POOL,
    pool_slots=1
)
def process_diksi_file(file_path: str):
    """Process a single Diski file"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            logging.warning(f"Skipping empty file: {file_path}")
            # Move empty file to archive with _EMPTY suffix
            archive_empty_file(file_path, empty=True)
            return

        df = read_diksi_file(file_path, max_rows=10000)
        if df is None or df.empty:
            logging.error(f"Failed to read data from file: {file_path}")
            # Move problematic file to archive with _ERROR suffix
            archive_empty_file(file_path, error=True)
            return

        # Database operations
        engine_test = get_engine('test')
        try:
            from diksi.create_table_and_upload import create_table_and_upload_diksi
            table_name = create_table_and_upload_diksi(file_path, engine=engine_test)
            logging.info(f"Diski data loaded to raw.{table_name}")
        except exc.SQLAlchemyError as e:
            engine_test.dispose()
            logging.error(f"Raw load error: {str(e)}")
            archive_empty_file(file_path, error=True)
            raise RuntimeError(f"Raw load error: {str(e)}")

        engine_stage = get_engine('stage')
        try:
            from diksi.convert_raw_to_stage import convert_raw_to_stage
            convert_raw_to_stage(
                table_name=table_name,
                raw_engine=engine_test,
                stage_engine=engine_stage,
                stage_schema=DiksiConfig.STAGE_SCHEMA,
                limit = 10000
            )
            logging.info("Diksi data loaded to stage")
        except exc.SQLAlchemyError as e:
            engine_stage.dispose()
            logging.error(f"Stage load error: {str(e)}")
            archive_empty_file(file_path, error=True)
            raise RuntimeError(f"Stage load error: {str(e)}")
        finally:
            engine_test.dispose()
            engine_stage.dispose()

        # Archive successfully processed file
        archive_empty_file(file_path)

    except Exception as e:
        logging.error(f"Diksi file processing error: {file_path} - {str(e)}", exc_info=True)
        archive_empty_file(file_path, error=True)
        raise

def archive_empty_file(file_path: str, empty: bool = False, error: bool = False):
    """Archive file with appropriate suffix (keeping original name and extension)"""
    os.makedirs(DiksiConfig.ARCHIVE_DIR, exist_ok=True)
    
    # Оставляем оригинальное имя файла
    archive_name = os.path.basename(file_path)
    archive_path = os.path.join(DiksiConfig.ARCHIVE_DIR, archive_name)

    try:
        shutil.move(file_path, archive_path)
        logging.info(f"File archived: {archive_path}")
    except Exception as e:
        logging.error(f"Failed to archive file {file_path}: {str(e)}")

        
default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'depends_on_past': False,
    'max_active_tis_per_dag': DiksiConfig.MAX_CONCURRENT_TASKS,
}

with DAG(
    dag_id="diksi_sales_pipeline_optimized",
    default_args=default_args,
    start_date=datetime(2025, 7, 1),
    schedule_interval=None,
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