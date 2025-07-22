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

class MagnitConfig:
    MAX_CONCURRENT_TASKS = 2
    MAX_FILES_PER_RUN = 4
    TASK_TIMEOUT = timedelta(minutes=120)
    MAX_FILE_SIZE = 3 * 1024 * 1024 * 1024  # 3GB
    POOL_SLOTS = 8  # Увеличить с 6

    DATA_DIR = "/opt/airflow/data"
    ARCHIVE_DIR = "/opt/airflow/archive"
    ALLOWED_EXT = {'.csv', '.xlsx', '.xls', '.xlsb'}  # Added .xlsb
    FILE_PREFIX = "magnit_"
    STAGE_SCHEMA = "magnit"

    @staticmethod
    def get_processing_pool():
        try:
            pools = Variable.get("airflow_pools", default_var="default_pool")
            return "magnit_file_pool" if "magnit_file_pool" in pools.split(",") else "default_pool"
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
            'application_name': 'airflow_magnit_loader'
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

            _engine_cache[db_type] = create_engine(conn_str, **MagnitConfig.CONN_SETTINGS)
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

def preprocess_magnit_file(file_path: str) -> str:
    """Pre-process Magnit files that might have parsing issues"""
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



def read_magnit_file(file_path: str, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Read Magnit file with optional row limit"""
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
                nrows=max_rows,  # Ограничение строк для Excel,
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
                return pd.read_csv(file_path, chunksize=50000)
            except:
                continue
    
    processed_path = preprocess_magnit_file(file_path)
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
    task_id="scan_magnit_files",
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=MagnitConfig.TASK_TIMEOUT,
    pool=MagnitConfig.PROCESSING_POOL,
    pool_slots=1
)
def scan_magnit_files() -> List[str]:
    """Scan for Magnit files matching criteria"""
    try:
        valid_files = []
        total_size = 0

        logging.info(f"Scanning directory: {MagnitConfig.DATA_DIR}")
        if not os.path.exists(MagnitConfig.DATA_DIR):
            raise FileNotFoundError(f"Directory not found: {MagnitConfig.DATA_DIR}")

        for root, _, files in os.walk(MagnitConfig.DATA_DIR):
            for f in files:
                if ('magnit' in f.lower() and 
                    os.path.splitext(f)[1].lower() in MagnitConfig.ALLOWED_EXT):
                    
                    file_path = os.path.join(root, f)
                    file_size = os.path.getsize(file_path)
                    
                    if file_size == 0:
                        logging.warning(f"Skipping empty file: {file_path}")
                        continue

                    if total_size + file_size > MagnitConfig.MAX_FILE_SIZE:
                        logging.warning(f"File size limit exceeded. Skipping {file_path}")
                        continue

                    valid_files.append(file_path)
                    total_size += file_size

                    if len(valid_files) >= MagnitConfig.MAX_FILES_PER_RUN:
                        break

            if len(valid_files) >= MagnitConfig.MAX_FILES_PER_RUN:
                break

        logging.info(f"Found {len(valid_files)} Magnit files (total size: {total_size / 1024 / 1024:.2f} MB)")
        return valid_files[:MagnitConfig.MAX_FILES_PER_RUN]

    except Exception as e:
        logging.error(f"File scanning error: {str(e)}", exc_info=True)
        raise

@task(
    task_id="process_magnit_file",
    retries=3,
    retry_delay=timedelta(minutes=5),
    execution_timeout=MagnitConfig.TASK_TIMEOUT,
    pool=MagnitConfig.PROCESSING_POOL,
    pool_slots=1
)
def process_magnit_file(file_path: str):
    """Process a single Magnit file"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            logging.warning(f"Skipping empty file: {file_path}")
            # Move empty file to archive with _EMPTY suffix
            archive_empty_file(file_path, empty=True)
            return

        df = read_magnit_file(file_path)
        if df is None or df.empty:
            logging.error(f"Failed to read data from file: {file_path}")
            # Move problematic file to archive with _ERROR suffix
            archive_empty_file(file_path, error=True)
            return

        # Database operations
        engine_test = get_engine('test')
        try:
            from magnit.create_table_and_upload import create_magnit_table_and_upload
            table_name = create_magnit_table_and_upload(file_path, engine=engine_test)
            logging.info(f"Magnit data loaded to raw.{table_name}")
        except exc.SQLAlchemyError as e:
            engine_test.dispose()
            logging.error(f"Raw load error: {str(e)}")
            archive_empty_file(file_path, error=True)
            raise RuntimeError(f"Raw load error: {str(e)}")

        engine_stage = get_engine('stage')
        try:
            from magnit.convert_raw_to_stage import convert_raw_to_stage
            convert_raw_to_stage(
                table_name=table_name,
                raw_engine=engine_test,
                stage_engine=engine_stage,
                stage_schema=MagnitConfig.STAGE_SCHEMA
            )
            logging.info("Magnit data loaded to stage")
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
        logging.error(f"Magnit file processing error: {file_path} - {str(e)}", exc_info=True)
        archive_empty_file(file_path, error=True)
        raise

def archive_empty_file(file_path: str, empty: bool = False, error: bool = False):
    """Archive file without changing its name or extension"""
    os.makedirs(MagnitConfig.ARCHIVE_DIR, exist_ok=True)
    
    file_name = os.path.basename(file_path)  # Имя файла с расширением
    archive_path = os.path.join(MagnitConfig.ARCHIVE_DIR, file_name)
    
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
    'max_active_tis_per_dag': MagnitConfig.MAX_CONCURRENT_TASKS,
}

with DAG(
    dag_id="magnit_sales_pipeline_optimized",
    default_args=default_args,
    start_date=datetime(2025, 7, 1),
    schedule_interval=None,
    catchup=False,
    max_active_tasks=MagnitConfig.MAX_CONCURRENT_TASKS,
    concurrency=MagnitConfig.MAX_CONCURRENT_TASKS,
    tags=["magnit", "optimized"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    scanned_files = scan_magnit_files()

    with TaskGroup("magnit_file_processing_group") as processing_group:
        process_magnit_file.expand(file_path=scanned_files)

    start >> scanned_files >> processing_group >> end