from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import os
import shutil
import logging
from sqlalchemy import create_engine, exc
from typing import List

# Optimized logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configuration constants
class Config:
    DATA_DIR = "/opt/airflow/data"
    ARCHIVE_DIR = "/opt/airflow/archive"
    ALLOWED_EXT = {'.csv', '.xlsx', '.xls', '.xlsb'}
    FILE_PREFIX = "x5_"
    STAGE_SCHEMA = "x5"
    
    # Connection pool settings
    CONN_SETTINGS = {
        'pool_size': 5,
        'max_overflow': 2,
        'pool_timeout': 30,
        'pool_recycle': 3600
    }

    DEFAULT_CONN_TEST = (
        "mssql+pyodbc://airflow_agent:123@host.docker.internal/Test"
        "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
    )
    DEFAULT_CONN_STAGE = (
        "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
        "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
    )

# Engine cache for better performance
_engine_cache = {}

def get_engine(db_type: str):
    """Optimized engine creation with caching."""
    if db_type in _engine_cache:
        return _engine_cache[db_type]
    
    try:
        conn_str = Variable.get(f"MSSQL_CONN_STR_{db_type.upper()}",
                              default_var=getattr(Config, f"DEFAULT_CONN_{db_type.upper()}"))
        
        engine = create_engine(conn_str, **Config.CONN_SETTINGS)
        _engine_cache[db_type] = engine
        return engine
    except Exception as e:
        logger.error(f"Engine creation error for {db_type}: {str(e)}")
        raise

@task(
    task_id="scan_files",
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=timedelta(minutes=5)
)
def scan_files() -> List[str]:
    """Optimized file scanning with error handling."""
    try:
        x5_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(Config.DATA_DIR)
            for f in files
            if f.lower().startswith(Config.FILE_PREFIX) 
            and os.path.splitext(f)[1].lower() in Config.ALLOWED_EXT
        ]
        
        logger.info(f"Found {len(x5_files)} X5 files to process")
        return x5_files
        
    except Exception as e:
        logger.error(f"File scanning error: {str(e)}")
        raise

@task(
    task_id="process_file",
    retries=3,
    retry_delay=timedelta(minutes=5),
    execution_timeout=timedelta(hours=1),
    pool="file_processing_pool"
)
def process_file(file_path: str):
    """Optimized file processing with resource management."""
    try:
        logger.info(f"Starting processing: {file_path}")
        
        # 1. Convert to CSV if needed
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {'.xlsx', '.xls', '.xlsb'}:
            from common.convert_xlsx_to_csv import convert_excel_to_csv
            file_path = convert_excel_to_csv(file_path)
            logger.info(f"Converted to CSV: {file_path}")

        # 2. Load to raw
        try:
            from x5.create_table_and_upload import create_table_and_upload
            engine_test = get_engine('test')
            table_name = create_table_and_upload(file_path, engine=engine_test)
            logger.info(f"Data loaded to raw.{table_name}")
        except exc.SQLAlchemyError as db_error:
            logger.error(f"DB error loading to raw: {str(db_error)}")
            raise

        # 3. Convert to stage
        try:
            from x5.convert_raw_to_stage import convert_raw_to_stage
            engine_stage = get_engine('stage')
            convert_raw_to_stage(
                table_name=table_name,
                raw_engine=engine_test,
                stage_engine=engine_stage,
                stage_schema=Config.STAGE_SCHEMA
            )
            logger.info(f"Data loaded to {Config.STAGE_SCHEMA}.{table_name}")
        except exc.SQLAlchemyError as db_error:
            logger.error(f"DB error loading to stage: {str(db_error)}")
            raise

        # 4. Archive with checks
        try:
            os.makedirs(Config.ARCHIVE_DIR, exist_ok=True)
            shutil.move(file_path, os.path.join(Config.ARCHIVE_DIR, os.path.basename(file_path)))
            logger.info(f"File archived: {file_path}")
        except OSError as file_error:
            logger.error(f"Archiving error: {str(file_error)}")
            raise

    except Exception as e:
        logger.error(f"Critical error processing {file_path}: {str(e)}", exc_info=True)
        raise

# DAG configuration with optimized parameters
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
    'max_active_tis_per_dag': 4
}

with DAG(
    dag_id="x5_sales_pipeline",
    default_args=default_args,
    description="ETL pipeline for X5 sales data",
    start_date=datetime(2025, 7, 7),
    schedule_interval=None,
    catchup=False,
    tags=["x5", "sales", "retail"],
    doc_md=__doc__,
    max_active_tasks=4,
    concurrency=6,
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    # Optimized workflow
    files = scan_files()
    processed = process_file.expand(file_path=files)
    
    start >> files >> processed >> end