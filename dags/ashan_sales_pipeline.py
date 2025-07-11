from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from datetime import datetime
import os
import shutil
import logging
from sqlalchemy import create_engine

from common.utils import get_files_from_directory
from common.convert_xlsx_to_csv import convert_excel_to_csv
from ashan.create_table_and_upload import create_table_and_upload
from ashan.convert_raw_to_stage import convert_raw_to_stage  # <- Обрати внимание на этот импорт

logger = logging.getLogger(__name__)

DATA_DIR = "/opt/airflow/data"
ARCHIVE_DIR = "/opt/airflow/archive"

DEFAULT_CONN_TEST = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Test"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)
DEFAULT_CONN_STAGE = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)

def get_engine_test():
    try:
        conn_str = Variable.get("MSSQL_CONN_STR_TEST")
    except KeyError:
        logger.warning("Using default connection string for TEST")
        conn_str = DEFAULT_CONN_TEST
    return create_engine(conn_str)

def get_engine_stage():
    try:
        conn_str = Variable.get("MSSQL_CONN_STR_STAGE")
    except KeyError:
        logger.warning("Using default connection string for STAGE")
        conn_str = DEFAULT_CONN_STAGE
    return create_engine(conn_str)

@task
def scan_files():
    allowed_ext = ['.csv', '.xlsx', '.xls', '.xlsb']
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8188243 (Приведение столбцов к общему виду для X5, правки к ДАГу Ашана)
    ashan_files = []
    
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().startswith("aushan_") and os.path.splitext(f)[1].lower() in allowed_ext:
                full_path = os.path.join(root, f)
                ashan_files.append(full_path)
    
    logger.info(f"Ашан файлы для обработки: {ashan_files}")
    return ashan_files
<<<<<<< HEAD
=======
    filtered = [
        f for f in files
        if os.path.basename(f).lower().startswith("aushan")
        and os.path.splitext(f)[1].lower() in allowed_ext
    ]
    logger.info(f"Ашан-файлы для обработки: {filtered}")
    return filtered
>>>>>>> 66c7e5c (Дополнил данные и пометил дальнейший план)
=======
>>>>>>> 8188243 (Приведение столбцов к общему виду для X5, правки к ДАГу Ашана)

@task
def process_file(file_path: str):
    try:
        logger.info(f"Начинаем обработку файла Ашан: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.xlsx', '.xls', '.xlsb']:
            file_path = convert_excel_to_csv(file_path)

        engine_test = get_engine_test()
        table_name = create_table_and_upload(file_path, engine=engine_test)

        engine_stage = get_engine_stage()
        convert_raw_to_stage(
            table_name=table_name,
            raw_engine=engine_test,
            stage_engine=engine_stage,
            stage_schema="ashan"  # <- Название схемы
        )

        if not os.path.exists(ARCHIVE_DIR):
            os.makedirs(ARCHIVE_DIR)
        shutil.move(file_path, os.path.join(ARCHIVE_DIR, os.path.basename(file_path)))

        logger.info(f"Файл Ашан успешно обработан и перемещён в архив: {file_path}")

    except Exception as e:
        logger.error(f"Ошибка при обработке файла Ашан {file_path}: {e}", exc_info=True)
        raise

with DAG(
    dag_id="ashan_sales_pipeline",
    start_date=datetime(2025, 7, 11),
    schedule_interval="@daily",
    catchup=False,
    tags=["ashan", "sales"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    files = scan_files()
    processed = process_file.expand(file_path=files)

    start >> files >> processed >> end
