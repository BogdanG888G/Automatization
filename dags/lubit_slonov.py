from airflow import DAG
from airflow.operators import PythonOperator
from datetime import datetime

with DAG(dag_id = 'lol', start_date = '24.09.2025', schedule_interval = '@daily', )