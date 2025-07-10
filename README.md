#TODO подправить readme.md!

# Автоматизация загрузки данных в SQL Server с помощью Apache Airflow

![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=Apache%20Airflow&logoColor=white)

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

![SQL Server](https://img.shields.io/badge/Microsoft%20SQL%20Server-CC2927?style=for-the-badge&logo=microsoft%20sql%20server&logoColor=white)

Этот проект автоматизирует процесс загрузки данных из CSV/XLSX файлов в Microsoft SQL Server с использованием Apache Airflow. Система включает в себя:

- Автоматическое обнаружение новых файлов
- Конвертацию XLSX в CSV
- Создание таблиц в базе данных
- Загрузку данных в SQL Server
- Вызов хранимых процедур для обработки данных
- Архивирование обработанных файлов

## 📦 Требования

- Docker Desktop
- Docker Compose
- Microsoft SQL Server

## 🚀 Быстрый старт

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/your-username/airflow-data-pipeline.git
   cd airflow-data-pipeline
   ```

2. **Настройте окружение:**
   - Создайте `.env` файл (пример в `.env.example`)
   - Настройте подключение к MSSQL в Airflow Variables (Admin → Variables)

3. **Запустите проект:**
   ```bash
   docker-compose up -d --build
   ```

4. **Откройте Airflow UI:**
   ```
   http://localhost:8080
   ```
   Логин: `airflow`  
   Пароль: `airflow`

5. **Поместите файлы для обработки:**
   Скопируйте CSV/XLSX файлы в папку `./data`

## 🗂 Структура проекта

```
├── dags/                   # Airflow DAGs
│   └── x5_sales_pipeline.py
├── scripts/                # Вспомогательные скрипты
│   ├── utils.py
│   ├── convert_xlsx_to_csv.py
│   ├── create_table_and_upload.py
│   └── call_stored_procedure.py
├── data/                   # Входные данные (автоматически обрабатываются)
├── archive/                # Архив обработанных файлов
├── docker-compose.yml      # Docker Compose конфигурация
├── Dockerfile              # Кастомный образ Airflow
├── requirements.txt        # Зависимости Python
├── entrypoint.sh           # Скрипт инициализации
└── README.md
```

## ⚙️ Настройка подключения к MSSQL

1. В Airflow UI перейдите в **Admin → Variables**
2. Создайте переменную:
   - **Key**: `MSSQL_CONN_STR`
   - **Value**: 
     ```
     mssql+pyodbc://<username>:<password>@<server>/<database>?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes
     ```

Пример для Windows:
```
mssql+pyodbc://sales_user:123@host.docker.internal/MSSQLSERVER01/Test?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes
```

## 🔧 Конфигурация DAG

Основной DAG (`x5_sales_pipeline.py`) настроен со следующими параметрами:

```python
with DAG(
    dag_id="x5_sales_pipeline",
    start_date=datetime(2025, 7, 7),
    schedule_interval="@daily",  # Ежедневный запуск
    catchup=False,               # Отключить дозапуск
    tags=["x5", "sales"],
) as dag:
```

## 📊 Пример файла данных

Файлы должны иметь следующий формат имени:
```
<сеть>_<месяц>_<год>.csv
```

Пример:
```
x5_january_2025.csv
```

Структура CSV:
```csv
store_id;product_id;quantity;price
1001;P001;150;29.99
1002;P002;200;49.99
...
```

## 🔄 Рабочий процесс

1. **Scan Files**  
   Сканирует папку `./data` на наличие новых файлов
   
2. **Process File**  
   Для каждого файла выполняет:
   - Конвертацию XLSX → CSV (если необходимо)
   - Создание таблицы в SQL Server
   - Загрузку данных
   - Вызов хранимой процедуры
   - Архивирование файла

## 🐛 Устранение неполадок

**Проверьте логи:**
```bash
docker-compose logs -f airflow
```

**Пересоберите проект:**
```bash
docker-compose down --volumes
docker-compose build --no-cache
docker-compose up -d
```

**Проверьте подключение к MSSQL из контейнера:**
```bash
docker exec -it airflow_webserver_1 bash
python -c "from sqlalchemy import create_engine; engine = create_engine('$MSSQL_CONN_STR'); conn = engine.connect(); print(conn.execute('SELECT 1').scalar())"
```
