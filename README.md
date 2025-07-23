# Автоматизация загрузки данных в SQL Server с помощью Apache Airflow

![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=Apache%20Airflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![SQL Server](https://img.shields.io/badge/Microsoft%20SQL%20Server-CC2927?style=for-the-badge&logo=microsoft%20sql%20server&logoColor=white)

Проект автоматизирует процесс загрузки данных из CSV/XLSX/XLSB файлов в Microsoft SQL Server с использованием Apache Airflow. Решение обеспечивает сквозную обработку данных от обнаружения файлов до архивирования с поддержкой структурирования данных для различных розничных сетей

## 🌟 Ключевые особенности
- Автоматическое обнаружение новых файлов в директории
- Конвертация XLSX → CSV
- Динамическое создание таблиц в SQL Server
- Пакетная загрузка данных
- Динамичная обработка данных
- Интегрированное архивирование
- Поддержка мультисетевой структуры данных
- Управление через Docker-контейнеры

## 📦 Требования
- Docker Desktop 20.10+
- Docker Compose 2.0+
- Microsoft SQL Server 2019+
- 4 ГБ свободной оперативной памяти

## 🚀 Быстрый старт
1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/BogdanG888G/Automatization.git
cd airflow-data-pipeline
```

2. **Настройте окружение:**
```bash
cp .env.example .env
# Отредактируйте .env файл под вашу конфигурацию
```

3. **Запустите проект:**
```bash
docker-compose up -d --build
```

4. **Доступ к Airflow UI:**
```
http://localhost:8080
```
Логин: `airflow`  
Пароль: `airflow`

5. **Поместите файлы для обработки:**
```bash
# Пример для Magnit
cp ваш_файл.csv ./data/magnit_december_2024.csv
```

## 🗂 Структура проекта
```
AUTOMATIZATION/
├── archive/                  # Архив обработанных файлов
├── data/                     # Входные данные
├── dags/                     # Индивидуальные DAG для каждой сети
│   ├── ashan_sales_pipeline.py
│   ├── diksi_sales_pipeline.py
│   ├── magnit_sales_pipeline.py
│   ├── okey_sales_pipeline.py
│   ├── perekrestok_sales_pipeline.py
│   ├── pyaterochka_sales_pipeline.py
│   └── x5_sales_pipeline.py
├── scripts/                  # Сетевые специфичные скрипты
│   ├── ashan/
│   │   ├── convert_raw_to_stage.py
│   │   └── create_table_and_upload.py
│   ├── diksi/
│   │   ├── convert_raw_to_stage.py
│   │   └── create_table_and_upload.py
│   ├── magnit/
│   │   ├── convert_raw_to_stage.py
│   │   └── create_table_and_upload.py
│   ├── okey/
│   │   ├── convert_raw_to_stage.py
│   │   └── create_table_and_upload.py
│   ├── perekrestok/
│   │   ├── convert_raw_to_stage.py
│   │   └── create_table_and_upload.py
│   ├── pyaterochka/
│   │   ├── convert_raw_to_stage.py
│   │   └── create_table_and_upload.py
│   └── x5/
│   │   ├── convert_raw_to_stage.py
│   │   └── create_table_and_upload.py
│   └── common/
│   │   ├── convert_xlsx_to_csv.py
│   │   └── utils.py                # Общие утилиты
├── .dockerignore
├── .env                      # Переменные окружения
├── .gitignore
├── docker-compose.yml
├── config.py
├── Dockerfile
├── requirements.txt
├── TODO.md
├── entrypoint.sh
└── README.md
```

## ⚙️ Настройка подключения к MSSQL
1. В Airflow UI: **Admin → Variables**
2. Создайте переменную:
   - **Key**: `MSSQL_CONN_STR`
   - **Value**:
```
mssql+pyodbc://<user>:<password>@<host>/<database>?driver=ODBC+Driver+17+for+SQL+Server
```

Пример для Windows:
```
mssql+pyodbc://airflow_agent:Pass123@host.docker.internal/SalesDB?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes
```

## 🔧 Конфигурация DAG
Основной DAG настроен с параметрами:
```python
with DAG(
    dag_id="retail_data_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",  # Ежедневный запуск
    catchup=False,
    tags=["retail", "data_processing"],
) as dag:
```

## 📊 Форматы данных для розничных сетей

### 🛒 Magnit
```sql
CREATE TABLE [Stage].[magnit].[magnit_<месяц>_<год>] (
    [month] NVARCHAR(50),
    [format] NVARCHAR(50),
    [store_name] NVARCHAR(255),
    [store_id] INT,
    [store_address] NVARCHAR(500),
    [level_1] NVARCHAR(255),
    [level_2] NVARCHAR(255),
    [level_3] NVARCHAR(255),
    [level_4] NVARCHAR(255),
    [supplier] NVARCHAR(255),
    [brand] NVARCHAR(255),
    [product_name] NVARCHAR(255),
    [product_id] NVARCHAR(50),
    [barcode] NVARCHAR(50),
    [revenue_rub] DECIMAL(18,2),
    [revenue_qty] INT,
    [purchase_price] DECIMAL(18,2)
```

### 🛒 Ashan
```sql
CREATE TABLE [Stage].[ashan].[ashan_<месяц>_<год>] (
    [date_raw] DATE,
    [product_name] NVARCHAR(255),
    [store] NVARCHAR(255),
    [city] NVARCHAR(100),
    [address] NVARCHAR(500),
    [month_raw] NVARCHAR(50),
    [ср_цена_продажи] DECIMAL(18,2),
    [списания_руб] DECIMAL(18,2),
    [списания_шт] INT,
    [продажи_c_ндс] DECIMAL(18,2),
    [продажи_шт] INT,
    [ср_цена_покупки] DECIMAL(18,2),
    [маржа_руб] DECIMAL(18,2),
    [потери_руб] DECIMAL(18,2),
    [потери_шт] INT,
    [промо_продажи_c_ндс] DECIMAL(18,2),
    [sale_month] INT,
    [sale_year] INT,
    [sale_date] DATE,
    [product_segment] NVARCHAR(100),
    [product_family_code] NVARCHAR(50),
    [product_family_name] NVARCHAR(255),
    [product_article] NVARCHAR(50),
    [supplier_code] NVARCHAR(50),
    [supplier_name] NVARCHAR(255),
    [store_format] NVARCHAR(50)
```

### 🛒 Diksi
```sql
CREATE TABLE [Stage].[diksi].[diksi_<месяц>_<год>] (
    [sale_year] INT,
    [sale_month] INT,
    [level_3] NVARCHAR(255),
    [level_4] NVARCHAR(255),
    [level_5] NVARCHAR(255),
    [vtm] NVARCHAR(255),
    [product] NVARCHAR(255),
    [address] NVARCHAR(500),
    [product_code] NVARCHAR(50),
    [stores] NVARCHAR(255),
    [quantity_first_week] INT,
    [cost_with_vat_first_week] DECIMAL(18,2),
    [amount_with_vat_first_week] DECIMAL(18,2),
    [quantity_second_week] INT,
    [cost_with_vat_second_week] DECIMAL(18,2),
    [amount_with_vat_second_week] DECIMAL(18,2),
    [quantity_third_week] INT,
    [cost_with_vat_third_week] DECIMAL(18,2),
    [amount_with_vat_third_week] DECIMAL(18,2),
    [quantity_fourth_week] INT,
    [cost_with_vat_fourth_week] DECIMAL(18,2),
    [amount_with_vat_fourth_week] DECIMAL(18,2),
    [quantity_fifth_week] INT,
    [cost_with_vat_fifth_week] DECIMAL(18,2),
    [amount_with_vat_fifth_week] DECIMAL(18,2),
    [quantity_summary] INT,
    [cost_with_vat_summary] DECIMAL(18,2),
    [amount_with_vat_summary] DECIMAL(18,2)
```

### 🛒 Okey
```sql
CREATE TABLE [Stage].[okey].[okey_<месяц>_<год>] (
    [retail_chain] NVARCHAR(50),
    [product_category] NVARCHAR(255),
    [product_type] NVARCHAR(255),
    [supplier_name] NVARCHAR(255),
    [brand] NVARCHAR(255),
    [product_name] NVARCHAR(255),
    [product_unified_name] NVARCHAR(255),
    [product_weight_g] INT,
    [product_flavor] NVARCHAR(100),
    [sales_quantity] INT,
    [sales_amount_rub] DECIMAL(18,2),
    [sales_weight_kg] DECIMAL(18,3),
    [cost_price_rub] DECIMAL(18,2),
    [sales_month] INT,
    [load_dt] DATETIME
```

### 🛒 Pereкrestok
```sql
CREATE TABLE [Stage].[perekrestok].[perekrestok_<месяц>_<год>] (
    [period] DATE,
    [retail_chain] NVARCHAR(50),
    [category] NVARCHAR(255),
    [supplier] NVARCHAR(255),
    [brand] NVARCHAR(255),
    [product_name] NVARCHAR(255),
    [weight_g] INT,
    [flavor] NVARCHAR(100),
    [sale_year] INT,
    [sale_month] INT,
    [cost_rub] DECIMAL(18,2),
    [sales_qty] INT,
    [sales_rub] DECIMAL(18,2),
    [sales_tons] DECIMAL(18,3),
    [category_lvl2] NVARCHAR(255),
    [product_name_uni] NVARCHAR(255)
```

### 🛒 Pyaterochka
```sql
CREATE TABLE [Stage].[pyaterochka].[pyaterochka_<месяц>_<год>] (
    [retail_chain] NVARCHAR(50),
    [brand] NVARCHAR(255),
    [product_name] NVARCHAR(255),
    [sales_quantity] INT,
    [sales_amount_rub] DECIMAL(18,2),
    [sales_weight_kg] DECIMAL(18,3),
    [cost_price_rub] DECIMAL(18,2),
    [sales_month] INT,
    [product_category] NVARCHAR(255),
    [product_type] NVARCHAR(255),
    [supplier_name] NVARCHAR(255),
    [product_unified_name] NVARCHAR(255),
    [product_weight_g] INT,
    [product_flavor] NVARCHAR(100),
    [load_dt] DATETIME
```

### 🛒 X5 Retail Group
```sql
CREATE TABLE [Stage].[x5].[x5_<месяц>_<год>] (
    [retail_chain] NVARCHAR(50),
    [branch] NVARCHAR(255),
    [region] NVARCHAR(100),
    [city] NVARCHAR(100),
    [address] NVARCHAR(500),
    [factory] NVARCHAR(255),
    [factory_2] NVARCHAR(255),
    [prod_level_2] NVARCHAR(255),
    [prod_level_3] NVARCHAR(255),
    [prod_level_4] NVARCHAR(255),
    [material] NVARCHAR(255),
    [material_2] NVARCHAR(255),
    [brand] NVARCHAR(255),
    [vendor] NVARCHAR(255),
    [main_supplier] NVARCHAR(255),
    [warehouse_supplier] NVARCHAR(255),
    [quantity] INT,
    [gross_turnover] DECIMAL(18,2),
    [gross_cost] DECIMAL(18,2),
    [avg_cost_price] DECIMAL(18,2),
    [avg_sell_price] DECIMAL(18,2),
    [sale_month] INT,
    [sale_year] INT
```

## 🔄 Рабочий процесс
```mermaid
graph TD
    A[Scan ./data directory] --> B{New file found?}
    B -->|Yes| C[Determine retail chain]
    B -->|No| Z[End process]
    C --> D[Convert XLSX to CSV]
    D --> E[Create SQL table]
    E --> F[Upload data to Test base]
    F --> G[Modify data with new types (INT, DECIMAL, NVARCHAR)]
    G --> H[Upload data to Stage base]
    H --> I[Move file to archive]
    I --> J[Log success]
```

## 🐛 Устранение неполадок
**Проверка логов:**
```bash
docker-compose logs -f airflow-worker
```


**Пересборка проекта:**
```bash
docker-compose down --volumes --remove-orphans
docker-compose build --no-cache
docker-compose up -d
```

**Очистка Docker:**
```bash
docker system prune -a --volumes
```
