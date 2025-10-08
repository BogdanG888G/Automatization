# Автоматизация загрузки данных в SQL Server с помощью Apache Airflow

![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=Apache%20Airflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![SQL Server](https://img.shields.io/badge/Microsoft%20SQL%20Server-CC2927?style=for-the-badge&logo=microsoft%20sql%20server&logoColor=white)

Проект автоматизирует процесс загрузки данных из CSV/XLSX/XLSB файлов в Microsoft SQL Server с использованием Apache Airflow. Решение обеспечивает сквозную обработку данных от обнаружения файлов до архивирования с поддержкой структурирования данных для различных розничных сетей

## 🌟 Ключевые особенности
- Автоматическое обнаружение новых файлов в директори
- Динамическое создание таблиц в SQL Server
- Пакетная загрузка данных
- Обогащение 7 столбцов при помощи ML
- Динамичная обработка данных
- Интегрированное архивирование

**Доступ к Airflow UI:**

Логин: `airflow`  
Пароль: `airflow`


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

## Документация по стандартизации данных розничных продаж

### 1. Общие положения

Данная документация описывает стандартный формат представления данных о розничных продажах для последующей аналитики. Все исходные таблицы должны быть преобразованы к указанному формату.

### 2. Структура таблицы

#### 2.1. Основные разделы данных

1. **Идентификация сети**
2. **Временные метки**
3. **Географические данные**
4. **Товарные характеристики**
5. **Иерархия продуктов**
6. **Коды и артикулы**
7. **Данные о производителях**
8. **Финансовые показатели**
9. **Дополнительные метрики**

#### 2.2. Детальное описание полей

| Группа | Поле | Тип данных | Описание | Обязательность | Пример значения |
|--------|------|------------|----------|----------------|-----------------|
| **Сеть** | retail_chain | NVARCHAR(50) | Название розничной сети | Обязательно | 'Магнит' |
| **Время** | sale_year | INT | Год продажи | Обязательно | 2024 |
|  | sale_month | INT | Месяц продажи (1-12) | Обязательно | 12 |
|  | sale_date | DATE | Дата продажи (первый день месяца) | Обязательно | '2024-12-01' |
| **География** | branch | NVARCHAR(100) | Филиал/отделение | Опционально | 'Филиал Приволжский' |
|  | region | NVARCHAR(100) | Регион | Опционально | 'Челябинская обл.' |
|  | city | NVARCHAR(100) | Город | Опционально | 'г.Челябинск' |
|  | address | NVARCHAR(200) | Адрес магазина | Обязательно | 'ул. Ворошилова, дом № 7' |
|  | store_format | NVARCHAR(50) | Формат магазина | Обязательно | 'МД' |
|  | store_name | NVARCHAR(100) | Название магазина | Обязательно | 'Скрепка' |
| **Товар** | product_name | NVARCHAR(200) | Наименование товара | Обязательно | 'Семечки отб с морской солью' |
|  | brand | NVARCHAR(100) | Бренд товара | Обязательно | 'От Мартина' |
|  | flavor | NVARCHAR(100) | Вкус/аромат | Опционально | 'Соль копченая' |
|  | weight | NVARCHAR(50) | Вес/объем | Опционально | '200' |
|  | product_type | NVARCHAR(100) | Тип продукта | Опционально | 'Приссованные рифленые' |
| **Иерархия** | product_level_1 | NVARCHAR(100) | Уровень 1 категории | Обязательно | 'Снэки' |
|  | product_level_2 | NVARCHAR(100) | Уровень 2 категории | Обязательно | 'Семечки' |
|  | product_level_3 | NVARCHAR(100) | Уровень 3 категории | Обязательно | 'Семечки подсолнечника' |
|  | product_level_4 | NVARCHAR(100) | Уровень 4 категории | Опционально | 'Семечки подсолнечника неочищенные' |
| **Коды** | product_family_code | INT | Код товарной группы | Опционально | NULL |
|  | product_family_name | NVARCHAR(100) | Название товарной группы | Опционально | NULL |
|  | product_article | BIGINT | Артикул товара | Обязательно | 1000229533 |
|  | product_code | BIGINT | Код товара | Обязательно | 1000229533 |
|  | barcode | NVARCHAR(50) | Штрих-код | Обязательно | '4607012351999' |
| **Производитель** | factory_code | INT | Код завода | Опционально | NULL |
|  | factory_name | NVARCHAR(100) | Название завода | Опционально | NULL |
|  | material | INT | Код материала | Опционально | NULL |
| **Поставщики** | vendor | NVARCHAR(100) | Производитель | Опционально | NULL |
|  | supplier | NVARCHAR(100) | Поставщик | Обязательно | 'Мартин ООО' |
|  | warehouse_supplier | NVARCHAR(100) | Склад поставщика | Опционально | NULL |
| **Финансы** | sales_quantity | INT | Количество продаж | Обязательно | 37 |
|  | sales_amount_rub | NUMERIC(18,2) | Сумма продаж в рублях | Обязательно | 4879.65 |
|  | avg_cost_price | NUMERIC(18,4) | Средняя закупочная цена | Обязательно | 82.8660 |
|  | avg_sell_price | NUMERIC(18,4) | Средняя цена продажи | Обязательно | 131.8824 |
| **Метрики** | margin_amount_rub | NUMERIC(18,2) | Маржа (руб) | Обязательно | 1813.91 |
|  | sales_tons | NUMERIC(18,3) | Продажи в тоннах | Опционально | 0.074 |
|  | sales_weight_kg | NUMERIC(18,3) | Продажи в кг | Опционально | 74.000 |

### 4. Контроль качества данных

1. **Проверка полноты**:
   - Обязательные поля не должны содержать NULL значений
   - Ключевые идентификаторы (product_article, barcode) должны быть уникальными

2. **Проверка согласованности**:
   - sales_amount_rub ≈ avg_sell_price * sales_quantity
   - margin_amount_rub ≈ sales_amount_rub - (avg_cost_price * sales_quantity)

3. **Проверка точности**:
   - Весовые показатели должны соответствовать заявленному весу продукта
   - Даты должны находиться в разумных временных рамках

### 5. Рекомендации по использованию

1. Для анализа использовать только представления [clear].[*]
2. При соединении таблиц использовать product_article или barcode в качестве ключа
3. Для временных анализов использовать комбинацию sale_year + sale_month
4. Для географического анализа использовать иерархию: branch → region → city → address



## 🔄 Рабочий процесс
```mermaid
graph TD
    A[Сканирование папки ./data] --> B{Найден новый файл?}
    B -->|Да| C[Определить розничную сеть]
    B -->|Нет| Z[Завершить процесс]
    C --> D[Конвертировать XLSX в CSV]
    D --> E[Создать таблицу в тестовой БД]
    E --> F[Загрузить данные в тестовую БД]
    F --> G[Преобразовать типы данных INT, DECIMAL, NVARCHAR]
    G --> H[Загрузить данные в Stage БД]
    H --> I[Переместить файл в архив]
    I --> J[Записать успешное выполнение в лог]
```
