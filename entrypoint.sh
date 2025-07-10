#!/bin/bash
set -e

# Явно указываем путь к Airflow
export AIRFLOW_HOME=/opt/airflow
export PATH="${AIRFLOW_HOME}/bin:${PATH}"

echo "✅ Проверяем доступность Airflow..."
if ! command -v airflow &> /dev/null; then
    echo "❌ Ошибка: airflow не найден!"
    echo "Поиск в системе:"
    find / -name airflow 2>/dev/null || true
    exit 1
fi

echo "✅ Airflow найден: $(which airflow)"
echo "✅ Версия Airflow: $(airflow version)"

echo "✅ Инициализация базы данных..."
airflow db upgrade

echo "✅ Создание пользователя..."
airflow users create \
  --username airflow \
  --password airflow \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com 2>/dev/null || echo "ℹ️ Пользователь уже существует"

echo "✅ Запуск планировщика..."
airflow scheduler &

echo "✅ Запуск веб-сервера..."
exec airflow webserver