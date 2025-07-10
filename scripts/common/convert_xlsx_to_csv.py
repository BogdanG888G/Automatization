# /opt/airflow/scripts/common/convert_xlsx_to_csv.py

import os
import pandas as pd
import csv

def convert_excel_to_csv(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.xlsb':
        df = pd.read_excel(file_path, dtype=str, engine='pyxlsb')
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, dtype=str)
    else:
        raise ValueError(f"Неподдерживаемое расширение файла: {file_path}")

    print(df.columns.tolist())

    df.columns = [
        col.strip()
           .replace('[', '')
           .replace(']', '')
           .replace('(', '')
           .replace(')', '')
           .replace('"', '')
           .replace("'", '')
        for col in df.columns
    ]

    new_path = file_path.replace(ext, '.csv')
    print(f"Число колонок: {len(df.columns)}")
    print(f"Пример строки: {len(df.iloc[0].values)} значений")

    df.to_csv(new_path, index=False, sep=';', encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    os.remove(file_path)

    return new_path
