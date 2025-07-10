import os
import pandas as pd
import csv

def convert_xlsx_to_csv(file_path):
    df = pd.read_excel(file_path, dtype=str)
    print(df.columns.tolist())
    df.columns = [col.strip().replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('"', '').replace("'", '') for col in df.columns]
    new_path = file_path.replace('.xlsx', '.csv')
    print(f"Число колонок: {len(df.columns)}")
    print(f"Пример строки: {len(df.iloc[0].values)} значений")
    df.to_csv(new_path, index=False, sep=';', encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    os.remove(file_path)
    return new_path
