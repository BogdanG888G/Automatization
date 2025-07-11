import os
import pandas as pd
import logging
import csv
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ExcelToCSVConverter:
    """Оптимизированный конвертер Excel/XLSB в CSV с обработкой больших файлов."""
    
    CHUNKSIZE = 100000  # Размер чанка для обработки больших файлов
    CSV_SEPARATOR = ';'
    ENCODING = 'utf-8-sig'
    
    @classmethod
    def clean_column_name(cls, col: str) -> str:
        """Очистка названий колонок."""
        remove_chars = str.maketrans('', '', '[]()"\'')
        return col.strip().translate(remove_chars)
    
    @classmethod
    def convert_to_csv(cls, file_path: str) -> Optional[str]:
        """Основной метод конвертации."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.xlsx', '.xls', '.xlsb']:
                raise ValueError(f"Неподдерживаемый формат файла: {ext}")
            
            if os.path.getsize(file_path) > 50 * 1024 * 1024:  # >50MB
                return cls._process_large_file(file_path, ext)
            return cls._process_normal_file(file_path, ext)
            
        except Exception as e:
            logger.error(f"Ошибка конвертации файла {file_path}: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def _process_normal_file(cls, file_path: str, ext: str) -> str:
        """Обработка файлов обычного размера."""
        engine = 'pyxlsb' if ext == '.xlsb' else 'openpyxl'
        df = pd.read_excel(file_path, engine=engine, dtype=str)
        
        # Очистка названий колонок
        df.columns = [cls.clean_column_name(col) for col in df.columns]
        
        # Сохранение в CSV
        new_path = file_path.replace(ext, '.csv')
        df.to_csv(
            new_path,
            index=False,
            sep=cls.CSV_SEPARATOR,
            encoding=cls.ENCODING,
            quoting=csv.QUOTE_NONNUMERIC
        )
        
        os.remove(file_path)
        return new_path
    
    @classmethod
    def _process_large_file(cls, file_path: str, ext: str) -> str:
        """Обработка больших файлов с чанкованием."""
        new_path = file_path.replace(ext, '.csv')
        engine = 'pyxlsb' if ext == '.xlsb' else 'openpyxl'
        
        # Читаем заголовки
        headers = pd.read_excel(
            file_path,
            engine=engine,
            nrows=1,
            dtype=str
        ).columns
        
        # Очищаем заголовки
        headers = [cls.clean_column_name(col) for col in headers]
        
        # Записываем заголовки в CSV
        with open(new_path, 'w', encoding=cls.ENCODING, newline='') as f:
            writer = csv.writer(f, delimiter=cls.CSV_SEPARATOR, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(headers)
        
        # Читаем и записываем данные по частям
        chunk_size = cls.CHUNKSIZE
        skip_rows = 1  # Пропускаем заголовок
        
        while True:
            df_chunk = pd.read_excel(
                file_path,
                engine=engine,
                skiprows=skip_rows,
                nrows=chunk_size,
                dtype=str,
                header=None
            )
            
            if df_chunk.empty:
                break
                
            # Записываем чанк в CSV
            df_chunk.to_csv(
                new_path,
                mode='a',
                index=False,
                sep=cls.CSV_SEPARATOR,
                encoding=cls.ENCODING,
                quoting=csv.QUOTE_NONNUMERIC,
                header=False
            )
            
            skip_rows += chunk_size
        
        os.remove(file_path)
        return new_path

def convert_excel_to_csv(file_path: str) -> str:
    """Точка входа для конвертации файлов."""
    return ExcelToCSVConverter.convert_to_csv(file_path)