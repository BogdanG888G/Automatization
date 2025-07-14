import os
import pandas as pd
import logging
import csv
from typing import Optional, List

logger = logging.getLogger(__name__)


class ExcelToCSVConverter:
    """Оптимизированный конвертер Excel/XLSB в CSV с обработкой больших файлов и всех листов."""

    CHUNKSIZE = 100_000  # Размер чанка для обработки больших файлов
    CSV_SEPARATOR = ';'
    ENCODING = 'utf-8-sig'

    @classmethod
    def clean_column_name(cls, col: str) -> str:
        """Очистка названий колонок от лишних символов."""
        remove_chars = str.maketrans('', '', '[]()"\'')
        return col.strip().translate(remove_chars)

    @classmethod
    def convert_to_csv(cls, file_path: str, max_rows: Optional[int] = None) -> List[str]:
        """Конвертация всех листов Excel-файла в отдельные CSV-файлы."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.xlsx', '.xls', '.xlsb']:
                raise ValueError(f"Неподдерживаемый формат файла: {ext}")

            engine = 'pyxlsb' if ext == '.xlsb' else 'openpyxl'
            sheet_names = pd.ExcelFile(file_path, engine=engine).sheet_names

            result_paths = []
            for sheet in sheet_names:
                logger.info(f"Обработка листа '{sheet}' из файла {file_path}")
                if os.path.getsize(file_path) > 50 * 1024 * 1024:  # >50MB
                    path = cls._process_large_file(file_path, ext, sheet, max_rows)
                else:
                    path = cls._process_normal_file(file_path, ext, sheet, max_rows)
                result_paths.append(path)

            os.remove(file_path)
            return result_paths

        except Exception as e:
            logger.error(f"Ошибка конвертации файла {file_path}: {str(e)}", exc_info=True)
            raise

    @classmethod
    def _generate_csv_path(cls, file_path: str, ext: str, sheet_name: str) -> str:
        """Создаёт путь к CSV-файлу с учётом имени листа."""
        base_path = file_path.replace(ext, '')
        sheet_suffix = sheet_name.replace(" ", "_").replace("/", "_")
        return f"{base_path}__{sheet_suffix}.csv"

    @classmethod
    def _process_normal_file(cls, file_path: str, ext: str, sheet_name: str, max_rows: Optional[int] = None) -> str:
        """Обработка листа обычного Excel-файла."""
        engine = 'pyxlsb' if ext == '.xlsb' else 'openpyxl'
        df = pd.read_excel(file_path, engine=engine, dtype=str, sheet_name=sheet_name, nrows=max_rows)

        df.columns = [cls.clean_column_name(col) for col in df.columns]

        new_path = cls._generate_csv_path(file_path, ext, sheet_name)
        df.to_csv(
            new_path,
            index=False,
            sep=cls.CSV_SEPARATOR,
            encoding=cls.ENCODING,
            quoting=csv.QUOTE_NONNUMERIC
        )

        return new_path

    @classmethod
    def _process_large_file(cls, file_path: str, ext: str, sheet_name: str, max_rows: Optional[int] = None) -> str:
        """Обработка большого листа Excel-файла по частям."""
        engine = 'pyxlsb' if ext == '.xlsb' else 'openpyxl'
        new_path = cls._generate_csv_path(file_path, ext, sheet_name)

        # Заголовки
        headers = pd.read_excel(
            file_path,
            engine=engine,
            sheet_name=sheet_name,
            nrows=1,
            dtype=str
        ).columns
        headers = [cls.clean_column_name(col) for col in headers]

        with open(new_path, 'w', encoding=cls.ENCODING, newline='') as f:
            writer = csv.writer(f, delimiter=cls.CSV_SEPARATOR, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(headers)

        chunk_size = cls.CHUNKSIZE
        skip_rows = 1
        rows_written = 0

        while True:
            if max_rows is not None and rows_written >= max_rows:
                break

            rows_left = max_rows - rows_written if max_rows is not None else chunk_size
            chunk_rows = min(chunk_size, rows_left)

            df_chunk = pd.read_excel(
                file_path,
                engine=engine,
                sheet_name=sheet_name,
                skiprows=skip_rows,
                nrows=chunk_rows,
                dtype=str,
                header=None
            )

            if df_chunk.empty:
                break

            df_chunk.to_csv(
                new_path,
                mode='a',
                index=False,
                sep=cls.CSV_SEPARATOR,
                encoding=cls.ENCODING,
                quoting=csv.QUOTE_NONNUMERIC,
                header=False
            )

            skip_rows += chunk_rows
            rows_written += len(df_chunk)

        return new_path


def convert_excel_to_csv(file_path: str, max_rows: Optional[int] = None) -> List[str]:
    """Точка входа для конвертации Excel-файлов. Возвращает список путей к CSV-файлам."""
    return ExcelToCSVConverter.convert_to_csv(file_path, max_rows=max_rows)
