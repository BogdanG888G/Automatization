import logging
import re
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import engine, text

logger = logging.getLogger(__name__)

_RE_JUNK_NUM = re.compile(r"[^\d.eE+\-]")
_RE_CONTROL = re.compile(r"[\x00-\x1F\x7F-\x9F]")

def _clean_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.replace(",", ".", regex=False)
    s = s.str.replace(_RE_JUNK_NUM, "", regex=True).fillna("0").replace("", "0")
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def _convert_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        try:
            # Пробуем перевести в число
            converted = _clean_numeric_series(df[col])
            # Если больше 90% значений успешно сконвертировались — ок
            non_zero_ratio = (converted != 0.0).mean()
            if non_zero_ratio > 0.5:
                df[col] = converted
            else:
                raise ValueError("Too many non-numeric values")
        except Exception:
            # fallback to string
            df[col] = (
                df[col].astype("string")
                .str.replace(_RE_CONTROL, "", regex=True)
                .str.strip()
                .fillna("")
            )
    return df


def _infer_column_type(series: pd.Series) -> str:
    try:
        pd.to_numeric(series.dropna().unique(), errors="raise")
        return "DECIMAL(18,2)"
    except Exception:
        return "NVARCHAR(MAX)"

def _create_or_truncate_stage_table(
    conn: engine.Connection,
    table_name: str,
    schema: str,
    drop_if_exists: bool,
    df_example: pd.DataFrame,
) -> None:
    if drop_if_exists:
        drop_sql = f"IF OBJECT_ID(N'[{schema}].[{table_name}]', 'U') IS NOT NULL DROP TABLE [{schema}].[{table_name}];"
        conn.execute(text(drop_sql))
        logger.info("Dropped existing table [%s].[%s].", schema, table_name)

    table_exists = (
        conn.execute(
            text(
                "SELECT COUNT(*) FROM sys.tables t "
                "JOIN sys.schemas s ON t.schema_id = s.schema_id "
                "WHERE s.name = :schema AND t.name = :tbl"
            ),
            {"schema": schema, "tbl": table_name},
        ).scalar() > 0
    )

    if not table_exists:
        columns_sql = ",\n".join(
            [f"[{c}] {_infer_column_type(df_example[c])}" for c in df_example.columns]
        )
        create_sql = f"""
            CREATE TABLE [{schema}].[{table_name}] (
                {columns_sql},
                load_dt DATETIME DEFAULT GETDATE()
            );
        """
        conn.execute(text(create_sql))
        logger.info("Created new table [%s].[%s].", schema, table_name)

def _prepare_chunk(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
        except Exception:
            df[col] = df[col].astype(str).replace({"nan": "", "None": ""})
    return df

def _chunk_to_tuples(df: pd.DataFrame) -> List[Tuple]:
    return list(df.itertuples(index=False, name=None))

def _bulk_insert_chunk(conn: engine.Connection, table_name: str, df: pd.DataFrame, schema: str) -> None:
    if df.empty:
        return
    cols = df.columns.tolist()
    insert_sql = f"INSERT INTO [{schema}].[{table_name}] ({', '.join(f'[{c}]' for c in cols)}) VALUES ({', '.join(['?'] * len(cols))})"
    raw_conn = conn.connection
    cursor = raw_conn.cursor()
    try:
        cursor.fast_executemany = True
        cursor.executemany(insert_sql, _chunk_to_tuples(df))
        raw_conn.commit()
    finally:
        cursor.close()

def convert_raw_to_stage(
    table_name: str,
    raw_engine: engine.Engine,
    stage_engine: engine.Engine,
    stage_schema: str = "test",
    limit: Optional[int] = None,
    drop_stage_if_exists: bool = False,
    chunk_size: int = 100000,
) -> None:
    start = datetime.now()
    logger.info("[Stage] Start processing table: %s", table_name)

    with raw_engine.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM raw.{table_name}")).scalar()
    logger.info("[Stage] Total rows to process: %d", total)

    with raw_engine.connect().execution_options(stream_results=True) as raw_conn, stage_engine.connect() as stage_conn:
        query = f"SELECT * FROM raw.{table_name}"
        if limit:
            query = f"SELECT TOP ({limit}) * FROM raw.{table_name}"

        chunks = pd.read_sql(text(query), raw_conn, chunksize=chunk_size, dtype="object")
        processed = 0
        for i, chunk in enumerate(chunks):
            if chunk.empty:
                continue

            chunk.columns = [c.strip() for c in chunk.columns]
            chunk = _convert_all_columns(chunk)

            if i == 0:
                _create_or_truncate_stage_table(stage_conn, table_name, stage_schema, drop_stage_if_exists, chunk)

            chunk = _prepare_chunk(chunk)

            if limit:
                remaining = limit - processed
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]

            _bulk_insert_chunk(stage_conn, table_name, chunk, stage_schema)
            processed += len(chunk)
            logger.info("[Stage] Processed rows: %d", processed)
            if limit and processed >= limit:
                break

        try:
            stage_conn.commit()
        except Exception:
            pass

    logger.info("[Stage] Finished. Total rows: %d. Duration: %.2f sec", processed, (datetime.now() - start).total_seconds())

# ------------------------------------------------------------------ #
# If run as script                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse
    from sqlalchemy import create_engine

    parser = argparse.ArgumentParser(description="Convert raw Okey sales tables to stage.")
    parser.add_argument("table_name", help="Source table name in schema raw")
    parser.add_argument("--raw-conn", required=True, help="SQLAlchemy connection string to raw DB")
    parser.add_argument("--stage-conn", required=True, help="SQLAlchemy connection string to stage DB")
    parser.add_argument("--stage-schema", default="okey", help="Destination schema name")
    parser.add_argument("--limit", type=int, default=None, help="Row limit for testing")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Rows per chunk")
    parser.add_argument("--drop", action="store_true", help="Drop stage table if exists before load")
    args = parser.parse_args()

    raw_engine = create_engine(args.raw_conn, fast_executemany=True)
    stage_engine = create_engine(args.stage_conn, fast_executemany=True)

    convert_raw_to_stage(
        table_name=args.table_name,
        raw_engine=raw_engine,
        stage_engine=stage_engine,
        stage_schema=args.stage_schema,
        limit=args.limit,
        drop_stage_if_exists=args.drop,
        chunk_size=args.chunk_size,
    )
