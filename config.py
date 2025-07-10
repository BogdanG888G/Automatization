import os
import urllib

SQL_CONNECTION_STRING = urllib.parse.quote_plus(
    f"Driver={{ODBC Driver 18 for SQL Server}};"
    f"Server={os.getenv('DB_SERVER')};"
    f"Database={os.getenv('DB_DATABASE')};"
    f"Trusted_Connection={os.getenv('DB_TRUSTED_CONNECTION')};"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)
