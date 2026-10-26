from io import BytesIO
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


AVAILABLE_DATA = {
    "Diabetes data": DATA_DIR / "diabetes.csv",
}

SUPPORTED = ["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"]
EXCEL = (".xlsx", ".xls")


def is_excel(name: str) -> bool:
    return name.lower().endswith(EXCEL)


def excel_sheets(source) -> list:
    """Sheet names of an Excel file (path or file-like)."""
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.ExcelFile(source).sheet_names


def read_any(source, name: str, sheet=None) -> pd.DataFrame:
    """Read a CSV / Excel / JSON / TXT / Parquet file from a path or file-like object."""
    if hasattr(source, "seek"):
        source.seek(0)
    n = name.lower()
    if n.endswith(".csv"):
        return pd.read_csv(source)
    if n.endswith(".tsv"):
        return pd.read_csv(source, sep="\t")
    if n.endswith(".txt"):
        return pd.read_csv(source, sep=None, engine="python")
    if n.endswith(EXCEL):
        return pd.read_excel(source, sheet_name=sheet or 0)
    if n.endswith(".json"):
        return pd.read_json(source)
    if n.endswith(".parquet"):
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported file type: {name}")


def load_available(label: str) -> pd.DataFrame:
    """Load one of the built-in datasets listed in AVAILABLE_DATA."""
    path = AVAILABLE_DATA[label]
    return read_any(path, path.name)


def list_folder(folder: str) -> list:
    path = Path(folder).expanduser()
    if not path.is_dir():
        raise ValueError("Folder not found on the machine running Streamlit.")
    files = [p for p in sorted(path.iterdir()) if p.suffix.lower().lstrip(".") in SUPPORTED]
    if not files:
        raise ValueError("No supported data files in this folder.")
    return files


def load_url(url: str) -> pd.DataFrame:
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("The URL must start with http:// or https://")
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    clean = url.split("?")[0].split("#")[0]
    return read_any(BytesIO(r.content), clean if "." in Path(clean).name else "data.csv")


def combine(frames: dict) -> pd.DataFrame:
    """Stack several files; adds a 'source_file' column."""
    parts = []
    for name, df in frames.items():
        df = df.copy()
        df["source_file"] = name
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


# ---- Databases ----
def database_engine(kind: str, v: dict):
    if kind == "SQLite":
        return create_engine(f"sqlite:///{Path(v['path']).expanduser()}")
    port = int(v["port"]) if str(v.get("port", "")).strip() else None
    if kind == "MySQL":
        url = URL.create("mysql+pymysql", v["user"], v["password"], v["host"] or "localhost",
                         port or 3306, v["database"])
    elif kind == "PostgreSQL":
        url = URL.create("postgresql+psycopg2", v["user"], v["password"], v["host"] or "localhost",
                         port or 5432, v["database"])
    elif kind == "SQL Server":
        conn = (f"DRIVER={{{v['driver']}}};SERVER={v['host']},{port or 1433};DATABASE={v['database']};"
                f"UID={v['user']};PWD={v['password']};TrustServerCertificate=yes")
        return create_engine("mssql+pyodbc:///?odbc_connect=" + quote_plus(conn))
    else:
        raise ValueError("Unsupported database.")
    return create_engine(url)


def get_tables(engine) -> list:
    insp = inspect(engine)
    return insp.get_table_names() + insp.get_view_names()


def load_table(engine, table: str, limit: int = 200_000) -> pd.DataFrame:
    q = text(f'SELECT * FROM "{table}"' if engine.dialect.name in ("sqlite", "postgresql")
             else f"SELECT * FROM `{table}`" if engine.dialect.name == "mysql"
             else f"SELECT * FROM [{table}]")
    with engine.connect() as c:
        return pd.read_sql(q, c).head(limit)


def run_query(engine, sql: str) -> pd.DataFrame:
    if not sql.strip().lower().startswith(("select", "with")):
        raise ValueError("Only SELECT queries are allowed.")
    with engine.connect() as c:
        return pd.read_sql(text(sql), c)
