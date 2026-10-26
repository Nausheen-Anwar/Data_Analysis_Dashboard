import io
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils import data as dt
from utils.analysis import (
    PARAMETERS, correlation_table, density_curves, hypothesis_test,
    numeric_columns, pdf_samples, summary_table
)
from utils.auth import login_screen, logout
from utils.plots import (
    CHARTS, MIME, NEEDS_ONE, NEEDS_XY, density_figure, fig_bytes, make_chart
)
from utils.report import build_report
from utils.theme import inject_css
from utils import lab

import matplotlib.pyplot as plt

st.set_page_config(page_title="UST Software India Pvt Ltd - Data Analysis Dashboard",
                   page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "ust_logo.jpg"
FEEDBACK_DB = BASE_DIR / "feedback.db"

# ---- Branding: logo, company name, feedback storage ----
def brand_header():
    left, mid, right = st.columns([0.15, 0.7, 0.15], vertical_alignment="center")
    with left:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=70)
    with mid:
        st.markdown('<div class="ust-title">Data Analysis Using Streamlit in Python</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="ust-title">UST Software India Pvt Ltd</div>',
                    unsafe_allow_html=True)
    st.divider()


def init_feedback_db():
    with sqlite3.connect(FEEDBACK_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                rating INTEGER NOT NULL,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def save_feedback(user_id, rating, comment):
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB) as conn:
        conn.execute(
            "INSERT INTO feedback(user_id, rating, comment) VALUES (?, ?, ?)",
            (user_id, int(rating), comment.strip())
        )
        conn.commit()


def k(name):
    return f"{name}_{st.session_state.data_id}"


def remove_uploads(names):
    """Forget uploaded files. If the loaded table came from one of them, clear it too."""
    for n in names:
        st.session_state.uploads.pop(n, None)
    if st.session_state.data_name in names:
        st.session_state.data, st.session_state.data_name = None, ""
        st.session_state.data_id += 1


# ---- Session defaults ----
for key, default in {
    "logged_in": False, "user_id": "", "data": None, "data_name": "",
    "data_id": 0, "db_engine": None, "db_tables": [],
    "uploads": {}, "uploader_nonce": 0,
    "last_chart_png": None, "last_chart_title": "",
    "last_stats": None, "last_corr": None, "last_hypothesis": None,
    "last_density_png": None, "last_density_title": "",
}.items():
    st.session_state.setdefault(key, default)

inject_css(logged_in=st.session_state.logged_in)

# ---- Nothing below is shown until the user has logged in ----
if not st.session_state.logged_in:
    login_screen()
    st.stop()

brand_header()

# ---- Sidebar: home, data source, data selection ----
sb = st.sidebar
sb.markdown(f"**👤 {st.session_state.user_id}**")
c1, c2 = sb.columns(2)
if c1.button("Logout", use_container_width=True):
    logout()
    st.rerun()
if c2.button("Clear data", use_container_width=True):
    st.session_state.data, st.session_state.data_name = None, ""
    st.session_state.data_id += 1
    st.rerun()

with sb.expander("1 · Home", expanded=True):
    st.text_input("Dashboard title", key="dashboard_title",
                  placeholder="e.g. Sales analysis 2026")

with sb.expander("2 · Data source", expanded=st.session_state.data is None):
    source = st.selectbox(
        "Source",
        ["Available data", "Upload file(s)", "Local folder", "Online URL", "Database"],
        index=None, placeholder="Choose a source"
    )

    if source == "Available data":
        dataset_name = "Diabetes data"
        st.markdown("**Diabetes data**")
        st.caption("Built-in sample dataset · ready to load · no upload required.")
        if st.button(
            "Load diabetes data", type="primary", use_container_width=True,
            key="load_diabetes_data"
        ):
            try:
                st.session_state.data = dt.load_available(dataset_name)
                st.session_state.data_name = dataset_name
                st.session_state.data_id += 1
                st.rerun()
            except Exception as e:
                st.error(f"Could not load the diabetes data. Please try again. Details: {e}")

    elif source == "Upload file(s)":
        new_files = st.file_uploader(
            "CSV, Excel, JSON, TXT, Parquet", type=dt.SUPPORTED,
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_nonce}"
        )
        if new_files:
            for f in new_files:
                st.session_state.uploads[f.name] = f.getvalue()
            st.session_state.uploader_nonce += 1
            st.rerun()

        uploads = st.session_state.uploads
        if uploads:
            st.markdown("**Uploaded files**")
            chosen = []
            for name, raw in list(uploads.items()):
                c_use, c_rm = st.columns([0.62, 0.38], vertical_alignment="center")
                if c_use.checkbox(
                    f"Use: {name}", key=f"use_{name}",
                    help=f"{len(raw) / 1024:,.0f} KB - tick to use this file"
                ):
                    chosen.append(name)
                if c_rm.button(
                    "Remove file", key=f"rm_{name}", use_container_width=True,
                    help=f"Remove {name} from the uploaded files"
                ):
                    remove_uploads([name])
                    st.rerun()

            merge = len(chosen) > 1 and st.checkbox("Stack all chosen files into one table")
            sheet = None
            if len(chosen) == 1 and dt.is_excel(chosen[0]):
                sheet = st.selectbox(
                    "Sheet", dt.excel_sheets(io.BytesIO(uploads[chosen[0]])),
                    index=None, placeholder="First sheet if none chosen"
                )
            b_load, b_clear = st.columns(2)
            load_clicked = b_load.button(
                "Load data", type="primary", use_container_width=True,
                disabled=not chosen or (len(chosen) > 1 and not merge)
            )
            if b_clear.button("Remove all", use_container_width=True):
                remove_uploads(list(uploads))
                st.rerun()
            if load_clicked:
                try:
                    picked = {n: io.BytesIO(uploads[n]) for n in chosen}
                    if merge:
                        frames = {n: dt.read_any(f, n) for n, f in picked.items()}
                        st.session_state.data = dt.combine(frames)
                        st.session_state.data_name = f"{len(frames)} files combined"
                    else:
                        n, f = next(iter(picked.items()))
                        st.session_state.data = dt.read_any(f, n, sheet)
                        st.session_state.data_name = n
                    if st.session_state.data.empty:
                        raise ValueError("The selected source contains no rows.")
                    st.session_state.data_id += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not read the file: {e}")

    elif source == "Local folder":
        st.caption("Reads a folder on the computer where Streamlit is running.")
        folder = st.text_input(
            "Folder path", placeholder=r"C:\Users\me\data   or   /home/me/data"
        )
        if folder:
            try:
                paths = dt.list_folder(folder)
                pick = st.selectbox(
                    "File", [p.name for p in paths], index=None,
                    placeholder="Choose a file"
                )
                path = next((p for p in paths if p.name == pick), None)
                sheet = (
                    st.selectbox(
                        "Sheet", dt.excel_sheets(path), index=None,
                        placeholder="First sheet if none chosen"
                    )
                    if path and dt.is_excel(pick) else None
                )
                if st.button(
                    "Load data", type="primary", use_container_width=True,
                    disabled=not pick
                ):
                    try:
                        st.session_state.data = dt.read_any(path, pick, sheet)
                        st.session_state.data_name = pick
                        if st.session_state.data.empty:
                            raise ValueError("The selected source contains no rows.")
                        st.session_state.data_id += 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not read the file: {e}")
            except Exception as e:
                st.warning(str(e))

    elif source == "Online URL":
        url = st.text_input(
            "Direct link to a CSV / Excel / JSON file",
            placeholder="https://.../data.csv"
        )
        if st.button(
            "Load data", type="primary", use_container_width=True, disabled=not url
        ):
            try:
                with st.spinner("Downloading..."):
                    st.session_state.data = dt.load_url(url)
                    st.session_state.data_name = url.split("/")[-1].split("?")[0] or url
                    st.session_state.data_id += 1
                st.rerun()
            except Exception as e:
                st.error(f"Could not load the URL: {e}")

    elif source == "Database":
        db = st.selectbox(
            "Database type", ["MySQL", "PostgreSQL", "SQL Server", "SQLite"],
            index=None, placeholder="Choose a database"
        )
        if db is None:
            vals = {}
        elif db == "SQLite":
            vals = {"path": st.text_input("Database file path", placeholder="example.db")}
        else:
            a, b = st.columns(2)
            vals = {
                "host": a.text_input("Host"),
                "port": b.text_input("Port"),
                "user": a.text_input("Username"),
                "password": b.text_input("Password", type="password"),
                "database": st.text_input("Database name"),
                "driver": (
                    st.text_input(
                        "ODBC driver",
                        placeholder="e.g. ODBC Driver 17 for SQL Server"
                    ) if db == "SQL Server" else ""
                ),
            }
        if db and st.button("Connect", use_container_width=True):
            try:
                eng = dt.database_engine(db, vals)
                st.session_state.db_tables = dt.get_tables(eng)
                st.session_state.db_engine = eng
                st.success(f"Connected - {len(st.session_state.db_tables)} tables found.")
            except Exception as e:
                st.session_state.db_engine = None
                st.error(f"Connection failed: {e}")

        if db and st.session_state.db_engine is not None:
            by_sql = st.checkbox("Write my own SELECT query")
            try:
                if by_sql:
                    sql = st.text_area(
                        "SQL (SELECT only)",
                        placeholder="SELECT * FROM table_name LIMIT 1000"
                    )
                    if st.button(
                        "Run query", type="primary", use_container_width=True,
                        disabled=not sql
                    ):
                        st.session_state.data = dt.run_query(st.session_state.db_engine, sql)
                        st.session_state.data_name = "SQL query"
                        st.session_state.data_id += 1
                        st.rerun()
                else:
                    table = st.selectbox(
                        "Table", st.session_state.db_tables, index=None,
                        placeholder="Choose a table"
                    )
                    if st.button(
                        "Load table", type="primary", use_container_width=True,
                        disabled=not table
                    ):
                        st.session_state.data = dt.load_table(
                            st.session_state.db_engine, table
                        )
                        st.session_state.data_name = table
                        st.session_state.data_id += 1
                        st.rerun()
            except Exception as e:
                st.error(str(e))

# ---- Apply the column and row selection ----
df = st.session_state.data
if df is not None and df.empty:
    # An empty table would break the row-range inputs, so drop it.
    st.session_state.data, df = None, None
    sb.warning("The loaded source has no rows. Choose another file or table.")
sel = None
if df is not None:
    with sb.expander("3 · Data selection", expanded=True):
        cols = st.multiselect(
            "Columns (empty = all)", df.columns.tolist(), key=k("cols")
        )
        work = df[cols] if cols else df
        n = len(work)
        a, b = st.columns(2)
        r0 = a.number_input(
            "First row", 1, n, value=None, placeholder="all", key=k("r0")
        )
        r1 = b.number_input(
            "Last row", 1, n, value=None, placeholder="all", key=k("r1")
        )
        r0 = 1 if r0 is None else r0
        r1 = n if r1 is None else r1
        if r0 > r1:
            st.warning("First row is after last row - they were swapped.")
            r0, r1 = r1, r0
        sel = work.iloc[int(r0) - 1:int(r1)]
        st.caption(f"{len(sel):,} rows × {len(sel.columns)} columns selected")

# ---- Main page ----
if st.session_state.get("dashboard_title"):
    st.title(st.session_state.dashboard_title)

if not isinstance(sel, pd.DataFrame):
    st.info("👈 Select **Data source** and browse files and folders in the sidebar.")
    st.stop()

st.caption(f"Source: **{st.session_state.data_name}**")
num_cols = numeric_columns(sel)
all_cols = sel.columns.tolist()

with st.container(key="kpi_row"):
    m = st.columns(4)
    m[0].metric("Rows selected", f"{len(sel):,}")
    m[1].metric("Columns", len(all_cols))
    m[2].metric("Numeric columns", len(num_cols))
    m[3].metric("Missing values", f"{int(sel.isna().sum().sum()):,}")

with st.expander("Preview selected data", expanded=False):
    st.dataframe(sel, use_container_width=True, height=300)
    st.download_button(
        "⬇ Download selection (CSV)", sel.to_csv(index=False).encode(),
        "selected_data.csv", "text/csv"
    )


# ---- 4. Missing data ----
with st.container(border=True):
    st.subheader("4 · Missing Data")
    st.caption("Find missing values, then fill, interpolate or drop them. Your original data is never modified.")
    st.write("Missing values by column")
    missing_summary = pd.DataFrame({"Missing count": sel.isna().sum(), "Missing %": (sel.isna().mean() * 100).round(2)})
    st.dataframe(missing_summary, use_container_width=True)
    missing_method = st.selectbox("Handle missing values", ["Show missing values only", "Fill numeric columns with mean", "Fill numeric columns with median", "Fill each column with its mode", "Fill with a value I enter", "Interpolate numeric columns", "Drop rows containing missing values", "Drop columns containing missing values"], index=None, placeholder="Choose a method", key=k("missing_method"))
    cleaned = sel.copy()
    if missing_method:
        fill_value = None
        if missing_method == "Fill with a value I enter":
            fill_value = st.text_input("Replacement value", value="0", key=k("missing_fill_value"))
        if missing_method == "Fill numeric columns with mean":
            cols = numeric_columns(cleaned)
            if cols: cleaned[cols] = cleaned[cols].fillna(cleaned[cols].mean())
        elif missing_method == "Fill numeric columns with median":
            cols = numeric_columns(cleaned)
            if cols: cleaned[cols] = cleaned[cols].fillna(cleaned[cols].median())
        elif missing_method == "Fill each column with its mode":
            for col in cleaned.columns:
                modes = cleaned[col].mode(dropna=True)
                if not modes.empty: cleaned[col] = cleaned[col].fillna(modes.iloc[0])
        elif missing_method == "Fill with a value I enter":
            if fill_value is not None:
                # Turn "0" into the number 0 where possible. Filling with the raw
                # text would convert numeric columns to text and break the steps below.
                parsed_value = pd.to_numeric(pd.Series([fill_value]), errors="coerce").iloc[0]
                replacement_value = parsed_value if pd.notna(parsed_value) else fill_value
                cleaned = cleaned.fillna(replacement_value)
        elif missing_method == "Interpolate numeric columns":
            cols = numeric_columns(cleaned)
            if cols: cleaned[cols] = cleaned[cols].apply(pd.to_numeric, errors="coerce").interpolate()
        elif missing_method == "Drop rows containing missing values":
            cleaned = cleaned.dropna(axis=0)
        elif missing_method == "Drop columns containing missing values":
            cleaned = cleaned.dropna(axis=1)
        st.write(f"Rows: {len(sel):,} → {len(cleaned):,} · Remaining missing values: {int(cleaned.isna().sum().sum()):,}")
        st.dataframe(cleaned, use_container_width=True)
        code_map = {
            "Show missing values only": "print(data.isna().sum())",
            "Fill numeric columns with mean": "columns = data.select_dtypes(include='number').columns\ndata[columns] = data[columns].fillna(data[columns].mean())",
            "Fill numeric columns with median": "columns = data.select_dtypes(include='number').columns\ndata[columns] = data[columns].fillna(data[columns].median())",
            "Fill each column with its mode": "for column in data.columns:\n    mode = data[column].mode(dropna=True)\n    if not mode.empty:\n        data[column] = data[column].fillna(mode.iloc[0])",
            "Fill with a value I enter": f"data = data.fillna({fill_value!r})",
            "Interpolate numeric columns": "columns = data.select_dtypes(include='number').columns\ndata[columns] = data[columns].interpolate()",
            "Drop rows containing missing values": "data = data.dropna(axis=0)",
            "Drop columns containing missing values": "data = data.dropna(axis=1)",
        }
        st.code("import pandas as pd\ndata = pd.read_csv('your_data.csv')\n" + code_map[missing_method] + "\nprint(data.isna().sum())", language="python")
        st.download_button("⬇ Download data after missing-value handling (CSV)", cleaned.to_csv(index=False).encode("utf-8"), "missing_data_handled.csv", "text/csv", key=k("download_missing"))
        use_cleaned_data = st.checkbox(
            "Use this handled data for normalization and the operations below",
            value=False,
            key=k("use_cleaned_data"),
            help="When selected, downstream operations use the handled preview instead of the original selected data. The source data remains unchanged.",
        )
        if use_cleaned_data:
            if isinstance(cleaned, pd.DataFrame):
                sel = cleaned.copy()
                num_cols = numeric_columns(sel)
                all_cols = sel.columns.tolist()
                st.success("Handled data is now being used by normalization and the operations below.")
            else:
                st.error("The handled data is not a valid table, so it cannot be used for downstream operations.")


# ---- 5. Data normalization ----
with st.container(border=True):
    st.subheader("5 · Data Normalization")
    st.caption("Scale or encode columns of the current data. Your original data is kept unchanged.")
    norm_cols = st.multiselect("Numeric columns to transform", num_cols, placeholder="Choose numeric columns", key=k("norm_cols"))
    norm_method = st.selectbox(
        "Method",
        ["Min–max normalization", "Divide by maximum value", "StandardScaler (mean 0, standard deviation 1)",
         "Mean removal / preprocessing.scale", "Binarizer (threshold)", "L1 normalization (row sums of absolute values = 1)",
         "One-hot encode categorical column(s)", "Label encode categorical column"],
        index=None, placeholder="Choose a method", key=k("norm_method")
    )
    if norm_method == "Binarizer (threshold)":
        threshold = st.number_input("Binarizer threshold", value=1.4, key=k("norm_threshold"))
    else:
        threshold = 1.4
    if norm_method and (norm_cols or norm_method in ("One-hot encode categorical column(s)", "Label encode categorical column")):
        try:
            normalized = sel.copy()
            if norm_method == "One-hot encode categorical column(s)":
                cat_cols = st.multiselect("Categorical columns to encode", [c for c in all_cols if not pd.api.types.is_numeric_dtype(sel[c])], key=k("norm_cat_cols"))
                if cat_cols:
                    normalized = pd.get_dummies(normalized, columns=cat_cols, dtype=int)
                    st.dataframe(normalized, use_container_width=True)
                    st.code("# Based on preprocessing.OneHotEncoder / one-hot encoding\nimport pandas as pd\ndata = pd.read_csv('your_data.csv')\ndata = pd.get_dummies(data, columns=" + repr(cat_cols) + ", dtype=int)\nprint(data.head())", language="python")
            elif norm_method == "Label encode categorical column":
                cat_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(sel[c])]
                if cat_cols:
                    cat_col = st.selectbox("Categorical column", cat_cols, key=k("norm_label_col"))
                    from sklearn.preprocessing import LabelEncoder
                    normalized[cat_col] = LabelEncoder().fit_transform(normalized[cat_col].astype(str))
                    st.dataframe(normalized, use_container_width=True)
                    st.code("# Based on sklearn.preprocessing.LabelEncoder\nimport pandas as pd\nfrom sklearn.preprocessing import LabelEncoder\ndata = pd.read_csv('your_data.csv')\ncolumn = " + repr(cat_col) + "\ndata[column] = LabelEncoder().fit_transform(data[column].astype(str))\nprint(data.head())", language="python")
                else:
                    st.info("No categorical columns are available to label encode.")
            else:
                X = normalized[norm_cols].apply(pd.to_numeric, errors="coerce")
                if norm_method == "Min–max normalization":
                    lo, hi = X.min(), X.max(); span = hi - lo
                    scaled = (X - lo) / span.replace(0, np.nan)
                    # A constant column has no spread, so it becomes 0 (blanks stay blank).
                    for c in span.index[span.eq(0)]:
                        scaled[c] = X[c] - lo[c]
                    normalized[norm_cols] = scaled
                    code = "from sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler(feature_range=(0, 1))\ndata[columns] = scaler.fit_transform(data[columns])"
                elif norm_method == "Divide by maximum value":
                    # Each chosen column is divided by its own maximum.
                    hi = X.max(skipna=True)
                    valid_cols = [c for c in norm_cols if pd.notna(hi[c]) and hi[c] != 0]
                    invalid_cols = [c for c in norm_cols if c not in valid_cols]
                    if valid_cols:
                        normalized[valid_cols] = X[valid_cols].div(hi[valid_cols])
                    if invalid_cols:
                        st.warning(
                            "Skipped these columns because division by their maximum is undefined "
                            "(maximum is zero or the column has no numeric values): "
                            + ", ".join(map(str, invalid_cols))
                            + ". Other valid columns were processed."
                        )
                    code = "# Divide each column by its maximum\ndata[columns] = data[columns] / data[columns].max()"
                elif norm_method == "StandardScaler (mean 0, standard deviation 1)":
                    from sklearn.preprocessing import StandardScaler
                    normalized[norm_cols] = StandardScaler().fit_transform(X)
                    code = "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ndata[columns] = scaler.fit_transform(data[columns])"
                elif norm_method == "Mean removal / preprocessing.scale":
                    from sklearn.preprocessing import scale
                    normalized[norm_cols] = scale(X)
                    code = "from sklearn import preprocessing\ndata[columns] = preprocessing.scale(data[columns])"
                elif norm_method == "Binarizer (threshold)":
                    from sklearn.preprocessing import Binarizer
                    normalized[norm_cols] = Binarizer(threshold=threshold).transform(X.fillna(0))
                    code = f"from sklearn.preprocessing import Binarizer\ndata[columns] = Binarizer(threshold={threshold!r}).transform(data[columns].fillna(0))"
                else:
                    from sklearn.preprocessing import normalize
                    normalized[norm_cols] = normalize(X.fillna(0), norm="l1")
                    code = "from sklearn import preprocessing\ndata[columns] = preprocessing.normalize(data[columns].fillna(0), norm='l1')"
                st.dataframe(normalized, use_container_width=True)
                st.code("import pandas as pd\ndata = pd.read_csv('your_data.csv')\ncolumns = " + repr(norm_cols) + "\n" + code + "\nprint(data.head())", language="python")
            st.download_button("⬇ Download transformed data (CSV)", normalized.to_csv(index=False).encode("utf-8"), "transformed_data.csv", "text/csv", key=k("download_normalized"))
        except Exception as e:
            st.error(f"Could not apply this preprocessing method: {e}")
    elif not num_cols:
        st.info("The selected data has no numeric columns. Categorical encoding can still be used when categorical columns are available.")

# ---- 6. Basic data operations in Python ----
with st.container(key="section_six", border=True):
    st.subheader("6 · Basic Data Operations in Python")
    st.caption("Try common Python operations with values you enter, and see the matching code.")
    topic = st.selectbox("Choose a topic", ["Decision", "Dictionary", "List", "Loops", "Math", "numbers", "PandasNumpyMatplot", "PreProcessing", "PythonBasic", "ReadWrite", "Tuple", "Visualization"], index=None, placeholder="Choose a topic", key=k("basic_topic"))
    if topic in ("List", "Tuple"):
        raw = st.text_input("Enter values separated by commas", placeholder="10, 20, apple, 40", key=k("basic_seq"))
        vals = [x.strip() for x in raw.split(",") if x.strip()]
        if topic == "List":
            op = st.selectbox("List operation", ["Create and inspect", "Append", "Insert", "Extend", "Remove value", "Pop by index", "Sort ascending (numeric)", "Reverse", "Count value", "Find value", "Slice list", "List comprehension: keep values containing text"], index=None, placeholder="Choose", key=k("list_op_v2"))
            extra = st.text_input("Value / search text", key=k("list_extra_v2"))
            idx = st.number_input("Index / slice start", min_value=0, step=1, value=0, key=k("list_idx_v2"))
            if op:
                result = vals.copy()
                try:
                    if op == "Append" and extra: result.append(extra)
                    elif op == "Insert" and extra: result.insert(int(idx), extra)
                    elif op == "Extend" and extra: result.extend([x.strip() for x in extra.split(",") if x.strip()])
                    elif op == "Remove value" and extra: result.remove(extra)
                    elif op == "Pop by index":
                        if not result:
                            st.warning("The list is empty; there is no item to pop.")
                        elif idx >= len(result):
                            st.warning(f"Index {int(idx)} is outside the list (valid indexes: 0 to {len(result) - 1}).")
                        else:
                            result.pop(int(idx))
                    elif op == "Sort ascending (numeric)": result = sorted(float(x) for x in result)
                    elif op == "Reverse": result.reverse()
                    elif op == "Slice list": result = result[int(idx):]
                    elif op == "List comprehension: keep values containing text": result = [x for x in result if extra.lower() in x.lower()]
                    st.write("Result:", result)
                    if op == "Count value": st.write("Count:", result.count(extra))
                    if op == "Find value": st.write("Index:", result.index(extra) if extra in result else "Value not found")
                    code_ops = {"Create and inspect":"print(items)\nprint(len(items))", "Append":f"items.append({extra!r})", "Insert":f"items.insert({int(idx)}, {extra!r})", "Extend":f"items.extend({[x.strip() for x in extra.split(',') if x.strip()]!r})", "Remove value":f"items.remove({extra!r})", "Pop by index":f"items.pop({int(idx)})", "Sort ascending (numeric)":"items = sorted(float(x) for x in items)", "Reverse":"items.reverse()", "Count value":f"print(items.count({extra!r}))", "Find value":f"print(items.index({extra!r}) if {extra!r} in items else 'Value not found')", "Slice list":f"print(items[{int(idx)}:])", "List comprehension: keep values containing text":f"items = [x for x in items if {extra.lower()!r} in x.lower()]"}
                    st.code("items = " + repr(vals) + "\n" + code_ops.get(op, "print(items)") + "\nprint(items)", language="python")
                except Exception as e: st.warning(f"Could not perform this operation: {e}")
        else:
            tup = tuple(vals)
            op = st.selectbox("Tuple operation", ["Create and inspect", "Access by index", "Count value", "Find index", "Slice tuple", "Unpack tuple"], index=None, placeholder="Choose", key=k("tuple_op_v2"))
            idx = st.number_input("Index / slice start", min_value=0, step=1, value=0, key=k("tuple_idx_v2"))
            target = st.text_input("Value to find/count", key=k("tuple_target_v2"))
            st.write("Tuple:", tup, "· Length:", len(tup))
            if op == "Access by index": st.write(tup[int(idx)] if idx < len(tup) else "Index is outside this tuple")
            elif op == "Count value": st.write("Count:", tup.count(target))
            elif op == "Find index": st.write("Index:", tup.index(target) if target in tup else "Value not found")
            elif op == "Slice tuple": st.write(tup[int(idx):])
            elif op == "Unpack tuple": st.write("Unpacking requires the same number of variables as tuple values:", len(tup))
            snippets = {"Create and inspect":"print(items)\nprint(len(items))", "Access by index":f"print(items[{int(idx)}])", "Count value":f"print(items.count({target!r}))", "Find index":f"print(items.index({target!r}) if {target!r} in items else 'Value not found')", "Slice tuple":f"print(items[{int(idx)}:])", "Unpack tuple":"# Example: a, b = items[:2]\nprint(items[:2])"}
            st.code("items = " + repr(tup) + "\n" + snippets.get(op, "print(items)"), language="python")
    elif topic == "Dictionary":
        raw = st.text_area("Enter one key=value pair per line", value="product=Notebook\nprice=120\nquantity=5", key=k("dict_raw_v2"))
        dct = {}
        for line in raw.splitlines():
            if "=" in line:
                key0, val0 = line.split("=", 1); dct[key0.strip()] = val0.strip()
        op = st.selectbox("Dictionary operation", ["Display", "Look up key", "Add/update key", "Delete key", "Show keys", "Show values", "Loop through items"], index=None, placeholder="Choose", key=k("dict_op_v2"))
        key0 = st.text_input("Key", key=k("dict_key_v2")); val0 = st.text_input("Value", key=k("dict_val_v2"))
        result = dct.copy()
        if op == "Add/update key" and key0: result[key0] = val0
        elif op == "Delete key" and key0: result.pop(key0, None)
        if op == "Look up key": st.write(result.get(key0, "Key not found"))
        elif op == "Show keys": st.write(list(result.keys()))
        elif op == "Show values": st.write(list(result.values()))
        elif op == "Loop through items": st.write([f"{a}: {b}" for a, b in result.items()])
        else: st.write(result)
        code = {"Display":"print(items)", "Look up key":f"print(items.get({key0!r}, 'Key not found'))", "Add/update key":f"items[{key0!r}] = {val0!r}\nprint(items)", "Delete key":f"items.pop({key0!r}, None)\nprint(items)", "Show keys":"print(items.keys())", "Show values":"print(items.values())", "Loop through items":"for key, value in items.items():\n    print(key, value)"}
        st.code("items = " + repr(dct) + "\n" + code.get(op, "print(items)"), language="python")
    elif topic in ("Math", "Decision"):
        a, b = st.columns(2); av = a.number_input("Number A", value=12.0, key=k("math_a_v2")); bv = b.number_input("Number B", value=3.0, key=k("math_b_v2"))
        if topic == "Math":
            op = st.selectbox("Math operation", ["Add", "Subtract", "Multiply", "Divide", "Power", "Remainder", "Floor division", "Square root of A", "Absolute value of A"], index=None, placeholder="Choose", key=k("math_op_v2"))
            if op:
                result = {"Add":av+bv,"Subtract":av-bv,"Multiply":av*bv,"Divide":av/bv if bv else "Cannot divide by zero","Power":av**bv,"Remainder":av%bv if bv else "Cannot use zero as divisor","Floor division":av//bv if bv else "Cannot use zero as divisor","Square root of A":np.sqrt(av) if av >= 0 else "Requires non-negative A","Absolute value of A":abs(av)}[op]
                st.write("Result:", result)
                symbols = {"Add":"+","Subtract":"-","Multiply":"*","Divide":"/","Power":"**","Remainder":"%","Floor division":"//"}
                code = f"a = {av}\nb = {bv}\nresult = a {symbols[op]} b" if op in symbols else f"import math\na = {av}\nresult = math.sqrt(a)" if op == "Square root of A" else f"a = {av}\nresult = abs(a)"
                st.code(code + "\nprint(result)", language="python")
        else:
            op = st.selectbox("Decision / comparison", ["A > B", "A >= B", "A < B", "A <= B", "A == B", "A != B", "A is positive", "A is even"], index=None, placeholder="Choose", key=k("decision_op_v2"))
            if op:
                result = {"A > B":av>bv,"A >= B":av>=bv,"A < B":av<bv,"A <= B":av<=bv,"A == B":av==bv,"A != B":av!=bv,"A is positive":av>0,"A is even":float(av).is_integer() and int(av)%2==0}[op]
                st.write("Result:", result)
                condition_code = {"A > B":"a > b", "A >= B":"a >= b", "A < B":"a < b", "A <= B":"a <= b", "A == B":"a == b", "A != B":"a != b", "A is positive":"a > 0", "A is even":"int(a) % 2 == 0"}[op]
                st.code(f"a = {av}\nb = {bv}\nprint({condition_code})", language="python")
    elif topic == "Loops":
        raw = st.text_input("Values to loop over (comma-separated)", placeholder="red, blue, green", key=k("loop_raw_v2")); vals = [x.strip() for x in raw.split(",") if x.strip()]
        loop_type = st.selectbox("Loop example", ["for loop with enumerate", "while loop countdown", "break when value matches", "continue to skip a value"], index=None, placeholder="Choose", key=k("loop_type_v2"))
        stop_value = st.text_input("Target value for break/continue", key=k("loop_target_v2"))
        if vals and loop_type:
            if loop_type == "for loop with enumerate": output = [f"{i}: {v}" for i,v in enumerate(vals)]; code = "for index, value in enumerate(values):\n    print(index, value)"
            elif loop_type == "while loop countdown": output = list(range(len(vals), 0, -1)); code = f"count = {len(vals)}\nwhile count > 0:\n    print(count)\n    count -= 1"
            elif loop_type == "break when value matches": output=[]
            else: output=[]
            if loop_type in ("break when value matches", "continue to skip a value"):
                for v in vals:
                    if loop_type == "break when value matches" and v == stop_value: break
                    if loop_type == "continue to skip a value" and v == stop_value: continue
                    output.append(v)
                loop_action = "break" if loop_type == "break when value matches" else "continue"
                code = f"target = {stop_value!r}\nfor value in values:\n    if value == target:\n        {loop_action}\n    print(value)"
            st.write("Output:", output); st.code(f"values = {vals!r}\n" + code, language="python")
    elif topic == "numbers":
        raw = st.text_input("Enter numbers separated by commas", placeholder="2, 4, 6, 8", key=k("number_raw_v2"))
        try:
            vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
            op = st.selectbox("Number operation", ["Display", "Sum", "Minimum", "Maximum", "Mean", "Median", "Range", "Variance", "Standard deviation", "Sort", "Unique values"], index=None, placeholder="Choose", key=k("number_op_v2"))
            if vals and op:
                arr=np.array(vals); result={"Display":vals,"Sum":float(arr.sum()),"Minimum":float(arr.min()),"Maximum":float(arr.max()),"Mean":float(arr.mean()),"Median":float(np.median(arr)),"Range":float(arr.max()-arr.min()),"Variance":float(arr.var()),"Standard deviation":float(arr.std()),"Sort":sorted(vals),"Unique values":list(dict.fromkeys(vals))}[op]
                st.write("Result:", result)
                number_code = {"Display":"print(numbers)", "Sum":"print(sum(numbers))", "Minimum":"print(min(numbers))", "Maximum":"print(max(numbers))", "Mean":"print(np.mean(arr))", "Median":"print(np.median(arr))", "Range":"print(max(numbers) - min(numbers))", "Variance":"print(np.var(arr))", "Standard deviation":"print(np.std(arr))", "Sort":"print(sorted(numbers))", "Unique values":"print(list(dict.fromkeys(numbers)))"}[op]
                st.code(f"import numpy as np\nnumbers = {vals!r}\narr = np.array(numbers)\n" + number_code, language="python")
        except ValueError: st.warning("Enter numeric values separated by commas.")
    elif topic == "PythonBasic":
        st.markdown("Variables, strings, type conversion, arithmetic and input.")
        text_value = st.text_input("Text input", key=k("pb_text")); number_value = st.number_input("Numeric input", value=0.0, key=k("pb_number"))
        st.write({"text":text_value,"uppercase":text_value.upper(),"character_count":len(text_value),"number":number_value,"number_as_integer_if_whole":int(number_value) if number_value.is_integer() else number_value})
        st.code(f"text = {text_value!r}\nnumber = {number_value}\nprint(text.upper())\nprint(len(text))\nprint(number)", language="python")
    elif topic == "PandasNumpyMatplot":
        st.markdown("Pandas and NumPy operations applied to your selected data.")
        op = st.selectbox("Operation", ["DataFrame info and describe", "Select columns", "Filter rows by numeric threshold", "Sort by column", "NumPy array summary", "Correlation matrix", "Group by category and mean"], index=None, placeholder="Choose", key=k("pandas_op_v2"))
        if op == "DataFrame info and describe": st.dataframe(sel.describe(include="all").transpose(), use_container_width=True); st.code("print(data.info())\nprint(data.describe(include='all'))", language="python")
        elif op == "Select columns":
            cols=st.multiselect("Columns", all_cols, key=k("pandas_cols_v2"))
            if cols: st.dataframe(sel[cols], use_container_width=True); st.code(f"print(data[{cols!r}])", language="python")
        elif op == "Filter rows by numeric threshold":
            col=st.selectbox("Numeric column", num_cols, key=k("pandas_filter_col")); threshold=st.number_input("Keep rows where value is at least", value=0.0, key=k("pandas_threshold"))
            filtered=sel[pd.to_numeric(sel[col],errors="coerce")>=threshold] if col else sel.iloc[0:0]; st.dataframe(filtered,use_container_width=True); st.code(f"filtered = data[data[{col!r}] >= {threshold}]\nprint(filtered)",language="python")
        elif op == "Sort by column":
            col=st.selectbox("Sort column", all_cols, key=k("pandas_sort_col")); asc=st.checkbox("Ascending",value=True,key=k("pandas_ascending")); sorted_df=sel.sort_values(col,ascending=asc); st.dataframe(sorted_df,use_container_width=True); st.code(f"data = data.sort_values({col!r}, ascending={asc})",language="python")
        elif op == "NumPy array summary":
            cols=st.multiselect("Numeric columns", num_cols, key=k("np_cols_v2"))
            if cols:
                arr=sel[cols].apply(pd.to_numeric,errors="coerce").to_numpy(); st.write("Shape",arr.shape,"Mean",np.nanmean(arr,axis=0),"Std",np.nanstd(arr,axis=0)); st.code(f"import numpy as np\narr = data[{cols!r}].to_numpy()\nprint(arr.shape)\nprint(np.nanmean(arr, axis=0))",language="python")
        elif op == "Correlation matrix":
            if len(num_cols)>=2: st.dataframe(sel[num_cols].corr(numeric_only=True),use_container_width=True); st.code("print(data.corr(numeric_only=True))",language="python")
            else: st.info("At least two numeric columns are required.")
        elif op == "Group by category and mean":
            cat_cols=[c for c in all_cols if c not in num_cols]; cat=st.selectbox("Category column",cat_cols,key=k("group_cat")); metric=st.selectbox("Numeric column",num_cols,key=k("group_metric"))
            if cat and metric: grouped=sel.groupby(cat,dropna=False)[metric].mean().reset_index(); st.dataframe(grouped,use_container_width=True); st.code(f"print(data.groupby({cat!r})[{metric!r}].mean())",language="python")
    elif topic == "ReadWrite":
        st.markdown("Read/write examples adapted for uploaded or selected data. No local sample file is required.")
        st.dataframe(sel.head(20), use_container_width=True)
        st.download_button("Write selected data to CSV", sel.to_csv(index=False).encode("utf-8"), "data_export.csv", "text/csv", key=k("readwrite_csv"))
        st.code("import pandas as pd\ndata = pd.read_csv('your_data.csv')\nprint(data.head())\ndata.to_csv('output.csv', index=False)", language="python")
    elif topic == "PreProcessing":
        st.markdown("Binarization, scaling, L1 normalization and encoders. More normalization options are in section 5 above.")
        pre_cols=st.multiselect("Numeric columns",num_cols,key=k("pre_cols")); pre_op=st.selectbox("Operation",["Binarizer", "MinMaxScaler with custom range", "L1 normalize rows", "One-hot encode categorical columns", "Label encode one categorical column"],index=None,placeholder="Choose",key=k("pre_op"))
        if pre_op == "Binarizer" and pre_cols:
            threshold=st.number_input("Threshold",value=1.4,key=k("pre_bin_threshold")); from sklearn.preprocessing import Binarizer; out=sel.copy(); out[pre_cols]=Binarizer(threshold=threshold).transform(out[pre_cols].fillna(0)); st.dataframe(out,use_container_width=True); st.code(f"from sklearn.preprocessing import Binarizer\ndata[{pre_cols!r}] = Binarizer(threshold={threshold}).transform(data[{pre_cols!r}].fillna(0))",language="python")
        elif pre_op == "MinMaxScaler with custom range" and pre_cols:
            lo,hi=st.columns(2); low=lo.number_input("Output minimum",value=0.0,key=k("pre_lo")); high=hi.number_input("Output maximum",value=1.0,key=k("pre_hi"))
            if high <= low:
                st.warning("Output maximum must be greater than output minimum.")
            else:
                try:
                    from sklearn.preprocessing import MinMaxScaler
                    out = sel.copy()
                    numeric = out[pre_cols].apply(pd.to_numeric, errors="coerce")
                    if np.isinf(numeric.to_numpy(dtype=float)).any():
                        st.error("Selected columns contain positive or negative infinity. Replace those values before scaling.")
                    else:
                        out[pre_cols] = MinMaxScaler(feature_range=(low, high)).fit_transform(numeric)
                        st.dataframe(out, use_container_width=True)
                        st.code(f"from sklearn.preprocessing import MinMaxScaler\ndata[{pre_cols!r}] = MinMaxScaler(feature_range=({low}, {high})).fit_transform(data[{pre_cols!r}])", language="python")
                except Exception as e:
                    st.error(f"Min–max scaling failed: {e}")
        elif pre_op == "L1 normalize rows" and pre_cols:
            from sklearn.preprocessing import normalize; out=sel.copy(); out[pre_cols]=normalize(out[pre_cols].fillna(0),norm="l1"); st.dataframe(out,use_container_width=True); st.code(f"from sklearn.preprocessing import normalize\ndata[{pre_cols!r}] = normalize(data[{pre_cols!r}].fillna(0), norm='l1')",language="python")
        elif pre_op in ("One-hot encode categorical columns", "Label encode one categorical column"):
            cat_cols=[c for c in all_cols if c not in num_cols]
            if pre_op == "One-hot encode categorical columns":
                selected_cat=st.multiselect("Categorical columns",cat_cols,key=k("pre_ohe_cols"))
                if selected_cat: out=pd.get_dummies(sel,columns=selected_cat,dtype=int); st.dataframe(out,use_container_width=True); st.code(f"data = pd.get_dummies(data, columns={selected_cat!r}, dtype=int)",language="python")
            elif cat_cols:
                cat=st.selectbox("Column to encode",cat_cols,key=k("pre_label_col")); from sklearn.preprocessing import LabelEncoder; out=sel.copy(); out[cat]=LabelEncoder().fit_transform(out[cat].astype(str)); st.dataframe(out,use_container_width=True); st.code(f"from sklearn.preprocessing import LabelEncoder\ndata[{cat!r}] = LabelEncoder().fit_transform(data[{cat!r}].astype(str))",language="python")
    elif topic == "Visualization":
        st.markdown("Basic, statistical and array plots in Matplotlib, Seaborn, Plotly, Bokeh or Pygal. Choose a library and a plot type; plots use your selected data or the values you enter.")
        library = st.selectbox("Visualization library", ["Matplotlib", "Seaborn", "Plotly", "Bokeh", "Pygal"], index=0, key=k("viz_library"))
        plot_type = st.selectbox("Plot type", ["Line plot", "Bar chart", "Scatter plot", "Histogram", "Box plot", "Pie chart", "Violin plot", "Stack plot", "Step plot", "Stem plot", "Fill-between plot", "Error-bar plot", "Hexbin plot", "2D histogram", "Correlation heatmap", "Pair plot"], index=None, placeholder="Choose a plot", key=k("viz_plot_type"))
        viz_num = st.multiselect("Numeric column(s)", num_cols, default=num_cols[:2] if len(num_cols)>=2 else num_cols[:1], key=k("viz_num_cols"))
        viz_cat = st.selectbox("Optional category column (for bar/pie)", [None] + [c for c in all_cols if c not in num_cols], key=k("viz_cat_col"))
        if plot_type and viz_num:
            try:
                plot_df=sel.copy(); xcol=viz_num[0]; ycol=viz_num[1] if len(viz_num)>1 else None
                if plot_type in ("Bar chart","Pie chart") and viz_cat:
                    plot_data=plot_df.groupby(viz_cat,dropna=False)[xcol].sum().reset_index().head(30); xvals=plot_data[viz_cat].astype(str).tolist(); yvals=plot_data[xcol].tolist(); xlabel=viz_cat
                else:
                    plot_data=plot_df[viz_num].apply(pd.to_numeric,errors="coerce").dropna(); xvals=plot_data[xcol].tolist(); yvals=plot_data[ycol].tolist() if ycol else None; xlabel=xcol
                if library == "Matplotlib":
                    import matplotlib.pyplot as plt
                    fig,ax=plt.subplots(figsize=(9,5))
                    if plot_type == "Line plot": ax.plot(xvals,yvals if yvals is not None else xvals,marker="o")
                    elif plot_type == "Bar chart": ax.bar(xvals,yvals if yvals is not None else xvals) if yvals is not None else ax.bar(range(len(xvals)),xvals)
                    elif plot_type == "Scatter plot": ax.scatter(xvals,yvals if yvals is not None else range(len(xvals)))
                    elif plot_type == "Histogram": ax.hist(xvals,bins="auto",edgecolor="black")
                    elif plot_type == "Box plot": ax.boxplot([xvals]+([yvals] if yvals is not None else []),labels=[xcol]+([ycol] if ycol else []))
                    elif plot_type == "Pie chart": ax.pie(yvals if yvals is not None else xvals,labels=xvals if viz_cat else None,autopct="%1.1f%%")
                    elif plot_type == "Violin plot": ax.violinplot([xvals]+([yvals] if yvals is not None else []),showmeans=True)
                    elif plot_type == "Stack plot": ax.stackplot(range(len(xvals)),xvals,yvals if yvals is not None else xvals,labels=[xcol,ycol or xcol]); ax.legend()
                    elif plot_type == "Step plot": ax.step(range(len(xvals)),xvals,where="mid")
                    elif plot_type == "Stem plot": ax.stem(range(len(xvals)),xvals)
                    elif plot_type == "Fill-between plot": ax.fill_between(range(len(xvals)),xvals,alpha=.4)
                    elif plot_type == "Error-bar plot": ax.errorbar(range(len(xvals)),xvals,yerr=np.std(xvals) if len(xvals)>1 else 0,fmt="o")
                    elif plot_type == "Hexbin plot": ax.hexbin(xvals,yvals if yvals is not None else range(len(xvals)),gridsize=20,mincnt=1)
                    elif plot_type == "2D histogram": ax.hist2d(xvals,yvals if yvals is not None else range(len(xvals)),bins=20)
                    elif plot_type == "Correlation heatmap":
                        corr=plot_df[viz_num].corr(numeric_only=True); im=ax.imshow(corr,aspect="auto"); ax.set_xticks(range(len(corr.columns)),corr.columns,rotation=45,ha="right"); ax.set_yticks(range(len(corr.index)),corr.index); fig.colorbar(im,ax=ax)
                    elif plot_type == "Pair plot":
                        plt.close(fig); import seaborn as sns; pair=sns.pairplot(plot_df[viz_num].dropna().sample(min(500,len(plot_df.dropna())),random_state=1) if len(plot_df.dropna())>500 else plot_df[viz_num].dropna()); st.pyplot(pair.figure); plt.close(pair.figure); pair=None
                    if plot_type != "Pair plot":
                        ax.set_title(plot_type); ax.set_xlabel(xlabel); ax.grid(alpha=.2)
                        # Show a legend when the chart contains multiple named data series.
                        if plot_type in ("Line plot", "Scatter plot") and ycol:
                            ax.legend([xcol, ycol], loc="best", frameon=True, framealpha=.85)
                        elif plot_type == "Stack plot":
                            ax.legend(loc="best", frameon=True, framealpha=.85)
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                    st.code("import matplotlib.pyplot as plt\n# values are taken from your selected data\n" + {"Line plot":"plt.plot(data[x_column], data[y_column], marker='o')", "Bar chart":"plt.bar(data[x_column], data[y_column])", "Scatter plot":"plt.scatter(data[x_column], data[y_column])", "Histogram":"plt.hist(data[x_column], bins='auto')", "Box plot":"plt.boxplot([data[x_column], data[y_column]])", "Pie chart":"plt.pie(data[y_column], labels=data[x_column], autopct='%1.1f%%')", "Violin plot":"plt.violinplot([data[x_column], data[y_column]])", "Stack plot":"plt.stackplot(range(len(data)), data[x_column], data[y_column])", "Step plot":"plt.step(range(len(data)), data[x_column], where='mid')", "Stem plot":"plt.stem(range(len(data)), data[x_column])", "Fill-between plot":"plt.fill_between(range(len(data)), data[x_column])", "Error-bar plot":"plt.errorbar(range(len(data)), data[x_column], yerr=data[x_column].std(), fmt='o')", "Hexbin plot":"plt.hexbin(data[x_column], data[y_column], gridsize=20)", "2D histogram":"plt.hist2d(data[x_column], data[y_column], bins=20)", "Correlation heatmap":"plt.imshow(data[numeric_columns].corr())", "Pair plot":"import seaborn as sns\nsns.pairplot(data[numeric_columns])"}[plot_type],language="python")
                elif library == "Seaborn":
                    import seaborn as sns; import matplotlib.pyplot as plt
                    fig,ax=plt.subplots(figsize=(9,5))
                    if plot_type == "Histogram": sns.histplot(data=plot_df,x=xcol,kde=True,ax=ax)
                    elif plot_type == "Box plot": sns.boxplot(data=plot_df[viz_num],ax=ax)
                    elif plot_type == "Violin plot": sns.violinplot(data=plot_df[viz_num],ax=ax)
                    elif plot_type == "Correlation heatmap": sns.heatmap(plot_df[viz_num].corr(numeric_only=True),annot=True,cmap="coolwarm",ax=ax)
                    elif plot_type == "Pair plot":
                        plt.close(fig); pair=sns.pairplot(plot_df[viz_num].dropna().sample(min(500,len(plot_df.dropna())),random_state=1) if len(plot_df.dropna())>500 else plot_df[viz_num].dropna()); st.pyplot(pair.figure); plt.close(pair.figure); pair=None
                    elif plot_type == "Bar chart": sns.barplot(data=plot_df,x=viz_cat if viz_cat else xcol,y=xcol if viz_cat else (ycol or xcol),ax=ax)
                    elif plot_type == "Scatter plot" and ycol: sns.scatterplot(data=plot_df,x=xcol,y=ycol,ax=ax)
                    else: sns.lineplot(data=plot_df[viz_num],ax=ax)
                    if plot_type != "Pair plot":
                        ax.set_title(plot_type)
                        if plot_type in ("Line plot", "Scatter plot") and ycol:
                            ax.legend([xcol, ycol], loc="best", frameon=True, framealpha=.85)
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                    st.code(f"import seaborn as sns\nsns.{ {'Histogram':'histplot','Box plot':'boxplot','Violin plot':'violinplot','Correlation heatmap':'heatmap','Pair plot':'pairplot','Bar chart':'barplot','Scatter plot':'scatterplot'}.get(plot_type,'lineplot') }(data={('data['+repr(viz_num)+']') if plot_type in ('Box plot','Violin plot','Correlation heatmap','Pair plot') else 'data'}, x={repr(xcol)}, y={repr(ycol) if ycol else 'None'})",language="python")
                elif library == "Plotly":
                    import plotly.express as px
                    if plot_type == "Histogram": fig=px.histogram(plot_df,x=xcol)
                    elif plot_type == "Box plot": fig=px.box(plot_df,y=xcol)
                    elif plot_type == "Violin plot": fig=px.violin(plot_df,y=xcol)
                    elif plot_type == "Pie chart" and viz_cat: fig=px.pie(plot_data,names=viz_cat,values=xcol)
                    elif plot_type == "Bar chart": fig=px.bar(plot_data,x=xlabel,y=xcol) if viz_cat else px.bar(plot_df,x=xcol,y=ycol or xcol)
                    elif plot_type == "Scatter plot" and ycol: fig=px.scatter(plot_df,x=xcol,y=ycol)
                    elif plot_type == "Correlation heatmap": fig=px.imshow(plot_df[viz_num].corr(numeric_only=True),text_auto=True)
                    elif plot_type == "Pair plot": fig=px.scatter_matrix(plot_df[viz_num].dropna())
                    elif plot_type == "Step plot": fig=px.line(plot_df,y=xcol,line_shape="hv")
                    else: fig=px.line(plot_df,x=xcol,y=ycol or xcol)
                    st.plotly_chart(fig,use_container_width=True)
                    st.code(f"import plotly.express as px\nfig = px.{ {'Histogram':'histogram','Box plot':'box','Violin plot':'violin','Pie chart':'pie','Bar chart':'bar','Scatter plot':'scatter','Correlation heatmap':'imshow','Pair plot':'scatter_matrix','Step plot':'line'}.get(plot_type,'line') }(data, x={repr(xcol)}, y={repr(ycol) if ycol else 'None'})\nfig.show()",language="python")
                elif library == "Pygal":
                    import pygal
                    chart = None
                    if plot_type == "Pie chart":
                        chart = pygal.Pie(height=360, width=700, title="Pie chart (Pygal)")
                        for label, value in zip(xvals, yvals if yvals is not None else xvals): chart.add(str(label), float(value))
                    elif plot_type == "Histogram":
                        chart = pygal.Bar(height=360, width=700, title="Histogram (Pygal)")
                        counts, edges = np.histogram(xvals, bins="auto")
                        for i, count in enumerate(counts): chart.add(f"{edges[i]:.3g}–{edges[i+1]:.3g}", int(count))
                    elif plot_type == "Bar chart":
                        chart = pygal.Bar(height=360, width=700, title="Bar chart (Pygal)")
                        for label, value in zip(xvals, yvals if yvals is not None else xvals): chart.add(str(label), float(value))
                    else:
                        chart = pygal.Line(height=360, width=700, title=f"{plot_type} (Pygal)")
                        chart.add(xcol, xvals)
                        if yvals is not None: chart.add(ycol, yvals)
                    st.components.v1.html(chart.render(is_unicode=True), height=410, scrolling=True)
                    st.code("import pygal\nchart = pygal.Bar()\nchart.add('Series', values)\nchart.render_to_file('chart.svg')", language="python")
                else:
                    from bokeh.plotting import figure
                    from bokeh.models import ColumnDataSource
                    from bokeh.palettes import Category10
                    from bokeh.transform import cumsum
                    from math import pi
                    p=figure(height=420,width=800,title=f"{plot_type} (Bokeh)",x_axis_label=xlabel,toolbar_location="above")
                    xx=list(range(len(xvals))); yy=yvals if yvals is not None else xvals
                    if plot_type in ("Line plot","Step plot"): p.line(xx,yy,line_width=2); p.circle(xx,yy,size=6)
                    elif plot_type == "Bar chart": p.vbar(x=xx,top=yy,width=.8)
                    elif plot_type == "Scatter plot": p.scatter(x=xx,y=yy,size=8)
                    elif plot_type == "Histogram": hist,edges=np.histogram(xvals,bins="auto"); p.quad(top=hist,bottom=0,left=edges[:-1],right=edges[1:],alpha=.7)
                    elif plot_type == "Pie chart":
                        counts=np.array(yy,dtype=float); counts=np.abs(counts); total=counts.sum() or 1; angles=counts/total*2*pi; starts=np.cumsum(np.r_[0,angles[:-1]]); p.wedge(x=0,y=0,radius=.8,start_angle=starts,end_angle=starts+angles,color=Category10[10][:len(counts)],legend_label=[str(x) for x in xvals])
                    elif plot_type == "Box plot": p.vbar(x=[1],top=[np.nanpercentile(xvals,75)],bottom=[np.nanpercentile(xvals,25)],width=.5); p.segment(x0=[.75],y0=[np.median(xvals)],x1=[1.25],y1=[np.median(xvals)],line_width=3)
                    elif plot_type == "Violin plot": p.line(np.linspace(min(xvals),max(xvals),len(xvals)),sorted(xvals),line_width=2)
                    elif plot_type in ("Hexbin plot","2D histogram") and yvals is not None: p.scatter(x=xvals,y=yvals,size=8,alpha=.5)
                    else: p.line(xx,yy,line_width=2)
                    st.bokeh_chart(p,use_container_width=True)
                    st.code(f"from bokeh.plotting import figure, show\np = figure(title={plot_type!r})\n# values are taken from your selected data\np.line(range(len(data)), data[{xcol!r}])\nshow(p)",language="python")
            except Exception as e:
                st.error(f"Could not create this plot with {library}: {e}")

# ---- 7. Visualization ----
with st.container(border=True, key="ust_section_7"):
    st.subheader("7 · Visualization")
    left, right = st.columns([1, 2.2], gap="large")

    with left:
        graph_title = st.text_input(
            "Graph title", key=k("gtitle"),
            placeholder="Title shown above the graph"
        )
        kind = st.selectbox(
            "Visualization", CHARTS, index=None,
            placeholder="Choose a visualization", key=k("kind")
        )
        x = ys = col = heat = None
        if kind in NEEDS_XY:
            x = st.selectbox(
                "X axis", all_cols, index=None, placeholder="Choose X", key=k("x")
            )
            ys = st.multiselect(
                "Y axis (one or more numeric)", num_cols,
                placeholder="Choose Y", key=k("y")
            )
        elif kind in NEEDS_ONE:
            col = st.selectbox(
                "Column",
                all_cols if kind in ("Pie chart", "Count plot") else num_cols,
                index=None, placeholder="Choose a column", key=k("col")
            )
        elif kind:
            heat = st.multiselect(
                "Numeric columns", num_cols,
                placeholder="Choose 2 or more", key=k("heat")
            )

        with st.expander("Axis range & style"):
            xa, xb = st.columns(2)
            x_min = xa.number_input("X min", value=None, placeholder="auto", key=k("xmin"))
            x_max = xb.number_input("X max", value=None, placeholder="auto", key=k("xmax"))
            ya, yb = st.columns(2)
            y_min = ya.number_input("Y min", value=None, placeholder="auto", key=k("ymin"))
            y_max = yb.number_input("Y max", value=None, placeholder="auto", key=k("ymax"))
            grid = st.checkbox("Grid", key=k("grid"))
            palette = st.selectbox(
                "Colours",
                ["teal", "deep", "muted", "bright", "pastel", "dark", "colorblind"],
                index=None, placeholder="Automatic", key=k("pal")
            ) or "teal"

    with right:
        ready = {
            "xy": kind in NEEDS_XY and x and ys,
            "one": kind in NEEDS_ONE and col,
            "heat": (
                kind and kind not in NEEDS_XY | NEEDS_ONE and
                heat and len(heat) >= 2
            ),
        }
        if not any(ready.values()):
            st.info("Choose a visualization and its columns on the left.")
        elif (
            (x_min is not None and x_max is not None and x_min >= x_max) or
            (y_min is not None and y_max is not None and y_min >= y_max)
        ):
            st.warning("An axis minimum must be smaller than its maximum.")
        else:
            try:
                fig = make_chart(
                    sel, kind, x=x, ys=ys, col=col, heat_cols=heat,
                    title=graph_title, xlim=(x_min, x_max), ylim=(y_min, y_max),
                    grid=grid, palette=palette
                )
                st.pyplot(fig)
                base_name = (graph_title.strip() or kind).replace(" ", "_")
                d = st.columns(3)
                for c, fmt in zip(d, ("png", "pdf", "svg")):
                    c.download_button(
                        f"⬇ {fmt.upper()}", fig_bytes(fig, fmt),
                        f"{base_name}.{fmt}", MIME[fmt],
                        use_container_width=True, key=k(f"dl_{fmt}")
                    )

                st.session_state.last_chart_png = fig_bytes(fig, "png")
                st.session_state.last_chart_title = graph_title.strip() or kind
                plt.close(fig)
            except Exception as e:
                st.error(f"Could not draw this chart: {e}")

# ---- 8. Statistical parameters ----
with st.container(border=True):
    st.subheader("8 · Statistical parameters")
    if not num_cols:
        st.warning("The selected data has no numeric columns.")
    else:
        a, b = st.columns(2)
        s_cols = a.multiselect(
            "Columns", num_cols, placeholder="Choose columns", key=k("scols")
        )
        s_par = b.multiselect(
            "Parameters", list(PARAMETERS),
            placeholder="Choose parameters", key=k("spar")
        )
        if s_cols and s_par:
            table = summary_table(sel, s_cols, s_par)
            st.dataframe(
                table.style.format(precision=4, na_rep="-"),
                use_container_width=True
            )
            st.download_button(
                "⬇ Download statistics (CSV)", table.to_csv().encode(),
                "statistics.csv", "text/csv"
            )
            st.session_state.last_stats = table
        else:
            st.info("Choose at least one column and one parameter.")

# ---- 9. Correlation amongst columns ----
with st.container(border=True):
    st.subheader("9 · Correlation amongst columns")
    st.caption("Choose two or more numeric columns to calculate Pearson correlation.")
    if len(num_cols) < 2:
        st.warning("At least two numeric columns are required.")
    else:
        corr_cols = st.multiselect(
            "Columns for correlation", num_cols,
            placeholder="Choose 2 or more", key=k("corrcols")
        )
        if len(corr_cols) >= 2:
            corr = correlation_table(sel, corr_cols)
            st.dataframe(
                corr.style.format(precision=3, na_rep="-"),
                use_container_width=True
            )
            st.download_button(
                "⬇ Download correlation (CSV)",
                corr.to_csv().encode(), "correlation.csv", "text/csv"
            )
            st.session_state.last_corr = corr
        else:
            st.info("Choose at least two numeric columns.")

# ---- 10. Regression  ----
lab.regression_models_section(sel, num_cols, k)

# ---- 11. Probability density ----
with st.container(border=True, key="ust_section_11"):
    st.subheader("11 · Probability density")
    if not num_cols:
        st.warning("The selected data has no numeric columns.")
    else:
        left, right = st.columns([1, 2.2], gap="large")
        values, label = None, ""
        with left:
            mode = st.radio(
                "Take the sample from a", ["Column", "Row"],
                index=None, horizontal=True, key=k("pmode")
            )
            if mode == "Column":
                pcol = st.selectbox(
                    "Column", num_cols, index=None,
                    placeholder="Choose a column", key=k("pcol")
                )
                if pcol:
                    values = pdf_samples(sel, "Column", pcol)
                    label = pcol
            elif mode == "Row":
                prow = st.number_input(
                    "Row number", 1, len(sel), value=None,
                    placeholder="Type a row number", key=k("prow")
                )
                pcols = st.multiselect(
                    "Columns to include", num_cols,
                    placeholder="Choose columns", key=k("prcols")
                )
                if prow and pcols:
                    values = pdf_samples(
                        sel, "Row", prow, columns=pcols
                    )
                    label = f"row {int(prow)}"

            method = st.selectbox(
                "Density",
                ["Kernel density estimate", "Normal distribution PDF", "Both"],
                index=None, placeholder="Choose a density", key=k("pmethod")
            )
            ptitle = st.text_input(
                "Density graph title", key=k("ptitle")
            )

        with right:
            if values is None or method is None:
                st.info("Choose a column or row, and a density, on the left.")
            elif len(values) < 3:
                st.warning("At least 3 numeric values are needed to estimate a density.")
            elif values.std(ddof=1) == 0:
                st.warning("All values are identical (standard deviation 0) - no density curve.")
            else:
                x_g, normal, kde, mu, sigma, _ = density_curves(values)
                fig = density_figure(
                    values, x_g, normal, kde, label, method, mu, sigma, ptitle
                )
                st.pyplot(fig)

                base_name = (
                    ptitle.strip() or f"probability_density_{label}"
                ).replace(" ", "_")
                d = st.columns(4)
                for c, fmt in zip(d[:3], ("png", "pdf", "svg")):
                    c.download_button(
                        f"⬇ {fmt.upper()}", fig_bytes(fig, fmt),
                        f"{base_name}.{fmt}", MIME[fmt],
                        use_container_width=True, key=k(f"pdl_{fmt}")
                    )
                curve = pd.DataFrame({
                    "x": x_g, "normal_pdf": normal, "kde_pdf": kde
                })
                d[3].download_button(
                    "⬇ Values (CSV)", curve.to_csv(index=False).encode(),
                    f"{base_name}_values.csv", "text/csv",
                    use_container_width=True, key=k("pdl_csv")
                )

                st.session_state.last_density_png = fig_bytes(fig, "png")
                st.session_state.last_density_title = ptitle.strip() or f"Probability density - {label}"
                plt.close(fig)

# ---- 12. Probability distributions ----
lab.distributions_section(sel, num_cols, k)

# ---- 13. Hypothesis testing (last analysis section) ----
with st.container(key="section_hypothesis", border=True):
    st.subheader("13 · Hypothesis testing")
    if not num_cols:
        st.warning("Numeric columns are required for hypothesis testing.")
    else:
        a, b, c = st.columns(3)
        test_name = a.selectbox(
            "Test", [
                "One-sample t-test",
                "Independent two-sample t-test",
                "Paired t-test",
                "Pearson correlation significance"
            ],
            index=None, placeholder="Choose a test", key=k("htest")
        )
        col1 = b.selectbox(
            "First numeric column", num_cols, index=None,
            placeholder="Choose column", key=k("hcol1")
        )
        col2 = None
        if test_name in {
            "Independent two-sample t-test",
            "Paired t-test",
            "Pearson correlation significance"
        }:
            col2 = c.selectbox(
                "Second numeric column", num_cols, index=None,
                placeholder="Choose column", key=k("hcol2")
            )
        else:
            null_value = c.number_input(
                "Hypothesized mean (H₀)", value=0.0, key=k("hnull")
            )

        alpha = st.selectbox(
            "Significance level (α)", [0.01, 0.05, 0.10],
            index=1, key=k("halpha")
        )
        ready_h = bool(test_name and col1 and (
            (test_name == "One-sample t-test") or col2
        ))
        if ready_h:
            try:
                result = hypothesis_test(
                    sel, test_name, col1, col2,
                    null_value if test_name == "One-sample t-test" else 0.0,
                    alpha
                )
                h1, h2, h3 = st.columns(3)
                h1.metric("Test statistic", f"{result['statistic']:.4g}")
                h2.metric("p-value", f"{result['p_value']:.4g}")
                h3.metric("Decision", result["decision"])
                st.markdown(f"**H₀:** {result['h0']}")
                st.markdown(f"**H₁:** {result['ha']}")
                st.session_state.last_hypothesis = result
            except Exception as e:
                st.error(f"Could not run the hypothesis test: {e}")
        else:
            st.info("Choose a test and the required column(s).")
    st.divider()
    lab.tails_section(sel, num_cols, k)
    st.divider()
    lab.anova_section(sel, num_cols, k)   # one-way / two-way / N-way ANOVA (book p.115)


# ---- Full PDF report ----
with st.container(key="whole_report_section", border=True):
    st.subheader("Save whole page / report")
    st.caption(
        "Create one PDF report containing the current dashboard title, selected-data preview, "
        "visualization, statistics, correlation, hypothesis testing, and probability density."
    )
    report_ready = (
        st.session_state.last_chart_png is not None or
        st.session_state.last_stats is not None or
        st.session_state.last_corr is not None or
        st.session_state.last_hypothesis is not None or
        st.session_state.last_density_png is not None
    )
    if st.button(
        "📄 Generate complete PDF report",
        type="primary", use_container_width=True,
        disabled=not report_ready
    ):
        try:
            pdf_bytes = build_report(
                title=st.session_state.get("dashboard_title", ""),
                source_name=st.session_state.data_name,
                selected_df=sel,
                logo_path=LOGO_PATH,
                chart_png=st.session_state.last_chart_png,
                chart_title=st.session_state.last_chart_title,
                stats_df=st.session_state.last_stats,
                corr_df=st.session_state.last_corr,
                hypothesis=st.session_state.last_hypothesis,
                density_png=st.session_state.last_density_png,
                density_title=st.session_state.last_density_title,
            )
            st.download_button(
                "⬇ Download complete report (PDF)",
                pdf_bytes,
                file_name="UST_data_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_complete_report"
            )
        except Exception as e:
            st.error(f"Could not generate the report: {e}")
    else:
        st.info("Generate at least one analysis above, then create the complete report.")

# ---- 14. Feedback (last section) ----
with st.container(border=True, key="ust_section_14"):
    st.subheader("14 · Feedback")
    with st.form("feedback_form"):
        rating = st.radio(
            "How would you rate this dashboard?",
            [1, 2, 3, 4, 5], horizontal=True,
            format_func=lambda x: "★" * x + "☆" * (5 - x)
        )
        comment = st.text_area(
            "Feedback / suggestions",
            placeholder="Tell us what was useful or what should be improved."
        )
        submitted = st.form_submit_button(
            "Submit feedback", type="primary", use_container_width=True
        )
        if submitted:
            try:
                save_feedback(st.session_state.user_id, rating, comment)
                st.success("Thank you! Your feedback has been saved.")
            except Exception as e:
                st.error(f"Could not save feedback: {e}")