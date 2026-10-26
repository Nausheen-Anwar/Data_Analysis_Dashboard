# Data Analysis Dashboard 

A Streamlit dashboard for loading data, selecting columns, and viewing data preparation, visualization, and statistical analysis on one page.

## Run the application

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

On the first launch, open the **Create account** tab on the login screen, create an account, and then log in. 

## Layout

The sidebar contains the controls. Results and analysis sections are displayed together on the main page.

| Location / section | Purpose |
|---|---|
| **1 · Home** | Dashboard title, KPI cards, data preview, and CSV download |
| **2 · Data Source** | Load data from a file, folder, URL, or database |
| **3 · Data Selection** | Select columns and available row/range options |
| **4 · Missing Data** | Review and handle missing values |
| **5 · Data Normalization** | Apply available data normalization or scaling options |
| **6 · Basic Data Operations in Python** | Perform the supported basic data operations |
| **7 · Visualization** | Create a chart and download it in PNG, PDF, or SVG format where available |
| **8 · Statistical Parameters** | View calculated statistical measures and download available results as CSV |
| **9 · Correlation Amongst Columns** | View the Pearson correlation matrix for suitable numeric columns |
| **10 · Regression** | Simple linear (MSE, MAE, RMSE, R²), multiple linear, polynomial and overfitting, logistic regression |
| **11 · Probability Density** | Plot a Kernel Density Estimate (KDE), a normal probability density function, or both |
| **12 · Probability Distributions** | Normal, probability, binomial, Poisson, central limit theorem |
| **13 · Hypothesis Testing** | t-tests, correlation significance, one/two-tailed tests, Type I/II error, one-way / two-way / N-way ANOVA |
| **Save Whole Page / Report** | Download a PDF report containing the current data preview and available analysis results |
| **14 · Feedback** | Submit a rating from 1 to 5 and optional comments |


## Data sources

Available source options include:

- **Upload files:** CSV, TSV, TXT, Excel, JSON, and Parquet.
- **Excel sheet selection:** choose a worksheet when loading an Excel workbook.
- **Multiple files:** upload more than one file and combine/stack them into one table where supported.
- **Local folder:** enter a folder path, then choose a file listed from that folder.
- **Online URL:** load a CSV, Excel, or JSON file from a URL.
- **Database:** connect to SQLite, MySQL, PostgreSQL, or SQL Server through ODBC. Choose a table or provide your own `SELECT` query.

Database access depends on the database service, connection settings, and required drivers being available.

## Data selection and initial state

Titles, data sources, columns, ranges, and chart options start empty. Nothing is pre-filled or pre-selected. Select the required data and options before running an analysis.

## Visualization

- Set the chart options and **Graph title**.
- Download the chart in PNG, PDF, or SVG format when the corresponding option is available.

## Statistical analysis

- **Statistical parameters:** calculate descriptive statistical values for the selected data.
- **Correlation:** display a Pearson correlation matrix for suitable numeric columns.
- **Hypothesis testing (last analysis section):** one-sample, two-sample and paired t-tests, Pearson correlation significance, one/two-tailed tests with Type I and II error, and ANOVA (one-way, two-way and N-way, from the book, page 115).
- **Probability density:** choose a sample from a column or a single row, then display a KDE, a normal PDF, or both. Available downloads include PNG, PDF, SVG, and CSV.

Some analysis require numeric columns or enough valid observations. Results depend on the data and options selected.

## PDF report

Use **Save whole page / report** to download a PDF containing the current selected-data preview and analysis sections available in the current session.

## Feedback

The final Feedback section accepts a rating from **1 to 5** and optional comments. 

## Colours

The whole dashboard uses two colours only, set in `utils/palette.py` (and the same values in `.streamlit/config.toml`):
`BG` (page background, the "The Team Behind UST" colour on ustindia.com) and `ACCENT` (all other surfaces, the "Our Business Partners" colour).
