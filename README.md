# Task1
Data Cleaning and Preprocessing
This project **task1_cleainng.py** cleans the raw **marketing_campaign.csv** dataset.

The original dataset was **tab-separated** and contained 2240 rows × 29 columns with no missing values but required:
- Column name standardization
- NA-like string normalization
- Duplicate removal (none found)
- Date parsing (`dt_customer`)
- Numeric conversion (income, recency, Mnt* columns, etc.)
- Imputation (median for numeric, mode/'Unknown' for categorical)
- Outlier clipping at 1st–99th percentile

- **Cleaning script**: `task1_cleaning.py` (Python, pandas)
- **Cleaned dataset**: `marketing_campaign_cleaned.csv`
