# Stock-Market-ETF-Recommendation-System



## What This Project Does

This project builds an ETL by analyzing over 7 million rows of historical stock market data from the NYSE and NASDAQ exchanges. You give it a list of stocks you own (ex: Apple, Microsoft and NVIDIA) and it finds ETFs that act most similar to those stocks in the past. This lets investors find ETFs that may not have been on their radar that fit their existing investment style – without requiring any financial expertise to use. The system is entirely built on Apache Spark, which is a distributed computing framework intended to process data sets that are too large for one machine.



## Prerequisites

Before running anything, make sure the following are installed on your machine:

### 1. Python 3.8 or higher
Check if you have it:
```bash
python3 --version
```
If not installed, download it from https://www.python.org/downloads/



### 2. Java 8 or Java 11

Check if you have it:
```bash
java -version
```

If not installed:
- **Mac:** `brew install openjdk@11`
- **Ubuntu/Debian:** `sudo apt install openjdk-11-jdk`
- **Windows:** Download from https://adoptium.net/



### 3. PySpark (Apache Spark for Python)
Once Python is installed, install PySpark with:
```bash
pip install pyspark==3.4.0
```


### 4. A Free Kaggle Account
The dataset is hosted on Kaggle. You need a free account to download it.
Sign up at: https://www.kaggle.com

## Step 1 — Get Your Kaggle API Key

The dataset is 1.8 GB, so we use the Kaggle command-line tool to download it automatically rather than clicking through a browser.

# To get your API key:
1. Log into https://www.kaggle.com
2. Click your profile picture in the top-right corner
3. Click settings
4. Scroll down to the API section
5. Click Create New Token
6. A pop-up screen with your API token key will appear. Copy that API token key.

# To install the Kaggle CLI and place your key:
```bash
# Install the Kaggle command-line tool
pip install kaggle

# Create the .kaggle folder in your home directory (if it doesn't exist)
mkdir -p ~/.kaggle

# Move the downloaded kaggle.json file into that folder
# (adjust the path below to wherever your Downloads folder is)
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

# Download the kaggle dataset
# Change {YOUR_USERNAME} with the username you signed into kaggle with
# Change {YOUR_API_TOKE_KEY} with the API token key you previously copied
echo '{"username":"{YOUR_USERNAME}","api_key":"{YOUR_API_TOKEN_KEY}"}' > ~/.kaggle/kaggle.json

# Lock down the file permissions — Kaggle requires this for security
chmod 600 ~/.kaggle/kaggle.json
```

> **Windows users:** Place `kaggle.json` in `C:\Users\YourName\.kaggle\kaggle.json` instead.



## Step 2 — Download the Dataset

Run the following commands from your terminal. Make sure you are inside the project folder first.

```bash
# Navigate into the project folder
cd path/to/stock_project

# Download the dataset from Kaggle (~1.8 GB, may take a few minutes)
kaggle datasets download -d jacksoncrow/stock-market-dataset

# Unzip the downloaded file into the data/ folder
unzip stock-market-dataset.zip -d data/

# You can delete the zip file afterward to save space
rm stock-market-dataset.zip
```

After unzipping, your `data/` folder should look like this:
```
data/
├── stocks/        ← ~6,000 CSV files, one per stock
└── etfs/          ← ~2,000 CSV files, one per ETF
```

Each CSV file contains daily trading data with these columns:

| Column    | Description                                      |
|-----------|--------------------------------------------------|
| Date      | Trading date (YYYY-MM-DD)                        |
| Open      | Price at market open                             |
| High      | Highest price during the day                     |
| Low       | Lowest price during the day                      |
| Close     | Price at market close                            |
| Adj Close | Close price adjusted for dividends/stock splits  |
| Volume    | Number of shares traded that day                 |



## Step 3 — (Optional) Set Environment Variables

By default, both scripts look for data in `data/` and write results to `output/` relative to your current folder. If your data is stored somewhere else, you can override these paths without editing the code:

```bash
# If your data folder is somewhere other than data/
export STOCK_DATA_PATH="/path/to/your/data/"

# If you want results saved somewhere other than output/
export OUTPUT_PATH="/path/to/your/output/"
```

Skip this step if you downloaded the data into the default `data/` folder inside the project.



## Step 4 — Run Task 1: Data Engineering

This script loads all ~8,000 CSV files, cleans and filters the data, engineers features, and saves a processed dataset for Task 2 to use.

```bash
spark-submit src/task1_data_engineering.py

# If required, use the sudo command and enter a password of your choosing
sudo spark-submit src/task1_data_engineering.py
```



# What this script does, step by step:
1. Loads all CSVs from `data/stocks/` and `data/etfs/` in parallel
2. Attaches the ticker symbol (e.g., `AAPL`) and asset type (`STOCK` or `ETF`) to every row
3. Prints a full descriptive summary: row counts, date ranges, null counts, per-ticker statistics
4. Filters out data before January 1, 2010 (pre-2010 data is sparse and from a different market era)
5. Removes tickers with fewer than 252 trading days (~1 year) of data — too little history to be useful
6. Removes rows with missing prices or zero trading volume
7. Computes a log-return for each row: `ln(today's close / yesterday's close)` — this is the core signal used for similarity
8. Saves the cleaned dataset to `output/cleaned_stocks/` in Parquet format (a compressed, fast-read format)

# Expected output files:
```
output/
└── cleaned_stocks/     ← processed dataset in Parquet format (~4.1 million rows)
```

# Expected runtime: 5–15 minutes depending on your machine



## Step 5 — Run Task 2: Recommendation System

This script reads the cleaned dataset from Task 1 and runs the full recommendation pipeline.

```bash
spark-submit src/task2_recommendation.py

# If required, use the sudo command and enter a password of your choosing
sudo spark-submit src/task2_recommendation.py

```

> **Important:** Task 1 must complete successfully before running Task 2. Task 2 reads from `output/cleaned_stocks/` which Task 1 creates.

# What this script does, step by step:**
1. Loads the cleaned Parquet dataset from Task 1
2. Builds a feature vector for each ticker: monthly average return and monthly return volatility across all available months — summarizing each stock/ETF's historical behavior as a list of numbers
3. Standardizes all feature vectors so no single variable dominates the math
4. Runs K-means clustering, sweeping k from 5 to 25, to group all tickers by behavioral similarity. Picks the best k using the Silhouette score (a measure of cluster quality)
5. Builds a MinHash LSH index — a data structure that allows fast approximate similarity search across all 5,800+ tickers without comparing every pair
6. Runs three example recommendation scenarios (see below) and prints results
7. Saves all results as CSV files

# Example scenarios run automatically:**
- Tech investor — holds AAPL, MSFT, NVDA → recommends tech-sector ETFs like QQQ, VGT
- Defensive investor — holds JNJ, PG, KO, JPM → recommends consumer/health ETFs like XLV, VDC
- Energy investor — holds XOM, CVX → recommends energy ETFs like XLE, VDE

# Expected output files:
```
output/
├── ticker_clusters/              ← cluster assignment for every ticker (CSV)
├── recommendations_tech/         ← ETF recommendations for tech portfolio (CSV)
├── recommendations_diversified/  ← ETF recommendations for defensive portfolio (CSV)
└── recommendations_energy/       ← ETF recommendations for energy portfolio (CSV)
```

# Expected runtime: 15–30 minutes depending on your machine

---

## Full Project Structure

```
stock_project/
├── README.md                        ← this file
├── src/
│   ├── task1_data_engineering.py    ← Step 4: cleans and prepares the data
│   └── task2_recommendation.py      ← Step 5: runs the recommendation system
├── data/                            ← place downloaded Kaggle data here
│   ├── stocks/                      ← one CSV per stock ticker
│   └── etfs/                        ← one CSV per ETF ticker
└── output/                          ← created automatically when scripts run
```
