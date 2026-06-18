"""
Task 1: Data Engineering
Dataset: Kaggle "Stock Market Dataset"
URL: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
import os

# 1. Initialize Spark
spark = (
    SparkSession.builder
    .appName("StockMarket_DataEngineering")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# 2. Load Raw Data
DATA_ROOT = os.environ.get("STOCK_DATA_PATH", "data/")
OUTPUT_DIR = os.environ.get("OUTPUT_PATH", "output/")

def load_partition(path, source_label):
    """Load all CSVs in a directory; attach ticker from filename."""
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
        .withColumn("source", F.lit(source_label))
        .withColumn("_file", F.input_file_name())
        .withColumn(
            "ticker",
            F.regexp_extract(F.col("_file"), r"([^/\\]+)\.csv$", 1)
        )
        .drop("_file")
    )
    return df

print("Loading stock and ETF data...")
stocks_df = load_partition(f"{DATA_ROOT}stocks/*.csv", "STOCK")
etfs_df   = load_partition(f"{DATA_ROOT}etfs/*.csv",   "ETF")
raw_df    = stocks_df.unionByName(etfs_df)

# 3. Descriptive Analysis
print("\n=== RAW DATASET OVERVIEW ===")
raw_df.printSchema()

total_rows    = raw_df.count()
total_tickers = raw_df.select("ticker").distinct().count()
date_range    = raw_df.agg(F.min("Date").alias("earliest"), F.max("Date").alias("latest")).collect()[0]
null_counts   = raw_df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in raw_df.columns
])

print(f"\nTotal rows       : {total_rows:,}")
print(f"Total tickers    : {total_tickers:,}")
print(f"Date range       : {date_range['earliest']} → {date_range['latest']}")
print("\nNull counts per column:")
null_counts.show()

# Per-ticker summary statistics (numerical)
print("\n=== PER-TICKER DESCRIPTIVE STATS ===")
numeric_stats = raw_df.groupBy("ticker", "source").agg(
    F.count("*").alias("trading_days"),
    F.round(F.mean("Volume"), 0).alias("avg_volume"),
    F.round(F.stddev("Volume"), 0).alias("std_volume"),
    F.round(F.mean("Close"), 4).alias("avg_close"),
    F.round(F.stddev("Close"), 4).alias("std_close"),
    F.round(F.min("Close"), 4).alias("min_close"),
    F.round(F.max("Close"), 4).alias("max_close"),
    F.min("Date").alias("start_date"),
    F.max("Date").alias("end_date"),
)
numeric_stats.show(20, truncate=False)

# Distribution of trading days (data completeness check)
print("\n=== TRADING DAY DISTRIBUTION ===")
numeric_stats.select("trading_days").describe().show()


# 4. Data Cleaning & Reduction
'''
Rationale: .
    a) DATE FILTER: Filter for >= 2010-01-01. Pre-2010 data is less dense, reflects a different market regime 
    (pre-HFT, pre-ETF proliferation), and adds noise for a recommendation use case grounded in modern portfolios.

   b) MINIMUM HISTORY: Require >= 252 trading days (~1 year)  Tickers with less rows have not enough 
   signal tocalculate return / volatility.

   c) NULL / ZERO REMOVAL: Null Close or Zero Volume rows are days when there is no trading (halts, delistings) 
   and would distort returns.

   d) FEATURE ENGINEERING: Daily log-returns per ticker - the main signal for clustering and recommendations. 
   Log-returns are preferred to raw prices since they are additive in time and better satisfy the normality 
   assumptions in portfolio theory.

   e) COLUMN REDUCTION: Drop Open/High/Low/Adj Close — only need Date, ticker, source, Close, Volume, 
   and derived return for Task 2.
'''

MIN_DATE   = "2010-01-01"
MIN_DAYS   = 252

print(f"\nApplying filters: date >= {MIN_DATE}, trading_days >= {MIN_DAYS} ...")

# Step a: date filter
cleaned = raw_df.filter(F.col("Date") >= MIN_DATE)

# Step b and c: remove null/zero prices
cleaned = (
    cleaned
    .filter(F.col("Close").isNotNull())
    .filter(F.col("Volume").isNotNull() & (F.col("Volume") > 0))
)

# Step b: minimum history filter (compute per ticker, then join back)
day_counts = cleaned.groupBy("ticker").agg(F.count("*").alias("n_days"))
valid_tickers = day_counts.filter(F.col("n_days") >= MIN_DAYS).select("ticker")
cleaned = cleaned.join(valid_tickers, on="ticker", how="inner")

# Step d: log-return feature
window_spec = Window.partitionBy("ticker").orderBy("Date")
cleaned = (
    cleaned
    .withColumn("prev_close", F.lag("Close", 1).over(window_spec))
    .withColumn(
        "log_return",
        F.when(
            F.col("prev_close").isNotNull() & (F.col("prev_close") > 0),
            F.log(F.col("Close") / F.col("prev_close"))
        ).otherwise(None)
    )
    .filter(F.col("log_return").isNotNull())
)

# Step e: column reduction
cleaned = cleaned.select(
    F.col("Date").cast("date"),
    "ticker",
    "source",
    F.col("Close").cast(DoubleType()),
    F.col("Volume").cast(DoubleType()),
    F.col("log_return").cast(DoubleType()),
)

# 5. Report on Intermediate Dataset
reduced_rows    = cleaned.count()
reduced_tickers = cleaned.select("ticker").distinct().count()
etf_count       = cleaned.filter(F.col("source") == "ETF").select("ticker").distinct().count()
stock_count     = cleaned.filter(F.col("source") == "STOCK").select("ticker").distinct().count()
date_range2     = cleaned.agg(F.min("Date"), F.max("Date")).collect()[0]

print("\n=== REDUCED DATASET SUMMARY ===")
print(f"Rows             : {reduced_rows:,}  (was {total_rows:,})")
print(f"Tickers          : {reduced_tickers:,}  (was {total_tickers:,})")
print(f"  - Stocks       : {stock_count:,}")
print(f"  - ETFs         : {etf_count:,}")
print(f"Date range       : {date_range2[0]} → {date_range2[1]}")

print("\n=== LOG-RETURN STATISTICS ===")
cleaned.select("log_return").describe().show()

# 6. Persist Intermediate Dataset
out_path = f"{OUTPUT_DIR}cleaned_stocks"
cleaned.write.mode("overwrite").parquet(out_path)
print(f"\nIntermediate dataset saved to: {out_path}")

spark.stop()
