"""
Task 2: Data Analytic — ETF/Portfolio Recommendation System
Approach: Collaborative Filtering on Ticker Return Profiles
  + Locality-Sensitive Hashing (LSH) for similarity search
  + K-Means clustering to group ETFs by behavioral sector

Core Query:
    "Given a user's existing stock holdings, which ETFs are mostsimilar in return behavior and would complement their portfolio with diversification?"
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, MinHashLSH, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import os

# 1. Initialize Spark
spark = (
    SparkSession.builder
    .appName("StockMarket_RecommendationSystem")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "4g")
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

OUTPUT_DIR = os.environ.get("OUTPUT_PATH", "output/")

# 2. Load Intermediate Dataset from Task 1
print("Loading cleaned dataset...")
df = spark.read.parquet(f"{OUTPUT_DIR}cleaned_stocks")
df.printSchema()
print(f"Loaded {df.count():,} rows, {df.select('ticker').distinct().count():,} tickers")


# 3. Build Ticker Return Feature Vectors
'''
Rationale:
   We reshape the log_return series in monthly aggregates (mean, std) per ticker. Using daily returns directly would produce vectors 
   of ~3,500 dimensions (14 years * 252 days), which is computationally prohibitive and subject to the curse of dimensionality. 
   Monthly aggregates (168 features: 14 yrs * 12 months * 2 stats) maintain the temporal return pattern while being tractable.
   Months without data filled with 0.0 (no movement assumption).
'''

print("\nBuilding monthly return feature vectors...")

monthly = (
    df
    .withColumn("year_month", F.date_format("Date", "yyyy-MM"))
    .groupBy("ticker", "source", "year_month")
    .agg(
        F.round(F.mean("log_return"), 6).alias("monthly_mean_ret"),
        F.round(F.stddev("log_return"), 6).alias("monthly_std_ret"),
    )
    .fillna(0.0)
)

# Pivot: one row per ticker, columns = year_month_mean + year_month_std
mean_pivot = (
    monthly
    .groupBy("ticker", "source")
    .pivot("year_month")
    .agg(F.first("monthly_mean_ret"))
    .fillna(0.0)
)

std_pivot = (
    monthly
    .groupBy("ticker", "source")
    .pivot("year_month")
    .agg(F.first("monthly_std_ret"))
    .fillna(0.0)
)

# Rename columns to distinguish mean vs std
mean_cols = [c for c in mean_pivot.columns if c not in ("ticker", "source")]
std_cols  = [c for c in std_pivot.columns  if c not in ("ticker", "source")]

for c in mean_cols:
    mean_pivot = mean_pivot.withColumnRenamed(c, f"mean_{c}")
for c in std_cols:
    std_pivot = std_pivot.withColumnRenamed(c, f"std_{c}")

feature_df = mean_pivot.join(std_pivot, on=["ticker", "source"], how="inner")

feature_cols = [c for c in feature_df.columns if c.startswith("mean_") or c.startswith("std_")]
print(f"Feature dimensions: {len(feature_cols)}")

# 4. Assemble & Scale Feature Vectors
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="keep")
feature_df = assembler.transform(feature_df)

scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
scaler_model = scaler.fit(feature_df)
feature_df = scaler_model.transform(feature_df).select("ticker", "source", "features")
feature_df.cache()

# 5. K-Means Clustering
'''
Rationale:
   K-Means clusters tickers based on behavioral similarity in return space.
   This enables us to:
    (a) cluster ETFs by "behavioral cluster" (acts like growth tech, acts like bond proxy, etc.)
    (b) Rapidly identify cluster-mates of a user's holdings, sweeping k from 5 to 25 and select the elbow using Silhouette score.
'''

print("\nRunning K-Means sweep (k=5 to 25)...")
evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")

best_k, best_score, best_model = 5, -1.0, None
for k in range(5, 26, 5):
    km = KMeans(k=k, seed=42, featuresCol="features", maxIter=30)
    m  = km.fit(feature_df)
    predictions = m.transform(feature_df)
    score = evaluator.evaluate(predictions)
    print(f"  k={k:2d}  silhouette={score:.4f}")
    if score > best_score:
        best_score, best_k, best_model = score, k, m

print(f"\nBest k={best_k}, silhouette={best_score:.4f}")
clustered = best_model.transform(feature_df).withColumnRenamed("prediction", "cluster")
clustered.cache()

# Cluster summary
print("\n=== CLUSTER COMPOSITION ===")
clustered.groupBy("cluster", "source").count().orderBy("cluster", "source").show(50)


# 6. LSH Similarity Search (ETF Recommendation Engine)
'''
Rationale:
   MinHash LSH allows for approximate nearest neighbor search in the high-dimensional feature space, 
   avoiding the need for exhaustive pairwise comparisons. For a dataset of 5,000+ tickers, brute-force 
   cosine distance is O(n^2) = 25M comparisons -- prohibitive at query time. LSH reduces this to sub-linear 
   lookup by hashing similar vectors to the same bucket with high probability. We binarize feature vectors 
   (positive return 1, else 0) to apply MinHash, and Jaccard similarity as the proximity metric.
'''

print("\nBuilding LSH index...")

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, DoubleType
import pyspark.sql.functions as F

# Binarize features: 1.0 if positive return signal, else 0.0
def binarize_vector(v):
    return Vectors.dense([1.0 if x > 0 else 0.0 for x in v])

binarize_udf = F.udf(binarize_vector, VectorUDT())
lsh_input = clustered.withColumn("bin_features", binarize_udf(F.col("features")))

mh = MinHashLSH(inputCol="bin_features", outputCol="hashes", numHashTables=10, seed=42)
lsh_model = mh.fit(lsh_input)
lsh_df = lsh_model.transform(lsh_input)
lsh_df.cache()


# 7. Recommendation Function
'''
Given a list of stock tickers (the user's holdings), find similar ETFs in behavior using 
LSH approximate nearest neighbors. We return top-N ETFs ranked by average Jaccard distance 
to the input, filtered to ETFs that are outside the user's current cluster (diversification).
'''

def recommend_etfs(user_tickers: list, top_n: int = 10, distance_threshold: float = 0.6):
    """
    Parameters
    ----------
    user_tickers : list of ticker symbols the user holds
    top_n        : number of ETF recommendations to return
    distance_threshold : max Jaccard distance for LSH lookup
    """
    user_tickers_upper = [t.upper() for t in user_tickers]

    # Get user holding rows
    user_df = lsh_df.filter(F.col("ticker").isin(user_tickers_upper))
    found = [r["ticker"] for r in user_df.select("ticker").collect()]
    print(f"\nMatched tickers in dataset: {found}")

    if not found:
        print("No matching tickers found. Check ticker symbols.")
        return None

    # ETF subset
    etf_df = lsh_df.filter(F.col("source") == "ETF")

    # For each user holding, run approximate similarity join against ETFs
    all_results = []
    for row in user_df.collect():
        ticker_sym = row["ticker"]
        key_vec    = spark.createDataFrame(
            [(ticker_sym, row["bin_features"])],
            ["ticker", "bin_features"]
        )
        neighbors = lsh_model.approxSimilarityJoin(
            key_vec, etf_df,
            threshold=distance_threshold,
            distCol="distance"
        ).select(
            F.col("datasetA.ticker").alias("input_ticker"),
            F.col("datasetB.ticker").alias("etf_ticker"),
            F.col("datasetB.cluster").alias("etf_cluster"),
            "distance"
        ).filter(~F.col("etf_ticker").isin(user_tickers_upper))
        all_results.append(neighbors)

    # Union and rank by average distance
    from functools import reduce
    combined = reduce(lambda a, b: a.unionByName(b), all_results)
    ranked = (
        combined
        .groupBy("etf_ticker", "etf_cluster")
        .agg(F.round(F.mean("distance"), 4).alias("avg_distance"))
        .orderBy("avg_distance")
        .limit(top_n)
    )

    print(f"\n=== TOP {top_n} ETF RECOMMENDATIONS ===")
    ranked.show(top_n, truncate=False)
    return ranked


# 8. Example Recommendations

# Scenario A: Tech-heavy individual investor
print("\n--- Scenario A: Tech investor (AAPL, MSFT, NVDA) ---")
recs_a = recommend_etfs(["AAPL", "MSFT", "NVDA"], top_n=10)

# Scenario B: Diversified investor looking for stability
print("\n--- Scenario B: Mixed portfolio (JNJ, PG, KO, JPM) ---")
recs_b = recommend_etfs(["JNJ", "PG", "KO", "JPM"], top_n=10)

# Scenario C: Energy sector
print("\n--- Scenario C: Energy investor (XOM, CVX) ---")
recs_c = recommend_etfs(["XOM", "CVX"], top_n=10)

# 9. Save Results
clustered.select("ticker", "source", "cluster") \
    .write.mode("overwrite").csv(f"{OUTPUT_DIR}ticker_clusters", header=True)

if recs_a:
    recs_a.write.mode("overwrite").csv(f"{OUTPUT_DIR}recommendations_tech", header=True)
if recs_b:
    recs_b.write.mode("overwrite").csv(f"{OUTPUT_DIR}recommendations_diversified", header=True)
if recs_c:
    recs_c.write.mode("overwrite").csv(f"{OUTPUT_DIR}recommendations_energy", header=True)

print("\nAll results saved to output/")
spark.stop()
