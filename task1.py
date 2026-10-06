import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
accuracy_score,
f1_score,
confusion_matrix
)

from sklearn.cluster import KMeans
print("=" * 60)
print("TASK 1 - DATA AUDIT AND MODEL EVALUATION")
print("=" * 60)

# LOAD DATA
df = pd.read_csv("tickets_raw.csv")

print("\nSTEP 1: LOAD DATA")
print("-" * 60)

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nFirst 5 rows:")
print(df.head())

# DATA AUDIT

print("\n")
print("=" * 60)
print("STEP 2: DATA QUALITY AUDIT")
print("=" * 60)
print("\nMissing values:")
print(df.isnull().sum())
print("\nMissing percentage:")
missing_percentage = df.isnull().mean() * 100
print(missing_percentage)
print("\nNumber of duplicate rows:")
duplicates = df.duplicated().sum()
print(duplicates)
print("\nCategory balance:")
print(df["category"].value_counts(dropna=False))
print("\nCategory values before cleaning:")
print(df["category"].unique())

# STEP 3: CLEAN DATA
print("\n")
print("=" * 60)
print("STEP 3: CLEAN DATA")
print("=" * 60)
clean_df = df.copy()
clean_df = clean_df.dropna(
subset=["text", "category"]
)
clean_df = clean_df.drop_duplicates()
clean_df["category"] = (
clean_df["category"]
.str.strip()
.str.title()
)
clean_df["text"] = (
clean_df["text"]
.str.strip()
)
print("\nRows before cleaning:", len(df))
print("Rows after cleaning:", len(clean_df))
print("\nDuplicates after cleaning:")
print(clean_df.duplicated().sum())
print("\nMissing values after cleaning:")
print(clean_df.isnull().sum())
print("\nCategory balance after cleaning:")
print(clean_df["category"].value_counts())
clean_df.to_csv(
"tickets_clean.csv",
index=False
)
print("\ntickets_clean.csv created successfully.")

# STEP 4: PREPARE DATA
print("\n")
print("=" * 60)
print("STEP 4: PREPARE DATA")
print("=" * 60)
X = clean_df["text"]
y = clean_df["category"]

# STEP 5: TRAIN TEST SPLIT

print("\n")
print("=" * 60)
print("STEP 5: TRAIN TEST SPLIT")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 6: BASELINE MODEL
print("\n")
print("=" * 60)
print("STEP 6: BASELINE MODEL")
print("=" * 60)
most_common_category = y_train.value_counts().idxmax()
print("\nMost common category:")
print(most_common_category)
baseline_predictions = [
most_common_category
] * len(y_test)
baseline_accuracy = accuracy_score(
y_test,
baseline_predictions
)
baseline_f1 = f1_score(
y_test,
baseline_predictions,
average="macro"
)
baseline_cm = confusion_matrix(
y_test,
baseline_predictions,
labels=sorted(y.unique())
)
print("\nBaseline Accuracy:")
print(round(baseline_accuracy, 2))
print("\nBaseline Macro F1:")
print(round(baseline_f1, 2))
print("\nBaseline Confusion Matrix:")
print(baseline_cm)

# STEP 7: TF-IDF
print("\n")
print("=" * 60)
print("STEP 7: TF-IDF")
print("=" * 60)
vectorizer = TfidfVectorizer(
lowercase=True,
stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(
X_train
)
X_test_tfidf = vectorizer.transform(
X_test
)
print("Training TF-IDF shape:")
print(X_train_tfidf.shape)
print("\nTesting TF-IDF shape:")
print(X_test_tfidf.shape)

# STEP 8: LOGISTIC REGRESSION
print("\n")
print("=" * 60)
print("STEP 8: LOGISTIC REGRESSION")
print("=" * 60)
model = LogisticRegression(
max_iter=1000,
random_state=42
)
model.fit(
X_train_tfidf,
y_train
)
predictions = model.predict(
X_test_tfidf
)
print("Model training completed.")

# STEP 9: MODEL EVALUATION
print("\n")
print("=" * 60)
print("STEP 9: MODEL EVALUATION")
print("=" * 60)
model_accuracy = accuracy_score(
y_test,
predictions
)
model_f1 = f1_score(
y_test,
predictions,
average="macro"
)
model_cm = confusion_matrix(
y_test,
predictions,
labels=sorted(y.unique())
)
print("\nModel Accuracy:")
print(round(model_accuracy, 2))
print("\nModel Macro F1:")
print(round(model_f1, 2))
print("\nModel Confusion Matrix:")
print(model_cm)
print("\n")
print("=" * 60)
print("STEP 10: BASELINE VS ML MODEL")
print("=" * 60)
print(
f"\nBaseline Accuracy : {baseline_accuracy:.2f}"
)
print(
f"ML Model Accuracy : {model_accuracy:.2f}"
)
print(
f"\nBaseline Macro F1 : {baseline_f1:.2f}"
)
print(
f"ML Model Macro F1 : {model_f1:.2f}"
)
print("\n")
print("=" * 60)
print("STEP 11: MISCLASSIFIED TICKETS")
print("=" * 60)
results = pd.DataFrame({

"text": X_test.values,

"actual": y_test.values,

"predicted": predictions

})
misclassified = results[
results["actual"] != results["predicted"]
]
if len(misclassified) == 0:

    print("\nNo misclassified tickets.")

else:

    print("\nMisclassified tickets:")

print(
misclassified
.head(3)
.to_string(index=False)
)
print("\nPossible reasons:")
print(
"1. Some tickets contain similar words."
)
print(
"2. Short tickets provide limited information."
)
print(
"3. Account and Technical tickets can have similar language."
)
print("\n")
print("=" * 60)
print("STEP 12: KMEANS CLUSTERING")
print("=" * 60)
all_tfidf = vectorizer.fit_transform(
clean_df["text"]
)
kmeans = KMeans(
n_clusters=3,
random_state=42,
n_init=10
)

clusters = kmeans.fit_predict(
all_tfidf
)
clean_df["cluster"] = clusters
cluster_table = pd.crosstab(
clean_df["cluster"],
clean_df["category"]
)
print("\nCluster vs Real Categories:")
print(cluster_table)
print("\n")
print("Cluster comparison:")
for cluster in sorted(
clean_df["cluster"].unique()
):

    cluster_data = clean_df[
clean_df["cluster"] == cluster
]
category_counts = (
cluster_data["category"]
.value_counts()
)
main_category = category_counts.idxmax()
print(
f"Cluster {cluster} mostly contains "
f"{main_category} tickets."
)
print("\n")
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(
f"\nRows: {len(df)} -> {len(clean_df)}"
)
print(
f"Baseline Accuracy: {baseline_accuracy:.2f}"
)
print(
f"ML Model Accuracy: {model_accuracy:.2f}"
)
print(
f"ML Model Macro F1: {model_f1:.2f}"
)
print("\nKMeans Clustering: Completed")
print("\nTASK 1 COMPLETED SUCCESSFULLY!")

