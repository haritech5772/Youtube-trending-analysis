import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")

df = pd.read_csv("INvideos.csv", encoding="latin1")

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)
print("Rows, Columns:", df.shape)
print(df.head())

print("\nColumns:")
print(df.columns)

df.drop_duplicates(subset="video_id", keep="last", inplace=True)

df["publish_time"] = pd.to_datetime(df["publish_time"])


df["publish_hour"] = df["publish_time"].dt.hour
df["publish_day"] = df["publish_time"].dt.day_name()
df["title_length"] = df["title"].astype(str).apply(len)

df.dropna(inplace=True)

print("\nAfter Cleaning:", df.shape)

with open("IN_category_id.json", "r", encoding="utf-8") as f:
    category_data = json.load(f)

print("JSON Loaded Successfully")

category_map = {
    int(item["id"]): item["snippet"]["title"]
    for item in category_data["items"]
}

df["category"] = df["category_id"].map(category_map)

df["like_ratio"] = (
    df["likes"] / df["views"]
) * 100

df["comment_ratio"] = (
    df["comment_count"] / df["views"]
) * 100

df.replace([float("inf"), -float("inf")], 0, inplace=True)


top_categories = (
    df["category"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_categories.values,
    y=top_categories.index
)

plt.title("Top 10 Trending Categories")
plt.xlabel("Trending Videos Count")
plt.ylabel("Category")

plt.tight_layout()
plt.savefig("chart1_top_categories.png")
plt.show()


corr = df[
    ["views", "likes", "dislikes", "comment_count"]
].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Views vs Likes Correlation")

plt.tight_layout()
plt.savefig("chart2_correlation.png")
plt.show()


day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_counts = (
    df["publish_day"]
    .value_counts()
    .reindex(day_order)
)

plt.figure(figsize=(10, 5))

sns.barplot(
    x=day_counts.index,
    y=day_counts.values
)

plt.title("Trending Videos by Publish Day")
plt.xlabel("Day")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("chart3_publish_day.png")
plt.show()


hour_counts = (
    df["publish_hour"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 5))

plt.plot(
    hour_counts.index,
    hour_counts.values,
    marker="o"
)

plt.title("Trending Videos by Publish Hour")
plt.xlabel("Hour")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("chart4_publish_hour.png")
plt.show()


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="title_length",
    y="views",
    alpha=0.4
)

plt.title("Title Length vs Views")

plt.tight_layout()
plt.savefig("chart5_title_length.png")
plt.show()


like_category = (
    df.groupby("category")["like_ratio"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=like_category.values,
    y=like_category.index
)

plt.title("Top Categories by Like Ratio")
plt.xlabel("Average Like Ratio (%)")
plt.ylabel("Category")

plt.tight_layout()
plt.savefig("chart6_like_ratio.png")
plt.show()


top10 = df.nlargest(10, "views")

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top10["views"],
    y=top10["title"]
)

plt.title("Top 10 Most Viewed Trending Videos")

plt.tight_layout()
plt.savefig("chart7_top10_videos.png")
plt.show()


print("\n")
print("=" * 50)
print("KEY INSIGHTS")
print("=" * 50)

print(
    "Top Trending Category:",
    top_categories.index[0]
)

print(
    "Best Publish Day:",
    day_counts.idxmax()
)

print(
    "Best Publish Hour:",
    hour_counts.idxmax()
)

top10_percent = df.nlargest(
    int(len(df) * 0.10),
    "views"
)

print(
    "Average Title Length (Top 10% Videos):",
    round(top10_percent["title_length"].mean(), 2)
)

print(
    "Views-Likes Correlation:",
    round(corr.loc["views", "likes"], 2)
)

print(
    "Average Like Ratio:",
    round(df["like_ratio"].mean(), 2)
)

print("\nAnalysis Completed Successfully")
print("Charts Saved Successfully")