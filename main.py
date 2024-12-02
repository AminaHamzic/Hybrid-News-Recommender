import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the training datasets
train_behaviors = pd.read_csv('data/MINDsmall_train/behaviors.tsv', sep='\t', header=None)
train_news = pd.read_csv('data/MINDsmall_train/news.tsv', sep='\t', header=None)

# Load the development datasets
dev_behaviors = pd.read_csv('data/MINDsmall_dev/behaviors.tsv', sep='\t', header=None)
dev_news = pd.read_csv('data/MINDsmall_dev/news.tsv', sep='\t', header=None)


# Display basic information about the datasets
print(train_behaviors.info())
print(train_news.info())

print(dev_behaviors.info())
print(dev_news.info())

# Display summary statistics of the datasets
print(train_behaviors.describe())
print(train_news.describe())
print(dev_behaviors.describe())
print(dev_news.describe())

# Display the first few rows of each dataset
print(train_behaviors.head())
print(train_news.head())
print(dev_behaviors.head())
print(dev_news.head())

# Value counts for categorical columns in the training datasets
print("Value counts for categorical columns in train_behaviors:")
print(train_behaviors[1].value_counts())
print(train_behaviors[2].value_counts())
print(train_behaviors[4].value_counts())

print("Value counts for categorical columns in train_news:")
print(train_news[1].value_counts())
print(train_news[2].value_counts())
print(train_news[5].value_counts())

# Value counts for categorical columns in the development datasets
print("Value counts for categorical columns in dev_behaviors:")
print(dev_behaviors[1].value_counts())
print(dev_behaviors[2].value_counts())
print(dev_behaviors[4].value_counts())

print("Value counts for categorical columns in dev_news:")
print(dev_news[1].value_counts())
print(dev_news[2].value_counts())
print(dev_news[5].value_counts())

# Unique values in each column
print("Unique values in each column for train_behaviors:")
print(train_behaviors.nunique())
print("Unique values in each column for train_news:")
print(train_news.nunique())
print("Unique values in each column for dev_behaviors:")
print(dev_behaviors.nunique())
print("Unique values in each column for dev_news:")
print(dev_news.nunique())

# Select only numeric columns for correlation analysis
numeric_columns = train_behaviors.select_dtypes(include=['number'])
numeric_columns_dev = dev_behaviors.select_dtypes(include=['number'])

# Calculate the correlation matrices
corr_train = numeric_columns.corr()
corr_dev = numeric_columns_dev.corr()

# # Visualize the correlation matrices using heatmaps
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_train, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix for Train Behaviors')
# plt.show()
#
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_dev, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix for Dev Behaviors')
# plt.show()

# Distribution of User Interactions
plt.figure(figsize=(10, 6))
sns.histplot(train_behaviors[1].value_counts(), bins=50, kde=True)
plt.title('Distribution of User Interactions in Train Behaviors')
plt.xlabel('Number of Interactions')
plt.ylabel('Frequency')
plt.show()

# Distribution of Timestamps
train_behaviors[2] = pd.to_datetime(train_behaviors[2])
plt.figure(figsize=(10, 6))
sns.histplot(train_behaviors[2], bins=50, kde=True)
plt.title('Distribution of Timestamps in Train Behaviors')
plt.xlabel('Timestamp')
plt.ylabel('Frequency')
plt.show()

# Category Distribution in Train News
plt.figure(figsize=(10, 6))
sns.countplot(y=train_news[1], order=train_news[1].value_counts().index)
plt.title('Category Distribution in Train News')
plt.xlabel('Count')
plt.ylabel('Category')
plt.show()