import pandas as pd
import numpy as np

## Task 1: Download the metadata.csv file into a pandas DataFrame
df = pd.read_csv('metadata.csv')
# Display the first 5 rows of the DataFrame
print("First 5 rows:")
print(df.head())

# Check the dimensions (rows, columns)
print("\nDataFrame dimensions:")
print(df.shape)

# Identify data types of each column
print("\nData types of each column:")
print(df.info())

## Task 2 basic data exploration.
# Check for missing values in important columns
# For example, let's check 'title', 'abstract', and 'publish_time'
print("\nMissing values in key columns:")
print(df[['title', 'abstract', 'publish_time']].isnull().sum())

# Generate basic statistics for numerical columns
# 'cord_uid' is an object, so let's focus on other potential numerical data
print("\nBasic statistics for numerical columns:")
print(df.describe(include=np.number))






### Part 2: Data Cleaning and Preparation
# Task 1 Identify columns with a large number of missing values
missing_counts = df.isnull().sum().sort_values(ascending=False)
print("\nMissing value counts per column:")
print(missing_counts[missing_counts > 0])

# Drop rows where 'title' or 'abstract' is missing
# This is a common strategy to ensure we have meaningful content for analysis
cleaned_df = df.dropna(subset=['title', 'abstract'])

# Check the new shape of the cleaned DataFrame
print("\nShape of the DataFrame after dropping rows with missing titles or abstracts:")
print(cleaned_df.shape)

# Task 2: Prepare Data for Analysis
# Convert 'publish_time' to datetime format
cleaned_df['publish_time'] = pd.to_datetime(cleaned_df['publish_time'], errors='coerce')

# Extract the year for time-based analysis
cleaned_df['publish_year'] = cleaned_df['publish_time'].dt.year

# Create a new column for abstract word count
cleaned_df['abstract_word_count'] = cleaned_df['abstract'].apply(lambda x: len(str(x).split()))

# Display the new columns and their data types
print("\nDataFrame with new 'publish_year' and 'abstract_word_count' columns:")
print(cleaned_df[['publish_time', 'publish_year', 'abstract_word_count']].head())
print(cleaned_df[['publish_time', 'publish_year', 'abstract_word_count']].info())






## Part 3: Data Analysis and Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud

sns.set_style("whitegrid")
# perform basic analysis
# Count papers by publication year
papers_by_year = cleaned_df['publish_year'].value_counts().sort_index()
print("\nNumber of papers published per year:")
print(papers_by_year)

# Plot number of publications over time
plt.figure(figsize=(10, 6))
papers_by_year.plot(kind='bar', color='skyblue')
plt.title('Number of Publications Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.xticks(rotation=45)
plt.show()

# Identify and plot top journals
top_journals = cleaned_df['journal'].value_counts().head(10)
print("\nTop 10 publishing journals:")
print(top_journals)

plt.figure(figsize=(12, 8))
top_journals.plot(kind='barh', color='lightcoral')
plt.title('Top 10 Publishing Journals')
plt.xlabel('Number of Papers')
plt.ylabel('Journal')
plt.tight_layout()
plt.show()

# Simple word frequency analysis for titles
all_titles = ' '.join(cleaned_df['title'].str.lower().dropna())
words = re.findall(r'\b\w+\b', all_titles)
stop_words = {'the', 'a', 'an', 'of', 'in', 'on', 'with', 'and', 'for', 'is', 'from', 'as', 'by', 'at', 'to', 'or'}
filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
word_counts = Counter(filtered_words)

# Generate and display a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud of Paper Titles')
plt.show()






## Part 4: Streamlit Application
# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import re

# Load and clean data (using cache to avoid re-running on every interaction)
@st.cache_data
def load_data():
    df = pd.read_csv('metadata.csv')
    df_cleaned = df.dropna(subset=['title', 'abstract'])
    df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')
    df_cleaned['publish_year'] = df_cleaned['publish_time'].dt.year
    return df_cleaned

df_cleaned = load_data()

st.title("CORD-19 Data Explorer 🔬")
st.write("Simple exploration of COVID-19 research papers using Streamlit.")

# Add interactive widgets
year_range = st.slider("Select Year Range", 
                       int(df_cleaned['publish_year'].min()), 
                       int(df_cleaned['publish_year'].max()), 
                       (2020, 2021))

# Filter data based on slider input
filtered_df = df_cleaned[(df_cleaned['publish_year'] >= year_range[0]) & 
                         (df_cleaned['publish_year'] <= year_range[1])]

# Display visualizations based on filtered data
st.header("Publications Over Time")
papers_by_year = filtered_df['publish_year'].value_counts().sort_index()
fig, ax = plt.subplots()
papers_by_year.plot(kind='bar', ax=ax, color='skyblue')
ax.set_title('Publications by Year')
ax.set_xlabel('Year')
ax.set_ylabel('Count')
st.pyplot(fig)

st.header("Top Publishing Journals")
top_journals = filtered_df['journal'].value_counts().head(10)
fig, ax = plt.subplots()
top_journals.plot(kind='barh', ax=ax, color='lightcoral')
ax.set_title('Top 10 Publishing Journals')
ax.set_xlabel('Count')
ax.set_ylabel('Journal')
st.pyplot(fig)

# Display a sample of the data
st.header("Sample Data")
st.dataframe(filtered_df[['title', 'journal', 'publish_year']].head(10))

# Display Word Cloud
st.header("Word Cloud of Titles")
all_titles = ' '.join(filtered_df['title'].str.lower().dropna())
words = re.findall(r'\b\w+\b', all_titles)
stop_words = {'the', 'a', 'an', 'of', 'in', 'on', 'with', 'and', 'for', 'is', 'from', 'as', 'by', 'at', 'to', 'or'}
filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
word_counts = Counter(filtered_words)

if word_counts:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)
else:
    st.write("Not enough data to generate a word cloud.")