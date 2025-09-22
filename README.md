
### CORD-19 Research Paper Analysis & Streamlit App

This project is a complete data analysis workflow for the CORD-19 research dataset's `metadata.csv` file. It goes from **data loading** and **cleaning** to **analysis**, **visualization**, and finally building a simple, interactive **Streamlit web application**.

-----

### Project Features

  - **Data Exploration:** Initial checks on data dimensions, types, and missing values.
  - **Data Cleaning:** Handling of missing values and conversion of date columns.
  - **Feature Engineering:** Creating new columns like `publish_year` and `abstract_word_count`.
  - **Data Analysis:** Counting publications by year, identifying top journals, and analyzing frequent words in paper titles.
  - **Visualizations:** Bar charts, horizontal bar charts, and word clouds to visualize key trends.
  - **Streamlit Application:** A dynamic web dashboard with interactive filters to explore the data and visualizations.

-----

### How to Run the Project

#### 1\. Prerequisites

First, install the necessary Python libraries.

```bash
pip install pandas matplotlib seaborn streamlit wordcloud
```

#### 2\. Data Download 📥

Download the **`metadata.csv`** file from the official Kaggle dataset page and place it in the same directory as your Python script.

  - **Dataset URL:** `https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge`

#### 3\. File Structure

Your project directory should look like this:

```
.
├── metadata.csv
└── framework.py
```

#### 4\. Running the Streamlit App

Open your terminal, navigate to your project directory, and run the following command. This will launch the web application in your default browser.

```bash
streamlit run framework.py
```

-----

### Code Explanation

The `framework.py` file contains all the project code. It uses `st.cache_data` for efficiency, `st.slider` for interactive filtering, and `st.pyplot()` to display `matplotlib` plots within the app.

-----

### Data Source

The data used is the **COVID-19 Open Research Dataset (CORD-19)** from the Allen Institute for AI.
