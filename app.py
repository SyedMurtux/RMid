import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Page configuration
st.set_page_config(page_title="Automobile Data Analysis", layout="wide")

# Load the dataset
@st.cache
def load_data():
    path = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(path)

df = load_data()

# Identify numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Sections", [
    "Introduction",
    "Data Overview",
    "Feature Visualization",
    "Descriptive Statistics",
    "Correlation and Causation",
    "Grouping and Pivot Tables"
])

# Section: Introduction
if options == "Introduction":
    st.title("Automobile Data Analysis")
    st.markdown("""
    Welcome to the Automobile Data Analysis app! This app explores key patterns, correlations, and insights from the dataset.
    
    **Navigation:** Use the sidebar to explore different sections.
    
    **Dataset:** Cleaned dataset of automobiles with features like price, engine size, body style, etc.
    """)

# Section: Data Overview
elif options == "Data Overview":
    st.title("Data Overview")
    st.write("### First 5 Rows of Data")
    st.write(df.head())
    st.write("### Data Types")
    st.write(df.dtypes)
    st.write("### Statistical Summary")
    st.write(df.describe())

# Section: Feature Visualization
elif options == "Feature Visualization":
    st.title("Feature Visualization")
    
    st.write("### Select Feature for Scatterplot")
    x_feature = st.selectbox("Select X-axis feature", numerical_columns, index=0)
    y_feature = st.selectbox("Select Y-axis feature", numerical_columns, index=1)

    # Scatterplot with regression line
    st.write(f"### Scatterplot of {x_feature} vs {y_feature}")
    fig, ax = plt.subplots()
    sns.regplot(x=x_feature, y=y_feature, data=df, ax=ax)
    st.pyplot(fig)

    # Correlation coefficient
    corr_value = df[[x_feature, y_feature]].corr().iloc[0, 1]
    st.write(f"Correlation between {x_feature} and {y_feature}: **{corr_value:.2f}**")

# Section: Descriptive Statistics
elif options == "Descriptive Statistics":
    st.title("Descriptive Statistics")
    st.write("### Numerical Variables")
    st.write(df.describe())
    st.write("### Categorical Variables")
    st.write(df.describe(include=['object']))

    st.write("### Value Counts for Categorical Features")
    categorical_columns = df.select_dtypes(include=['object']).columns
    selected_cat_col = st.selectbox("Select Categorical Feature", categorical_columns)
    st.write(df[selected_cat_col].value_counts())

# Section: Correlation and Causation
elif options == "Correlation and Causation":
    st.title("Correlation and Causation")
    st.write("### Pearson Correlation Coefficients")
    selected_columns = st.multiselect("Select Features for Correlation Analysis", numerical_columns, default=list(numerical_columns[:4]))
    if len(selected_columns) > 1:
        correlation_matrix = df[selected_columns].corr()
        st.write("### Correlation Matrix")
        st.write(correlation_matrix)

        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.write("### Pearson Correlation with P-values")
    x_feature = st.selectbox("Select X Feature", numerical_columns, index=0)
    y_feature = st.selectbox("Select Y Feature", numerical_columns, index=1)
    pearson_coef, p_value = stats.pearsonr(df[x_feature], df[y_feature])
    st.write(f"Pearson Correlation Coefficient: **{pearson_coef:.2f}**")
    st.write(f"P-value: **{p_value:.3e}**")

# Section: Grouping and Pivot Tables
elif options == "Grouping and Pivot Tables":
    st.title("Grouping and Pivot Tables")
    st.write("### Group by Features")
    group_by_columns = st.multiselect("Select Columns to Group By", df.columns, default=["drive-wheels", "body-style"])
    if group_by_columns:
        grouped_data = df.groupby(group_by_columns)["price"].mean().reset_index()
        st.write("### Grouped Data")
        st.write(grouped_data)

    st.write("### Pivot Table")
    if len(group_by_columns) >= 2:
        pivot_table = grouped_data.pivot(index=group_by_columns[0], columns=group_by_columns[1], values="price")
        st.write(pivot_table)

        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
