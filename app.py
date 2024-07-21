from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save_model
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull, save_model as clf_save_model
from pycaret.clustering import setup as clu_setup, create_model as clu_create, pull as clu_pull, save_model as clu_save_model
import pandas as pd
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://image.freepik.com/free-vector/artificial-intelligence-concept_23-2148623474.jpg")
    st.title("Your ML Companion")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Regression", "Classification", "Clustering", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    
    st.header('Data Summary')
    st.write(df.describe())
    
    st.header('Missing Values')
    st.write(df.isnull().sum())

    st.header('Correlation Matrix')
    corr_matrix = df.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr)

    st.header('Distribution of Numerical Features')
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_features:
        fig = px.histogram(df, x=col, title=f'Distribution of {col}')
        st.plotly_chart(fig)
    
    st.header('Distribution of Categorical Features')
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        fig = px.histogram(df, x=col, title=f'Distribution of {col}')
        st.plotly_chart(fig)

    st.header('Scatter Plots')
    chosen_target = st.selectbox('Choose the Target Column for Scatter Plots', numerical_features)
    for col in numerical_features:
        if col != chosen_target:
            fig = px.scatter(df, x=col, y=chosen_target, title=f'{col} vs {chosen_target}')
            st.plotly_chart(fig)

if choice == "Regression": 
    st.title("Regression Modelling")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Regression Modelling'): 
        reg_setup(df, target=chosen_target, verbose=True)
        setup_df = reg_pull()
        st.dataframe(setup_df)
        best_model = reg_compare()
        compare_df = reg_pull()
        st.dataframe(compare_df)
        reg_save_model(best_model, 'best_model_regression')

if choice == "Classification": 
    st.title("Classification Modelling")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Classification Modelling'): 
        clf_setup(df, target=chosen_target, verbose=True)
        setup_df = clf_pull()
        st.dataframe(setup_df)
        best_model = clf_compare()
        compare_df = clf_pull()
        st.dataframe(compare_df)
        clf_save_model(best_model, 'best_model_classification')

if choice == "Clustering": 
    st.title("Clustering Modelling")
    if st.button('Run Clustering Modelling'): 
        clu_setup(df, verbose=True)
        setup_df = clu_pull()
        st.dataframe(setup_df)
        best_model = clu_create('kmeans')
        compare_df = clu_pull()
        st.dataframe(compare_df)
        clu_save_model(best_model, 'best_model_clustering')

if choice == "Download": 
    st.title("Download Your Model")
    model_type = st.selectbox("Choose Model Type", ["Regression", "Classification", "Clustering"])
    model_file = f'best_model_{model_type.lower()}.pkl'
    with open(model_file, 'rb') as f: 
        st.download_button('Download Model', f, file_name=model_file)
