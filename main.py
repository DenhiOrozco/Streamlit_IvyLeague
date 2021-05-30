import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


siteHeader = st.beta_container()
with siteHeader:
    st.title('Modelo de Regresión Lineal')
    st.markdown(""" En este proyecto se busca encontrar cuáles son los
    parámetros principales que pueden predecir que una persona sea aceptada en una Ivy League""")
     
newFeatures = st.beta_container()
with newFeatures:
    st.header('Parámetros: ')
    st.markdown('Set de Datos de Kaggle, utilizando el dataset de "Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019"')
 
with st.beta_expander("Variables ", expanded=False):
    st.write(
        """
                - Puntuación GRE (max 340)
                - Puntuación TOEFL (max 120)
                - Ranking de la universidad (1 a 5)
                - Carta intención o ensayo (1 a 5)
                - Fortaleza de la carta de recomendación (1 a 5)
                - Promedio de la universidad o GPA (escala de 10)
                - Experiencia en investigación (1 para sí o 0 para no)
                - Posibilidad de admisión.
        """
    )

df= pd.read_csv('/Admission_Predict_Ver1.1.csv')
df.head()
df.shape
df.describe()
