# app.py

# -----------------------------
# 🧱 Importar librerías necesarias
# -----------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 🧭 Configuración inicial de la app
# -----------------------------
st.set_page_config(page_title="Análisis Titanic - EVAL 2", layout="wide")
st.title("🚢 Evaluación 2: Visualización de Datos - Titanic")
st.write("Aplicación desarrollada para explorar el conjunto de datos del Titanic de forma visual e interactiva.")

# -----------------------------
# 📁 Cargar el dataset (archivo local titanic.csv)
# -----------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv("MDAS-HVD_EVAL_2_Datos.csv")  # Usa el nombre real entregado

df = cargar_datos()

# Mostrar vista previa
st.subheader("👁️ Vista previa del dataset")
st.dataframe(df.head())
