# app.py

# -----------------------------
# Importar librerías necesarias
# -----------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Configuración inicial de la app
# -----------------------------
st.set_page_config(page_title="Análisis Titanic - EVAL 2", layout="wide")
st.title("Evaluación 2: Visualización de Datos - Titanic")
st.write("Aplicación desarrollada para explorar el conjunto de datos del Titanic de forma visual e interactiva.")

# -----------------------------
# Cargar el dataset
# -----------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv("MDAS-HVD_EVAL_2_Datos.csv")

df = cargar_datos()

# -----------------------------
# Vista previa
# -----------------------------
st.subheader("Vista previa del dataset")
st.dataframe(df.head())

# -----------------------------
# Parte 1 - Pregunta 1: Proporción de sobrevivientes según clase
# -----------------------------
st.header("Parte 1 - Visualizaciones Analíticas")
st.subheader("¿Cuál es la proporción de sobrevivientes según la clase del pasajero?")

# Agrupar datos
prop_clase = df.groupby(['pclass', 'survived']).size().reset_index(name='count')

# -------- Estática con Seaborn --------
st.markdown("### Versión estática (Seaborn)")
fig1, ax1 = plt.subplots()
sns.barplot(data=prop_clase, x='pclass', y='count', hue='survived', palette='Set2', ax=ax1)
ax1.set_title("Supervivencia por clase")
ax1.set_xlabel("Clase")
ax1.set_ylabel("Cantidad")
st.pyplot(fig1)

# -------- Interactiva con Plotly --------
st.markdown("### Versión interactiva (Plotly)")
fig2 = px.bar(prop_clase, x='pclass', y='count', color='survived',
              title="Supervivencia por clase (interactivo)",
              labels={'pclass': 'Clase', 'count': 'Cantidad', 'survived': 'Sobrevivió'})
st.plotly_chart(fig2)

# -----------------------------
# Parte 1 - Pregunta 2: Distribución de edad según sobrevivencia
# -----------------------------
st.subheader("¿Cómo varía la distribución de edad entre los pasajeros que sobrevivieron y los que no?")

# -------- Estática con Seaborn --------
st.markdown("### Versión estática (Seaborn)")
fig3, ax2 = plt.subplots()
sns.kdeplot(data=df, x='age', hue='survived', fill=True, common_norm=False, palette='Set1', alpha=0.5, ax=ax2)
ax2.set_title("Distribución de edades según sobrevivencia")
ax2.set_xlabel("Edad")
st.pyplot(fig3)

# -------- Interactiva con Plotly --------
st.markdown("### Versión interactiva (Plotly)")
fig4 = px.histogram(df, x="age", color="survived", marginal="density", barmode="overlay",
                    title="Distribución de edades según sobrevivencia (interactivo)",
                    labels={'age': 'Edad', 'survived': 'Sobrevivió'})
st.plotly_chart(fig4)

# -----------------------------
# Parte 2 - Visualización PCA 3D
# -----------------------------
st.header("Parte 2 - PCA Interactivo 3D")
st.markdown("Esta visualización se basa solo en las variables numéricas (cuantitativas).")

# Seleccionar solo variables numéricas
numericas = df.select_dtypes(include='number').drop(columns=['survived'], errors='ignore')

# Estandarizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numericas)

# Aplicar PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
pca_df["survived"] = df["survived"]

# Filtro interactivo
grupo = st.radio("Selecciona el grupo a visualizar", ["Todos", "Solo sobrevivientes", "Solo no sobrevivientes"])
if grupo == "Solo sobrevivientes":
    pca_df_filtrado = pca_df[pca_df["survived"] == 1]
elif grupo == "Solo no sobrevivientes":
    pca_df_filtrado = pca_df[pca_df["survived"] == 0]
else:
    pca_df_filtrado = pca_df

# Gráfico 3D interactivo
fig5 = px.scatter_3d(pca_df_filtrado, x="PC1", y="PC2", z="PC3", color=pca_df_filtrado["survived"].astype(str),
                     title="PCA 3D interactivo de variables cuantitativas",
                     labels={"color": "Sobrevivió"})
st.plotly_chart(fig5)

