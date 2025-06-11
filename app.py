# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuración general
st.set_page_config(
    page_title="Visualización Titanic - Evaluación 2",
    page_icon="🚢",
    layout="wide"
)

# Sidebar con detalles
with st.sidebar:
    st.title("🚢 Titanic Explorer")
    st.markdown("**Evaluación 2 - Magíster en Ciencia de Datos**")
    st.markdown("**Autor:** Tomás Briceño")
    st.info("Sube el archivo CSV original del Titanic")

# Carga de datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("MDAS-HVD_EVAL_2_Datos.csv")

df = cargar_datos()

# Conversión segura de columna binaria a string
df["Survived"] = df["Survived"].map({0: "No sobrevivió", 1: "Sobrevivió"})

# Vista previa
st.title("📊 Explorador Interactivo del Titanic")
st.dataframe(df.head())

tab1, tab2, tab3 = st.tabs(["🧍‍♂️ Supervivencia por Clase", "🎂 Edad y Supervivencia", "🧬 PCA 3D"])

# ---------- TAB 1 ----------
with tab1:
    st.header("🧍‍♂️ Supervivencia por Clase")

    prop_clase = df.groupby(['Pclass', 'Survived']).size().reset_index(name='count')

    st.subheader("Versión Estática (Seaborn)")
    fig1, ax1 = plt.subplots()
    sns.barplot(data=prop_clase, x='Pclass', y='count', hue='Survived', palette=["#d62728", "#2ca02c"], ax=ax1)
    ax1.set_title("Distribución de Supervivencia por Clase")
    ax1.set_xlabel("Clase")
    ax1.set_ylabel("Cantidad")
    st.pyplot(fig1)

    st.subheader("Versión Interactiva (Plotly)")
    fig2 = px.bar(prop_clase, x='Pclass', y='count', color='Survived',
                  labels={'Pclass': 'Clase', 'count': 'Cantidad', 'Survived': 'Sobrevivencia'},
                  color_discrete_map={"No sobrevivió": "#d62728", "Sobrevivió": "#2ca02c"},
                  title="Supervivencia por Clase (Interactivo)")
    st.plotly_chart(fig2)

# ---------- TAB 2 ----------
with tab2:
    st.header("🎂 Distribución de Edad según Supervivencia")

    df_age = df[["Age", "Survived"]].dropna()

    st.subheader("Versión Estática (Seaborn - KDE)")
    fig3, ax2 = plt.subplots()
    sns.kdeplot(data=df_age, x="Age", hue="Survived", fill=True,
                common_norm=False, palette=["#d62728", "#2ca02c"], ax=ax2)
    ax2.set_title("Densidad de Edad por Supervivencia")
    ax2.set_xlabel("Edad")
    st.pyplot(fig3)

    st.subheader("Versión Interactiva (Plotly)")
    fig4 = px.histogram(df_age, x="Age", color="Survived", marginal="density",
                        labels={'Age': 'Edad', 'Survived': 'Sobrevivencia'},
                        color_discrete_map={"No sobrevivió": "#d62728", "Sobrevivió": "#2ca02c"},
                        title="Distribución de Edad por Supervivencia (Interactivo)")
    st.plotly_chart(fig4)

# ---------- TAB 3 ----------
with tab3:
    st.header("🧬 PCA 3D sobre variables numéricas")

    df_numeric = df.select_dtypes(include='number').dropna()
    X = StandardScaler().fit_transform(df_numeric)
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2", "PC3"])
    pca_df["Survived"] = df.loc[df_numeric.index, "Survived"]

    filtro = st.radio("Filtrar por condición", ["Todos", "Solo sobrevivientes", "Solo no sobrevivientes"])
    if filtro == "Solo sobrevivientes":
        pca_df = pca_df[pca_df["Survived"] == "Sobrevivió"]
    elif filtro == "Solo no sobrevivientes":
        pca_df = pca_df[pca_df["Survived"] == "No sobrevivió"]

    fig5 = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Survived",
                         color_discrete_map={"No sobrevivió": "#d62728", "Sobrevivió": "#2ca02c"},
                         title="PCA 3D interactivo sobre variables cuantitativas")
    st.plotly_chart(fig5)

# Footer
st.markdown("---")
st.markdown("*App desarrollada por Tomás Briceño para la Evaluación 2 del Magíster en Ciencia de Datos*")
