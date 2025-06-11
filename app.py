import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Análisis Visual del Titanic", layout="wide")

@st.cache_data
def cargar_datos():
    return pd.read_csv("MDAS-HVD_EVAL_2_Datos.csv")

df = cargar_datos()
df["Sobreviviente"] = df["Survived"].map({0: "No", 1: "Sí"})

st.title("Análisis Visual del Titanic")
st.markdown("""
Este panel interactivo presenta un análisis visual del conjunto de datos del Titanic.
Permite explorar relaciones entre variables clave como la clase del pasajero, la edad y la supervivencia.
""")
st.dataframe(df.head())

tab1, tab2, tab3 = st.tabs(["Supervivencia por Clase", "Distribución de Edad", "Análisis PCA 3D"])

# ------------------- TAB 1 -------------------
with tab1:
    st.header("Supervivencia por Clase de Pasajero")
    st.markdown("Este gráfico compara la cantidad de personas que sobrevivieron y no sobrevivieron según su clase de ticket.")

    resumen_clase = df.groupby(["Pclass", "Sobreviviente"]).size().reset_index(name="Cantidad")

    fig1, ax1 = plt.subplots()
    sns.barplot(data=resumen_clase, x="Pclass", y="Cantidad", hue="Sobreviviente",
                palette=["#d62728", "#2ca02c"], ax=ax1)
    ax1.set_title("Supervivencia según Clase de Pasaje")
    ax1.set_xlabel("Clase")
    ax1.set_ylabel("Cantidad de Pasajeros")
    st.pyplot(fig1)

    fig2 = px.bar(resumen_clase, x="Pclass", y="Cantidad", color="Sobreviviente",
                  color_discrete_map={"No": "#d62728", "Sí": "#2ca02c"},
                  labels={"Pclass": "Clase", "Cantidad": "Cantidad de Pasajeros"},
                  title="Supervivencia por Clase (Interactivo)")
    st.plotly_chart(fig2)

# ------------------- TAB 2 -------------------
with tab2:
    st.header("Distribución de Edad y Supervivencia")
    st.markdown("Este gráfico muestra cómo varía la edad entre quienes sobrevivieron y quienes no.")

    df_edad = df[["Age", "Sobreviviente"]].dropna()

    fig3, ax2 = plt.subplots()
    sns.kdeplot(data=df_edad, x="Age", hue="Sobreviviente", fill=True,
                palette=["#d62728", "#2ca02c"], ax=ax2)
    ax2.set_title("Densidad de Edad por Supervivencia")
    ax2.set_xlabel("Edad")
    ax2.set_ylabel("Densidad")
    st.pyplot(fig3)

    try:
        df_edad["Sobreviviente"] = df_edad["Sobreviviente"].astype(str)
        fig4 = px.histogram(df_edad, x="Age", color="Sobreviviente",
                            color_discrete_map={"No": "#d62728", "Sí": "#2ca02c"},
                            labels={"Age": "Edad", "Sobreviviente": "Supervivencia"},
                            title="Histograma de Edad por Supervivencia (Interactivo)")
        st.plotly_chart(fig4)
    except Exception as e:
        st.error("No se pudo generar el histograma interactivo.")
        st.exception(e)

# ------------------- TAB 3 -------------------
with tab3:
    st.header("Reducción de Dimensiones (PCA 3D)")
    st.markdown("Este gráfico tridimensional representa las características numéricas de cada pasajero \
utilizando Análisis de Componentes Principales (PCA), agrupadas por su condición de supervivencia.")

    datos_numericos = df.select_dtypes(include="number").dropna()
    X = StandardScaler().fit_transform(datos_numericos)
    pca = PCA(n_components=3)
    componentes = pca.fit_transform(X)
    pca_df = pd.DataFrame(componentes, columns=["CP1", "CP2", "CP3"])
    pca_df["Sobreviviente"] = df.loc[datos_numericos.index, "Sobreviviente"]

    filtro = st.radio("Filtrar por:", ["Todos", "Solo sobrevivientes", "Solo no sobrevivientes"])
    if filtro == "Solo sobrevivientes":
        pca_df = pca_df[pca_df["Sobreviviente"] == "Sí"]
    elif filtro == "Solo no sobrevivientes":
        pca_df = pca_df[pca_df["Sobreviviente"] == "No"]

    fig5 = px.scatter_3d(pca_df, x="CP1", y="CP2", z="CP3", color="Sobreviviente",
                         color_discrete_map={"No": "#d62728", "Sí": "#2ca02c"},
                         title="Visualización 3D PCA por Supervivencia")
    st.plotly_chart(fig5)

st.markdown("---")
st.caption("Aplicación desarrollada para la Evaluación 2 del Magíster en Ciencia de Datos.")
