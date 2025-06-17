
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuración general de la página
st.set_page_config(page_title="📉 Historias del Titanic", layout="wide")

# Título principal y narrativa introductoria
st.title("🌊 Historias del Titanic: Visualización Interactiva")
st.markdown("Explora los datos del Titanic desde una perspectiva visual y reflexiva. "
            "Cada gráfico representa decisiones, vidas y realidades distintas.")
# Carga de datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("MDAS-HVD_EVAL_2_Datos.csv")

if archivo is not None:
    df = pd.read_csv(archivo)

    # Limpieza preliminar
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    st.subheader("🔍 Vista previa de los datos")
    st.dataframe(df.head(10))

    # Conversión para gráficos
    df["Survived"] = df["Survived"].map({0: "No sobrevivió", 1: "Sobrevivió"})

    # Gráfico 1: Barras de proporción por clase
    st.markdown("## 📊 Proporción de Supervivientes por Clase")
    prop_clase = df.groupby(['Pclass', 'Survived']).size().reset_index(name='count')
    fig1 = px.bar(prop_clase, x='Pclass', y='count', color='Survived',
                  barmode='group',
                  labels={"Pclass": "Clase", "count": "Cantidad"},
                  title="Supervivencia por Clase")
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2: Histograma de edades
    st.markdown("## 🧒 Distribución de Edades según Supervivencia")
    if "Age" in df.columns:
        df_age = df[["Age", "Survived"]].dropna()
        fig2 = px.histogram(df_age, x="Age", color="Survived", marginal="density",
                            nbins=30,
                            labels={"Age": "Edad", "Survived": "Sobrevivencia"},
                            title="Distribución de Edad por Supervivencia")
        st.plotly_chart(fig2, use_container_width=True)

    # Gráfico 3: PCA interactivo
    st.markdown("## 🧬 Análisis de Componentes Principales (PCA) 3D")
    columnas_numericas = df.select_dtypes(include='number').drop(columns=['PassengerId'], errors='ignore')
    columnas_numericas = columnas_numericas.dropna()
    X_std = StandardScaler().fit_transform(columnas_numericas)
    pca = PCA(n_components=3)
    componentes = pca.fit_transform(X_std)

    df_pca = pd.DataFrame(componentes, columns=["PC1", "PC2", "PC3"])
    df_pca["Survived"] = df["Survived"][:len(df_pca)].reset_index(drop=True)

    fig3 = px.scatter_3d(df_pca, x="PC1", y="PC2", z="PC3", color="Survived",
                         title="PCA 3D Interactivo - Titanic",
                         labels={"Survived": "Sobrevivencia"})
    st.plotly_chart(fig3, use_container_width=True)

    # Conclusión narrativa
    st.markdown("---")
    st.markdown("### 📝 Reflexión Final")
    st.markdown("Los datos del Titanic no solo son cifras, son historias humanas. A través de esta visualización, "
                "esperamos haber ofrecido una mirada comprensiva y empática de este evento histórico.")
    st.markdown("<small><i>Optimizado para pantallas de PC. Visualización interactiva sujeta a capacidad del navegador.</i></small>", unsafe_allow_html=True)

else:
    st.info("Por favor, sube un archivo CSV válido para comenzar.")
