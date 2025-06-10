# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------
# Configuración inicial
# -------------------------
st.set_page_config(
    page_title="Titanic Visual App",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Sidebar informativo
# -------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/Titanic_Plan_1st_Class_Accommodation.png", width=260)
    st.markdown("## 📌 Navegación")
    st.markdown("- Parte 1: Análisis visual")
    st.markdown("- Parte 2: PCA Interactivo 3D")
    st.markdown("---")
    st.markdown("📧 Contacto académico: [tomas.briceno@ejemplo.com](mailto:tomas.briceno@ejemplo.com)")

# -------------------------
# Título de la app
# -------------------------
st.title("🧭 Titanic - Visualización Interactiva")
st.write("Aplicación para explorar el dataset del Titanic mediante gráficos analíticos y reducción de dimensionalidad.")

# -------------------------
# Cargar datos
# -------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv("MDAS-HVD_EVAL_2_Datos.csv")

df = cargar_datos()

st.markdown("### 🗂️ Vista previa del dataset")
st.dataframe(df.head())
st.markdown(f"**Columnas disponibles:** {', '.join(df.columns)}")

# -------------------------
# Tabs para visualizaciones
# -------------------------
tab1, tab2, tab3 = st.tabs(["📊 Supervivencia por clase", "📉 Edad y supervivencia", "🧬 PCA 3D"])

# -------------------------
# TAB 1: Supervivencia por clase
# -------------------------
with tab1:
    st.subheader("📊 ¿Cuál es la proporción de sobrevivientes según la clase del pasajero?")

    prop_clase = df.groupby(['Pclass', 'Survived']).size().reset_index(name='count')

    st.markdown("#### Versión estática (Seaborn)")
    fig1, ax1 = plt.subplots()
    sns.barplot(data=prop_clase, x='Pclass', y='count', hue='Survived', palette='Set2', ax=ax1)
    ax1.set_title("Distribución de Supervivencia por Clase")
    ax1.set_xlabel("Clase")
    ax1.set_ylabel("Cantidad")
    st.pyplot(fig1)

    st.markdown("#### Versión interactiva (Plotly)")
    fig2 = px.bar(prop_clase, x='Pclass', y='count', color='Survived',
                  title="Supervivencia por clase - Interactivo",
                  labels={'Pclass': 'Clase', 'count': 'Cantidad', 'Survived': 'Sobrevivió'})
    st.plotly_chart(fig2)

# -------------------------
# TAB 2: Distribución de edad según supervivencia
# -------------------------
with tab2:
    st.subheader("📉 ¿Cómo varía la distribución de edad entre los pasajeros que sobrevivieron y los que no?")

    st.markdown("#### Versión estática (Seaborn)")
    fig3, ax2 = plt.subplots()
    sns.kdeplot(data=df, x='Age', hue='Survived', fill=True, common_norm=False,
                palette='Set1', alpha=0.5, ax=ax2)
    ax2.set_title("Distribución de edades por condición de sobrevivencia")
    ax2.set_xlabel("Edad")
    st.pyplot(fig3)

    st.markdown("#### Versión interactiva (Plotly)")
    fig4 = px.histogram(df, x="Age", color="Survived", marginal="density", barmode="overlay",
                        title="Distribución de Edad por Supervivencia - Interactivo",
                        labels={'Age': 'Edad', 'Survived': 'Sobrevivió'})
    st.plotly_chart(fig4)

# -------------------------
# TAB 3: PCA Interactivo 3D
# -------------------------
with tab3:
    st.subheader("🧬 PCA Interactivo 3D sobre variables cuantitativas")

    numericas = df.select_dtypes(include='number').drop(columns=['Survived'], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numericas)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    pca_df["Survived"] = df["Survived"]

    grupo = st.radio("🎛️ Filtro de visualización", ["Todos", "Solo sobrevivientes", "Solo no sobrevivientes"])
    if grupo == "Solo sobrevivientes":
        pca_df_filtrado = pca_df[pca_df["Survived"] == 1]
    elif grupo == "Solo no sobrevivientes":
        pca_df_filtrado = pca_df[pca_df["Survived"] == 0]
    else:
        pca_df_filtrado = pca_df

    fig5 = px.scatter_3d(pca_df_filtrado, x="PC1", y="PC2", z="PC3",
                         color=pca_df_filtrado["Survived"].astype(str),
                         title="PCA 3D interactivo de variables cuantitativas",
                         labels={"color": "Sobrevivió"})
    st.plotly_chart(fig5)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("📘 Trabajo de Visualización de Datos - Evaluación 2 | Magíster en Ciencia de Datos | Junio 2025")
