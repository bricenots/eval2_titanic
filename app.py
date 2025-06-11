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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar con navegación y contexto
with st.sidebar:
    st.title("🚢 Titanic Explorer")
    st.markdown("### Evaluación 2 - Magíster en Ciencia de Datos")
    st.markdown("**Autor:** Tomás Briceño")
    st.markdown("**Institución:** [Nombre de tu universidad]")
    st.markdown("---")
    st.markdown("📂 Este visualizador permite explorar:")
    st.markdown("- Supervivencia por clase")
    st.markdown("- Distribución de edad")
    st.markdown("- Análisis PCA en 3D")
    st.markdown("---")
    st.info("Sube el archivo original como `MDAS-HVD_EVAL_2_Datos.csv`")

# Carga de datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("MDAS-HVD_EVAL_2_Datos.csv")

df = cargar_datos()

# Vista previa
st.title("📊 Explorador Interactivo del Titanic")
st.write("Visualizaciones analíticas basadas en el dataset real de pasajeros del Titanic.")
st.markdown("### 🗂️ Vista previa de los datos")
st.dataframe(df.head())
st.write(f"**Columnas detectadas:** {', '.join(df.columns)}")

# Tabs principales
tab1, tab2, tab3 = st.tabs(["🧍‍♂️ Supervivencia por Clase", "🎂 Edad y Supervivencia", "🧬 PCA 3D"])

# ------------------- TAB 1 -------------------
with tab1:
    st.header("🧍‍♂️ Supervivencia por Clase")
    prop_clase = df.groupby(['Pclass', 'Survived']).size().reset_index(name='count')

    st.subheader("Gráfico de barras (Seaborn)")
    fig1, ax1 = plt.subplots()
    sns.barplot(data=prop_clase, x='Pclass', y='count', hue='Survived', palette='Set2', ax=ax1)
    ax1.set_title("Distribución de Supervivencia por Clase")
    st.pyplot(fig1)

    st.subheader("Gráfico interactivo (Plotly)")
    fig2 = px.bar(prop_clase, x='Pclass', y='count', color='Survived',
                  labels={'Pclass': 'Clase', 'count': 'Cantidad', 'Survived': 'Sobrevivió'},
                  title="Supervivencia por Clase (Interactivo)")
    st.plotly_chart(fig2)

# ------------------- TAB 2 -------------------
with tab2:
    st.header("🎂 Distribución de Edad según Supervivencia")

    df_age = df[["Age", "Survived"]].dropna()

    st.subheader("Distribución KDE (Seaborn)")
    fig3, ax2 = plt.subplots()
    sns.kdeplot(data=df_age, x="Age", hue="Survived", fill=True, common_norm=False, palette='Set1', alpha=0.5, ax=ax2)
    ax2.set_title("Densidad de Edad por Supervivencia")
    st.pyplot(fig3)

    st.subheader("Histograma Interactivo (Plotly)")
    fig4 = px.histogram(df_age, x="Age", color="Survived", marginal="density",
                        labels={'Age': 'Edad', 'Survived': 'Sobrevivió'},
                        title="Distribución de Edad por Supervivencia (Interactivo)")
    st.plotly_chart(fig4)

# ------------------- TAB 3 -------------------
with tab3:
    st.header("🧬 Visualización PCA 3D")

    # Solo columnas numéricas y sin nulos
    numeric = df.select_dtypes(include='number').drop(columns=['Survived'], errors='ignore')
    numeric = numeric.dropna()

    # Escalado + PCA
    X_scaled = StandardScaler().fit_transform(numeric)
    pca = PCA(n_components=3)
    components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2", "PC3"])
    pca_df["Survived"] = df.loc[numeric.index, "Survived"]

    # Filtro de grupo
    filtro = st.radio("Filtrar por sobrevivencia", ["Todos", "Solo sobrevivientes", "Solo no sobrevivientes"])
    if filtro == "Solo sobrevivientes":
        pca_df = pca_df[pca_df["Survived"] == 1]
    elif filtro == "Solo no sobrevivientes":
        pca_df = pca_df[pca_df["Survived"] == 0]

    # Gráfico
    fig5 = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color=pca_df["Survived"].astype(str),
                         title="PCA 3D sobre variables cuantitativas",
                         labels={"color": "Sobrevivió"})
    st.plotly_chart(fig5)

# Footer
st.markdown("---")
st.markdown("🧠 *App desarrollada por Tomás Briceño - Evaluación 2 - Magíster Ciencia de Datos*")

