
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Análisis Visual del Titanic", layout="wide")

# Advertencia para móviles
st.markdown("""
<style>
@media screen and (max-width: 800px) {
    .mobile-warning {
        display: block;
        padding: 1em;
        background-color: #fff3cd;
        border-left: 6px solid #ffa500;
        font-size: 15px;
        color: #856404;
        margin-bottom: 1.5em;
        border-radius: 5px;
    }
}
</style>
<div class='mobile-warning'>
📱 <strong>Advertencia:</strong> Esta aplicación ha sido optimizada para computadores de escritorio. En móviles algunos gráficos y funciones pueden no mostrarse correctamente.
</div>
""", unsafe_allow_html=True)

# Sidebar mejorado
st.sidebar.markdown("### 🧭 Explorador del Titanic")
idioma = st.sidebar.radio("Selecciona idioma:", ["Español"], index=0, help="Función decorativa para mantener visible el panel lateral")
st.sidebar.markdown("Explora la tragedia del Titanic a través de datos históricos.")
st.sidebar.markdown("**Colores:** 🟩 Sí sobrevivió | 🟥 No sobrevivió")

# Carga de datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("MDAS-HVD_EVAL_2_Datos.csv")

df = cargar_datos()
df["Sobreviviente"] = df["Survived"].map({0: "No", 1: "Sí"})

# Título y descripción
st.title("El Titanic: Más que datos, vidas")
st.markdown("> Cada gráfico representa decisiones humanas. Esta app busca entender, desde los datos, qué marcó la diferencia en la tragedia.")

tab1, tab2, tab3 = st.tabs(["Supervivencia por Clase", "Distribución de Edad", "PCA 3D"])

# Tab 1: Clase
with tab1:
    st.header("Impacto de la clase social en la supervivencia")
    resumen_clase = df.groupby(["Pclass", "Sobreviviente"]).size().reset_index(name="Cantidad")

    fig1, ax = plt.subplots()
    sns.barplot(data=resumen_clase, x="Pclass", y="Cantidad", hue="Sobreviviente", palette=["#d62728", "#2ca02c"], ax=ax)
    ax.set_title("Supervivencia según Clase de Pasaje")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad")
    st.pyplot(fig1)

    fig2 = px.bar(resumen_clase, x="Pclass", y="Cantidad", color="Sobreviviente",
                  color_discrete_map={"No": "#d62728", "Sí": "#2ca02c"},
                  title="Supervivencia por Clase (Interactivo)",
                  labels={"Pclass": "Clase", "Cantidad": "Cantidad de Pasajeros"})
    st.plotly_chart(fig2)

# Tab 2: Edad
with tab2:
    st.header("Distribución de Edad y Supervivencia")
    df_edad = df[["Age", "Sobreviviente"]].dropna()

    fig3, ax2 = plt.subplots()
    sns.kdeplot(data=df_edad, x="Age", hue="Sobreviviente", fill=True, palette=["#d62728", "#2ca02c"], ax=ax2)
    ax2.set_title("Densidad de Edad por Supervivencia")
    ax2.set_xlabel("Edad")
    ax2.set_ylabel("Densidad")
    st.pyplot(fig3)

    try:
        fig4 = px.histogram(df_edad, x="Age", color="Sobreviviente",
                            marginal="rug", opacity=0.7,
                            color_discrete_map={"No": "#d62728", "Sí": "#2ca02c"},
                            labels={"Age": "Edad", "Sobreviviente": "Supervivencia"},
                            title="Histograma de Edad (Interactivo)")
        st.plotly_chart(fig4)
    except Exception as e:
        st.warning("No se pudo generar el histograma interactivo.")

# Tab 3: PCA 3D
with tab3:
    st.header("Agrupamiento multivariable con PCA")
    datos_numericos = df.select_dtypes(include="number").dropna()
    X = StandardScaler().fit_transform(datos_numericos)
    pca = PCA(n_components=3)
    componentes = pca.fit_transform(X)
    pca_df = pd.DataFrame(componentes, columns=["CP1", "CP2", "CP3"])
    pca_df["Sobreviviente"] = df.loc[datos_numericos.index, "Sobreviviente"]

    filtro = st.radio("Filtrar por supervivencia:", ["Todos", "Solo sobrevivientes", "Solo no sobrevivientes"])
    if filtro == "Solo sobrevivientes":
        pca_df = pca_df[pca_df["Sobreviviente"] == "Sí"]
    elif filtro == "Solo no sobrevivientes":
        pca_df = pca_df[pca_df["Sobreviviente"] == "No"]

    fig5 = px.scatter_3d(pca_df, x="CP1", y="CP2", z="CP3", color="Sobreviviente",
                         color_discrete_map={"No": "#d62728", "Sí": "#2ca02c"},
                         title="PCA 3D: Agrupación por Supervivencia")
    st.plotly_chart(fig5)

# Cierre
st.markdown("---")
st.markdown("🧠 Cada gráfico representa vidas. Esta visualización invita a reflexionar más allá de los datos.")
