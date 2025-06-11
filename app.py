import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuración inicial
st.set_page_config(page_title="Análisis Visual del Titanic", layout="wide")

# 🔔 Advertencia para móviles
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
📱 <strong>Advertencia:</strong> Esta aplicación ha sido optimizada para computadores de escritorio. En móviles algunos gráficos y funciones pueden no mostrarse correctamente. Para mejor experiencia, usa una pantalla grande.
</div>
""", unsafe_allow_html=True)

# Función para cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("MDAS-HVD_EVAL_2_Datos.csv")

df = cargar_datos()
df["Sobreviviente"] = df["Survived"].map({0: "No", 1: "Sí"})

# 📝 Storytelling de inicio
st.title("El Titanic: Más que datos, vidas")
st.markdown("""
> En abril de 1912, el **RMS Titanic**, orgullo de la ingeniería naval, partió en su viaje inaugural desde Southampton a Nueva York. Más de 2.200 personas iban a bordo. Solo 710 sobrevivieron.

Esta aplicación busca **dar sentido a los datos** de esa tragedia, explorando cómo la clase social, la edad y otras variables influyeron en las posibilidades de sobrevivir.

Los datos son reales. Cada punto representa una vida. Comencemos.
""")

st.dataframe(df.head())

tab1, tab2, tab3 = st.tabs(["Supervivencia por Clase", "Distribución de Edad", "Análisis PCA 3D"])

# ---------- TAB 1 ----------
with tab1:
    st.header("¿Sobrevivir o no? La importancia de la clase")
    st.markdown("""
Durante el naufragio, la **clase del pasaje** influyó drásticamente en las posibilidades de escapar.

¿Fue el acceso a los botes? ¿La ubicación en el barco? ¿La velocidad de respuesta?

Este gráfico muestra cómo la clase social marcó una diferencia.
""")

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

# ---------- TAB 2 ----------
with tab2:
    st.header("La edad no perdona... ¿o sí?")
    st.markdown("""
Uno podría pensar que los niños tendrían prioridad. ¿Pero fue así?

Estos gráficos permiten explorar si hubo diferencias en la supervivencia según la edad de los pasajeros.
""")

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

# ---------- TAB 3 ----------
with tab3:
    st.header("¿Y si lo vemos en tres dimensiones?")
    st.markdown("""
El Análisis de Componentes Principales (PCA) permite reducir múltiples variables a solo tres dimensiones visuales.

Aquí podrás observar una representación simplificada de los pasajeros, agrupados según si sobrevivieron o no.
""")

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

# ---------- CIERRE ----------
st.markdown("---")
st.markdown("""
### 🎯 Reflexión Final

Esta visualización no solo muestra datos, sino también decisiones humanas.

Cada punto representa una historia. Esta aplicación es un intento de entender, desde los datos, qué factores hicieron la diferencia aquella noche.

Los datos no hablan por sí solos. Somos nosotros quienes debemos **darles sentido**.
""")

st.caption("Aplicación desarrollada por Tomás Briceño — Evaluación 2, Magíster en Ciencia de Datos.")
