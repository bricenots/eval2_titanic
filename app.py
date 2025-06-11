import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Titanic Visualizador", page_icon="🚢", layout="wide")

@st.cache_data
def cargar_datos():
    return pd.read_csv("MDAS-HVD_EVAL_2_Datos.csv")

df = cargar_datos()
df["Survived"] = df["Survived"].map({0: "No sobrevivió", 1: "Sobrevivió"})

st.title("📊 Explorador del Titanic")
st.dataframe(df.head())

tab1, tab2, tab3 = st.tabs(["🧍‍♂️ Supervivencia por Clase", "🎂 Edad y Supervivencia", "🧬 PCA 3D"])

# ------------------- TAB 1 -------------------
with tab1:
    st.header("🧍‍♂️ Supervivencia por Clase")
    prop_clase = df.groupby(["Pclass", "Survived"]).size().reset_index(name="count")

    st.subheader("Gráfico Estático")
    fig1, ax1 = plt.subplots()
    sns.barplot(data=prop_clase, x="Pclass", y="count", hue="Survived",
                palette=["#d62728", "#2ca02c"], ax=ax1)
    ax1.set_title("Distribución por Clase")
    st.pyplot(fig1)

    st.subheader("Gráfico Interactivo")
    fig2 = px.bar(prop_clase, x="Pclass", y="count", color="Survived",
                  color_discrete_map={"No sobrevivió": "#d62728", "Sobrevivió": "#2ca02c"},
                  labels={"Pclass": "Clase", "count": "Cantidad"})
    st.plotly_chart(fig2)

# ------------------- TAB 2 -------------------
with tab2:
    st.header("🎂 Edad y Supervivencia")
    df_age = df[["Age", "Survived"]].dropna()

    st.subheader("Densidad KDE (Seaborn)")
    fig3, ax2 = plt.subplots()
    sns.kdeplot(data=df_age, x="Age", hue="Survived", fill=True,
                palette=["#d62728", "#2ca02c"], ax=ax2)
    ax2.set_title("Distribución de Edad")
    st.pyplot(fig3)

    st.subheader("Histograma Interactivo (Plotly)")
    fig4 = px.histogram(df_age, x="Age", color="Survived",
                        color_discrete_map={"No sobrevivió": "#d62728", "Sobrevivió": "#2ca02c"},
                        labels={"Age": "Edad"})
    st.plotly_chart(fig4)

# ------------------- TAB 3 -------------------
with tab3:
    st.header("🧬 PCA 3D")
    df_numeric = df.select_dtypes(include="number").dropna()
    X = StandardScaler().fit_transform(df_numeric)
    pca = PCA(n_components=3)
    comps = pca.fit_transform(X)
    pca_df = pd.DataFrame(comps, columns=["PC1", "PC2", "PC3"])
    pca_df["Survived"] = df.loc[df_numeric.index, "Survived"]

    filtro = st.radio("Filtrar por condición", ["Todos", "Solo sobrevivientes", "Solo no sobrevivientes"])
    if filtro == "Solo sobrevivientes":
        pca_df = pca_df[pca_df["Survived"] == "Sobrevivió"]
    elif filtro == "Solo no sobrevivientes":
        pca_df = pca_df[pca_df["Survived"] == "No sobrevivió"]

    fig5 = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Survived",
                         color_discrete_map={"No sobrevivió": "#d62728", "Sobrevivió": "#2ca02c"},
                         title="PCA 3D Interactivo")
    st.plotly_chart(fig5)

st.markdown("---")
st.markdown("*Desarrollado por Tomás Briceño para Evaluación 2*")
