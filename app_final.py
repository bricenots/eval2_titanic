
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Visualización Titanic", layout="wide")
st.sidebar.markdown("⚠️ Esta aplicación está diseñada para usarse en PC o pantallas grandes.")

st.title("🚢 Exploración Visual del Titanic")
st.markdown("""
Esta aplicación permite explorar visualmente el conjunto de datos del Titanic mediante gráficos analíticos 
y una representación 3D basada en reducción de dimensionalidad (PCA).
""")

@st.cache_data
def cargar_datos():
    url = "https://github.com/bricenots/eval2_titanic/raw/main/MDAS-HVD_EVAL_2_Datos.csv"
    df = pd.read_csv(url)
    df.rename(columns={"Survived": "Sobrevivencia", "Pclass": "Clase", "Age": "Edad"}, inplace=True)
    df["Sobrevivencia"] = df["Sobrevivencia"].map({0: "No", 1: "Sí"})
    return df

df = cargar_datos()

tab1, tab2 = st.tabs(["📊 Análisis Exploratorio", "🧠 Visualización PCA 3D"])

with tab1:
    st.subheader("1️⃣ Proporción de Sobrevivientes según Clase del Pasajero")
    prop_clase = df.groupby(["Clase", "Sobrevivencia"]).size().reset_index(name="Cantidad")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.barplot(data=prop_clase, x="Clase", y="Cantidad", hue="Sobrevivencia", ax=ax)
        ax.set_title("Sobrevivencia por Clase (Estático)")
        st.pyplot(fig)

    with col2:
        fig2 = px.bar(prop_clase, x="Clase", y="Cantidad", color="Sobrevivencia", barmode="group",
                      color_discrete_map={"Sí": "green", "No": "red"},
                      title="Sobrevivencia por Clase (Interactivo)")
        st.plotly_chart(fig2)

    st.subheader("2️⃣ Distribución de Edad según Sobrevivencia")
    df_age = df.dropna(subset=["Edad"])

    col3, col4 = st.columns(2)
    with col3:
        fig3, ax3 = plt.subplots()
        sns.kdeplot(data=df_age, x="Edad", hue="Sobrevivencia", fill=True, ax=ax3)
        ax3.set_title("Distribución de Edad (KDE)")
        st.pyplot(fig3)

    with col4:
        try:
            fig4 = px.histogram(df_age, x="Edad", color="Sobrevivencia", marginal="density",
                                nbins=30, opacity=0.7,
                                color_discrete_map={"Sí": "green", "No": "red"},
                                title="Distribución de Edad (Interactivo)")
            st.plotly_chart(fig4)
        except Exception as e:
            st.error("Error al generar gráfico interactivo de edad.")

with tab2:
    st.subheader("🧠 PCA - Visualización 3D de Datos")
    df_pca = df[["Edad", "Clase"]].dropna()
    df_pca_std = StandardScaler().fit_transform(df_pca)
    pca = PCA(n_components=3)
    components = pca.fit_transform(df_pca_std)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2", "PC3"])
    pca_df["Sobrevivencia"] = df.loc[df_pca.index, "Sobrevivencia"].values

    filtro = st.radio("Filtrar por sobrevivencia:", ["Todos", "Sí", "No"])
    if filtro != "Todos":
        pca_df = pca_df[pca_df["Sobrevivencia"] == filtro]

    fig_pca = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Sobrevivencia",
                            color_discrete_map={"Sí": "green", "No": "red"},
                            title="Visualización 3D (PCA)")
    st.plotly_chart(fig_pca)
