
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Título y presentación
# -------------------------------
st.set_page_config(page_title="Explorador Titanic", layout="wide")
st.title("🚢 Análisis Visual del Titanic")
st.markdown("### Evaluación 2 - Herramientas de Visualización de Datos")
st.markdown("Esta aplicación permite explorar visualmente el dataset del Titanic con visualizaciones interactivas.")

st.sidebar.markdown("## Navegación")
st.sidebar.markdown("Usa las secciones para responder preguntas analíticas y observar una proyección 3D del dataset.")
st.sidebar.info("⚠️ Esta app está optimizada para visualizarse en PC.")

# -------------------------------
# Cargar datos desde GitHub
# -------------------------------
DATA_URL = "https://raw.githubusercontent.com/bricenots/eval2_titanic/main/MDAS-HVD_EVAL_2_Datos.csv"
df = pd.read_csv(DATA_URL)

# -------------------------------
# Renombrar columnas para visualización
# -------------------------------
df.rename(columns={
    'Survived': 'Sobrevivencia',
    'Pclass': 'Clase',
    'Age': 'Edad',
    'Sex': 'Sexo',
    'Fare': 'Tarifa',
    'SibSp': 'Hermanos/Pareja',
    'Parch': 'Padres/Hijos'
}, inplace=True)

# -------------------------------
# Gráfico 1 - Proporción de sobrevivientes por clase
# -------------------------------
st.subheader("1️⃣ ¿Cuál es la proporción de sobrevivientes según la clase del pasajero?")

prop_clase = df.groupby(['Clase', 'Sobrevivencia']).size().reset_index(name='Cantidad')
fig1 = px.bar(
    prop_clase,
    x='Clase',
    y='Cantidad',
    color='Sobrevivencia',
    barmode='group',
    text='Cantidad',
    labels={'Clase': 'Clase del Pasajero', 'Cantidad': 'Cantidad de Personas', 'Sobrevivencia': 'Sobrevivencia'},
    title="Proporción de Sobrevivientes por Clase"
)
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# Gráfico 2 - Distribución de edad por sobrevivencia
# -------------------------------
st.subheader("2️⃣ ¿Cómo varía la distribución de edad entre sobrevivientes y no sobrevivientes?")

df_age = df[df['Edad'].notnull()]
fig2 = px.histogram(
    df_age,
    x="Edad",
    color="Sobrevivencia",
    marginal="density",
    labels={'Edad': 'Edad', 'Sobrevivencia': 'Sobrevivencia'},
    title="Distribución de Edad por Sobrevivencia"
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Gráfico 3D PCA - Visualización con reducción de dimensionalidad
# -------------------------------
st.subheader("3️⃣ Representación del Titanic en 3D (PCA)")

df_pca = df[['Edad', 'Tarifa', 'Hermanos/Pareja', 'Padres/Hijos']].dropna()
pca_features = StandardScaler().fit_transform(df_pca)
pca_result = PCA(n_components=3).fit_transform(pca_features)

df_viz = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
df_viz['Sobrevivencia'] = df.loc[df_pca.index, 'Sobrevivencia'].astype(str)

fig3 = px.scatter_3d(
    df_viz,
    x='PC1',
    y='PC2',
    z='PC3',
    color='Sobrevivencia',
    title='Reducción de Dimensiones con PCA (3D)',
    labels={'Sobrevivencia': 'Sobrevivencia'}
)
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Creado por **Tomás Briceño** para la Evaluación 2 de MDAS-HVD | Magíster Ciencia de Datos 🌐")
