
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Configuración general de la página
# -------------------------------
st.set_page_config(page_title="Análisis Titanic - Evaluación 2", layout="wide")
st.markdown("### 🚢 Evaluación 2 - Análisis Visual del Titanic")
st.markdown("Esta aplicación está diseñada para visualizar datos del Titanic y presentar hallazgos relevantes.")
st.markdown("🔍 *Optimizado para pantallas de computador.*")
st.markdown("---")

# -------------------------------
# Carga directa del CSV desde el entorno
# -------------------------------
try:
    df = pd.read_csv("MDAS-HVD_EVAL_2_Datos.csv")
    st.success("✅ Datos del Titanic cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# Traducción de columnas para mejor lectura
df.rename(columns={
    'survived': 'Sobrevivencia',
    'pclass': 'Clase',
    'sex': 'Sexo',
    'age': 'Edad',
    'sibsp': 'Hermanos/Pareja',
    'parch': 'Padres/Hijos',
    'fare': 'Tarifa'
}, inplace=True)

# -------------------------------
# Gráfico 1: Proporción de sobrevivientes según clase
# -------------------------------
st.markdown("#### 🎯 Proporción de Sobrevivientes por Clase")

prop_clase = df.groupby(['Clase', 'Sobrevivencia']).size().reset_index(name='Cantidad')
fig1 = px.bar(prop_clase, x='Clase', y='Cantidad', color='Sobrevivencia', barmode='group',
              labels={'Cantidad': 'Cantidad de Personas'}, title="Sobrevivencia por Clase")
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# Gráfico 2: Distribución de edad según sobrevivencia
# -------------------------------
st.markdown("#### 👶📈 Distribución de Edad según Sobrevivencia")

df_age = df[['Edad', 'Sobrevivencia']].dropna()
fig2 = px.histogram(df_age, x="Edad", color="Sobrevivencia", marginal="density",
                    labels={'Edad': 'Edad', 'Sobrevivencia': 'Sobrevivencia'},
                    color_discrete_map={"No": "#d62728", "Sí": "#2ca02c"},
                    title="Distribución de Edad por Supervivencia (Interactivo)")
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Gráfico 3: PCA 3D con variables cuantitativas
# -------------------------------
st.markdown("#### 🧬 Análisis de Componentes Principales (PCA)")

df_pca = df[['Edad', 'Hermanos/Pareja', 'Padres/Hijos', 'Tarifa']].dropna()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pca)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df_scaled)

df_pca_plot = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
df_pca_plot['Sobrevivencia'] = df.loc[df_pca.index, 'Sobrevivencia'].values

fig3 = px.scatter_3d(df_pca_plot, x='PC1', y='PC2', z='PC3',
                     color='Sobrevivencia', title="PCA 3D Interactivo",
                     labels={'Sobrevivencia': 'Sobrevivencia'})
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# Conclusión narrativa
# -------------------------------
st.markdown("---")
st.markdown("### 📌 Conclusiones")
st.markdown("""
- Las personas en **clase alta** tenían mayor tasa de supervivencia.
- Los **niños** y **jóvenes adultos** sobrevivieron más que personas mayores.
- El análisis PCA sugiere que existe una correlación visual entre las variables cuantitativas y la probabilidad de sobrevivencia.

> Esta app es parte de una evaluación académica, desarrollada con fines educativos bajo los lineamientos de visualización de datos.
""")
