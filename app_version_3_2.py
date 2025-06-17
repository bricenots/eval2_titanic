
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Configuración general de la App
# -------------------------------
st.set_page_config(page_title="Análisis Titanic - Evaluación 2", layout="wide")
st.title("🚢 Supervivencia en el Titanic")
st.write("Esta aplicación está diseñada para ser visualizada desde un computador de escritorio. Presenta gráficos exploratorios sobre los pasajeros del Titanic, con enfoque visual y explicativo.")

# -------------------------------
# Cargar datos desde GitHub
# -------------------------------
DATA_URL = "https://raw.githubusercontent.com/TomasBricenoC/MDS/main/MDAS-HVD_EVAL_2_Datos.csv"
df = pd.read_csv(DATA_URL)

# -------------------------------
# Renombrar columnas para visualización
# -------------------------------
df = df.rename(columns={
    "Pclass": "Clase",
    "Survived": "Sobrevivencia"
})

# -------------------------------
# Gráfico 1: Proporción de sobrevivientes por clase
# -------------------------------
st.header("1️⃣ Proporción de Sobrevivientes por Clase")
st.markdown("Comparamos la proporción de personas que sobrevivieron o no según la clase del pasaje.")

prop_clase = df.groupby(['Clase', 'Sobrevivencia']).size().reset_index(name='Cantidad')
fig1 = px.bar(prop_clase, x="Clase", y="Cantidad", color="Sobrevivencia", barmode="group",
              labels={"Clase": "Clase", "Cantidad": "Número de pasajeros", "Sobrevivencia": "¿Sobrevivió?"},
              title="Sobrevivencia por Clase - Interactivo")
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# Gráfico 2: Distribución de edad según sobrevivencia
# -------------------------------
st.header("2️⃣ Distribución de Edad según Sobrevivencia")
st.markdown("Visualizamos cómo se distribuye la edad de los pasajeros que sobrevivieron o no.")

df_age = df.dropna(subset=["Age"])
df_age["Sobrevivencia"] = df_age["Sobrevivencia"].map({0: "No sobrevivió", 1: "Sobrevivió"})
fig2 = px.histogram(df_age, x="Age", color="Sobrevivencia", marginal="density",
                    labels={'Age': 'Edad', 'Sobrevivencia': 'Sobrevivencia'},
                    title="Distribución de Edad por Supervivencia (Interactivo)")
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Gráfico 3: PCA 3D sobre variables numéricas
# -------------------------------
st.header("3️⃣ Análisis de Componentes Principales (PCA)")
st.markdown("Se aplica PCA sobre las variables numéricas para observar posibles agrupaciones entre sobrevivientes y no sobrevivientes.")

# Variables numéricas
variables = ['Age', 'Fare', 'SibSp', 'Parch']
df_pca = df.dropna(subset=variables + ['Sobrevivencia'])
X = df_pca[variables]
y = df_pca['Sobrevivencia']
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=3)
components = pca.fit_transform(X_scaled)
df_pca_result = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
df_pca_result['Sobrevivencia'] = y.values

fig3 = px.scatter_3d(df_pca_result, x='PC1', y='PC2', z='PC3', color='Sobrevivencia',
                     title="PCA 3D Interactivo",
                     labels={'Sobrevivencia': 'Sobrevivencia'})
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# Conclusión
# -------------------------------
st.markdown("---")
st.subheader("🧠 Conclusión Final")
st.markdown("La clase del pasajero y la edad parecen influir significativamente en la probabilidad de sobrevivir. Las visualizaciones interactivas y el análisis PCA permiten identificar patrones importantes de forma clara.")
