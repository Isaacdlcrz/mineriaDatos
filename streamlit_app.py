import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import random
from skimage import io
import os
from datetime import datetime
import osmnx as ox
import networkx as nx

# Ruta de la carpeta 
dataDirectory = './DatosAbiertosMiBici'

# Obtener la lista de nombres de archivos
files_name = os.listdir(dataDirectory)
files_name = [f for f in files_name if os.path.isfile(os.path.join(dataDirectory, f)) and f.startswith('datos_abiertos')]
files_name.sort()

st.title("Programación Para Minería de Datos")
st.subheader(":blue[Análisis de Datos de MiBici]")

# Seleccion de archivo para analizar
selected_file = st.selectbox('Seleccione archivo a analizar:', files_name)
st.divider()
try:
    datos_df = pd.read_csv(f'./DatosAbiertosMiBici/{selected_file}', parse_dates=['Inicio_del_viaje', 'Fin_del_viaje'], encoding='utf-8-sig')
except UnicodeDecodeError:
    datos_df = pd.read_csv(f'./DatosAbiertosMiBici/{selected_file}', parse_dates=['Inicio_del_viaje', 'Fin_del_viaje'], encoding='latin-1')

# Archivo nomenclatura
nomenclatura_df = pd.read_csv(f'./DatosAbiertosMiBici/nomenclatura_2024_07.csv', encoding='latin-1')

# -----------------------
# Imputacion de datos
# -----------------------
# Count null values and values less than 1900 before the conversion
null_count_before = datos_df['Año_de_nacimiento'].isnull().sum()
less_than_1900_count_before = (datos_df['Año_de_nacimiento'] < 1900).sum()

# Calculate the average of valid birth years (non-null and ≥ 1900)
valid_birth_years = datos_df.loc[(datos_df['Año_de_nacimiento'] >= 1900) & (~datos_df['Año_de_nacimiento'].isna()), 'Año_de_nacimiento']
birth_year_avg = valid_birth_years.mean()

# Fill the null values and the values less than 1900 with the average
datos_df['Año_de_nacimiento'] = datos_df['Año_de_nacimiento'].apply(lambda x: round(birth_year_avg) if pd.isnull(x) or x < 1900 else x)

# Count null values and values less than 1900 after the conversion
null_count_after = datos_df['Año_de_nacimiento'].isnull().sum()
less_than_1900_count_after = (datos_df['Año_de_nacimiento'] < 1900).sum()
avg_after = datos_df['Año_de_nacimiento'].mean()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Antes de la imputación:")
    st.text(f'Valores con año de nacimiento nulos: {null_count_before}')
    st.text(f'Valores menores a 1900: {less_than_1900_count_before}')
    st.text(f'Media: {birth_year_avg}')

with col2:
    st.subheader("Después de la imputación:")
    st.text(f'Valores con año de nacimiento nulos: {null_count_after}')
    st.text(f'Valores menores a 1900: {less_than_1900_count_after}')
    st.text(f'Media: {avg_after}')



# -----------------------

datos_df['Duracion'] = (datos_df['Fin_del_viaje'] - datos_df['Inicio_del_viaje']).dt.total_seconds() / 60 

def calcular_dinero_gastado(duracion):
    if duracion <= 30:
        return 0
    elif duracion <= 60:
        return 28
    else:
        # Restamos 60 minutos, luego contamos cuántas medias horas adicionales hay
        medias_horas_extra = (duracion - 60) / 30
        return 28 + 39 * np.ceil(medias_horas_extra)

# Aplicar la función para calcular el dinero gastado en base a la duración del viaje
datos_df['Dinero_gastado'] = datos_df['Duracion'].apply(calcular_dinero_gastado)

current_year = datetime.now().year
datos_df['Edad'] = current_year - datos_df['Año_de_nacimiento'] 

# Join de datos nomenclatura
# Realizar la primera unión por la columna 'Origen_Id'
datos_df = pd.merge(
    datos_df, nomenclatura_df, how='left', left_on='Origen_Id', right_on='id',
    suffixes=('', '_origen')
)

# Renombrar las columnas relevantes para el origen
datos_df = datos_df.rename(
    columns={
        'name': 'Origen_Nombre',
        'latitude': 'Origen_Latitud',
        'longitude': 'Origen_Longitud'
    }
)

# Realizar la segunda unión por la columna 'Destino_Id'
datos_df = pd.merge(
    datos_df, nomenclatura_df, how='left', left_on='Destino_Id', right_on='id',
    suffixes=('', '_destino')
)

# Renombrar las columnas relevantes para el destino
datos_df = datos_df.rename(
    columns={
        'name': 'Destino_Nombre',
        'latitude': 'Destino_Latitud',
        'longitude': 'Destino_Longitud'
    }
)

# Eliminar las columnas 'id' que ya no son necesarias
datos_df = datos_df.drop(columns=['id', 'id_destino'])

# ------------------------------------------------------
#                   En mantenimiento
#                 Cálculo de distancias
# ------------------------------------------------------
# # Calculando distancia de viajes
# # https://github.com/Project-OSRM/osrm-backend
# # Step 1: Download the street network for biking in a specific city
# G = ox.graph_from_place('Guadalajara, Jalisco, Mexico', network_type='bike')
# # fig, ax = plt.subplots(figsize=(10, 10))
# # ox.plot_graph(G, ax=ax, show=False, close=False, edge_color='lightgrey')
# # all_routes = []

# # Create a mapping of coordinates to nearest nodes
# def get_nearest_nodes(coords):
#     return [ox.distance.nearest_nodes(G, lon, lat) for lat, lon in coords]

# # Precompute nearest nodes for origins and destinations
# orig_coords = datos_df[['Origen_Latitud', 'Origen_Longitud']].drop_duplicates().values.tolist()
# dest_coords = datos_df[['Destino_Latitud', 'Destino_Longitud']].drop_duplicates().values.tolist()

# # Get the nearest nodes
# orig_nodes = get_nearest_nodes(orig_coords)
# dest_nodes = get_nearest_nodes(dest_coords)

# # Create a mapping from coordinates to nodes
# orig_node_map = dict(zip(map(tuple, orig_coords), orig_nodes))
# dest_node_map = dict(zip(map(tuple, dest_coords), dest_nodes))
# print(orig_node_map)
# print(dest_node_map)

# def calculate_route_length(row):
#     orig_node = orig_node_map.get((row['Origen_Latitud'], row['Origen_Longitud']))
#     dest_node = dest_node_map.get((row['Destino_Latitud'], row['Destino_Longitud']))
    
#     if orig_node is None or dest_node is None:
#         return np.nan  # No nearest node found

#     # Calculate the shortest path length
#     try:
#         # shortest_route = nx.shortest_path(G, orig_node, dest_node, weight='length')
#         route_length = nx.shortest_path_length(G, orig_node, dest_node, weight='length')
#         # all_routes.append(shortest_route)
#         print(route_length)
#         return route_length
#     except nx.NetworkXNoPath:
#         return np.nan  # Return NaN if no path exists

# # Apply the function to each row and create a new column for route lengths
# datos_df['route_length'] = datos_df.apply(calculate_route_length, axis=1)

# # # Initialize session state for the current index
# # if 'current_index' not in st.session_state:
# #     st.session_state.current_index = 0

# # # Create buttons for navigation
# # col1, col2, col3 = st.columns(3)

# # with col1:
# #     if st.button('Previous'):
# #         if st.session_state.current_index > 0:
# #             st.session_state.current_index -= 1

# # with col2:
# #     st.write(f"Current Route: {st.session_state.current_index + 1} / {len(all_routes)}")

# # with col3:
# #     if st.button('Next'):
# #         if st.session_state.current_index < len(all_routes) - 1:
# #             st.session_state.current_index += 1

# # # Get the current route based on the index
# # current_route = all_routes[st.session_state.current_index] if all_routes else None

# # # Plot the graph
# # fig, ax = plt.subplots(figsize=(10, 10))
# # ox.plot_graph(G, ax=ax, show=False, close=False, edge_color='lightgrey')

# # if current_route is not None:
# #     # Plot the current route
# #     ox.plot_graph_route(G, current_route, ax=ax, route_linewidth=3, node_size=0, color='blue', show=False)
# # else:
# #     st.warning("No route available for the selected index.")

# # # Step 5 (Optional): Visualize the route
# # st.pyplot(fig)
# # ------

st.dataframe(datos_df)

st.divider()


#----- HISTOGRAMA Viajes por género  -----------------------------------------
#Título para el gráfico
st.header('Viajes por Género')
st.markdown('Análisis de duración de viajes y cantidad por viajes agrupados por género')
st.subheader('Cantidad de Viajes por Género')

#Inicialización del gráfico
fig1, ax1 = plt.subplots()

#Generación del gráfico
sns.set(style = "darkgrid")
sns.histplot(data=datos_df, x='Genero')
ax1.set_title('Histograma de Valores')
ax1.set_xlabel('Género')
ax1.set_ylabel('Cantidad de Viajes')

#Renderización del gráfico
st.pyplot(fig1)

#Título para el gráfico
st.subheader('Duración de Viajes por Género')

#Inicialización del gráfico
fig1, ax1 = plt.subplots()

#Generación del gráfico
sns.set(style = "darkgrid")
sns.barplot(data=datos_df, x='Genero', y='Duracion', estimator='mean', errorbar=None)
ax1.set_title('Histograma de Valores')
ax1.set_xlabel('Género')
ax1.set_ylabel('Duración de Viajes')

#Renderización del gráfico
st.pyplot(fig1)
st.divider()

#-----  Edades  -----------------------------------------
st.header('Edades de usuarios')
st.markdown('En estas gráficas se puede visualizar cuál es el grupo de edad que más utiliza el servicio de MiBici')
# Definir rangos de edad
edades_df = datos_df.copy()
bins = [0, 17, 24, 34, 44, 54, 64, 100]
labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
edades_df['Rango_Edad'] = pd.cut(edades_df['Edad'], bins=bins, labels=labels)

# Contar el número de usuarios en cada rango de edad
edad_counts = edades_df['Rango_Edad'].value_counts().sort_index()

# ---------------------
fig1, ax1 = plt.subplots()

edad_counts.plot(kind='pie', explode=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], autopct='%.0f%%', cmap='tab10')
ax1.set_title('Distribución de Usuarios por Rango de Edad')
ax1.axis('off')
st.pyplot(fig1)
# ---------------------



fig1, ax1 = plt.subplots()

sns.barplot(x=edad_counts.index, y=edad_counts.values, palette='viridis')
ax1.set_title('Distribución de Usuarios por Rango de Edad')
ax1.set_xlabel('Rango de Edad')
ax1.set_ylabel('Número de Usuarios')
ax1.set_xticklabels(edad_counts.index, rotation=45)

st.pyplot(fig1)

st.subheader('Duración Promedio de Viajes por Edad')

#Inicialización del gráfico
fig1, ax1 = plt.subplots()

# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# Cambiar a rango de edad vs duración
#Generación del gráfico
sns.set(style = "darkgrid")
sns.barplot(data=edades_df, x='Rango_Edad', y='Duracion', estimator='mean', errorbar=None)
ax1.set_title('Histograma de Valores')
ax1.set_xlabel('Rango de Edad')
ax1.set_ylabel('Duración de Viajes')

#Renderización del gráfico
st.pyplot(fig1)

# Crear gráfico de barras
st.divider()

#----- Estaciones Origen y Destino -----------------------
st.subheader('Top estaciones origen y destino')
st.markdown('Estaciones con mayor cantidad de viajes.')
col1, col2 = st.columns(2)
with col1:
    st.subheader("Estaciones origen")

with col2:
    st.subheader("Estaciones destino")
# Contar las veces que cada estación fue utilizada como origen
origen_counts = datos_df['Origen_Id'].value_counts().reset_index()
origen_counts.columns = ['Estacion_Origen', 'Conteo_Origen']

# Contar las veces que cada estación fue utilizada como destino
destino_counts = datos_df['Destino_Id'].value_counts().reset_index()
destino_counts.columns = ['Estacion_Destino', 'Conteo_Destino']

# Crear figura para los dos gráficos
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Graficar estaciones de origen más populares
sns.barplot(x='Estacion_Origen', y='Conteo_Origen', data=origen_counts.head(10), ax=axes[0])
axes[0].set_title('Top 10 Estaciones de ORIGEN')
axes[0].set_xlabel('ID de la Estación')
axes[0].set_ylabel('Número de Viajes')

# Graficar estaciones de destino más populares
sns.barplot(x='Estacion_Destino', y='Conteo_Destino', data=destino_counts.head(10), ax=axes[1])
axes[1].set_title('Top 10 Estaciones de DESTINO')
axes[1].set_xlabel('ID de la Estación')
axes[1].set_ylabel('Número de Viajes')

# Ajustar el layout
plt.tight_layout()

# Mostrar los gráficos en Streamlit
st.pyplot(fig)

# ----
# Agrupar por Género y contar las estaciones de Origen y Destino
uso_estaciones_origen = datos_df.groupby(['Genero', 'Origen_Id']).size().unstack(fill_value=0)
uso_estaciones_destino = datos_df.groupby(['Genero', 'Destino_Id']).size().unstack(fill_value=0)

# Seleccionar el top 10 de estaciones más usadas en origen
top_10_origen = uso_estaciones_origen.sum().nlargest(10).index
uso_estaciones_origen_top10 = uso_estaciones_origen[top_10_origen]

# Seleccionar el top 10 de estaciones más usadas en destino
top_10_destino = uso_estaciones_destino.sum().nlargest(10).index
uso_estaciones_destino_top10 = uso_estaciones_destino[top_10_destino]

# Graficar el uso de estaciones de origen por género (top 10)
fig_origen, ax_origen = plt.subplots()
uso_estaciones_origen_top10.T.plot(kind='bar', stacked=True, ax=ax_origen, colormap='tab20')
ax_origen.set_title('Top 10 Estaciones de Origen')
ax_origen.set_xlabel('Estación de Origen')
ax_origen.set_ylabel('Número de Viajes')
ax_origen.legend(title='Género')

# Graficar el uso de estaciones de destino por género (top 10)
fig_destino, ax_destino = plt.subplots()
uso_estaciones_destino_top10.T.plot(kind='bar', stacked=True, ax=ax_destino, colormap='tab20')
ax_destino.set_title('Top 10 Estaciones de Destino')
ax_destino.set_xlabel('Estación de Destino')
ax_destino.set_ylabel('Número de Viajes')
ax_destino.legend(title='Género')

st.subheader('Top estaciones por género')

col1, col2 = st.columns(2)

with col1:
    st.pyplot(fig_origen)

with col2:
    st.pyplot(fig_destino)

st.divider()

#----- GRÁFICO DE CORRELACIÓN ------------------------
#Título para el gráfico
st.subheader('Matriz de Correlación')

#Inicialización del gráfico
fig3, ax3 = plt.subplots()

#Generación del gráfico
df_corr = datos_df[['Año_de_nacimiento', 'Duracion', 'Edad', 
                    'Origen_Latitud', 'Origen_Longitud', 
                    'Destino_Latitud', 'Destino_Longitud', 'Dinero_gastado']].corr()

sns.heatmap(df_corr, annot = True, fmt='.2f', cmap = 'inferno')

#Renderización del gráfico
st.pyplot(fig3)
st.divider()


# ------- Viajes por día de la semana ------
datos_df['Inicio_del_viaje'] = pd.to_datetime(datos_df['Inicio_del_viaje'])
datos_df['Día_de_la_semana'] = datos_df['Inicio_del_viaje'].dt.day_name()
st.header('Viajes por día de la semana')
viajes_por_dia_count = datos_df['Día_de_la_semana'].value_counts()

fig1, ax1 = plt.subplots()

viajes_por_dia_count.plot(kind='pie', explode=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], autopct='%.0f%%', cmap='tab10')
ax1.set_title('Cantidad de viajes por día de la semana')
ax1.axis('off')
st.pyplot(fig1)



# ----------- Series de tiempo -----------
st.header('Viajes a través del tiempo')
st.markdown('Cantidad de viajes realizados a través de los meses')
datos_df['Inicio_del_viaje'] = pd.to_datetime(datos_df['Inicio_del_viaje']).dt.date

grouped_df = datos_df.groupby(['Inicio_del_viaje']).size().to_frame(name='Count')

ts = grouped_df['Count']

fig6, ax6 = plt.subplots()
ax6.plot(ts, label = 'Serie de Tiempo')
plt.legend(loc='best')
plt.ylabel('Cantidad de viajes')
plt.title('Serie de Tiempo')
fig6.autofmt_xdate(rotation=45)
ax6.set_xlim(np.datetime64(ts.index.min()), np.datetime64(ts.index.max()))

st.pyplot(fig6)
st.divider()