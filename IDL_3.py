# Importar bibliotecas necesarias
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Cargar datos desde el archivo CSV
df = pd.read_csv("ventas.csv")

# Convertir la columna 'Fecha' a formato datetime para facilitar el análisis temporal
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Inicializar la aplicación Dash
app = Dash(__name__)

# Layout del dashboard
app.layout = html.Div([
    html.H1("Dashboard Interactivo de Ventas"),
    
    # Dropdown para seleccionar la región
    html.Div([
        html.Label("Selecciona una región:"),
        dcc.Dropdown(
            id='region-selector',
            options=[{'label': r, 'value': r} for r in df['Region'].unique()],
            value=df['Region'].unique()[0],  # Región inicial seleccionada
            clearable=False
        )
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    # Gráfico de barras: Ventas por región
    dcc.Graph(id='bar-chart'),
    
    # Gráfico de líneas: Tendencia de ventas
    dcc.Graph(id='line-chart'),
    
    # Gráfico de dispersión: Relación inversión vs ventas
    dcc.Graph(id='scatter-plot'),
])

# Callbacks para actualizar los gráficos basados en interacciones
@app.callback(
    Output('bar-chart', 'figure'),
    Input('region-selector', 'value')
)
def update_bar_chart(region):
    filtered_data = df[df['Region'] == region]
    fig = px.bar(filtered_data, x='Producto', y='Ventas',
                 title=f"Ventas por Producto en la Región {region}",
                 labels={'Ventas': 'Cantidad Vendida', 'Producto': 'Producto'})
    return fig

@app.callback(
    Output('line-chart', 'figure'),
    Input('region-selector', 'value')
)
def update_line_chart(region):
    filtered_data = df[df['Region'] == region]
    fig = px.line(filtered_data, x='Fecha', y='Ventas',
                  title=f"Tendencia de Ventas en {region}",
                  labels={'Ventas': 'Cantidad Vendida', 'Fecha': 'Fecha'})
    return fig

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('region-selector', 'value')
)
def update_scatter_plot(region):
    filtered_data = df[df['Region'] == region]
    fig = px.scatter(filtered_data, x='Inversion_Publicitaria', y='Ventas',
                     size='Satisfaccion_Cliente', color='Producto',
                     title=f"Inversión vs Ventas en {region}",
                     labels={'Inversion_Publicitaria': 'Inversión Publicitaria', 'Ventas': 'Cantidad Vendida'})
    return fig

# Ejecutar el servidor
if __name__ == '__main__':
    app.run_server(debug=True)
