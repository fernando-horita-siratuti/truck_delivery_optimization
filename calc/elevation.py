import requests
import pandas as pd
import numpy as np
import math

def get_elevation(lat, lon):
    """Busca elevação de um ponto usando a API"""
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        response = requests.get(url)
        data = response.json()
        return data['results'][0]['elevation']
    except:
        return None

def calculate_slope(elevation1, elevation2, distance):
    """Calcula inclinação em graus"""
    if distance == 0:
        return 0
    elevation_diff = elevation2 - elevation1
    slope_radians = math.atan(elevation_diff / distance)
    slope_degrees = math.degrees(slope_radians)
    return slope_degrees

# Carregar dados
df_edges = pd.read_csv('divinopolis_edges.csv')
df_nodes = pd.read_csv('divinopolis_nodes.csv')

# Criar dicionário de elevações para nós
node_elevations = {}
for _, node in df_nodes.iterrows():
    if pd.notna(node['latitude']) and pd.notna(node['longitude']):
        elevation = get_elevation(node['latitude'], node['longitude'])
        node_elevations[node['node_id']] = elevation
        print(f"Nó {node['node_id']}: {elevation}m")

# Calcular inclinação para cada aresta
edges_with_slope = []
for _, edge in df_edges.iterrows():
    source_id = edge['source_node']
    target_id = edge['target_node']
    
    if source_id in node_elevations and target_id in node_elevations:
        elevation1 = node_elevations[source_id]
        elevation2 = node_elevations[target_id]
        distance = edge['length']  # em metros
        
        if elevation1 and elevation2 and distance:
            slope = calculate_slope(elevation1, elevation2, distance)
            
            edges_with_slope.append({
                'source_node': source_id,
                'target_node': target_id,
                'length': distance,
                'elevation_start': elevation1,
                'elevation_end': elevation2,
                'elevation_diff': elevation2 - elevation1,
                'slope_degrees': slope,
                'name': edge['name']
            })

# Salvar dados com inclinação
df_slopes = pd.DataFrame(edges_with_slope)
df_slopes.to_csv('divinopolis_with_slopes.csv', index=False)

# Análise das inclinações
print(f"\n📊 Análise de Inclinações:")
print(f"Estradas analisadas: {len(df_slopes)}")
print(f"Inclinação máxima: {df_slopes['slope_degrees'].max():.2f}°")
print(f"Inclinação média: {df_slopes['slope_degrees'].mean():.2f}°")
print(f"Estradas com subida > 5°: {len(df_slopes[df_slopes['slope_degrees'] > 5])}")
print(f"Estradas com descida > 5°: {len(df_slopes[df_slopes['slope_degrees'] < -5])}")

# Mostrar as estradas mais íngremes
print(f"\n🔺 Estradas mais íngremes (subida):")
steepest_up = df_slopes.nlargest(5, 'slope_degrees')
print(steepest_up[['name', 'slope_degrees', 'elevation_diff', 'length']])

print(f"\n🔻 Estradas mais íngremes (descida):")
steepest_down = df_slopes.nsmallest(5, 'slope_degrees')
print(steepest_down[['name', 'slope_degrees', 'elevation_diff', 'length']])