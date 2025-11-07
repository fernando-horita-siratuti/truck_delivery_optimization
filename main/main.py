import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd

G = ox.graph_from_place("Divinópolis, MG, Brazil", network_type='drive')
print(f"Grafo: {len(G.nodes)} nós, {len(G.edges)} arestas")

nodes_data = []

for node_id, node_attrs in G.nodes(data=True):
    nodes_data.append({
        'node_id': node_id,
        'latitude': node_attrs.get('y', ''),
        'longitude': node_attrs.get('x', ''),
        'elevation': node_attrs.get('elevation', ''),
        'street_count': node_attrs.get('street_count', ''),
        'highway': node_attrs.get('highway', ''),
        'ref': node_attrs.get('ref', '')
    })

df_nodes = pd.DataFrame(nodes_data)
df_nodes.to_csv('divinopolis_nodes.csv', index=False)
print(f"✅ Dados dos {len(df_nodes)} nós salvos em 'divinopolis_nodes.csv'")

edges_data = []
for edge in G.edges(data=True):
    source, target, edge_attrs = edge
    edges_data.append({
        'source_node': source,
        'target_node': target,
        'length': edge_attrs.get('length', ''),
        'highway': edge_attrs.get('highway', ''),
        'name': edge_attrs.get('name', ''),
        'maxspeed': edge_attrs.get('maxspeed', ''),
        'oneway': edge_attrs.get('oneway', ''),
        'lanes': edge_attrs.get('lanes', ''),
        'surface': edge_attrs.get('surface', '')
    })

df_edges = pd.DataFrame(edges_data)
df_edges.to_csv('divinopolis_edges.csv', index=False)
print(f"✅ Dados das {len(df_edges)} arestas salvos em 'divinopolis_edges.csv'")
