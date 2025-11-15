import math
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from calculation.elevation import get_elevations

ROUND_PRECISION = 6
PARALLEL_WORKERS = 5
BATCH_SIZE = 50
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

G = ox.graph_from_place("Divinópolis, MG, Brazil", network_type='drive')
print(f"Grafo: {len(G.nodes)} nós, {len(G.edges)} arestas")

nodes_data: List[Dict[str, object]] = []
coords_to_fetch: List[Tuple[float, float]] = []
coord_to_indexes: Dict[Tuple[float, float], List[int]] = {}

def _round_coord(value: float) -> float:
    return round(value, ROUND_PRECISION)


def _is_missing_elevation(value) -> bool:
    if value in ('', None):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (int, float)):
        try:
            if math.isnan(float(value)):
                return True
        except (TypeError, ValueError):
            return True
    return False

for idx, (node_id, node_attrs) in enumerate(G.nodes(data=True)):
    latitude = node_attrs.get('y')
    longitude = node_attrs.get('x')
    elevation = node_attrs.get('elevation')

    elevation_value = '' if _is_missing_elevation(elevation) else elevation

    nodes_entry = {
        'node_id': node_id,
        'latitude': latitude if latitude is not None else '',
        'longitude': longitude if longitude is not None else '',
        'elevation': elevation_value,
        'street_count': node_attrs.get('street_count', ''),
    }

    if _is_missing_elevation(elevation) and latitude is not None and longitude is not None:
        key = (_round_coord(latitude), _round_coord(longitude))
        if key not in coord_to_indexes:
            coord_to_indexes[key] = [idx]
            coords_to_fetch.append((latitude, longitude))
        else:
            coord_to_indexes[key].append(idx)

    nodes_data.append(nodes_entry)

def _chunked(seq: List[Tuple[float, float]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

if coords_to_fetch:
    print(f"Consultando elevação para {len(coord_to_indexes)} coordenadas faltantes...")
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(get_elevations, batch): batch
            for batch in _chunked(coords_to_fetch, BATCH_SIZE)
        }
        for future in as_completed(futures):
            batch = futures[future]
            elevations = future.result()
            for (lat, lon), elevation in zip(batch, elevations):
                key = (_round_coord(lat), _round_coord(lon))
                indexes = coord_to_indexes.get(key, [])
                for node_idx in indexes:
                    nodes_data[node_idx]['elevation'] = elevation if elevation is not None else ''


df_nodes = pd.DataFrame(nodes_data)
nodes_csv = OUTPUT_DIR / "divinopolis_nodes.csv"
df_nodes.to_csv(nodes_csv, index=False)
print(f"Dados dos {len(df_nodes)} nós salvos em '{nodes_csv.name}'")

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
    })

df_edges = pd.DataFrame(edges_data)
edges_csv = OUTPUT_DIR / "divinopolis_edges.csv"
df_edges.to_csv(edges_csv, index=False)
print(f"Dados das {len(df_edges)} arestas salvos em '{edges_csv.name}'")
