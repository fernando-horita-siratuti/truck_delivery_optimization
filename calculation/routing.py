import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import networkx as nx
import numpy as np
from geopy.geocoders import Nominatim
# importa a função de inclinação (reaproveita seu módulo)
from elevation import street_steepness

# ========== PARÂMETROS (ajuste conforme desejar) ==========
BASE_L_PER_100KM = 10.0       # consumo base típico (L/100km) em velocidade moderada
SLOPE_COEF = 10.0             # quanto a subida aumenta o consumo (multiplicador por unidade de slope)
SPEED_PENALTY_COEF = 0.2      # penalidade por velocidades fora da referência (quadrática)
REF_SPEED_KMH = 50.0          # velocidade de referência para consumo (km/h)
TIME_WEIGHT = 0.5             # quantos "litros equivalentes" atribuímos a 1 minuto extra (fator multiplica)
# =========================================================

# tenta localizar a pasta data onde seu main.py salva os CSVs
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NODES_CSV = DEFAULT_DATA_DIR / "divinopolis_nodes.csv"
EDGES_CSV = DEFAULT_DATA_DIR / "divinopolis_edges.csv"


def _safe_float(val: object, fallback: float = 0.0) -> float:
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return fallback
        return float(val)
    except Exception:
        return fallback


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))


def parse_maxspeed(val, default=REF_SPEED_KMH):
    if pd.isna(val) or val == "":
        return default
    try:
        if isinstance(val, str):
            first = val.split(';')[0].strip()
            digits = ''.join(ch for ch in first if (ch.isdigit() or ch=='.'))
            return float(digits) if digits != "" else default
        else:
            return float(val)
    except:
        return default


def build_graph_from_csv(nodes_csv: Path = NODES_CSV, edges_csv: Path = EDGES_CSV) -> nx.DiGraph:
    if not nodes_csv.exists():
        raise FileNotFoundError(f"Nodes CSV não encontrado em: {nodes_csv}")
    if not edges_csv.exists():
        raise FileNotFoundError(f"Edges CSV não encontrado em: {edges_csv}")

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    G = nx.DiGraph()

    # adiciona nós com lat/lon/elev
    for _, r in nodes_df.iterrows():
        nid = int(r['node_id'])
        lat = _safe_float(r.get('latitude'), fallback=0.0)
        lon = _safe_float(r.get('longitude'), fallback=0.0)
        elev_val = r.get('elevation', '')
        elevation = _safe_float(elev_val, fallback=0.0)
        G.add_node(nid, y=lat, x=lon, elevation=elevation)

    # adiciona arestas
    for _, r in edges_df.iterrows():
        try:
            u = int(r['source_node'])
            v = int(r['target_node'])
        except Exception:
            continue
        length = _safe_float(r.get('length'), fallback=0.0)
        name = r.get('name', "") if pd.notna(r.get('name', "")) else ""
        maxspeed = parse_maxspeed(r.get('maxspeed', REF_SPEED_KMH), default=REF_SPEED_KMH)
        oneway = str(r.get('oneway', 'False')).lower() in ('true', '1', 't', 'yes')

        # ignore edges whose nodes are missing
        if u not in G.nodes or v not in G.nodes:
            continue

        G.add_edge(u, v, length=length, name=name, maxspeed_kmh=maxspeed, original=True)
        if not oneway:
            G.add_edge(v, u, length=length, name=name, maxspeed_kmh=maxspeed, original=True)

    _precompute_edge_costs(G)
    return G


def _precompute_edge_costs(G: nx.DiGraph) -> None:
    """Calcula fuel_liters, time_minutes e eco_cost para cada aresta do grafo.
       Usa street_steepness para obter a grade (mais robusto que diferença/length simples)."""
    base_per_m = BASE_L_PER_100KM / 100000.0  # L por metro
    ref_speed_kmh = REF_SPEED_KMH
    ref_speed_m_per_min = ref_speed_kmh * 1000.0 / 60.0
    liters_per_min_ref = base_per_m * ref_speed_m_per_min

    for u, v, data in list(G.edges(data=True)):
        length = float(data.get('length', 1.0))
        speed_kmh = float(data.get('maxspeed_kmh', REF_SPEED_KMH))

        lat_u = float(G.nodes[u].get('y', 0.0))
        lon_u = float(G.nodes[u].get('x', 0.0))
        elev_u = float(G.nodes[u].get('elevation', 0.0))

        lat_v = float(G.nodes[v].get('y', 0.0))
        lon_v = float(G.nodes[v].get('x', 0.0))
        elev_v = float(G.nodes[v].get('elevation', 0.0))

        # usa street_steepness para obter grade (dh/dist_horizontal)
        steep = street_steepness(lat_u, lon_u, elev_u, lat_v, lon_v, elev_v)
        grade = steep.get("grade")
        # se grade for None (dist_h == 0), set 0
        slope = grade if grade is not None else 0.0
        uphill = max(slope, 0.0)

        # fatores
        slope_multiplier = 1.0 + (SLOPE_COEF * uphill)
        speed_factor = 1.0 + SPEED_PENALTY_COEF * ((speed_kmh - ref_speed_kmh) / ref_speed_kmh) ** 2

        fuel_liters = base_per_m * length * slope_multiplier * speed_factor

        speed_m_per_min = speed_kmh * 1000.0 / 60.0
        time_minutes = length / speed_m_per_min if speed_m_per_min > 0 else float('inf')

        time_penalty_equiv_liters = TIME_WEIGHT * time_minutes * liters_per_min_ref

        eco_cost = fuel_liters + time_penalty_equiv_liters

        data['fuel_liters'] = fuel_liters
        data['time_minutes'] = time_minutes
        data['eco_cost'] = eco_cost
        data['slope'] = slope


def nearest_node_to_point(G: nx.DiGraph, lat: float, lon: float) -> int:
    """Busca o nó mais próximo por distância haversine (simples e robusto para cidade)."""
    nodes = list(G.nodes(data=True))
    coords = np.array([[n[1]['y'], n[1]['x']] for n in nodes])
    lat_arr = coords[:, 0].astype(float)
    lon_arr = coords[:, 1].astype(float)
    dists = np.array([haversine(lon, lat, lon_arr[i], lat_arr[i]) for i in range(len(lat_arr))])
    idx = int(np.argmin(dists))
    nearest_node = nodes[idx][0]
    return nearest_node


def geocode_address(address: str, user_agent: str = "meu_app") -> Tuple[float, float, str]:
    geolocator = Nominatim(user_agent=user_agent)
    loc = geolocator.geocode(address)
    if loc is None:
        raise ValueError(f"Geocoding falhou para: {address}")
    return loc.latitude, loc.longitude, loc.address


def _select_best_edge_between(G: nx.DiGraph, u: int, v: int) -> Optional[Dict]:
    """Se MultiGraph escolhe aresta com menor eco_cost; se DiGraph simples devolve atributos."""
    if G.is_multigraph():
        ed = G.get_edge_data(u, v)
        if not ed:
            return None
        best_data = None
        best_cost = float('inf')
        for k, attr in ed.items():
            cost = attr.get('eco_cost', float('inf'))
            if cost < best_cost:
                best_cost = cost
                best_data = attr
        return best_data
    else:
        return G[u][v] if G.has_edge(u, v) else None


def compress_street_segments(segments: List[Tuple[str, float, float, float]]) -> List[Tuple[str, float, float, float]]:
    """Agrega segmentos consecutivos com o mesmo nome."""
    if not segments:
        return []
    out = []
    cur_name, cur_len, cur_fuel, cur_time = segments[0]
    for name, length, fuel, time in segments[1:]:
        if name == cur_name:
            cur_len += length
            cur_fuel += fuel
            cur_time += time
        else:
            out.append((cur_name, cur_len, cur_fuel, cur_time))
            cur_name, cur_len, cur_fuel, cur_time = name, length, fuel, time
    out.append((cur_name, cur_len, cur_fuel, cur_time))
    return out


def route_ecological(G: nx.DiGraph, start_addr: str, dest_addr: str) -> Dict:
    start_lat, start_lon, _ = geocode_address(start_addr)
    dest_lat, dest_lon, _ = geocode_address(dest_addr)

    start_node = nearest_node_to_point(G, start_lat, start_lon)
    end_node = nearest_node_to_point(G, dest_lat, dest_lon)

    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='eco_cost', method='dijkstra')
    except nx.NetworkXNoPath:
        raise RuntimeError("Não há caminho entre os nós selecionados.")

    total_length = 0.0
    total_fuel = 0.0
    total_time_min = 0.0
    edges = []
    street_segments = []

    for i in range(len(path) - 1):
        u = path[i]; v = path[i + 1]
        data = _select_best_edge_between(G, u, v)
        if data is None:
            continue
        length = data.get('length', 0.0)
        fuel = data.get('fuel_liters', 0.0)
        time_min = data.get('time_minutes', 0.0)
        name = data.get('name') if data.get('name') else "unnamed"
        total_length += length
        total_fuel += fuel
        total_time_min += time_min
        edges.append((u, v, data))
        street_segments.append((name, length, fuel, time_min))

    street_segments_compressed = compress_street_segments(street_segments)

    return {
        'start_node': start_node,
        'end_node': end_node,
        'path_nodes': path,
        'edges': edges,
        'street_segments': street_segments_compressed,
        'total_length_m': total_length,
        'total_time_min': total_time_min,
        'total_fuel_liters': total_fuel
    }


if __name__ == "__main__":
    # exemplo de uso rápido
    print("Carregando grafo a partir dos CSVs (certifique-se que main.py já gerou os arquivos em data/)...")
    G = build_graph_from_csv()
    print(f"Grafo com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.\n")

    # coloque aqui os endereços de início/fim que deseja testar
    start_address = "Rua Padre Eustáquio, 710, Divinópolis, MG, Brasil"
    dest_address = "Rua Álvares de Azevedo, 400, Divinópolis, MG, Brasil"

    print("Calculando rota ecológica (pode demorar alguns segundos - depende do geocoding)...")
    result = route_ecological(G, start_address, dest_address)

    print("\n--- Resumo da rota ---")
    print(f"Start node: {result['start_node']}, End node: {result['end_node']}")
    print(f"Distância total: {result['total_length_m']:.1f} m")
    print(f"Tempo estimado: {result['total_time_min']:.1f} min")
    print(f"Consumo estimado: {result['total_fuel_liters']:.3f} L")

    print("\nTrechos por rua (agregado):")
    for idx, (name, length, fuel, time_min) in enumerate(result['street_segments'], start=1):
        print(f"{idx}. {name or 'unnamed'} — {length:.0f} m — {time_min:.1f} min — {fuel:.3f} L")