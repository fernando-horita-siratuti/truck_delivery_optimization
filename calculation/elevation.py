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

def horizontal_displacement_m(lat1, lon1, lat2, lon2):
    """Retorna (x, y, dist_horizontal) em metros.
    x = deslocamento leste (positivo para leste)
    y = deslocamento norte (positivo para norte)
    dist_horizontal = sqrt(x^2 + y^2)
    """
    R = 6371000.0 
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    λ1, λ2 = math.radians(lon1), math.radians(lon2)
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    x = dλ * math.cos((φ1 + φ2) / 2) * R
    y = dφ * R
    dist = math.hypot(x, y)
    return x, y, dist

def street_steepness(lat1, lon1, h1, lat2, lon2, h2):
    """Retorna um dicionário com:
       - dh: diferença de elevação (h2 - h1) em metros
       - dist_horizontal: distância plana em metros
       - dist_3d: distância entre os pontos em 3D (m)
       - grade: razão dh / dist_horizontal (None se indeterminado)
       - inclination_deg: ângulo de inclinação vertical em graus (positivo = sobe de p1 para p2)
       - inclination_percent: inclinação em porcentagem (None se indeterminado)
    """
    x, y, dist_h = horizontal_displacement_m(lat1, lon1, lat2, lon2)
    dh = h2 - h1
    # distância 3D (hipotenusa)
    dist_3d = math.hypot(dist_h, dh)

    # Inclinação: use atan2(dh, dist_h) — lida corretamente com sinais e com dist_h == 0
    inclination_rad = math.atan2(dh, dist_h)  # se dist_h == 0 e dh != 0 -> ±pi/2 (±90°)
    inclination_deg = math.fabs(math.degrees(inclination_rad))

    # grade (razão) e porcentagem: se dist_h == 0, fica indefinido (None)
    if dist_h == 0:
        grade = None
        inclination_percent = None
    else:
        grade = dh / dist_h
        inclination_percent = 100.0 * grade

    return {
        "dh_m": dh,
        "grade": grade,
        "inclination_deg": inclination_deg,
        "inclination_percent": inclination_percent
    }