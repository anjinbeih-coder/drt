"""Shared geometry helpers for the Yanjiao pipeline (lon/lat in degrees)."""
import numpy as np
from math import asin, cos, radians, sin, sqrt
from matplotlib.path import Path
from scipy.spatial import ConvexHull

EARTH_R = 6371000.0


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * asin(sqrt(a))


def meters_per_degree(lat_deg):
    """(m/deg_lon, m/deg_lat) at the given latitude."""
    return 111320.0 * cos(radians(lat_deg)), 110540.0


def shrunk_hull_path(points_lonlat, shrink_m):
    """Convex hull of lon/lat points, shrunk toward the centroid by shrink_m meters."""
    pts = np.asarray(points_lonlat, dtype=float)
    hull = ConvexHull(pts)
    verts = pts[hull.vertices]
    centroid = verts.mean(axis=0)
    m_lon, m_lat = meters_per_degree(centroid[1])
    out = []
    for v in verts:
        d_m = sqrt(((v[0] - centroid[0]) * m_lon) ** 2 + ((v[1] - centroid[1]) * m_lat) ** 2)
        f = max(0.0, 1.0 - shrink_m / d_m) if d_m > 0 else 0.0
        out.append(centroid + (v - centroid) * f)
    return Path(np.asarray(out))
