

import osmnx as ox
import numpy as np
import pyvista as pv
from shapely.geometry import Polygon, MultiPolygon
import random, pickle, warnings, os
import pandas as pd
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
LAT           = 48.8584
LON           = 2.2945
LOCATION_NAME = "Paris, France"
RADIUS        = 500

N_TOWERS      = 6
TOWER_HEIGHT  = 40.0
SIGNAL_DBM    = -60
FREQ_FACTOR   = 1.0
HEATMAP_RES   = 80
MODEL_PATH    = r"5g_model.pkl"
RECS_PATH     = r"tower_recommendations.csv"  # from train_tower_placement.py

SPHERE_COLORS = ['#00e5ff', '#ff00cc', '#ffee00', '#39ff14', '#ff6b00', '#4466ff']

# ── SIGNAL MODEL ──────────────────────────────────────────────────────────────
def signal_score(tx, ty, px, py, tower_idx=0, tower_positions=None, tower_h=TOWER_HEIGHT):
    dist  = max(np.sqrt((px - tx)**2 + (py - ty)**2), 1.0)
    decay = -np.log(40.0 / 100.0) / RADIUS
    base  = 100.0 * np.exp(-decay * dist)

    interference = 0.0
    if tower_positions:
        for j, (otx, oty) in enumerate(tower_positions):
            if otx == tx and oty == ty: continue
            od  = max(np.sqrt((px-otx)**2 + (py-oty)**2), 1.0)
            interference += 100.0 * np.exp(-decay * od) * 0.08

    tower_chars = [
        {'power': 1.00, 'height_bonus': 1.00},
        {'power': 1.10, 'height_bonus': 1.05},
        {'power': 0.85, 'height_bonus': 0.92},
        {'power': 1.15, 'height_bonus': 1.10},
        {'power': 0.90, 'height_bonus': 0.95},
        {'power': 1.05, 'height_bonus': 1.02},
    ]
    char     = tower_chars[tower_idx % len(tower_chars)]
    adjusted = base * char['power'] * char['height_bonus']
    return float(np.clip(adjusted - interference, 0, 100))

def best_signal(px, py, tower_positions):
    return max(
        signal_score(tx, ty, px, py, tower_idx=i, tower_positions=tower_positions)
        for i, (tx, ty) in enumerate(tower_positions)
    )

# ── ML-DRIVEN TOWER PLACEMENT ─────────────────────────────────────────────────
def load_ml_tower_positions(recs_path, n_towers, center_xy, radius):
    """
    Load recommended positions from train_tower_placement.py output.
    Maps the geographic bounding box of the recommendations to the 
    current simulation map extent so that the placement pattern is 
    preserved regardless of the city.
    """
    import pyproj
    transformer = pyproj.Transformer.from_crs(
        'epsg:4326', 'epsg:3857', always_xy=True)
    cx, cy = center_xy

    if os.path.exists(recs_path):
        print(f"\nLoading ML tower recommendations from {recs_path}...")
        recs = pd.read_csv(recs_path)

        # Sort by gap score (highest priority first)
        recs = recs.sort_values('Gap_Score', ascending=False).head(n_towers * 3)

        # Calculate bounding box of the recommendations in Web Mercator
        xs, ys = [], []
        for _, row in recs.iterrows():
            px, py = transformer.transform(row['Longitude'], row['Latitude'])
            xs.append(px)
            ys.append(py)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Avoid division by zero if all points are the same
        if max_x == min_x: max_x = min_x + 1
        if max_y == min_y: max_y = min_y + 1

        placed = []
        import math
        import random # make sure random is available
        random.seed(42)  # For consistent placement visually
        
        # force a larger minimum distance for visual dispersion
        min_dist = radius * 0.45

        for _, row in recs.iterrows():
            if len(placed) >= n_towers:
                break

            px, py = transformer.transform(row['Longitude'], row['Latitude'])

            # Normalize to 0-1
            norm_x = (px - min_x) / (max_x - min_x) if max_x != min_x else 0.5
            norm_y = (py - min_y) / (max_y - min_y) if max_y != min_y else 0.5

            # Map to the local city (scale to 1.6x radius to fill space well without overflowing)
            # Centered around cx, cy
            mapped_x = cx + (norm_x - 0.5) * (radius * 1.6)
            mapped_y = cy + (norm_y - 0.5) * (radius * 1.6)

            # --- Repulsion Mechanism ---
            # If the mapped point is too close to an existing tower, push it away
            # iteratively until it clears the minimum distance or max attempts reached
            attempts = 0
            too_close = True
            
            while too_close and attempts < 50:
                too_close = False
                for ep in placed:
                    dx, dy = mapped_x - ep[0], mapped_y - ep[1]
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < min_dist:
                        too_close = True
                        if dist < 1.0: # Identical location fallback
                            angle = random.uniform(0, 2 * math.pi)
                            dist = 1.0
                        else:
                            angle = math.atan2(dy, dx)
                        
                        # Apply outward force proportional to the overlap
                        force = (min_dist - dist) + 5.0
                        mapped_x += math.cos(angle) * force
                        mapped_y += math.sin(angle) * force
                attempts += 1
                
            # Keep it roughly within the simulated radius boundaries
            dist_to_center = math.sqrt((mapped_x - cx)**2 + (mapped_y - cy)**2)
            if dist_to_center > radius * 1.6:
                angle_to_center = math.atan2(mapped_y - cy, mapped_x - cx)
                mapped_x = cx + math.cos(angle_to_center) * (radius * 1.5)
                mapped_y = cy + math.sin(angle_to_center) * (radius * 1.5)

            placed.append((mapped_x, mapped_y))
            print(f"  Tower {len(placed)}: Gap_Score={row['Gap_Score']:.3f} "
                  f"Priority={row['Priority']} "
                  f"RSRP={row['Current_RSRP']:.0f}dBm")

        # If we didn't get enough from CSV, fill with geometric placement
        if len(placed) < n_towers:
            print(f"  Only {len(placed)} from CSV — filling {n_towers-len(placed)} geometrically...")
            extra = geometric_place_towers([], n_towers - len(placed), center_xy, radius, existing=placed)
            placed.extend(extra)

        print(f"  Total towers placed: {len(placed)} (ML-driven)")
        return placed

    else:
        print(f"\n[WARN] {recs_path} not found — using geometric placement.")
        return geometric_place_towers([], n_towers, center_xy, radius)

def geometric_place_towers(footprints, n_towers, center_xy, radius, existing=None):
    """Fallback: spread towers evenly across the area."""
    cx, cy = center_xy
    r = radius * 0.75
    placed = list(existing) if existing else []

    angles = np.linspace(0, 2 * np.pi, n_towers + 1)[:-1]
    for angle in angles:
        if len(placed) - (len(existing) if existing else 0) >= n_towers:
            break
        px = cx + np.cos(angle) * r * 0.6
        py = cy + np.sin(angle) * r * 0.6
        if not any(np.sqrt((px-ep[0])**2+(py-ep[1])**2) < 80 for ep in placed):
            placed.append((px, py))
    return placed

# ── OSM DATA ──────────────────────────────────────────────────────────────────
def extract_osm_data(lat, lon, radius):
    print(f"\nDownloading OSM data — {LOCATION_NAME} (r={radius}m)...")
    buildings = ox.features_from_point((lat, lon), tags={'building': True}, dist=radius)
    streets   = ox.graph_from_point((lat, lon), dist=radius,
                                     network_type='drive', simplify=False)
    buildings = buildings.to_crs(epsg=3857)
    streets   = ox.project_graph(streets, to_crs='epsg:3857')
    print(f"  Buildings: {len(buildings)}  |  Streets: {len(streets.edges)}")
    return buildings, streets

def get_footprints(buildings):
    footprints = []
    for _, row in buildings.iterrows():
        geom, height = row.geometry, None
        if 'height' in row and row['height'] is not None:
            try:
                h = float(str(row['height']).replace('m','').strip())
                if h > 0: height = h
            except: pass
        if height is None and 'building:levels' in row and row['building:levels'] is not None:
            try:
                lvl = float(row['building:levels'])
                if lvl > 0: height = lvl * 3.2
            except: pass
        if height is None:
            height = random.uniform(10, 28)
        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            footprints.append({'polygon': poly, 'height': height})
    print(f"  Footprints: {len(footprints)}")
    return footprints

# ── BUILDING MESH ─────────────────────────────────────────────────────────────
def make_walls(coords, height):
    if np.allclose(coords[0], coords[-1]): coords = coords[:-1]
    n   = len(coords)
    pts = [[x, y, 0] for x,y in coords] + [[x, y, height] for x,y in coords]
    pts = np.array(pts)
    faces  = [n] + list(range(n-1,-1,-1))
    faces += [n] + list(range(n, 2*n))
    for i in range(n):
        j = (i+1) % n
        faces += [4, i, j, j+n, i+n]
    return pts, faces

def build_city_mesh(footprints):
    print("\nExtruding buildings (unique realistic colors)...")
    city_mesh = pv.PolyData()
    base_colors = [
        [0.82, 0.76, 0.65], [0.78, 0.72, 0.60], [0.74, 0.68, 0.58],
        [0.68, 0.63, 0.54], [0.62, 0.58, 0.52], [0.55, 0.52, 0.48],
        [0.72, 0.60, 0.50], [0.65, 0.55, 0.46], [0.80, 0.70, 0.58],
        [0.58, 0.56, 0.54], [0.76, 0.65, 0.52], [0.85, 0.80, 0.70],
    ]
    for fp in footprints:
        poly, height = fp['polygon'], fp['height']
        pts, faces = make_walls(np.array(poly.exterior.coords), height)
        bldg  = pv.PolyData(pts, np.array(faces))
        base  = random.choice(base_colors)
        color = np.clip(np.array(base) + np.random.uniform(-0.04, 0.04, 3), 0, 1).tolist()
        n_pts = bldg.n_points
        colors = []
        for k in range(n_pts):
            if k < n_pts // 2:
                colors.append([max(0, c - 0.06) for c in color])
            else:
                colors.append(color)
        bldg['base_color'] = np.array(colors)
        city_mesh = bldg if city_mesh.n_points == 0 else \
                    city_mesh.merge(bldg, merge_points=False)
    print(f"  Done: {city_mesh.n_points} points")
    return city_mesh

def build_signal_overlay_for_tower(footprints, tower_pos, tower_idx, tower_positions):
    tx, ty  = tower_pos
    overlay = pv.PolyData()
    for fp in footprints:
        poly, height = fp['polygon'], fp['height']
        cx2, cy2 = poly.centroid.x, poly.centroid.y
        sc = signal_score(tx, ty, cx2, cy2, tower_idx=tower_idx,
                          tower_positions=tower_positions)
        pts, faces = make_walls(np.array(poly.exterior.coords), height + 0.3)
        bldg = pv.PolyData(pts, np.array(faces))
        bldg['signal_score'] = np.full(bldg.n_points, sc)
        overlay = bldg if overlay.n_points == 0 else \
                  overlay.merge(bldg, merge_points=False)
    return overlay

def make_street_mesh(st_graph):
    _, edges = ox.graph_to_gdfs(st_graph)
    pts_list = edges['geometry'].apply(
        lambda g: np.column_stack((g.xy[0], g.xy[1], np.zeros(len(g.xy[0]))))
    ).tolist()
    verts = np.concatenate(pts_list)
    lines, j = [], 0
    for pts in pts_list:
        n = len(pts)
        lines.append([n] + list(range(j, j+n)))
        j += n
    return pv.PolyData(verts, lines=np.hstack(lines))

def make_heatmap_for_tower(tower_pos, tower_idx, tower_positions,
                            center_xy, radius, res=50):
    tx, ty = tower_pos
    cx, cy = center_xy
    xs = np.linspace(cx-radius, cx+radius, res)
    ys = np.linspace(cy-radius, cy+radius, res)
    XX, YY = np.meshgrid(xs, ys)
    mask   = (XX-cx)**2 + (YY-cy)**2 <= radius**2
    scores = np.zeros(XX.shape)
    for i in range(res):
        for j in range(res):
            if not mask[i,j]: continue
            scores[i,j] = signal_score(tx, ty, XX[i,j], YY[i,j],
                                        tower_idx=tower_idx,
                                        tower_positions=tower_positions)
    flat_pts = np.column_stack([XX.ravel(), YY.ravel(), np.full(res*res, -0.8)])
    inside   = mask.ravel()
    grid     = pv.PolyData(flat_pts[inside])
    grid['signal_score'] = scores.ravel()[inside]
    return grid.delaunay_2d()

# ── VISUALIZE ─────────────────────────────────────────────────────────────────
def visualize(city_mesh, signal_overlays, street_mesh, ground_heatmaps,
              tower_positions, coverage_pct, placement_source):

    print("\nLaunching 3D viewer...")
    print("  >> Click a tower beacon to show its coverage")
    print("  >> Click same tower again or empty space to hide")

    pl = pv.Plotter(window_size=[1600, 950], lighting='three lights')
    pl.renderer.SetBackground(0.02, 0.05, 0.10)
    pl.renderer.SetBackground2(0.00, 0.02, 0.06)
    pl.renderer.SetGradientBackground(True)

    pl.remove_all_lights()
    pl.add_light(pv.Light(position=(0, 0, 3000),       color='white',   intensity=0.85))
    pl.add_light(pv.Light(position=(2000, 1000, 800),   color='#d0e8ff', intensity=0.45))
    pl.add_light(pv.Light(position=(-1000, -500, 400),  color='#fff4e0', intensity=0.30))

    # Base city
    if city_mesh.n_points > 0:
        pl.add_mesh(city_mesh, scalars='base_color', rgb=True,
                    show_edges=False, opacity=1.0, smooth_shading=True)
    pl.add_mesh(street_mesh, color='#1a3a52', line_width=2.0, opacity=0.85)

    # Per-tower hidden actors
    tower_actors = []
    cover_r = RADIUS * 0.45

    for i, (tx, ty) in enumerate(tower_positions):
        col    = SPHERE_COLORS[i % len(SPHERE_COLORS)]
        actors = []

        # Signal overlay
        a = pl.add_mesh(signal_overlays[i], scalars='signal_score', clim=[20, 100],
                        cmap='RdYlGn', opacity=0.40, smooth_shading=True,
                        show_edges=False, show_scalar_bar=(i == 0),
                        scalar_bar_args=dict(
                            title='  Signal Quality', color='#cde8f5',
                            title_font_size=13, label_font_size=10,
                            position_x=0.85, position_y=0.06,
                            width=0.12, height=0.32, n_labels=5,
                            fmt='%.0f', shadow=True))
        a.SetVisibility(False); actors.append(a)

        # Heatmap
        a = pl.add_mesh(ground_heatmaps[i], scalars='signal_score', clim=[20, 100],
                        cmap='RdYlGn', opacity=0.55, show_scalar_bar=False)
        a.SetVisibility(False); actors.append(a)

        # Hemispheres
        for r_frac, op in [(1.0, 0.06), (0.60, 0.10), (0.30, 0.22)]:
            h = pv.Sphere(radius=cover_r * r_frac, center=(tx, ty, 0),
                           theta_resolution=60, phi_resolution=60,
                           start_phi=0, end_phi=90)
            a = pl.add_mesh(h, color=col, opacity=op,
                            style='surface', smooth_shading=True)
            a.SetVisibility(False); actors.append(a)

        # Wireframe
        outer = pv.Sphere(radius=cover_r, center=(tx, ty, 0),
                           theta_resolution=60, phi_resolution=60,
                           start_phi=0, end_phi=90)
        a = pl.add_mesh(outer, color=col, opacity=0.55,
                        style='wireframe', line_width=0.7)
        a.SetVisibility(False); actors.append(a)

        tower_actors.append(actors)

    # Tower poles + beacons (always visible)
    for i, (tx, ty) in enumerate(tower_positions):
        col = SPHERE_COLORS[i % len(SPHERE_COLORS)]
        pole = pv.Cylinder(center=(tx, ty, TOWER_HEIGHT/2), direction=(0,0,1),
                            radius=3.5, height=TOWER_HEIGHT, resolution=16)
        pl.add_mesh(pole, color='#d0dce8', opacity=1.0, smooth_shading=True)
        for angle in [0, 90]:
            rad = np.deg2rad(angle)
            arm = pv.Cylinder(
                center=(tx + np.cos(rad)*8, ty + np.sin(rad)*8, TOWER_HEIGHT),
                direction=(np.cos(rad), np.sin(rad), 0),
                radius=1.2, height=16, resolution=8)
            pl.add_mesh(arm, color='#b8c8d8', opacity=1.0, smooth_shading=True)
        beacon = pv.Sphere(radius=9, center=(tx, ty, TOWER_HEIGHT + 10),
                            theta_resolution=16, phi_resolution=16)
        pl.add_mesh(beacon, color=col, opacity=1.0, smooth_shading=True,
                    show_scalar_bar=False)
        pl.add_point_labels(
            np.array([[tx, ty, TOWER_HEIGHT + 26]]),
            [f' T{i+1} '], point_size=1, font_size=12,
            text_color='#39ff14', bold=True, show_points=False, shadow=True)

    # Click state
    state = {'active': None}
    info_actor = pl.add_text("  Click a tower to inspect its coverage",
                              position='lower_right', color='#00e5ff',
                              font_size=10, shadow=True)

    tower_chars = [
        {'power': '23 dBm', 'band': 'n78 3.5GHz', 'type': 'Standard'},
        {'power': '26 dBm', 'band': 'n78 3.5GHz', 'type': 'Boosted'},
        {'power': '20 dBm', 'band': 'n77 4.0GHz', 'type': 'Low Power'},
        {'power': '27 dBm', 'band': 'n78 3.5GHz', 'type': 'High Power'},
        {'power': '22 dBm', 'band': 'n258 26GHz', 'type': 'mmWave'},
        {'power': '24 dBm', 'band': 'n78 3.5GHz', 'type': 'Standard+'},
    ]

    def hide_all():
        for acts in tower_actors:
            for a in acts: a.SetVisibility(False)
        state['active'] = None

    def show_tower(idx):
        hide_all()
        for a in tower_actors[idx]: a.SetVisibility(True)
        state['active'] = idx
        tx, ty = tower_positions[idx]
        char   = tower_chars[idx % len(tower_chars)]
        test_pts = [(tx + r*np.cos(ang), ty + r*np.sin(ang))
                    for r in [50,100,150,200,250,300]
                    for ang in np.linspace(0, 2*np.pi, 8)]
        avg_sc = np.mean([signal_score(tx, ty, px, py, tower_idx=idx,
                                        tower_positions=tower_positions)
                          for px, py in test_pts])
        peak   = signal_score(tx, ty, tx+1, ty, tower_idx=idx,
                               tower_positions=tower_positions)
        info_actor.SetInput(
            f"  T{idx+1} | {char['type']} | {char['band']} | "
            f"Power: {char['power']} | Peak: {peak:.0f} | "
            f"Avg: {avg_sc:.0f}/100 | Placement: {placement_source}")
        pl.render()

    def on_click(point):
        if point is None:
            hide_all()
            info_actor.SetInput("  Click a tower to inspect its coverage")
            pl.render()
            return
        px, py = point[0], point[1]
        min_d, nearest = float('inf'), None
        for i, (tx, ty) in enumerate(tower_positions):
            d = np.sqrt((px-tx)**2 + (py-ty)**2)
            if d < min_d: min_d, nearest = d, i
        if min_d < 40:
            if state['active'] == nearest:
                hide_all()
                info_actor.SetInput("  Click a tower to inspect its coverage")
                pl.render()
            else:
                show_tower(nearest)
        else:
            hide_all()
            info_actor.SetInput("  Click a tower to inspect its coverage")
            pl.render()

    pl.track_click_position(callback=on_click, side='left')

    # HUD
    pl.add_text(f"5G Signal Simulation  --  {LOCATION_NAME}",
                position='upper_left', color='#00e5ff', font_size=16, shadow=True)
    pl.add_text(
        f"  Towers: {len(tower_positions)}  |  Coverage (>40): {coverage_pct:.1f}%  |  "
        f"Placement: {placement_source}  |  Click tower to inspect",
        position='lower_left', color='#3a7a9e', font_size=9, shadow=True)
    pl.add_text("[R] Poor  [Y] Fair  [G] Good signal",
                position='upper_right', color='#7aaabb', font_size=9)

    cx2, cy2 = tower_positions[len(tower_positions)//2]
    pl.camera.position    = (cx2 + RADIUS*1.7, cy2 - RADIUS*1.7, RADIUS*2.0)
    pl.camera.focal_point = (cx2, cy2, 20)
    pl.camera.up          = (0, 0, 1)
    pl.show_axes()
    pl.show(title=f'5G Simulation -- {LOCATION_NAME}', auto_close=False)

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 62)
    print(f"  5G SIGNAL SIMULATION  --  {LOCATION_NAME}  (v5)")
    print(f"  Tower placement: ML gap detection via {RECS_PATH}")
    print("=" * 62)

    try:
        with open(MODEL_PATH, 'rb') as f:
            pickle.load(f)
        print("ML model loaded.")
    except FileNotFoundError:
        print("[WARN] 5g_model.pkl not found.")

    buildings, streets = extract_osm_data(LAT, LON, RADIUS)
    footprints = get_footprints(buildings)

    import pyproj
    transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True)
    cx, cy = transformer.transform(LON, LAT)
    center_xy = (cx, cy)
    print(f"Projected center: ({cx:.1f}, {cy:.1f})")

    # ML-driven tower placement
    tower_positions = load_ml_tower_positions(RECS_PATH, N_TOWERS, center_xy, RADIUS)
    placement_source = "ML Gap Detection" if os.path.exists(RECS_PATH) else "Geometric"

    city_mesh    = build_city_mesh(footprints)
    street_mesh  = make_street_mesh(streets)

    print("\nBuilding per-tower signal overlays...")
    signal_overlays, ground_heatmaps = [], []
    for i, tp in enumerate(tower_positions):
        print(f"  Tower {i+1}/{len(tower_positions)}...")
        signal_overlays.append(
            build_signal_overlay_for_tower(footprints, tp, i, tower_positions))
        ground_heatmaps.append(
            make_heatmap_for_tower(tp, i, tower_positions, center_xy, RADIUS))

    all_scores = [best_signal(fp['polygon'].centroid.x,
                               fp['polygon'].centroid.y,
                               tower_positions) for fp in footprints]
    cov = sum(s > 40 for s in all_scores) / len(all_scores) * 100

    visualize(city_mesh, signal_overlays, street_mesh, ground_heatmaps,
              tower_positions, cov, placement_source)