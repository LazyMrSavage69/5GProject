import osmnx as ox
import numpy as np 
import pyvista as pv 
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path 
import random 

def extract_osm_data_coords(lat, lon, radius=200):
    print(f"Downloading OSM data for ({lat}, {lon})...")
    center_point = (lat, lon)
    buildings = ox.features_from_point(
        center_point,
        tags={'building': True},
        dist=radius
    )
    streets = ox.graph_from_point(center_point, dist=radius,
                                   network_type='drive', simplify=False)
    
    buildings = buildings.to_crs(epsg=2154)
    streets = ox.project_graph(streets, to_crs='epsg:2154')
    print(f"Downloaded {len(buildings)} buildings")
    print(f"Downloaded {len(streets.edges)} street segments")
    return buildings, streets


def generate_footprints(buildings):
    footprints = []
    known_count = 0
    estimated_count = 0

    for _, row in buildings.iterrows():
        geom = row.geometry
        height = None
        height_source = "estimated"

        # ── 1. Use OSM 'height' field as-is (trust the data) ──────────────
        if 'height' in row and row['height'] is not None:
            try:
                h = float(str(row['height']).replace("m", "").strip())
                if h > 0:
                    height = h
                    height_source = "osm"
            except:
                pass

        # ── 2. Use OSM 'building:levels' as-is ────────────────────────────
        if height is None and 'building:levels' in row and row['building:levels'] is not None:
            try:
                levels = float(row['building:levels'])
                if levels > 0:
                    height = levels * 3.0   # standard ~3 m per floor
                    height_source = "osm_levels"
            except:
                pass

        # ── 3. No OSM height data → estimate 10–15 m ──────────────────────
        if height is None:
            height = random.uniform(10, 15)
            height_source = "estimated"

        # Track stats
        if height_source.startswith("osm"):
            known_count += 1
        else:
            estimated_count += 1

        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                footprints.append({
                    "polygon": poly,
                    "height": height,
                    "height_source": height_source
                })
        elif isinstance(geom, Polygon):
            footprints.append({
                "polygon": geom,
                "height": height,
                "height_source": height_source
            })

    print(f"Heights — OSM known: {known_count} | Estimated (10–15 m): {estimated_count}")
    return footprints


def create_watertight_buildings(coords, height):
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    
    n_points = len(coords)
    points = []
    
    for x, y in coords:
        points.append([x, y, 0])
    for x, y in coords:
        points.append([x, y, height])
    
    points = np.array(points)
    faces = []
    
    # Bottom face
    faces.append(n_points)
    for i in range(n_points - 1, -1, -1):
        faces.append(i)
    
    # Top face
    faces.append(n_points)
    for i in range(n_points):
        faces.append(i + n_points)
    
    # Side faces
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append(4)
        faces.append(i)
        faces.append(next_i)
        faces.append(next_i + n_points)
        faces.append(i + n_points)
    
    return points, faces


def generate_realistic_building_color():
    color_schemes = [
        [0.7, 0.7, 0.72],
        [0.6, 0.6, 0.62],
        [0.5, 0.52, 0.54],
        [0.7, 0.4, 0.3],
        [0.6, 0.35, 0.25],
        [0.75, 0.45, 0.35],
        [0.85, 0.85, 0.82],
        [0.8, 0.78, 0.75],
        [0.65, 0.6, 0.55],
        [0.4, 0.5, 0.6],
        [0.45, 0.5, 0.45],
    ]
    return random.choice(color_schemes)


def extrude_buildings(footprints):
    print("Extruding buildings...")

    city_mesh = pv.PolyData()
    instances_building = []

    for footprint_data in footprints:
        polygon = footprint_data["polygon"]
        height = footprint_data["height"]
        coords = np.array(polygon.exterior.coords)

        points, faces = create_watertight_buildings(coords, height)
        building = pv.PolyData(points, np.array(faces))

        color = generate_realistic_building_color()
        building['color'] = np.tile(color, (building.n_points, 1))

        instances_building.append(building)

        if city_mesh.n_points == 0:
            city_mesh = building
        else:
            city_mesh = city_mesh.merge(building, merge_points=False)

    print("Building extrusion complete")
    return city_mesh, instances_building


def save_to_obj(mesh, output_path):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving model to {output_path}")
    mesh.save(output_path)
    print("export completed")


def streetGraph_to_pyvista(st_graph):
    nodes, edges = ox.graph_to_gdfs(st_graph)
    pts_list = edges['geometry'].apply(lambda g: np.column_stack(
        (g.xy[0], g.xy[1], np.zeros(len(g.xy[0]))))).tolist()
    vertices = np.concatenate(pts_list)
    
    lines = []
    j = 0
    for i in range(len(pts_list)):
        pts = pts_list[i]
        vertex_length = len(pts)
        vertex_start = j
        vertex_end = j + vertex_length - 1
        vertex_arr = [vertex_length] + list(range(vertex_start, vertex_end + 1))
        lines.append(vertex_arr)
        j += vertex_length
    
    return pv.PolyData(vertices, lines=np.hstack(lines))


def cloudgify(location, mesh, street_mesh, file_path):
    pl = pv.Plotter(off_screen=True, image_scale=1)
    pl.background_color = 'k'
    
    actor = pl.add_text(
        location,
        position='upper_left',
        color='lightgrey',
        font_size=26,
    )
    
    pl.add_mesh(mesh, scalars=mesh['color'], cmap="tab20", show_edges=False)
    pl.add_mesh(street_mesh)
    pl.remove_scalar_bar()
    pl.show(auto_close=False)
    
    viewup = [0, 0, 1]
    output_dir = Path(file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path = pl.generate_orbital_path(n_points=40, shift=mesh.length,
                                     viewup=viewup, factor=3.0)
    pl.open_gif(str(output_dir / "model.gif"))
    pl.orbit_on_path(path, write_frames=True, viewup=viewup)
    pl.close()
    print("Export of GIF successful")


# ── Main ──────────────────────────────────────────────────────────────────────
lat = 48.8584
lon = 2.2945
location_name = "Paris, France"
radius = 1000

buildings, streets = extract_osm_data_coords(lat, lon, radius)
footprints = generate_footprints(buildings)
mesh, bd_instances = extrude_buildings(footprints)
street_mesh = streetGraph_to_pyvista(streets)

pl = pv.Plotter(border=False)
pl.add_mesh(mesh, scalars=mesh['color'], cmap="tab20", show_edges=False)
pl.remove_scalar_bar()
pl.show(title='3D Tech')

'''output_dir = "output/" + location_name.split(",")[0]
cloudgify(location_name, mesh, street_mesh, output_dir)

output_file = output_dir + "/buildings.obj"
output_streets = output_dir + "/streets.obj"
save_to_obj(mesh, output_file)
save_to_obj(street_mesh, output_streets)'''