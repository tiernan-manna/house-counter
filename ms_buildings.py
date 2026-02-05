"""
Microsoft Building Footprints module.
Queries building footprints from Overture Maps (which includes Microsoft's ML-derived buildings).
Uses the official overturemaps library for optimized access.
No API key required - data is publicly accessible.
"""
import math
from typing import List, Tuple
from dataclasses import dataclass
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import transform
import pyproj

# Import overturemaps library
import overturemaps


@dataclass
class Building:
    """Represents a building from building footprints data."""
    id: int
    lat: float
    lon: float
    area_sqm: float
    geometry: any  # Shapely polygon


def get_bounding_box(lat: float, lon: float, radius_meters: float) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box from center point and radius.
    Returns (min_lon, min_lat, max_lon, max_lat).
    """
    R = 6371000  # Earth's radius in meters
    angular_distance = radius_meters / R
    
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    min_lat = math.degrees(lat_rad - angular_distance)
    max_lat = math.degrees(lat_rad + angular_distance)
    
    delta_lon = math.asin(math.sin(angular_distance) / math.cos(lat_rad))
    min_lon = math.degrees(lon_rad - delta_lon)
    max_lon = math.degrees(lon_rad + delta_lon)
    
    return (min_lon, min_lat, max_lon, max_lat)


def query_ms_buildings_in_radius(
    lat: float,
    lon: float,
    radius_meters: float
) -> Tuple[List[Building], gpd.GeoDataFrame]:
    """
    Query building footprints from Overture Maps (includes Microsoft buildings).
    
    Uses the overturemaps library which provides optimized access to the data.
    No API key required.
    
    Args:
        lat: Center latitude
        lon: Center longitude
        radius_meters: Search radius in meters
        
    Returns:
        Tuple of (list of Building objects, GeoDataFrame with all buildings)
    """
    bbox = get_bounding_box(lat, lon, radius_meters)
    min_lon, min_lat, max_lon, max_lat = bbox
    
    print(f"Querying Overture Maps for buildings in bbox: {bbox}")
    print("This may take a moment...")
    
    try:
        # Use overturemaps library to query buildings
        # The library handles the partitioning and efficient querying
        bbox_tuple = (min_lon, min_lat, max_lon, max_lat)
        
        # Query the buildings type
        gdf = overturemaps.record_batch_reader("building", bbox_tuple).read_all().to_pandas()
        
        print(f"Query returned {len(gdf)} buildings from Overture Maps")
        
        if len(gdf) == 0:
            return [], gpd.GeoDataFrame()
        
        # Convert to GeoDataFrame if not already
        if 'geometry' in gdf.columns:
            # The geometry might be in WKB format
            from shapely import wkb
            if isinstance(gdf['geometry'].iloc[0], bytes):
                gdf['geometry'] = gdf['geometry'].apply(lambda x: wkb.loads(x) if x else None)
            
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')
        else:
            print("No geometry column found")
            return [], gpd.GeoDataFrame()
        
    except Exception as e:
        print(f"Error querying Overture Maps: {e}")
        import traceback
        traceback.print_exc()
        return [], gpd.GeoDataFrame()
    
    # Create search circle for more precise filtering
    center = Point(lon, lat)
    
    # Project to UTM for accurate distance calculation
    utm_zone = int((lon + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}" if lat >= 0 else f"EPSG:{32700 + utm_zone}"
    
    project_to_utm = pyproj.Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True).transform
    project_to_wgs = pyproj.Transformer.from_crs(utm_crs, 'EPSG:4326', always_xy=True).transform
    
    center_utm = transform(project_to_utm, center)
    search_circle_utm = center_utm.buffer(radius_meters)
    search_circle = transform(project_to_wgs, search_circle_utm)
    
    # Filter to buildings within the radius (the bbox is slightly larger)
    gdf = gdf[gdf.geometry.intersects(search_circle)]
    
    print(f"Buildings within {radius_meters}m radius: {len(gdf)}")
    
    # Convert to Building objects
    buildings = []
    for idx, row in gdf.iterrows():
        try:
            centroid = row.geometry.centroid
            
            # Calculate area in square meters
            geom_utm = transform(project_to_utm, row.geometry)
            area = geom_utm.area
            
            buildings.append(Building(
                id=hash(str(row.get('id', idx))) % (10**9),
                lat=centroid.y,
                lon=centroid.x,
                area_sqm=area,
                geometry=row.geometry
            ))
        except Exception as e:
            continue
    
    return buildings, gdf


def get_building_polygons_ms(
    lat: float,
    lon: float, 
    radius_meters: float
) -> List[dict]:
    """
    Get building polygons for visualization.
    Returns list of dicts compatible with the visualization module.
    """
    buildings, gdf = query_ms_buildings_in_radius(lat, lon, radius_meters)
    
    polygons = []
    for building in buildings:
        try:
            # Convert shapely geometry to list of coordinates
            if building.geometry.geom_type == 'Polygon':
                coords = [(y, x) for x, y in building.geometry.exterior.coords]
            elif building.geometry.geom_type == 'MultiPolygon':
                largest = max(building.geometry.geoms, key=lambda g: g.area)
                coords = [(y, x) for x, y in largest.exterior.coords]
            else:
                continue
        except:
            continue
            
        polygons.append({
            "id": building.id,
            "coordinates": coords,
            "type": "building",
            "center": (building.lat, building.lon),
            "area_sqm": building.area_sqm
        })
    
    return polygons
