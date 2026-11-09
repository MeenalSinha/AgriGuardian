"""Interactive farm location map component using Folium."""

from typing import Tuple, Optional
import folium
import streamlit as st
from streamlit_folium import st_folium


def render_farmer_location_map(
    lat: float = 28.6139,
    lon: float = 77.2090,
    crop: str = "Crop",
    farmer_name: str = "Farmer",
    height: int = 280,
) -> Tuple[float, float]:
    """Render an interactive map centered at the farm location.

    Returns:
        Tuple of (selected_lat, selected_lon).
    """
    m = folium.Map(
        location=[lat, lon],
        zoom_start=9,
        tiles="OpenStreetMap",
    )

    folium.Marker(
        [lat, lon],
        popup=f"<b>{farmer_name}'s Farm</b><br>Crop: {crop}",
        tooltip=f"{farmer_name} - {crop}",
        icon=folium.Icon(color="green", icon="leaf"),
    ).add_to(m)

    map_data = st_folium(m, height=height, width=None)

    # If the user clicks on the map, update coordinates
    if map_data and map_data.get("last_clicked"):
        click_lat = map_data["last_clicked"]["lat"]
        click_lng = map_data["last_clicked"]["lng"]
        return click_lat, click_lng

    return lat, lon

