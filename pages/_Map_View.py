import streamlit as st
import pandas as pd
import pydeck as pdk
from database import get_accidents_df

st.set_page_config(page_title="🗺️ Accident Map View", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d0f18 0%, #141824 100%); color: #e8eaf0; }
h1, h2, h3 { font-family: 'Rajdhani', sans-serif; }
.metric-card {
    background: linear-gradient(135deg,#1a1d2e,#1f2340);
    border:1px solid #2a2f4a; border-radius:12px; padding:14px 18px; margin-bottom:10px;
}
.metric-label { color:#8892b0; font-size:10px; font-weight:600; letter-spacing:1.2px; text-transform:uppercase; }
.metric-value { color:#e8eaf0; font-size:24px; font-family:'Rajdhani',sans-serif; font-weight:700; }
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a0c16 0%,#111422 100%);
    border-right:1px solid #1e2235;
}
div[data-testid="stSidebar"] * { color:#c8cee0 !important; }
.location-pill {
    display:inline-block; background:#1f2340;
    border:1px solid #3a7bd5; border-radius:20px;
    padding:4px 12px; font-size:12px; color:#64b5f6; margin:2px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🗺️ ACCIDENT MAP VIEW</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8892b0;margin-top:-10px;'>Real-time GPS location of accident detections with area names</p>",
            unsafe_allow_html=True)
st.markdown("---")

# ── Load data ────────────────────────────────────────────────────────────────
df = get_accidents_df()

if df.empty:
    st.info("No accident data yet. Run detection to generate logs.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
map_df = df.dropna(subset=["latitude","longitude"]).copy()

if map_df.empty:
    st.info("No GPS data found. Run detection — real location is logged automatically.")
    st.stop()

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
min_d = map_df["date"].min().date()
max_d = map_df["date"].max().date()
start = st.sidebar.date_input("Start Date", min_d, min_value=min_d, max_value=max_d)
end   = st.sidebar.date_input("End Date",   max_d, min_value=min_d, max_value=max_d)

filtered = map_df[
    (map_df["date"].dt.date >= start) &
    (map_df["date"].dt.date <= end)
].copy()

if filtered.empty:
    st.warning("No accidents in selected date range.")
    st.stop()

# ── Stats row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Total Accidents</div>
    <div class='metric-value' style='color:#ff4444;'>{len(filtered)}</div></div>""",
    unsafe_allow_html=True)

unique_locs = filtered[["latitude","longitude"]].drop_duplicates().shape[0]
col2.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Unique Spots</div>
    <div class='metric-value'>{unique_locs}</div></div>""", unsafe_allow_html=True)

avg_conf = filtered["confidence"].dropna().mean()
col3.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Avg Confidence</div>
    <div class='metric-value'>{f"{avg_conf:.1%}" if not pd.isna(avg_conf) else "—"}</div></div>""",
    unsafe_allow_html=True)

date_range = (end - start).days + 1
col4.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Days Covered</div>
    <div class='metric-value'>{date_range}</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Area name pills ──────────────────────────────────────────────────────────
if "location_name" in filtered.columns:
    location_names = filtered["location_name"].dropna().unique()
    if len(location_names):
        st.markdown("**📍 Accident Locations Detected:**")
        pills = " ".join(
            f"<span class='location-pill'>📍 {name}</span>"
            for name in location_names if name
        )
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown("")

# ── Map ──────────────────────────────────────────────────────────────────────
st.subheader(f"🗺️ Accident Hotspots — {start} to {end}")

filtered["radius"]  = (filtered["confidence"].fillna(0.5) * 120).clip(60, 180)
filtered["tooltip_time"]   = filtered["time"].astype(str)
filtered["tooltip_conf"]   = filtered["confidence"].apply(
    lambda x: f"{x:.1%}" if pd.notna(x) else "—")
filtered["tooltip_loc"]    = filtered.get("location_name", "—").fillna("—")
filtered["tooltip_source"] = filtered["source"].fillna("—")
filtered["tooltip_veh"]    = filtered.get("vehicles", "—").fillna("—")

center_lat = filtered["latitude"].mean()
center_lon = filtered["longitude"].mean()

# Scatter layer — accident pins
scatter = pdk.Layer(
    "ScatterplotLayer",
    data=filtered,
    get_position="[longitude, latitude]",
    get_color="[255, 60, 60, 210]",
    get_radius="radius",
    radius_scale=6,
    radius_min_pixels=8,
    radius_max_pixels=50,
    pickable=True,
    auto_highlight=True,
)

# Heatmap layer
heatmap = pdk.Layer(
    "HeatmapLayer",
    data=filtered,
    get_position="[longitude, latitude]",
    opacity=0.55,
    threshold=0.05,
    radiusPixels=80,
)

# Text layer — show area name on map
text_layer = pdk.Layer(
    "TextLayer",
    data=filtered.drop_duplicates(subset=["latitude","longitude"]),
    get_position="[longitude, latitude]",
    get_text="tooltip_loc",
    get_size=12,
    get_color=[255, 200, 200, 220],
    get_anchor="'middle'",
    get_alignment_baseline="'bottom'",
    get_pixel_offset=[0, -20],
    pickable=False,
)

view = pdk.ViewState(
    latitude=center_lat, longitude=center_lon,
    zoom=12, pitch=45, bearing=0
)

tooltip = {
    "html": """
    <div style='background:#1a1d2e;border:1px solid #ff4444;border-radius:10px;
        padding:12px 16px;font-family:Inter,sans-serif;color:#e8eaf0;
        font-size:12px;min-width:220px;'>
        <div style='color:#ff6666;font-weight:700;font-size:14px;margin-bottom:6px;'>
            🚨 Accident Event</div>
        <div>📍 <b>{tooltip_loc}</b></div>
        <div style='margin-top:4px;'>🕐 {date} at {tooltip_time}</div>
        <div>🎯 Confidence: <b style='color:#00e676;'>{tooltip_conf}</b></div>
        <div>🚗 Objects: {tooltip_veh}</div>
        <div style='margin-top:4px;color:#8892b0;font-size:10px;'>
            Source: {tooltip_source}</div>
    </div>""",
    "style": {"background": "transparent", "border": "none", "padding": "0"}
}

deck = pdk.Deck(
    layers=[heatmap, scatter, text_layer],
    initial_view_state=view,
    tooltip=tooltip,
    map_style="mapbox://styles/mapbox/dark-v11",
)
st.pydeck_chart(deck)

# ── Details table ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Full Accident Log Table"):
    show_cols = [c for c in ["date","time","location_name","source","label",
                              "confidence","vehicles","latitude","longitude"]
                 if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download Map Data (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name=f"accident_map_{start}_{end}.csv",
    mime="text/csv"
)