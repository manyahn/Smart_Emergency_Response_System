import streamlit as st
import pandas as pd
from database import get_accidents_df, clear_accidents

st.set_page_config(page_title="📅 Accident Log", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg,#0d0f18 0%,#141824 100%); color:#e8eaf0; }
h1, h2, h3 { font-family: 'Rajdhani', sans-serif; }
.metric-card {
    background:linear-gradient(135deg,#1a1d2e,#1f2340);
    border:1px solid #2a2f4a; border-radius:12px; padding:14px 18px; margin-bottom:10px;
}
.metric-label { color:#8892b0; font-size:10px; font-weight:600; letter-spacing:1.2px; text-transform:uppercase; }
.metric-value { color:#e8eaf0; font-size:26px; font-family:'Rajdhani',sans-serif; font-weight:700; }
div[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0a0c16 0%,#111422 100%);
    border-right:1px solid #1e2235;
}
div[data-testid="stSidebar"] * { color:#c8cee0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📅 ACCIDENT DETECTION LOG</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8892b0;margin-top:-10px;'>Full history of all detected accident events</p>",
            unsafe_allow_html=True)
st.markdown("---")

df = get_accidents_df()

if st.sidebar.button("🗑️ Clear All Data"):
    clear_accidents()
    st.success("All records deleted.")
    st.rerun()

if df.empty:
    st.info("No accident records yet. Start detection to generate logs.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
min_d = df["date"].min().date()
max_d = df["date"].max().date()
sel_date = st.sidebar.date_input("Select Date", max_d, min_value=min_d, max_value=max_d)
fdf = df[df["date"].dt.date == sel_date].copy()

# ── KPIs ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
total    = len(fdf)
avg_conf = fdf["confidence"].dropna().mean()
most_lbl = fdf["label"].mode().iloc[0] if not fdf.empty else "—"
top_loc  = (
    fdf["location_name"].dropna().mode().iloc[0]
    if "location_name" in fdf.columns and fdf["location_name"].notna().any()
    else "—"
)

for col, lbl, val, clr in [
    (c1, "Accidents on Date", total,                              "#ff4444"),
    (c2, "Avg Confidence",    f"{avg_conf:.1%}" if avg_conf==avg_conf else "—", "#00b894"),
    (c3, "Most Common Label", most_lbl,                           "#64b5f6"),
    (c4, "Top Location",      top_loc[:22]+"…" if len(str(top_loc))>22 else top_loc, "#ffb432"),
]:
    col.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>{lbl}</div>
        <div class='metric-value' style='color:{clr};font-size:18px;'>{val}</div></div>""",
        unsafe_allow_html=True)

st.markdown(f"### Accidents on {sel_date.strftime('%B %d, %Y')}")

if fdf.empty:
    st.success("✅ No accidents detected on this date.")
else:
    # Column order — put location_name prominently
    ordered_cols = [c for c in
        ["id","date","time","location_name","source","label","confidence","vehicles","latitude","longitude"]
        if c in fdf.columns]
    st.dataframe(fdf[ordered_cols], use_container_width=True, hide_index=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.download_button(
    "⬇️ Download Log (CSV)",
    fdf.to_csv(index=False).encode("utf-8"),
    file_name=f"accident_log_{sel_date}.csv",
    mime="text/csv"
)

# ── Quick insights ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Database Overview")
st.markdown(f"""
- **Total records in database:** {len(df)}  
- **Date range:** {min_d} → {max_d}  
- **Last logged event:** {df.iloc[0]['date'].date()} at {df.iloc[0]['time']}  
""")