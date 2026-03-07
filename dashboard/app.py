"""
app.py - AI Spectrum Sharing Simulator Dashboard
=================================================
Professional Network Operations Center (NOC) style dashboard.

Sections:
  1. Network Status Overview (KPI cards)
  2. Channel Status Panel (per-channel metrics)
  3. AI Congestion Prediction Panel
  4. Simulation Controls (Run Next Step / Run 5 Steps / Rebalance)
  5. Channel Load Visualization (bar chart)
  6. Network Topology (graph)
  7. Node Movement Log (expander)
  8. Channel Capacity Bars
  9. Channel Legend Panel
 10. NOC Health Monitor
 11. System Alerts (sidebar)
 12. Simulation Performance Graph (load over time)
"""

import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# Allow imports from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.network_simulation import (
    create_network,
    create_demo_network,
    simulate_next_step,
    get_channel_counts,
    get_congested_channels,
    get_congestion_level,
    get_average_load,
    CHANNELS,
    CONGESTION_THRESHOLD,
    CHANNEL_CAPACITY,
)
from src.channel_allocator import (
    rebalance_network,
    predict_future_congestion,
    early_rebalance_if_needed,
    generate_system_alerts,
    CHANNEL_FREQUENCY_INFO,
)
from src.predict_channel import load_model

# =============================================================================
#  6G SPECTRUM SLOT DISPLAY MAPPING  (UI only — backend unchanged)
# =============================================================================
CHANNEL_DISPLAY_MAP = {
    "channel_1":  "Spectrum Slot A",
    "channel_3":  "Spectrum Slot B",
    "channel_6":  "Spectrum Slot C",
    "channel_9":  "Spectrum Slot D",
    "channel_11": "Spectrum Slot E",
}

SLOT_DESCRIPTIONS = {
    "channel_1":  "Sub-THz Band",
    "channel_3":  "mmWave Band",
    "channel_6":  "Mid-Band Spectrum",
    "channel_9":  "Shared Spectrum",
    "channel_11": "Dynamic Spectrum Pool",
}

def ch_label(ch: str) -> str:
    """Return the 6G display name for an internal channel ID."""
    return CHANNEL_DISPLAY_MAP.get(ch, ch)

def ch_short(ch: str) -> str:
    """Return short slot letter (A-E) for compact displays."""
    return CHANNEL_DISPLAY_MAP.get(ch, ch).replace("Spectrum Slot ", "Slot ")

def filter_text_labels(text: str) -> str:
    """Replace all channel references with Slot names in logs and alerts."""
    for ch, slot in CHANNEL_DISPLAY_MAP.items():
        # Replace code-level references 'channel_1'
        text = text.replace(f"'{ch}'", slot)
        text = text.replace(ch, slot)
        # Replace typical strings 'Channel 1'
        ch_num = ch.split('_')[1]
        text = text.replace(f"Channel {ch_num}", slot)
    return text

# =============================================================================
#  PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AI Spectrum Sharing Simulator Dashboard",
    page_icon="📡",
    layout="wide",
)

# =============================================================================
#  CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 14px;
    padding: 18px 12px;
    color: white;
    text-align: center;
    margin-bottom: 8px;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3);
}
.kpi-card h2 { margin: 0; font-size: 2rem; }
.kpi-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }

/* Status colors */
.status-free      { background: linear-gradient(135deg, #27ae60, #2ecc71) !important; }
.status-moderate  { background: linear-gradient(135deg, #f39c12, #e67e22) !important; }
.status-congested { background: linear-gradient(135deg, #e74c3c, #c0392b) !important; }

/* NOC Panel */
.noc-panel {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 14px;
    padding: 20px;
    color: #e0e0e0;
    border: 1px solid #2c3e50;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.noc-panel h3 { color: #3498db; margin-top: 0; }
.noc-metric { font-size: 1.1rem; margin: 6px 0; }
.noc-ok     { color: #2ecc71; }
.noc-warn   { color: #f39c12; }
.noc-crit   { color: #e74c3c; }

/* Alert panel */
.alert-item {
    background: rgba(231, 76, 60, 0.1);
    border-left: 3px solid #e74c3c;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
}
.alert-ok {
    background: rgba(46, 204, 113, 0.1);
    border-left: 3px solid #2ecc71;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
}

/* Legend */
.legend-box {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 14px;
    color: #ccc;
    border: 1px solid #2c3e50;
}
.legend-box h4 { color: #3498db; margin-top: 0; }

/* Prediction */
.pred-safe     { color: #2ecc71; font-weight: bold; }
.pred-warning  { color: #f39c12; font-weight: bold; }
.pred-critical { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
#  TITLE
# =============================================================================
st.title("📡 AI Spectrum Sharing Simulator Dashboard")
st.caption("6G Spectrum Orchestration Platform | Real-time Wireless Spectrum Monitoring")

# =============================================================================
#  SIDEBAR - CONTROLS & ALERTS
# =============================================================================
st.sidebar.header("⚙️ Simulation Controls")
seed = st.sidebar.slider("Random Seed", 0, 100, 42)
demo_mode = st.sidebar.toggle("🎯 Demo Mode", value=False,
                                help="Start with intentional congestion to demo AI rebalancing")

# --- Load model ---------------------------------------------------------------
@st.cache_resource
def get_model():
    model_path = os.path.join(PROJECT_ROOT, "models", "trained_model.pkl")
    if not os.path.exists(model_path):
        st.error("Trained model not found! Run `python src/train_model.py` first.")
        st.stop()
    return load_model(model_path)


model = get_model()

# --- Session state init -------------------------------------------------------
needs_reset = (
    "network" not in st.session_state
    or st.session_state.get("seed") != seed
    or st.session_state.get("demo_mode") != demo_mode
)
if needs_reset:
    if demo_mode:
        st.session_state.network = create_demo_network(seed=seed)
    else:
        st.session_state.network = create_network(seed=seed)
    st.session_state.seed = seed
    st.session_state.demo_mode = demo_mode
    st.session_state.movements = []
    st.session_state.all_movements = []
    st.session_state.step_log = []

G = st.session_state.network
counts = get_channel_counts(G)
time_step = G.graph.get("time_step", 0)

# --- Sidebar: System Alerts (Feature 9) --------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🚨 System Alerts")

alerts = generate_system_alerts(G, model)
graph_alerts = G.graph.get("alerts", [])

# Combine real-time + historical alerts
all_display_alerts = alerts + graph_alerts[-5:]

for alert in all_display_alerts:
    alert_text = filter_text_labels(alert)
    if "nominal" in alert_text.lower():
        st.sidebar.markdown(f'<div class="alert-ok">✅ {alert_text}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<div class="alert-item">⚠️ {alert_text}</div>', unsafe_allow_html=True)

# =============================================================================
#  SECTION 1 — NETWORK STATUS OVERVIEW (KPI Cards)
# =============================================================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

congested_list = get_congested_channels(G)

with col1:
    st.markdown(f"""<div class="kpi-card">
        <p>Total Devices</p><h2>{G.number_of_nodes()}</h2>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="kpi-card">
        <p>Spectrum Slots</p><h2>{len(CHANNELS)}</h2>
    </div>""", unsafe_allow_html=True)

with col3:
    css = "status-congested" if congested_list else "status-free"
    st.markdown(f"""<div class="kpi-card {css}">
        <p>Congested Slots</p><h2>{len(congested_list)}</h2>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="kpi-card">
        <p>Current Time Step</p><h2>{time_step}</h2>
    </div>""", unsafe_allow_html=True)

# =============================================================================
#  SECTION 2 — CHANNEL STATUS PANEL
# =============================================================================
st.markdown("---")
st.subheader("📶 Spectrum Slot Status")

ch_cols = st.columns(len(CHANNELS))
STATUS_ICONS = {"free": "🟢", "moderate": "🟡", "congested": "🔴"}

for i, ch in enumerate(CHANNELS):
    cnt = counts.get(ch, 0)
    level = get_congestion_level(cnt)
    icon = STATUS_ICONS[level]
    with ch_cols[i]:
        st.metric(
            label=ch_label(ch),
            value=f"{cnt} devices",
            delta=f"{icon} {level.capitalize()}",
        )

# =============================================================================
#  SECTION 3 — AI CONGESTION PREDICTION PANEL
# =============================================================================
st.markdown("---")
st.subheader("🔮 AI Congestion Prediction")

predictions = predict_future_congestion(G, model)
pred_cols = st.columns(len(CHANNELS))

RISK_ICONS = {"safe": "✅", "warning": "⚠️", "critical": "🔴"}
RISK_CSS = {"safe": "pred-safe", "warning": "pred-warning", "critical": "pred-critical"}

for i, ch in enumerate(CHANNELS):
    pred = predictions[ch]
    icon = RISK_ICONS[pred["risk"]]
    css = RISK_CSS[pred["risk"]]
    with pred_cols[i]:
        st.markdown(f"**{ch_label(ch)}**")
        st.markdown(f'<span class="{css}">{icon} {pred["risk"].upper()}</span>',
                    unsafe_allow_html=True)
        st.caption(f"Load: {pred['load']} | Trend: {pred['trend']:+.1f} | Busy: {pred['busy_prob']:.0%}")

# =============================================================================
#  SECTION 4 — SIMULATION CONTROLS
# =============================================================================
st.markdown("---")
btn_col1, btn_col2, btn_col3 = st.columns(3)

with btn_col1:
    if st.button("▶️ Run Next Step", use_container_width=True, type="primary"):
        result = simulate_next_step(G, model=model, rebalance_fn=rebalance_network)
        early_moves = early_rebalance_if_needed(G, model)
        all_moves = result["movements"] + early_moves
        st.session_state.movements = all_moves
        st.session_state.all_movements.extend(all_moves)
        st.session_state.step_log.append(result)
        st.rerun()

with btn_col2:
    if st.button("⏭️ Run 5 Steps", use_container_width=True):
        last_moves = []
        for _ in range(5):
            result = simulate_next_step(G, model=model, rebalance_fn=rebalance_network)
            early_moves = early_rebalance_if_needed(G, model)
            all_moves = result["movements"] + early_moves
            st.session_state.all_movements.extend(all_moves)
            st.session_state.step_log.append(result)
            last_moves = all_moves
        st.session_state.movements = last_moves
        st.rerun()

with btn_col3:
    if st.button("🔄 Rebalance Network", use_container_width=True):
        movements = rebalance_network(G, model)
        st.session_state.movements = movements
        st.session_state.all_movements.extend(movements)
        st.rerun()

# =============================================================================
#  SECTION 5 — CHANNEL LOAD VISUALIZATION (Bar Chart)
# =============================================================================
st.markdown("---")
st.subheader("📊 Spectrum Load Visualization")

COLOR_MAP = {"free": "#27ae60", "moderate": "#f39c12", "congested": "#e74c3c"}

fig_bar, ax_bar = plt.subplots(figsize=(12, 4))
channel_labels = [ch_short(ch) for ch in CHANNELS]
channel_values = [counts.get(ch, 0) for ch in CHANNELS]
bar_colors = [COLOR_MAP[get_congestion_level(v)] for v in channel_values]

bars = ax_bar.bar(channel_labels, channel_values, color=bar_colors,
                  edgecolor="white", linewidth=1.5, width=0.6)
ax_bar.set_xticks(range(len(channel_labels)))
ax_bar.set_xticklabels(channel_labels)
ax_bar.axhline(y=CONGESTION_THRESHOLD, color="#e74c3c", linestyle="--",
               linewidth=1.2, label=f"Threshold ({CONGESTION_THRESHOLD})")
ax_bar.set_ylabel("Number of Devices", fontweight="bold")
ax_bar.set_xlabel("Spectrum Slot", fontweight="bold")
ax_bar.set_title("Devices per Spectrum Slot", fontweight="bold", fontsize=14)
ax_bar.legend()
ax_bar.set_ylim(0, max(max(channel_values) + 2, CHANNEL_CAPACITY + 1))

for bar, val in zip(bars, channel_values):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)

fig_bar.patch.set_alpha(0)
ax_bar.set_facecolor("#0e1117")
ax_bar.tick_params(colors="white")
ax_bar.xaxis.label.set_color("white")
ax_bar.yaxis.label.set_color("white")
ax_bar.title.set_color("white")
for spine in ax_bar.spines.values():
    spine.set_color("white")

st.pyplot(fig_bar)

# =============================================================================
#  SECTION 5B — CHANNEL CAPACITY BARS (Feature 5)
# =============================================================================
st.markdown("---")
st.subheader("🔋 Spectrum Slot Utilization")

cap_cols = st.columns(len(CHANNELS))
for i, ch in enumerate(CHANNELS):
    cnt = counts.get(ch, 0)
    utilization = min(cnt / CHANNEL_CAPACITY, 1.0)
    with cap_cols[i]:
        st.markdown(f"**{ch_short(ch)}**")
        st.progress(utilization, text=f"{cnt}/{CHANNEL_CAPACITY}")

# =============================================================================
#  SECTION 6 — NETWORK TOPOLOGY
# =============================================================================
st.markdown("---")
st.subheader("🌐 Network Topology")

CHANNEL_COLORS = {
    "channel_1":  "#3498db",   # blue
    "channel_3":  "#9b59b6",   # purple
    "channel_6":  "#e67e22",   # orange
    "channel_9":  "#e74c3c",   # red
    "channel_11": "#2ecc71",   # green
}

fig_net, ax_net = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42, k=2.0)

node_colors = [CHANNEL_COLORS.get(G.nodes[n].get("channel"), "#95a5a6") for n in G.nodes()]

nx.draw_networkx_edges(G, pos, ax=ax_net, alpha=0.12, edge_color="#555")
nx.draw_networkx_nodes(
    G, pos, ax=ax_net,
    node_color=node_colors,
    node_size=700,
    edgecolors="white",
    linewidths=1.5,
)
nx.draw_networkx_labels(
    G, pos, ax=ax_net,
    font_size=8,
    font_color="white",
    font_weight="bold",
)

legend_patches = [mpatches.Patch(color=c, label=ch_short(ch))
                  for ch, c in CHANNEL_COLORS.items()]
ax_net.legend(handles=legend_patches, loc="upper left", fontsize=9,
              framealpha=0.8, facecolor="#1a1a2e", labelcolor="white")
ax_net.set_title(
    f"6G Spectrum Topology (Step {time_step})",
    fontweight="bold", fontsize=14, color="white",
)
ax_net.axis("off")
fig_net.patch.set_alpha(0)
ax_net.set_facecolor("#0e1117")

st.pyplot(fig_net)

# =============================================================================
#  SECTION 7 — NODE MOVEMENT LOG
# =============================================================================
st.markdown("---")
st.subheader("📦 Node Movement Log")

recent_movements = st.session_state.all_movements[-10:] if st.session_state.all_movements else []

if recent_movements:
    with st.expander(f"Last {len(recent_movements)} movements", expanded=True):
        for m in reversed(recent_movements):
            from_slot = ch_label(m['from_channel'])
            to_slot = ch_label(m['to_channel'])
            reason_text = filter_text_labels(m['reason'])
            st.markdown(
                f"- **{m['node']}** moved from {from_slot} -> {to_slot}  \n"
                f"  _{reason_text}_"
            )
else:
    st.info("No node movements yet. Click **Run Next Step** or **Rebalance Network** to start.")

# =============================================================================
#  SECTION 8 — NOC HEALTH MONITOR (Feature 8)
# =============================================================================
st.markdown("---")
st.subheader("🏥 Network Health Monitor")

noc_col1, noc_col2 = st.columns(2)

with noc_col1:
    avg_load = get_average_load(G)
    total_devices = G.number_of_nodes()
    total_channels = len(CHANNELS)
    num_congested = len(congested_list)

    health_class = "noc-ok" if num_congested == 0 else ("noc-warn" if num_congested <= 1 else "noc-crit")
    health_icon = "🟢" if num_congested == 0 else ("🟡" if num_congested <= 1 else "🔴")

    st.markdown(f"""<div class="noc-panel">
        <h3>🏥 Network Health Monitor</h3>
        <div class="noc-metric">Total Devices: <strong>{total_devices}</strong></div>
        <div class="noc-metric">Spectrum Slots: <strong>{total_channels}</strong></div>
        <div class="noc-metric">Congested Slots: <strong class="{health_class}">{num_congested}</strong></div>
        <div class="noc-metric">Avg Slot Load: <strong>{avg_load}</strong></div>
        <div class="noc-metric">System Status: <span class="{health_class}">{health_icon} {"HEALTHY" if num_congested == 0 else "DEGRADED" if num_congested <= 1 else "CRITICAL"}</span></div>
    </div>""", unsafe_allow_html=True)

with noc_col2:
    # AI alerts summary
    ai_alerts = generate_system_alerts(G, model)
    alert_html = ""
    for a in ai_alerts[:6]:
        a_text = filter_text_labels(a)
        if "nominal" in a_text.lower():
            alert_html += f'<div class="alert-ok">✅ {a_text}</div>'
        else:
            alert_html += f'<div class="alert-item">⚠️ {a_text}</div>'

    st.markdown(f"""<div class="noc-panel">
        <h3>🚨 AI Alerts</h3>
        {alert_html}
    </div>""", unsafe_allow_html=True)

# =============================================================================
#  SECTION 10 — SIMULATION PERFORMANCE GRAPH (Feature 10)
# =============================================================================
st.markdown("---")
st.subheader("📈 Spectrum Slot Load History")

history = G.graph.get("load_history", {})
has_history = any(len(h) > 1 for h in history.values())

if has_history:
    fig_perf, ax_perf = plt.subplots(figsize=(12, 5))

    HISTORY_COLORS = {
        "channel_1":  "#3498db",
        "channel_3":  "#9b59b6",
        "channel_6":  "#e67e22",
        "channel_9":  "#e74c3c",
        "channel_11": "#2ecc71",
    }

    max_len = max(len(v) for v in history.values())
    x = list(range(max_len))

    for ch in CHANNELS:
        ch_data = history.get(ch, [])
        if ch_data:
            ax_perf.plot(range(len(ch_data)), ch_data,
                         color=HISTORY_COLORS.get(ch, "#888"),
                         linewidth=2, marker="o", markersize=4,
                         label=ch_short(ch), alpha=0.9)

    ax_perf.axhline(y=CONGESTION_THRESHOLD, color="#e74c3c", linestyle="--",
                     linewidth=1.2, alpha=0.7, label=f"Threshold ({CONGESTION_THRESHOLD})")
    ax_perf.set_xlabel("Time Step", fontweight="bold")
    ax_perf.set_ylabel("Device Count", fontweight="bold")
    ax_perf.set_title("Spectrum Slot Load History", fontweight="bold", fontsize=14)
    ax_perf.legend(fontsize=8, ncol=3)
    ax_perf.set_ylim(bottom=0)

    fig_perf.patch.set_alpha(0)
    ax_perf.set_facecolor("#0e1117")
    ax_perf.tick_params(colors="white")
    ax_perf.xaxis.label.set_color("white")
    ax_perf.yaxis.label.set_color("white")
    ax_perf.title.set_color("white")
    for spine in ax_perf.spines.values():
        spine.set_color("white")

    st.pyplot(fig_perf)
else:
    st.info("Run some simulation steps to see load history over time.")

# =============================================================================
#  SECTION 6B — CHANNEL LEGEND PANEL (Feature 6)
# =============================================================================
st.markdown("---")
st.subheader("📖 6G Spectrum Slot Legend")

legend_col1, legend_col2 = st.columns(2)

with legend_col1:
    slot_lines = ""
    for ch in CHANNELS:
        name = ch_label(ch)
        desc = SLOT_DESCRIPTIONS.get(ch, "Unknown")
        slot_lines += f"<div style='margin:5px 0;'>{name} &mdash; <strong>{desc}</strong></div>"

    st.markdown(f"""<div class="legend-box">
        <h4>🛰️ Future 6G Spectrum Slots</h4>
        {slot_lines}
    </div>""", unsafe_allow_html=True)

with legend_col2:
    st.markdown("""<div class="legend-box">
        <h4>🔭 Next-Gen Bands (Roadmap)</h4>
        <div style='margin:5px 0; opacity:0.5;'>Slot F &mdash; Visible Light Communication</div>
        <div style='margin:5px 0; opacity:0.5;'>Slot G &mdash; Satellite Direct-to-Cell</div>
        <div style='margin:5px 0; opacity:0.5;'>Slot H &mdash; Reconfigurable Intelligent Surface</div>
        <div style='margin:5px 0; opacity:0.5;'><em>Planned for v4.0</em></div>
    </div>""", unsafe_allow_html=True)

# =============================================================================
#  STEP LOG (collapsible)
# =============================================================================
if st.session_state.step_log:
    with st.expander("📋 Recent Simulation Steps", expanded=False):
        for entry in reversed(st.session_state.step_log[-8:]):
            added_str = ", ".join(f"{a['node']}->{ch_short(a['channel'])}" for a in entry["added"])
            removed_str = ", ".join(r["node"] for r in entry["removed"])
            cong = ", ".join(ch_short(c) for c in entry["congested"]) if entry["congested"] else "none"
            st.caption(
                f"**Step {entry['time_step']}** | "
                f"Added: {added_str or 'none'} | "
                f"Removed: {removed_str or 'none'} | "
                f"Congested: {cong}"
            )

# =============================================================================
#  FOOTER
# =============================================================================
st.markdown("---")
st.caption(
    "AI Spectrum Sharing Simulator v3.0 | "
    "6G Spectrum Orchestration Platform | "
    "Powered by scikit-learn, networkx & Streamlit"
)
