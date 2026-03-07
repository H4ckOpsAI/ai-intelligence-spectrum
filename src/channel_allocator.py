"""
channel_allocator.py
--------------------
Intelligent channel reallocation using ML predictions.

Functions:
  - rebalance_network(G, model) : move excess nodes from congested channels
  - predict_future_congestion(G, model) : estimate upcoming congestion risk
  - early_rebalance_if_needed(G, model) : pre-emptive rebalancing
  - generate_system_alerts(G, model) : AI-generated alert messages
"""

import os
import sys
import random
import numpy as np
import networkx as nx

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network_simulation import (
    CHANNELS,
    CONGESTION_THRESHOLD,
    CHANNEL_CAPACITY,
    get_channel_counts,
    get_congested_channels,
    get_nodes_on_channel,
    get_congestion_level,
    get_average_load,
)
from src.predict_channel import predict_channel_status, predict_channel_proba

# Representative signal characteristics per channel (used for ML prediction)
CHANNEL_SIGNAL_PROFILES = {
    "channel_1":  {"snr": -55.0, "signal_power": -55.0, "frequency": 2412},
    "channel_3":  {"snr": -52.0, "signal_power": -52.0, "frequency": 2422},
    "channel_6":  {"snr": -50.0, "signal_power": -50.0, "frequency": 2437},
    "channel_9":  {"snr": -53.0, "signal_power": -53.0, "frequency": 2452},
    "channel_11": {"snr": -60.0, "signal_power": -60.0, "frequency": 2462},
}

# Frequency band info for legend
CHANNEL_FREQUENCY_INFO = {
    "channel_1":  {"freq_mhz": 2412, "band": "2.4 GHz"},
    "channel_3":  {"freq_mhz": 2422, "band": "2.4 GHz"},
    "channel_6":  {"freq_mhz": 2437, "band": "2.4 GHz"},
    "channel_9":  {"freq_mhz": 2452, "band": "2.4 GHz"},
    "channel_11": {"freq_mhz": 2462, "band": "2.4 GHz"},
}


def _get_profile(channel: str) -> dict:
    """Safely get signal profile for a channel, with fallback."""
    return CHANNEL_SIGNAL_PROFILES.get(channel, {
        "snr": -55.0, "signal_power": -55.0, "frequency": 2437,
    })


def ai_predict_channel_status(model, channel: str, channel_load: int) -> str:
    """Use the ML model to predict whether *channel* is busy or free."""
    profile = _get_profile(channel)
    return predict_channel_status(
        model,
        snr=profile["snr"],
        signal_power=profile["signal_power"],
        frequency=profile["frequency"],
        channel_load=channel_load,
    )


def ai_predict_channel_proba(model, channel: str, channel_load: int) -> dict:
    """Return probability dict for a channel given its current load."""
    profile = _get_profile(channel)
    return predict_channel_proba(
        model,
        snr=profile["snr"],
        signal_power=profile["signal_power"],
        frequency=profile["frequency"],
        channel_load=channel_load,
    )


# ---------------------------------------------------------------------------
#  REBALANCE NETWORK  (with safety checks)
# ---------------------------------------------------------------------------

def rebalance_network(G: nx.Graph, model) -> list[dict]:
    """
    Move nodes from congested channels to the least-loaded channel.

    Safety checks:
      - Prevent moving node to the same channel
      - Prevent negative node counts
      - Ensure target channel exists

    Returns
    -------
    movements : list[dict]
        Each dict has keys: node, from_channel, to_channel, reason
    """
    movements: list[dict] = []
    counts = get_channel_counts(G)
    congested = get_congested_channels(G)

    if not congested:
        return movements

    for ch in congested:
        ai_status = ai_predict_channel_status(model, ch, counts.get(ch, 0))
        excess = counts.get(ch, 0) - CONGESTION_THRESHOLD

        if excess <= 0:
            continue

        nodes_to_move = get_nodes_on_channel(G, ch)[:excess]

        for node in nodes_to_move:
            # Find least loaded channel that is NOT the current channel
            candidates = [c for c in CHANNELS if c != ch]
            if not candidates:
                continue

            target = min(candidates, key=lambda c: counts.get(c, 0))

            # Safety: prevent same-channel move
            if target == ch:
                continue

            old_channel = ch
            new_channel = target

            # Update the graph
            G.nodes[node]["channel"] = new_channel
            counts[old_channel] = max(0, counts.get(old_channel, 0) - 1)
            counts[new_channel] = counts.get(new_channel, 0) + 1

            # Remove old edges and add new ones
            neighbors = list(G.neighbors(node))
            for nbr in neighbors:
                if G.has_edge(node, nbr):
                    G.remove_edge(node, nbr)
            for other_node, data in G.nodes(data=True):
                if other_node != node and data.get("channel") == new_channel:
                    G.add_edge(node, other_node)

            movements.append({
                "node": node,
                "from_channel": old_channel,
                "to_channel": new_channel,
                "reason": f"AI predicted '{old_channel}' as {ai_status}; moved to '{new_channel}'",
            })

    return movements


# ---------------------------------------------------------------------------
#  CONGESTION PREDICTION
# ---------------------------------------------------------------------------

def predict_future_congestion(G: nx.Graph, model) -> dict[str, dict]:
    """
    Estimate whether each channel will become congested soon.

    Uses three signals:
      1. Current channel_load
      2. Recent trend (change in device count over last 3 time steps)
      3. ML model probability of 'busy'

    Returns
    -------
    dict[str, dict] mapping channel -> {
        "load": int,
        "trend": float,
        "busy_prob": float,
        "risk": str,           ("safe", "warning", "critical")
        "predicted_congestion": bool,
    }
    """
    counts = get_channel_counts(G)
    history = G.graph.get("load_history", {})
    predictions: dict[str, dict] = {}

    for ch in CHANNELS:
        load = counts.get(ch, 0)

        # --- trend from load_history ---
        ch_history = history.get(ch, [load])
        if len(ch_history) >= 3:
            recent = ch_history[-3:]
        else:
            recent = ch_history
        trend = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)

        # --- ML probability ---
        proba = ai_predict_channel_proba(model, ch, load)
        busy_prob = proba.get("busy", 0.0)

        # --- risk classification ---
        predicted_load = load + trend

        if predicted_load >= CONGESTION_THRESHOLD or busy_prob > 0.7:
            risk = "critical"
            predicted_congestion = True
        elif predicted_load >= CONGESTION_THRESHOLD - 1 or busy_prob > 0.5:
            risk = "warning"
            predicted_congestion = True
        else:
            risk = "safe"
            predicted_congestion = False

        predictions[ch] = {
            "load": load,
            "trend": round(trend, 2),
            "busy_prob": round(busy_prob, 3),
            "risk": risk,
            "predicted_congestion": predicted_congestion,
        }

    return predictions


def early_rebalance_if_needed(G: nx.Graph, model) -> list[dict]:
    """
    Check congestion predictions and trigger early rebalancing if needed.

    Moves 1 device from channels predicted to become congested
    to the safest channel, BEFORE actual congestion occurs.

    Returns list of movements (may be empty).
    """
    predictions = predict_future_congestion(G, model)
    movements: list[dict] = []
    counts = get_channel_counts(G)

    at_risk = [ch for ch, p in predictions.items()
               if p["predicted_congestion"] and counts.get(ch, 0) > 4]

    if not at_risk:
        return movements

    # Find the safest target channel
    safe_channels = [ch for ch in CHANNELS if predictions[ch]["risk"] == "safe"]
    if not safe_channels:
        safe_channels = CHANNELS  # fallback

    safest = min(safe_channels, key=lambda c: counts.get(c, 0))

    for ch in at_risk:
        # Safety: don't move to same channel
        if ch == safest:
            continue

        nodes = get_nodes_on_channel(G, ch)
        if not nodes:
            continue

        node = nodes[0]

        G.nodes[node]["channel"] = safest
        counts[ch] = max(0, counts.get(ch, 0) - 1)
        counts[safest] = counts.get(safest, 0) + 1

        # Update edges
        for nbr in list(G.neighbors(node)):
            if G.has_edge(node, nbr):
                G.remove_edge(node, nbr)
        for other_node, data in G.nodes(data=True):
            if other_node != node and data.get("channel") == safest:
                G.add_edge(node, other_node)

        movements.append({
            "node": node,
            "from_channel": ch,
            "to_channel": safest,
            "reason": f"Early rebalance: '{ch}' predicted {predictions[ch]['risk']} "
                      f"(trend={predictions[ch]['trend']:+.1f}, "
                      f"busy_prob={predictions[ch]['busy_prob']:.0%})",
        })

    return movements


# ---------------------------------------------------------------------------
#  SYSTEM ALERTS  (Feature 9)
# ---------------------------------------------------------------------------

def generate_system_alerts(G: nx.Graph, model) -> list[str]:
    """
    Generate AI-driven system alerts based on current network state.

    Returns list of alert strings.
    """
    alerts: list[str] = []
    counts = get_channel_counts(G)
    congested = get_congested_channels(G)
    predictions = predict_future_congestion(G, model)
    avg_load = get_average_load(G)

    # High network load
    if avg_load > 5:
        alerts.append(f"High network load detected (avg: {avg_load} devices/channel)")

    # Multiple congested channels
    if len(congested) >= 2:
        alerts.append(f"Multiple congested channels: {', '.join(congested)}")

    # Individual channel congestion alerts
    for ch in congested:
        alerts.append(f"Channel {ch.split('_')[1]} is congested ({counts[ch]} devices)")

    # Predicted congestion
    for ch, pred in predictions.items():
        if pred["risk"] == "critical" and ch not in congested:
            alerts.append(f"Channel {ch.split('_')[1]} predicted congestion (prob: {pred['busy_prob']:.0%})")
        elif pred["risk"] == "warning":
            alerts.append(f"Channel {ch.split('_')[1]} congestion risk rising (trend: {pred['trend']:+.1f})")

    # Near capacity
    for ch, cnt in counts.items():
        if cnt >= CHANNEL_CAPACITY:
            alerts.append(f"Channel {ch.split('_')[1]} at full capacity ({cnt}/{CHANNEL_CAPACITY})")

    return alerts if alerts else ["All systems nominal"]


# -- quick test --------------------------------------------------------------
if __name__ == "__main__":
    from src.network_simulation import create_network, create_demo_network
    from src.predict_channel import load_model

    G = create_demo_network()
    model = load_model()

    print("Before rebalancing:")
    for ch, cnt in get_channel_counts(G).items():
        print(f"  {ch}: {cnt} devices [{get_congestion_level(cnt)}]")

    moves = rebalance_network(G, model)
    print(f"\n[MOVES] {len(moves)} node(s) moved:")
    for m in moves:
        print(f"  {m['node']}: {m['from_channel']} -> {m['to_channel']}")

    print("\nAfter rebalancing:")
    for ch, cnt in get_channel_counts(G).items():
        print(f"  {ch}: {cnt} devices [{get_congestion_level(cnt)}]")

    print("\nAlerts:")
    for a in generate_system_alerts(G, model):
        print(f"  {a}")
