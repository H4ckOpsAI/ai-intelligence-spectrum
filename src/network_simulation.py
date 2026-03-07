"""
network_simulation.py
---------------------
Simulate a wireless network using **networkx** with time-step progression.

- 20 device nodes
- 5 Wi-Fi channels: channel_1, channel_3, channel_6, channel_9, channel_11
- Random initial assignment with reproducible seeds
- Short node labels: D0, D1, D2, ...
- Congestion detection: channel is congested when devices >= 7
- Time-step simulation: devices connect/disconnect each step
- Demo mode: intentional congestion for AI rebalancing demonstration
"""

import random
import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
#  REPRODUCIBLE SEEDS
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
#  CONSTANTS
# ---------------------------------------------------------------------------
CHANNELS = ["channel_1", "channel_3", "channel_6", "channel_9", "channel_11"]
NUM_NODES = 20
CONGESTION_THRESHOLD = 7   # devices >= 7 -> congested
CHANNEL_CAPACITY = 10      # max devices per channel (for utilization bars)
_next_device_id = NUM_NODES


# ---------------------------------------------------------------------------
#  NETWORK CREATION
# ---------------------------------------------------------------------------

def create_network(num_nodes: int = NUM_NODES, seed: int | None = None) -> nx.Graph:
    """
    Build a network graph with *num_nodes* devices randomly assigned to channels.

    Each node has attributes:
        - channel : str  (one of CHANNELS)
        - device_id : int

    The graph also stores:
        - graph["time_step"] : int
        - graph["load_history"] : dict[str, list[int]]

    Returns
    -------
    nx.Graph
    """
    global _next_device_id

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.Graph()
    for i in range(num_nodes):
        ch = random.choice(CHANNELS)
        G.add_node(f"D{i}", channel=ch, device_id=i)

    _rebuild_edges(G)

    # Simulation metadata
    G.graph["time_step"] = 0
    G.graph["load_history"] = {ch: [get_channel_counts(G)[ch]] for ch in CHANNELS}
    G.graph["alerts"] = []

    _next_device_id = num_nodes
    return G


def create_demo_network(seed: int = 42) -> nx.Graph:
    """
    Create a network with intentional congestion for demo purposes.

    Distribution:
        channel_1  -> 9 devices  (congested)
        channel_3  -> 7 devices  (congested)
        channel_6  -> 5 devices  (moderate)
        channel_9  -> 4 devices  (free)
        channel_11 -> 3 devices  (free)
    Total: 28 devices
    """
    global _next_device_id
    random.seed(seed)
    np.random.seed(seed)

    demo_distribution = {
        "channel_1": 9,
        "channel_3": 7,
        "channel_6": 5,
        "channel_9": 4,
        "channel_11": 3,
    }

    G = nx.Graph()
    device_id = 0
    for ch, count in demo_distribution.items():
        for _ in range(count):
            G.add_node(f"D{device_id}", channel=ch, device_id=device_id)
            device_id += 1

    _rebuild_edges(G)

    G.graph["time_step"] = 0
    G.graph["load_history"] = {ch: [get_channel_counts(G)[ch]] for ch in CHANNELS}
    G.graph["alerts"] = ["[DEMO] Simulation initialized with intentional congestion"]

    _next_device_id = device_id
    return G


# ---------------------------------------------------------------------------
#  EDGE MANAGEMENT
# ---------------------------------------------------------------------------

def _rebuild_edges(G: nx.Graph):
    """Rebuild all edges so devices on the same channel are connected."""
    G.remove_edges_from(list(G.edges()))
    nodes_by_channel: dict[str, list[str]] = {ch: [] for ch in CHANNELS}
    for n, data in G.nodes(data=True):
        ch = data.get("channel")
        if ch in nodes_by_channel:
            nodes_by_channel[ch].append(n)
    for ch_nodes in nodes_by_channel.values():
        for i in range(len(ch_nodes)):
            for j in range(i + 1, len(ch_nodes)):
                G.add_edge(ch_nodes[i], ch_nodes[j])


# ---------------------------------------------------------------------------
#  CHANNEL QUERIES  (always safe — returns 0 for empty channels)
# ---------------------------------------------------------------------------

def get_channel_counts(G: nx.Graph) -> dict[str, int]:
    """Return {channel_name: device_count} for every channel (always all 5)."""
    counts = {ch: 0 for ch in CHANNELS}
    for _, data in G.nodes(data=True):
        ch = data.get("channel")
        if ch in counts:
            counts[ch] += 1
    return counts


def get_congested_channels(G: nx.Graph) -> list[str]:
    """Return list of channel names that are congested (>= threshold)."""
    counts = get_channel_counts(G)
    return [ch for ch, cnt in counts.items() if cnt >= CONGESTION_THRESHOLD]


def get_free_channels(G: nx.Graph) -> list[str]:
    """Return list of channel names that are NOT congested."""
    counts = get_channel_counts(G)
    return [ch for ch, cnt in counts.items() if cnt < CONGESTION_THRESHOLD]


def get_nodes_on_channel(G: nx.Graph, channel: str) -> list[str]:
    """Return list of node names assigned to *channel*."""
    return [n for n, d in G.nodes(data=True) if d.get("channel") == channel]


def get_congestion_level(count: int) -> str:
    """
    Classify congestion level for dashboard colour coding.

    - < 4   -> 'free'      (green)
    - < 7   -> 'moderate'  (yellow)
    - >= 7  -> 'congested' (red)
    """
    if count < 4:
        return "free"
    elif count < CONGESTION_THRESHOLD:
        return "moderate"
    else:
        return "congested"


def get_average_load(G: nx.Graph) -> float:
    """Return average device count across all channels."""
    counts = get_channel_counts(G)
    total = sum(counts.values())
    return round(total / max(len(CHANNELS), 1), 1)


# ---------------------------------------------------------------------------
#  TIME-STEP SIMULATION
# ---------------------------------------------------------------------------

def simulate_next_step(G: nx.Graph, model=None, rebalance_fn=None) -> dict:
    """
    Advance the simulation by one time step.

    Each step:
      1. Randomly adds 1-3 new devices to random channels
      2. Randomly removes 0-2 existing devices
      3. Updates channel_load counts
      4. Detects congestion
      5. Triggers rebalancing if a model and rebalance function are provided

    Parameters
    ----------
    G : nx.Graph              - the current network graph
    model : sklearn model     - trained ML model (optional)
    rebalance_fn : callable   - rebalance_network(G, model) -> list[dict]

    Returns
    -------
    dict with keys:
        time_step, added, removed, counts, congested, movements, alerts
    """
    global _next_device_id

    step = G.graph.get("time_step", 0) + 1
    G.graph["time_step"] = step
    step_alerts: list[str] = []

    # --- 1. Add new devices (1-3) ----------------------------------------
    num_add = random.randint(1, 3)
    added = []
    for _ in range(num_add):
        ch = random.choice(CHANNELS)
        name = f"D{_next_device_id}"
        G.add_node(name, channel=ch, device_id=_next_device_id)
        _next_device_id += 1
        added.append({"node": name, "channel": ch})

    # --- 2. Remove random devices (0-2) ----------------------------------
    num_remove = random.randint(0, min(2, max(G.number_of_nodes() - 1, 0)))
    removed = []
    if num_remove > 0:
        candidates = list(G.nodes())
        to_remove = random.sample(candidates, min(num_remove, len(candidates)))
        for node in to_remove:
            ch = G.nodes[node].get("channel", "unknown")
            removed.append({"node": node, "channel": ch})
            G.remove_node(node)

    # --- 3. Rebuild edges ------------------------------------------------
    _rebuild_edges(G)

    # --- 4. Update counts & history --------------------------------------
    counts = get_channel_counts(G)
    history = G.graph.setdefault("load_history", {ch: [] for ch in CHANNELS})
    for ch in CHANNELS:
        history.setdefault(ch, []).append(counts.get(ch, 0))

    # --- 5. Detect congestion & rebalance --------------------------------
    congested = get_congested_channels(G)
    movements = []

    if congested:
        step_alerts.append(f"Step {step}: High congestion on {', '.join(congested)}")

    if congested and model is not None and rebalance_fn is not None:
        movements = rebalance_fn(G, model)
        _rebuild_edges(G)
        counts = get_channel_counts(G)
        if movements:
            step_alerts.append(f"Step {step}: Rebalancing triggered - {len(movements)} node(s) moved")

    # Store alerts
    all_alerts = G.graph.setdefault("alerts", [])
    all_alerts.extend(step_alerts)
    # Keep only last 20 alerts
    if len(all_alerts) > 20:
        G.graph["alerts"] = all_alerts[-20:]

    return {
        "time_step": step,
        "added": added,
        "removed": removed,
        "counts": counts,
        "congested": congested,
        "movements": movements,
        "alerts": step_alerts,
    }


# -- quick test --------------------------------------------------------------
if __name__ == "__main__":
    print("=== Normal Network ===")
    G = create_network(seed=42)
    counts = get_channel_counts(G)
    for ch, cnt in counts.items():
        print(f"  {ch}: {cnt} devices  [{get_congestion_level(cnt)}]")
    print(f"  Average load: {get_average_load(G)}")

    print("\n=== Demo Network ===")
    G2 = create_demo_network()
    for ch, cnt in get_channel_counts(G2).items():
        print(f"  {ch}: {cnt} devices  [{get_congestion_level(cnt)}]")

    print("\n=== Simulate 3 steps ===")
    for _ in range(3):
        result = simulate_next_step(G)
        print(f"  Step {result['time_step']}: {result['counts']}")
