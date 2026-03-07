# 📡 AI Spectrum Sharing Simulator

An intelligent wireless spectrum allocation system that uses **Machine Learning** to predict channel congestion and **dynamically rebalance** device assignments across communication channels.

## 🎯 Project Goal

| Component | Role |
|-----------|------|
| **Dataset** | Cognitive radio spectrum sensing data (SNR, signal power, frequency, spectral entropy) |
| **ML Model** | Random Forest classifier predicts whether a channel is **busy** or **free** |
| **Network Sim** | 20 wireless devices across 3 channels – congestion detected when devices > 6 |
| **Reallocation** | AI-driven `rebalance_network()` moves devices from congested → free channels |
| **Dashboard** | Streamlit UI with live channel stats, movement logs, and network graph |

## 🧠 How AI Predicts Congestion

1. The model is trained on real spectrum sensing data with features:
   - **SNR** (Signal-to-Noise Ratio in dB)
   - **Signal Power** (received signal strength)
   - **Frequency Band** (MHz)
   - **Spectral Entropy**
2. Labels: `busy` (1) or `free` (0) — derived from Primary User presence
3. A **RandomForestClassifier** learns to classify channel states
4. At runtime, each channel's signal profile is fed to the model to predict load status

## 🔄 How Nodes Are Reassigned

1. Congested channels (> 6 devices) are identified
2. The ML model confirms whether the channel is predicted as "busy"
3. Excess nodes are moved to the **least-crowded** alternative channel
4. Graph edges are updated to reflect new communication links

## 🚀 Running the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train_model.py
```
This prints the accuracy and saves the model to `models/trained_model.pkl`.

### 3. Start the dashboard
```bash
streamlit run dashboard/app.py
```
Open the displayed URL (usually `http://localhost:8501`) to see:
- 📊 Channel congestion bar chart (green / yellow / red)
- 🔄 Rebalance button to trigger AI-driven node migration
- 🌐 Network topology graph coloured by channel

## 📁 Project Structure

```
spectrum-ai-project/
├── data/
│   └── spectrum_dataset.csv       # Preprocessed spectrum data
├── models/
│   └── trained_model.pkl          # Saved Random Forest model
├── src/
│   ├── train_model.py             # ML training pipeline
│   ├── predict_channel.py         # Channel prediction module
│   ├── network_simulation.py      # NetworkX wireless simulation
│   └── channel_allocator.py       # AI-driven channel rebalancing
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── utils/
│   └── data_loader.py             # Dataset loading & preprocessing
├── requirements.txt
└── README.md
```

## 📊 Technologies

- **pandas / numpy** – Data processing
- **scikit-learn** – Random Forest classifier
- **networkx** – Wireless network graph simulation
- **streamlit** – Interactive dashboard
- **matplotlib** – Visualisations
