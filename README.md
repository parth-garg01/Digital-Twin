# Digital Twin

Digital Twin is a Python prototype for simulating a coupled smart-city system where traffic flow and energy demand influence each other. The project builds a mixed graph of traffic and energy nodes, uses a spatiotemporal graph neural network to forecast node behavior, applies cross-domain coupling from mobility demand to energy demand, and evaluates optimization outcomes through travel-time and energy-variance metrics.

## Project Structure

- `main.py` runs scenario comparisons and plots travel-time and energy results.
- `graph/graph_builder.py` constructs the traffic, energy, and cross-domain graph.
- `simulation/digital_twin.py` coordinates data loading, model training, prediction, coupling, optimization, and metric collection.
- `data/real_loader.py` loads and normalizes the `METR-LA` traffic dataset stored in `data/metr_la.npz`.
- `models/` contains the STGCN model, an LSTM baseline module, and the STGCN training loop.
- `coupling/cross_domain.py` converts traffic demand into energy demand for connected energy nodes.
- `optimization/optimizer.py` adjusts traffic and energy states to reduce overload and imbalance.
- `evaluation/metrics.py` computes the summary metrics used in the experiments.
- `config/config.py` stores the main experiment settings.

## How It Works

1. A graph is created with traffic nodes, energy nodes, and cross-domain edges.
2. Real traffic observations are loaded from `data/metr_la.npz`.
3. An STGCN is trained to predict node-level behavior over the combined graph.
4. Predicted traffic is converted into energy demand through the coupling model.
5. The optimizer balances traffic and energy stress across the network.
6. The system records average travel time and energy-demand variance for each scenario.

## Scenarios

The entrypoint compares multiple experiment settings:

- `Traffic-only`
- `Energy-only`
- `Cross-domain`
- `Ablation`

It prints raw metrics, computes improvement versus the traffic-only baseline, and generates bar charts with Matplotlib.

## Requirements

The code uses Python with these main libraries:

- `numpy`
- `matplotlib`
- `networkx`
- `torch`
- `torch_geometric`

## Run

Create or activate a Python environment with the required packages installed, then run:

```bash
python main.py
```

## Notes

- The repository currently includes the `METR-LA` dataset file because the simulation depends on it directly.
- `.gitignore` excludes the local virtual environment and Python cache files so they are not committed.
