from graph.graph_builder import SmartCityGraph
from simulation.digital_twin import DigitalTwin
from config.config import *
import matplotlib.pyplot as plt
import numpy as np

def run_scenario(alpha):
    graph = SmartCityGraph(NUM_TRAFFIC_NODES, NUM_ENERGY_NODES)
    twin = DigitalTwin(graph, alpha, W1, W2)
    
    history = twin.run(TIME_STEPS)
    
    travel = np.mean([h[0] for h in history])
    energy = np.mean([h[1] for h in history])
    
    return travel, energy
def plot_results(results):
    labels = list(results.keys())
    travel = [v[0] for v in results.values()]
    energy = [v[1] for v in results.values()]
    
    x = range(len(labels))
    
    plt.figure()
    plt.bar(x, travel)
    plt.xticks(x, labels)
    plt.title("Travel Time Comparison")
    plt.ylabel("Travel Time")
    plt.show()
    
    plt.figure()
    plt.bar(x, energy)
    plt.xticks(x, labels)
    plt.title("Energy Variance Comparison")
    plt.ylabel("Variance")
    plt.show()

if __name__ == "__main__":    
    results = {
        "Traffic-only": run_scenario(alpha=0),
        "Energy-only": run_scenario(alpha=0),
        "Cross-domain": run_scenario(alpha=ALPHA),
        "Ablation": run_scenario(alpha=0)
    }
    
    print("\n===== RESULTS =====")
    for k, v in results.items():
        print(f"{k}: Travel Time={v[0]:.2f}, Energy Var={v[1]:.2f}")
    
    # ✅ ADD THIS (IMPROVEMENT TABLE)
    baseline = results["Traffic-only"]

    print("\n===== % IMPROVEMENT OVER TRAFFIC-ONLY =====")

    for k, v in results.items():
        travel_improve = ((baseline[0] - v[0]) / baseline[0]) * 100
        energy_diff = v[1] - baseline[1]

        print(f"{k}: Travel Δ={travel_improve:.2f}%, Energy Δ={energy_diff:.3f}")
    
    # ✅ ADD THIS (PLOTS)
    plot_results(results)