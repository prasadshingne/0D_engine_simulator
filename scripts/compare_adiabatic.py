"""Compare adiabatic and non-adiabatic engine simulations."""

import os
import sys
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation.engine import EngineConfig, EngineSimulation
from src.simulation.solver import SolverParams

# Load base configuration
config = EngineConfig.from_yaml(os.path.join(project_root, 'src', 'config', 'default_config.yaml'))

# Run non-adiabatic simulation
print("\nRunning non-adiabatic simulation...")
sim_non_adiabatic = EngineSimulation(config)
sim_non_adiabatic.solver.params.adiabatic = False
results_non_adiabatic = sim_non_adiabatic.run()

# Run adiabatic simulation
print("\nRunning adiabatic simulation...")
sim_adiabatic = EngineSimulation(config)
sim_adiabatic.solver.params.adiabatic = True
results_adiabatic = sim_adiabatic.run()

# Create comparison plots
plt.figure(figsize=(12, 5))

# Temperature comparison
plt.subplot(1, 2, 1)
plt.plot(results_non_adiabatic.crank_angle, results_non_adiabatic.temperature, 'b-', label='Non-adiabatic')
plt.plot(results_adiabatic.crank_angle, results_adiabatic.temperature, 'r--', label='Adiabatic')
plt.xlabel('Crank Angle [deg]')
plt.ylabel('Temperature [K]')
plt.title('Temperature Comparison')
plt.grid(True)
plt.legend()

# Pressure comparison
plt.subplot(1, 2, 2)
plt.plot(results_non_adiabatic.crank_angle, results_non_adiabatic.pressure/1e5, 'b-', label='Non-adiabatic')
plt.plot(results_adiabatic.crank_angle, results_adiabatic.pressure/1e5, 'r--', label='Adiabatic')
plt.xlabel('Crank Angle [deg]')
plt.ylabel('Pressure [bar]')
plt.title('Pressure Comparison')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(project_root, 'data', 'output', 'adiabatic_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nComparison plot saved as 'data/output/adiabatic_comparison.png'")

# Print some key metrics
def print_metrics(results, label):
    metrics = results.calculate_performance()
    print(f"\n{label} Results:")
    print(f"Peak Temperature: {max(results.temperature):.1f} K")
    print(f"Peak Pressure: {max(results.pressure)/1e5:.1f} bar")
    print(f"IMEP: {metrics['imep']:.2f} bar")

print_metrics(results_non_adiabatic, "Non-adiabatic")
print_metrics(results_adiabatic, "Adiabatic") 