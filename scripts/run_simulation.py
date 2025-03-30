"""Example script to run engine simulation."""

import sys
import os
from pathlib import Path

# Add parent directory to path and set project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.simulation.engine import EngineConfig, EngineSimulation

def main():
    """Run engine simulation example."""
    # Create results directory
    results_dir = project_root / 'data' / 'output'
    results_dir.mkdir(exist_ok=True)
    
    # Load configuration using absolute path
    config = EngineConfig.from_yaml(str(project_root / 'src' / 'config' / 'default_config.yaml'))
    
    # Update mechanism path to absolute path
    config.mechanism = str(project_root / 'data' / 'mechanisms' / 'Nissan_chem.yaml')
    
    # Create and run simulation
    sim = EngineSimulation(config)
    results = sim.run()
    
    # Plot results
    print("\nSimulation completed. Creating plots...")
    results.plot_interactive(str(results_dir / 'interactive_plots.png'))
    results.plot_minor_species(str(results_dir / 'minor_species.png'))
    
    # Calculate and display performance metrics
    perf = results.calculate_performance()
    print("\nEngine Performance Metrics:")
    print(f"Indicated Work: {perf['indicated_work']:.2f} J")
    print(f"IMEP: {perf['imep']:.2f} bar")
    print(f"Peak Pressure: {perf['peak_pressure']:.2f} bar")
    print(f"Peak Temperature: {perf['peak_temperature']:.2f} K")
    
    # Print summary of simulation settings
    print("\nSimulation Settings:")
    print(f"Fuel: {config.fuel}")
    print(f"Equivalence Ratio: {config.phi:.2f}")
    print(f"EGR Fraction: {config.egr:.2f}")
    print(f"Initial Temperature: {config.temperature:.0f} K")
    print(f"Initial Pressure: {config.pressure/1e5:.1f} bar")
    print(f"Compression Ratio: {config.comp_ratio:.1f}")
    print(f"Engine Speed: {config.speed:.0f} rpm")
    
if __name__ == '__main__':
    main() 