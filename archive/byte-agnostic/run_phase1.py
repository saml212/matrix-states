#!/usr/bin/env python3
"""Run Phase 1 experiment: proof of life on text data."""

import sys
import os

# Ensure we can import src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/phase1.yaml'
    model, history = train(config_path)

    # Run analysis after training
    print("\n" + "="*60)
    print("Training complete. Running analysis...")
    print("="*60 + "\n")

    from src.analyze import (
        plot_training_curves,
        visualize_boundaries,
        analyze_phm_algebra,
    )

    # Plot training curves
    plot_training_curves('results/phase1/history.json', 'results/phase1')

    # Visualize boundaries on sample text
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models process data in batches.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ]

    for i, text in enumerate(sample_texts):
        if len(text.encode('utf-8')) < model.max_len:
            visualize_boundaries(
                model, text,
                output_path=f'results/phase1/boundaries_sample_{i}.png',
                device=str(next(model.parameters()).device),
            )

    # Analyze PHM algebra
    algebra_results = analyze_phm_algebra(model, 'results/phase1')
    print("\nPHM Algebra Analysis:")
    for name, res in algebra_results.items():
        print(f"  {name}:")
        if res['quaternion_distance'] is not None:
            print(f"    Quaternion distance: {res['quaternion_distance']:.4f}")
        print(f"    Commutativity: {res['avg_commutativity']:.4f}")

    print("\nAll results saved to results/phase1/")
    print("Done!")
