"""
Run this once to train models for all states and save the best one per state.
Estimated time: 15-40 minutes depending on hardware (SARIMA is the slow part).

Usage:
    python train.py                  # train all states, skip already-trained ones
    python train.py --retrain        # force retrain everything from scratch
    python train.py --states "California" "Texas"   # specific states only
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import prepare_data, get_all_states, get_series_for_state
from src.model_selector import train_and_select, MODELS_DIR, REGISTRY_PATH


def get_trained_states():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return set(json.load(f).keys())
    return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", nargs="+", default=None)
    parser.add_argument("--retrain", action="store_true", help="Retrain even if model exists")
    args = parser.parse_args()

    print("Loading and preprocessing data...")
    df = prepare_data()
    all_states = get_all_states(df)

    if args.states:
        states = args.states
    else:
        states = all_states

    if not args.retrain:
        already_done = get_trained_states()
        states = [s for s in states if s not in already_done]
        if not states:
            print("All states already trained. Use --retrain to force re-training.")
            return
        print(f"Skipping {len(already_done)} already-trained state(s).")

    print(f"Training models for {len(states)} state(s).\n")

    t0 = time.time()
    failed = []

    for i, state in enumerate(states, 1):
        print(f"[{i}/{len(states)}] {state}")
        try:
            series = get_series_for_state(df, state)
            train_and_select(state, series, verbose=True)
        except Exception as e:
            print(f"  ERROR on {state}: {e}\n")
            failed.append(state)

    elapsed = time.time() - t0
    trained = len(states) - len(failed)
    print(f"\nDone. {trained}/{len(states)} states trained in {elapsed/60:.1f} minutes.")
    if failed:
        print(f"Failed states: {failed}")
    print("Start the API with: uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
