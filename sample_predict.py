"""Standalone CLI inference script for concrete compressive strength prediction."""

import argparse
import numpy as np
import pandas as pd
from src.data_loader import load_concrete_data, split_features_target
from src.pipeline import build_ensemble_pipeline


def main():
    parser = argparse.ArgumentParser(description="Predict concrete compressive strength (MPa).")
    parser.add_argument("--cement", type=float, default=280.0, help="Cement (kg/m3)")
    parser.add_argument("--slag", type=float, default=100.0, help="Blast furnace slag (kg/m3)")
    parser.add_argument("--fly_ash", type=float, default=0.0, help="Fly ash (kg/m3)")
    parser.add_argument("--water", type=float, default=180.0, help="Water (kg/m3)")
    parser.add_argument("--sp", type=float, default=5.0, help="Superplasticizer (kg/m3)")
    parser.add_argument("--coarse", type=float, default=950.0, help="Coarse aggregate (kg/m3)")
    parser.add_argument("--fine", type=float, default=750.0, help="Fine aggregate (kg/m3)")
    parser.add_argument("--age", type=float, default=28.0, help="Curing age in days")
    args = parser.parse_args()

    # Fast fit on training data
    df = load_concrete_data("data/raw/Concrete_Data.csv")
    X, y = split_features_target(df)
    pipe = build_ensemble_pipeline()
    pipe.fit(X.values, y.values)

    sample = np.array([[args.cement, args.slag, args.fly_ash, args.water, args.sp, args.coarse, args.fine, args.age]])
    pred = pipe.predict(sample)[0]

    print("=" * 55)
    print(" 🏗️ CONCRETE COMPRESSIVE STRENGTH PREDICTION")
    print("=" * 55)
    print(f" Cement:           {args.cement:.1f} kg/m³")
    print(f" Blast Slag:       {args.slag:.1f} kg/m³")
    print(f" Fly Ash:          {args.fly_ash:.1f} kg/m³")
    print(f" Water:            {args.water:.1f} kg/m³")
    print(f" Superplasticizer: {args.sp:.1f} kg/m³")
    print(f" Coarse Agg.:      {args.coarse:.1f} kg/m³")
    print(f" Fine Agg.:        {args.fine:.1f} kg/m³")
    print(f" Age:              {args.age:.0f} days")
    print("-" * 55)
    print(f" PREDICTED STRENGTH: {pred:.2f} MPa")
    print("=" * 55)


if __name__ == "__main__":
    main()
