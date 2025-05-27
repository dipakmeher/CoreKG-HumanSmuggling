import os
import argparse
import pandas as pd

def convert_parquet_to_csv(parquet_path):
    if not os.path.exists(parquet_path):
        print(f"Error: File not found at {parquet_path}")
        return

    # Load the Parquet file
    df = pd.read_parquet(parquet_path)

    # Create output path with same base name
    output_dir = os.path.dirname(parquet_path)
    base_name = os.path.splitext(os.path.basename(parquet_path))[0]
    csv_path = os.path.join(output_dir, f"{base_name}.csv")

    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Converted to CSV: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .parquet file to .csv")
    parser.add_argument("--parquet-file", required=True, help="Path to the .parquet file to convert")
    args = parser.parse_args()

    convert_parquet_to_csv(args.parquet_file)
