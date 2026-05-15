import json
import pandas as pd
from pathlib import Path

def extract_raw_kmeans_data(results_dir="./"):
    results_path = Path(results_dir)
    all_data = []

    if not results_path.exists():
        print(f"Directory not found: {results_path}")
        return pd.DataFrame()

    # Find any variation of 'KMeans' folder and 'results_baseline.jsonl'
    # Use rglob to be recursive and flexible with case
    files = list(results_path.glob("*/KMeans/results_baseline.jsonl"))
    
    if not files:
        # Fallback if the file is singular 'result_baseline.jsonl'
        files = list(results_path.glob("*/KMeans/result_baseline.jsonl"))

    print(f"Found {len(files)} JSONL files. Processing all rows...")

    for result_file in files:
        try:
            with open(result_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    row_dict = json.loads(line)
                    
                    # Flatten the 'metrics' dictionary into the main row
                    if 'metrics' in row_dict and isinstance(row_dict['metrics'], dict):
                        metrics = row_dict.pop('metrics')
                        for k, v in metrics.items():
                            row_dict[f"metrics.{k}"] = v
                    
                    all_data.append(row_dict)
        except Exception as e:
            print(f"Error in {result_file}: {e}")

    if not all_data:
        print("No rows found. Check file permissions or paths.")
        return pd.DataFrame()

    return pd.DataFrame(all_data)

# --- EXECUTION ---
df_kmeans = extract_raw_kmeans_data()

if not df_kmeans.empty:
    # Save to Parquet
    output_name = 'kmeans_baselines_full.parquet'
    df_kmeans.to_parquet(output_name, index=False)
    
    print(f"✅ Success! Extracted {len(df_kmeans)} total rows.")
    print(f"Columns: {list(df_kmeans.columns)}")
    print(f"Saved to: {output_name}")
else:
    print("❌ Failed: No data extracted.")

if __name__ == "__main__":
    df_kmeans = extract_raw_kmeans_data()
