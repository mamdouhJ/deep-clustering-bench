import subprocess
from pathlib import Path

# Match your existing directory setup
ROOT_DIR = Path.cwd().parent
id_files = [
    ROOT_DIR / "small_ids.txt",
    ROOT_DIR / "normal_ids.txt",
    ROOT_DIR / "big_ids.txt"
]

def main():
    for file_path in id_files:
        if not file_path.exists():
            print(f"[WARNING] File not found: {file_path}")
            continue
            
        with open(file_path, "r") as f:
            # Read all non-empty lines as IDs
            dids = [line.strip() for line in f if line.strip()]
            
        print(f"\n========== Processing {len(dids)} datasets from {file_path.name} ==========\n")
        
        for did in dids:
            print(f"\n--> Triggering KMeans for DID: {did}")
            # Call the KMeans script we created previously
            subprocess.run(["python", "run_kmeans.py", "--did", did])

    print("\n[DONE] Finished iterating over all ID files.")

if __name__ == "__main__":
    main()