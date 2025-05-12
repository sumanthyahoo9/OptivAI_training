import json
import glob
import os

# Automatically load data when the module is imported
# Change this path to your JSON files directory
JSON_DIR = "generated_responses_12May25/"

# Initialize the lists
queries = []
responses = []
ground_truths = []

# Get all JSON files and sort them for consistent ordering
json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))

# Load all the data
for file_path in json_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
        queries.append(data["query"])
        responses.append(data["response"])
        ground_truths.append(data["ground_truth"])

# Optional: print loading status
print(f"Loaded {len(queries)} entries from {len(json_files)} JSON files")