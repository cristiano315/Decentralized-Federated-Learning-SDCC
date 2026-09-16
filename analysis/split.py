import os
import re
import pandas as pd

# Path to the exported CloudWatch CSV file and target output directory
EXPORTED_CSV_PATH = "query-results.csv"
OUTPUT_DIRECTORY = "./separated_nodes"

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
df = pd.read_csv(EXPORTED_CSV_PATH)

# Normalize column names (stripping whitespace and leading '@' typical of CloudWatch exports)
df.columns = [col.strip().lstrip('@') for col in df.columns]

# Ensure chronological ordering (from earliest to latest event)
df = df.sort_values(by='timestamp', ascending=True)

# Group entries by each individual container/logStream
grouped = df.groupby('logStream')
print(f"Found {len(grouped)} distinct log streams in the file.")

for stream_name, group in grouped:
    full_text = " ".join(group['message'].astype(str))

    # Match the explicit Node ID using updated and legacy signatures
    match = (
        re.search(r'Init:\s*Node\s*ID:\s*(\d+)', full_text, re.IGNORECASE) or
        re.search(r'Client\s*(\d+):', full_text) or
        re.search(r'Respawned node registration success:\s*Node\s*(\d+)', full_text, re.IGNORECASE) or
        re.search(r'Registration success:\s*Node\s*(\d+)', full_text, re.IGNORECASE) or
        re.search(r'Node\s+(\d+)\s+successfully\s+(?:registered|unregistered)', full_text, re.IGNORECASE) or
        re.search(r'\[Init\]\s*Nodo ID:\s*(\d+)', full_text, re.IGNORECASE)
    )

    # Fallback to the last two digits found in the stream name if no ID is matched in log text
    stream_digits = re.sub(r'\D', '', str(stream_name))
    fallback_id = stream_digits[-2:] if len(stream_digits) >= 2 else (stream_digits or "unknown")
    node_num = match.group(1) if match else fallback_id

    output_file_path = os.path.join(OUTPUT_DIRECTORY, f"node_{node_num}.csv")

    # Export timestamp and message columns
    group[['timestamp', 'message']].to_csv(output_file_path, index=False)
    print(f"Generated file for Node {node_num}: {output_file_path} ({len(group)} rows)")

print("\nLog splitting completed successfully.")