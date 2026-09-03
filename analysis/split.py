import os
import re
import pandas as pd

CSV_ESPORTATO = "query-results.csv"  # Il file scaricato da CloudWatch
CARTELLA_DESTINAZIONE = "./nodi_separati"

os.makedirs(CARTELLA_DESTINAZIONE, exist_ok=True)
df = pd.read_csv(CSV_ESPORTATO)

# Uniforma i nomi delle colonne
df.columns = [c.strip().lstrip('@') for c in df.columns]

# Ordina cronologicamente (dal primo all'ultimo evento)
df = df.sort_values(by='timestamp', ascending=True)

# Raggruppa per ogni singolo container/logStream
grouped = df.groupby('logStream')
print(f"Trovati {len(grouped)} stream nel file.")

for stream_name, group in grouped:
    full_text = " ".join(group['message'].astype(str))
    
    # Cerca l'ID effettivo del nodo
    match = re.search(r'\[Init\]\s*Nodo ID:\s*(\d+)', full_text) or \
            re.search(r'Node\s*(\d+)\s*successfully', full_text) or \
            re.search(r'Client\s*(\d+):', full_text)
            
    node_num = match.group(1) if match else re.sub(r'\D', '', stream_name)[-2:]
    out_path = os.path.join(CARTELLA_DESTINAZIONE, f"{node_num}.csv")
    
    group[['timestamp', 'message']].to_csv(out_path, index=False)
    print(f"Generato file per Nodo {node_num}: {out_path} ({len(group)} righe)")

print("\nSuddivisione completata.")