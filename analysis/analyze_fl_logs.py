import os
import re
import sys
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione stile grafico
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

def parse_single_file(filepath):
    """
    Estrae le informazioni grezze e le metriche da un singolo file di log.
    Supporta timestamp Unix ms o ISO, file dritti o rovesciati.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
        raw_lines = [l for l in fp.readlines() if l.strip()]
        
    if not raw_lines:
        return None, None, ["File completamente vuoto"]

    header_offset = 1 if ('timestamp' in raw_lines[0].lower() and 'message' in raw_lines[0].lower()) else 0
    lines_content = raw_lines[header_offset:]
    if not lines_content:
        return None, None, ["File senza righe di log oltre l'intestazione"]

    def extract_ts(line):
        parts = line.split(',', 1)
        return parts[0].strip().strip('"').strip("'") if len(parts) > 0 else ""

    ts_first = extract_ts(lines_content[0])
    ts_last = extract_ts(lines_content[-1])

    # Se il timestamp del primo record è maggiore dell'ultimo, il file è a ritroso: inverti
    if ts_first and ts_last and ts_first > ts_last:
        lines = list(reversed(lines_content))
    else:
        lines = lines_content

    full_text = "".join(lines)

    # 1. Rileva ID esplicito nel log
    explicit_matches = (
        re.findall(r'\[Init\]\s*Nodo ID:\s*(\d+)', full_text) or
        re.findall(r'Registration success:\s*Node\s*(\d+)', full_text) or
        re.findall(r'Unregistration success:\s*Node\s*(\d+)', full_text) or
        re.findall(r'Node\s+(\d+)\s+successfully\s+(?:registered|unregistered)', full_text) or
        re.findall(r'Client\s+(\d+)\s*:', full_text)
    )
    explicit_id = int(explicit_matches[0]) if explicit_matches else None

    # 2. Peer con cui comunica (utile per dedurre l'ID mancante)
    sent_peers = set([int(x) for x in re.findall(r'peer with id\s+(\d+)', full_text)])
    received_peers = set([int(x) for x in re.findall(r'From\s+(\d+)', full_text)])
    all_peers = sent_peers.union(received_peers)

    # 3. Fallback nome file
    base = os.path.basename(filepath)
    cleaned_base = re.sub(r'log-events-viewer-result', '', base, flags=re.IGNORECASE)
    fn_match = re.search(r'(\d+)', cleaned_base)
    fn_id = int(fn_match.group(1)) if fn_match else None

    metrics = {
        'filepath': filepath,
        'filename': base,
        'explicit_id': explicit_id,
        'all_peers': all_peers,
        'fn_id': fn_id,
        'has_init_log': bool(re.search(r'\[Init\]|Registration success', full_text)),
        'has_final_eval': bool(re.search(r'\[RISULTATI GLOBALI\]|Valutazione finale', full_text)),
        'Loss': None,
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1-Score': None,
        'Cohens_Kappa': None,
        'ROC_AUC': None,
        'TN': None,
        'FP': None,
        'FN': None,
        'TP': None,
        'Avg_Calc_Time_s': None,
        'Avg_Net_Time_s': None
    }

    epochs_data = []
    current_round = 1

    for idx, line in enumerate(lines):
        m_round_hdr = re.search(r'ROUND\s*(\d+)/\d+', line, re.IGNORECASE)
        if m_round_hdr:
            current_round = int(m_round_hdr.group(1))

        if 'Loss:' in line and 'Acc:' in line and 'Epoch' not in line and '[Train]' not in line:
            m_loss_acc = re.search(r'Loss:\s*([\d\.]+)\s*\|\s*Acc:\s*([\d\.]+)', line)
            if m_loss_acc:
                metrics['Loss'] = float(m_loss_acc.group(1))
                metrics['Accuracy'] = float(m_loss_acc.group(2))

        m_prec_rec = re.search(r'Precision:\s*([\d\.]+)\s*\|\s*Recall:\s*([\d\.]+)', line)
        if m_prec_rec:
            metrics['Precision'] = float(m_prec_rec.group(1))
            metrics['Recall'] = float(m_prec_rec.group(2))

        m_ck_auc = re.search(r"Cohen's Kappa:\s*([\d\.]+)\s*\|\s*ROC-AUC:\s*([\d\.]+)", line)
        if m_ck_auc:
            metrics['Cohens_Kappa'] = float(m_ck_auc.group(1))
            metrics['ROC_AUC'] = float(m_ck_auc.group(2))

        m_neg = re.search(r'Actual Neg.*?(\d+)\s+(\d+)\s+\(TN,\s*FP\)', line)
        if m_neg:
            metrics['TN'] = int(m_neg.group(1))
            metrics['FP'] = int(m_neg.group(2))

        m_pos = re.search(r'Actual Pos.*?(\d+)\s+(\d+)\s+\(FN,\s*TP\)', line)
        if m_pos:
            metrics['FN'] = int(m_pos.group(1))
            metrics['TP'] = int(m_pos.group(2))

        m_t_calc = re.search(r'MEDIA Tempo di Calcolo per round:\s*([\d\.]+)s', line)
        if m_t_calc:
            metrics['Avg_Calc_Time_s'] = float(m_t_calc.group(1))
        m_t_net = re.search(r'MEDIA Tempo di Rete/Attesa per round:\s*([\d\.]+)s', line)
        if m_t_net:
            metrics['Avg_Net_Time_s'] = float(m_t_net.group(1))

        m_ep = re.search(r'Epoch\s*(\d+)\s*\|\s*Train Loss:\s*([\d\.]+)\s*\|\s*Val Loss:\s*([\d\.]+)\s*\|\s*Val Acc:\s*([\d\.]+)', line)
        if m_ep:
            ep_num = int(m_ep.group(1))
            tr_loss = float(m_ep.group(2))
            vl_loss = float(m_ep.group(3))
            vl_acc = float(m_ep.group(4))
            epochs_data.append({
                'filepath': filepath,
                'Round': current_round,
                'Local_Epoch': ep_num,
                'Global_Epoch': len(epochs_data) + 1,
                'Train_Loss': tr_loss,
                'Val_Loss': vl_loss,
                'Val_Acc': vl_acc
            })

    if metrics['Precision'] is not None and metrics['Recall'] is not None:
        p, r = metrics['Precision'], metrics['Recall']
        metrics['F1-Score'] = round(2 * p * r / (p + r) if (p + r) > 0 else 0.0, 4)

    return metrics, pd.DataFrame(epochs_data), []

def load_and_resolve_all_nodes(csv_files):
    parsed_items = []
    file_errors = {}

    for f in csv_files:
        m, p, errs = parse_single_file(f)
        if errs:
            file_errors[os.path.basename(f)] = errs
            continue
        if m and m['Loss'] is not None:
            parsed_items.append((m, p))
        else:
            file_errors[os.path.basename(f)] = ["Nessuna metrica di valutazione finale rilevata (job interrotto o log vuoto)"]

    if not parsed_items:
        return pd.DataFrame(), pd.DataFrame(), file_errors

    assigned = {}
    for m, _ in parsed_items:
        if m['explicit_id'] is not None:
            assigned[m['filepath']] = m['explicit_id']

    cluster_peers = set(assigned.values())
    for m, _ in parsed_items:
        cluster_peers.update(m['all_peers'])

    for m, _ in parsed_items:
        fp = m['filepath']
        if fp not in assigned:
            if m['all_peers'] and cluster_peers:
                diff = cluster_peers - m['all_peers']
                candidates = [x for x in diff if x not in assigned.values()]
                if len(candidates) == 1:
                    assigned[fp] = candidates[0]
            if fp not in assigned and m['fn_id'] is not None and m['fn_id'] not in assigned.values():
                assigned[fp] = m['fn_id']

    remaining = sorted(list(cluster_peers - set(assigned.values())))
    for m, _ in parsed_items:
        fp = m['filepath']
        if fp not in assigned:
            if remaining:
                assigned[fp] = remaining.pop(0)
            elif m['fn_id'] is not None:
                assigned[fp] = m['fn_id']
            else:
                assigned[fp] = 999999

    used = set()
    for m, _ in parsed_items:
        fp = m['filepath']
        val = assigned[fp]
        if val in used:
            val = max(used) + 1 if used else 1
            assigned[fp] = val
        used.add(val)
        m['Raw_Node_Num'] = val

    parsed_items.sort(key=lambda x: x[0]['Raw_Node_Num'])
    
    metrics_list = []
    prog_list = []

    for seq_idx, (m, df_p) in enumerate(parsed_items, start=1):
        m['Node_Num'] = seq_idx
        m['Node'] = f"Nodo {seq_idx}"
        m['Orig_Node'] = f"Nodo {m['Raw_Node_Num']}"
        metrics_list.append(m)

        if not df_p.empty:
            df_p['Node_Num'] = seq_idx
            df_p['Node'] = f"Nodo {seq_idx}"
            prog_list.append(df_p)

    df_metrics = pd.DataFrame(metrics_list)
    df_prog = pd.concat(prog_list, ignore_index=True) if prog_list else pd.DataFrame()

    return df_metrics, df_prog, file_errors

def detect_experiment_metadata(csv_files, df_prog=None):
    is_centralized = False
    is_decentralized = False
    total_rounds = None
    local_epochs = None

    for f in csv_files:
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
                c = fp.read()
            if re.search(r'coordinat', c, re.IGNORECASE) and not re.search(r'gossip', c, re.IGNORECASE):
                is_centralized = True
            if re.search(r'gossip|fanout|dai peer', c, re.IGNORECASE):
                is_decentralized = True
            if total_rounds is None:
                m_rounds = re.findall(r'ROUND\s*\d+/(\d+)', c, re.IGNORECASE)
                if m_rounds:
                    total_rounds = int(m_rounds[-1])
            if local_epochs is None:
                m_eps = re.findall(r'Epoch\s*(\d+)\s*\|', c)
                if m_eps:
                    local_epochs = max([int(x) for x in m_eps])
        except Exception:
            continue

    if is_centralized and not is_decentralized:
        arch_label = "Centralizzato"
    else:
        arch_label = "Decentralizzato (Gossip-based)"

    if df_prog is not None and not df_prog.empty:
        if local_epochs is None and 'Local_Epoch' in df_prog.columns:
            local_epochs = int(df_prog['Local_Epoch'].max())
        if total_rounds is None and 'Round' in df_prog.columns:
            total_rounds = int(df_prog['Round'].nunique())

    rounds_str = f"{total_rounds}" if total_rounds else "N/D"
    epochs_str = f"{local_epochs}" if local_epochs else "N/D"

    return arch_label, rounds_str, epochs_str

def check_missing_data_per_node(df_metrics, df_prog, file_errors, total_rounds, local_epochs):
    """
    Rileva quali file hanno dati mancanti, specificando epoche perse o avvii troncati.
    """
    integrity_report = {}
    
    # Errori di file non leggibili
    for fn, errs in file_errors.items():
        integrity_report[fn] = errs

    expected_total_epochs = None
    if total_rounds != "N/D" and local_epochs != "N/D":
        expected_total_epochs = int(total_rounds) * int(local_epochs)
    elif not df_prog.empty:
        expected_total_epochs = int(df_prog.groupby('Node')['Global_Epoch'].max().max())

    for _, row in df_metrics.iterrows():
        fn = row['filename']
        node = row['Node']
        issues = []

        if not row.get('has_init_log', True):
            issues.append("File troncato all'inizio (mancano i log di boot/registrazione container)")

        if not row.get('has_final_eval', True):
            issues.append("Mancano le metriche di test globali finali")

        # Controllo epoche
        node_prog = df_prog[df_prog['Node'] == node] if not df_prog.empty else pd.DataFrame()
        epochs_found = len(node_prog)

        if expected_total_epochs and epochs_found < expected_total_epochs:
            missing_qty = expected_total_epochs - epochs_found
            issues.append(f"Epoche parziali: trovate {epochs_found}/{expected_total_epochs} (mancano {missing_qty} epoche)")

        if issues:
            integrity_report[f"{fn} [{node}]"] = issues

    return integrity_report

def sort_legend_handles(ax):
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    by_label = dict(zip(labels, handles))
    
    def legend_sort_key(label):
        m = re.search(r'Nodo\s*(\d+)', label)
        if m:
            return (0, int(m.group(1)))
        return (1, label)
        
    sorted_labels = sorted(by_label.keys(), key=legend_sort_key)
    sorted_handles = [by_label[k] for k in sorted_labels]
    ax.legend(sorted_handles, sorted_labels, loc='best', framealpha=0.9)

def generate_report_files(df_metrics, df_prog, output_dir, arch_label, rounds_str, epochs_str, integrity_report):
    num_cols = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Cohens_Kappa', 'ROC_AUC', 'Avg_Calc_Time_s', 'Avg_Net_Time_s']
    df_sorted = df_metrics.sort_values(by='Node_Num').reset_index(drop=True)
    
    mean_series = df_sorted[num_cols].mean().round(4)
    std_series = df_sorted[num_cols].std().round(4)
    median_series = df_sorted[num_cols].median().round(4)
    
    df_summary = df_sorted.drop(columns=['Node_Num', 'Orig_Node', 'Raw_Node_Num', 'filepath', 'explicit_id', 'all_peers', 'fn_id', 'has_init_log', 'has_final_eval'], errors='ignore').copy()
    row_mean = {'Node': 'MEDIA FEDERATA'}
    row_std = {'Node': 'DEV. STANDARD'}
    row_med = {'Node': 'MEDIANA'}
    for c in num_cols:
        row_mean[c] = mean_series[c]
        row_std[c] = std_series[c]
        row_med[c] = median_series[c]
    
    df_summary = pd.concat([df_summary, pd.DataFrame([row_mean, row_std, row_med])], ignore_index=True)
    
    csv_path = os.path.join(output_dir, 'fl_report_metriche.csv')
    df_summary.to_csv(csv_path, index=False)
    
    txt_path = os.path.join(output_dir, 'fl_report_finale.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 85 + "\n")
        f.write(f"     REPORT FINALE FEDERATED LEARNING — ARCHITETTURA: {arch_label.upper()}\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"Configurazione: {len(df_sorted)} Nodi | {rounds_str} Round | {epochs_str} Epoche Locali per Round\n")
        f.write(f"Nodi rinumerati sequenzialmente da Nodo 1 a Nodo {len(df_sorted)}\n\n")

        # SEZIONE INTEGRITÀ DATI E FILE MANCANTI
        f.write("=" * 85 + "\n")
        f.write("VERIFICA INTEGRITÀ FILE E DATI MANCANTI:\n")
        f.write("-" * 85 + "\n")
        if not integrity_report:
            f.write("OTTIMO: Tutti i file sono completi al 100%. Nessun dato o epoca mancante!\n")
        else:
            f.write(f"ATTENZIONE: Trovate anomalie o dati incompleti in {len(integrity_report)} file:\n\n")
            for target, issues in integrity_report.items():
                f.write(f"• {target}:\n")
                for iss in issues:
                    f.write(f"   └── {iss}\n")
        f.write("=" * 85 + "\n\n")
        
        f.write("STATISTICHE AGGREGATE DELL'INTERA RETE:\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'Metrica':<22} | {'Media':<10} | {'Dev.Std':<10} | {'Mediana':<10} | {'Min':<10} | {'Max':<10}\n")
        f.write("-" * 85 + "\n")
        for col in num_cols:
            c_min = df_sorted[col].min()
            c_max = df_sorted[col].max()
            f.write(f"{col:<22} | {mean_series[col]:<10.4f} | {std_series[col]:<10.4f} | {median_series[col]:<10.4f} | {c_min:<10.4f} | {c_max:<10.4f}\n")
            
        f.write("\n" + "=" * 85 + "\n")
        f.write("TABELLA METRICHE PER SINGOLO NODO:\n")
        f.write("-" * 85 + "\n")
        cols_to_print = ['Node', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC_AUC', 'Avg_Calc_Time_s', 'Avg_Net_Time_s']
        f.write(df_sorted[cols_to_print].to_string(index=False))
        f.write("\n\n" + "=" * 85 + "\n")
        
        tot_tn = df_sorted['TN'].dropna().sum()
        tot_fp = df_sorted['FP'].dropna().sum()
        tot_fn = df_sorted['FN'].dropna().sum()
        tot_tp = df_sorted['TP'].dropna().sum()
        f.write("MATRICE DI CONFUSIONE CUMULATA (SOMMA SU TUTTI I NODI):\n")
        f.write(f"                   Pred Neg (0)      Pred Pos (1)\n")
        f.write(f"Actual Neg (0):    {int(tot_tn):<16}  {int(tot_fp):<16}\n")
        f.write(f"Actual Pos (1):    {int(tot_fn):<16}  {int(tot_tp):<16}\n")
        f.write("=" * 85 + "\n")
        
    return df_summary, csv_path, txt_path

def generate_visualizations(df_metrics, df_prog, output_dir, arch_label, rounds_str, epochs_str):
    n_nodes = len(df_metrics)
    is_large_scale = n_nodes > 8
    node_order = [f"Nodo {i}" for i in range(1, n_nodes + 1)]
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22)
    
    # 1. Loss
    ax1 = fig.add_subplot(gs[0, 0])
    if is_large_scale:
        sns.lineplot(data=df_prog, x='Global_Epoch', y='Train_Loss', units='Node', estimator=None, 
                     color='#3498db', alpha=0.18, lw=1, ax=ax1)
        sns.lineplot(data=df_prog, x='Global_Epoch', y='Val_Loss', units='Node', estimator=None, 
                     color='#e67e22', alpha=0.18, lw=1, linestyle='--', ax=ax1)
        sns.lineplot(data=df_prog, x='Global_Epoch', y='Train_Loss', color='#1f77b4', lw=3, 
                     errorbar='sd', ax=ax1, label='Media Train Loss (±1σ)')
        sns.lineplot(data=df_prog, x='Global_Epoch', y='Val_Loss', color='#d35400', lw=3, 
                     linestyle='--', errorbar='sd', ax=ax1, label='Media Val Loss (±1σ)')
        ax1.legend(loc='upper right', framealpha=0.9)
    else:
        sns.lineplot(data=df_prog, x='Global_Epoch', y='Train_Loss', hue='Node', hue_order=node_order, marker='o', ax=ax1, palette='tab10')
        sns.lineplot(data=df_prog, x='Global_Epoch', y='Val_Loss', hue='Node', hue_order=node_order, linestyle='--', marker='s', ax=ax1, palette='tab10', legend=False)
        mean_p = df_prog.groupby('Global_Epoch')[['Train_Loss', 'Val_Loss']].mean().reset_index()
        ax1.plot(mean_p['Global_Epoch'], mean_p['Train_Loss'], color='black', lw=2.5, label='Media Train')
        ax1.plot(mean_p['Global_Epoch'], mean_p['Val_Loss'], color='black', lw=2.5, linestyle='--', label='Media Val')
        sort_legend_handles(ax1)
        
    ax1.set_title(f"Andamento della Loss ({n_nodes} Nodi)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoca Globale")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    if is_large_scale:
        sns.lineplot(data=df_prog, x='Global_Epoch', y='Val_Acc', units='Node', estimator=None, 
                     color='#2ecc71', alpha=0.2, lw=1, ax=ax2)
        sns.lineplot(data=df_prog, x='Global_Epoch', y='Val_Acc', color='#27ae60', lw=3, 
                     errorbar='sd', ax=ax2, label='Media Val Accuracy (±1σ)')
        ax2.legend(loc='lower right', framealpha=0.9)
    else:
        sns.lineplot(data=df_prog, x='Global_Epoch', y='Val_Acc', hue='Node', hue_order=node_order, marker='^', ax=ax2, palette='tab10')
        mean_p = df_prog.groupby('Global_Epoch')['Val_Acc'].mean().reset_index()
        ax2.plot(mean_p['Global_Epoch'], mean_p['Val_Acc'], color='black', lw=2.5, label='Media Accuracy')
        sort_legend_handles(ax2)
        
    ax2.set_title(f"Progressione Validation Accuracy ({n_nodes} Nodi)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoca Globale")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    
    # 3. Metriche Finali
    ax3 = fig.add_subplot(gs[1, 0])
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC_AUC']
    
    if is_large_scale:
        melted = df_metrics.melt(id_vars=['Node', 'Node_Num'], value_vars=metrics_to_plot, var_name='Metrica', value_name='Punteggio')
        sns.boxplot(data=melted, x='Metrica', y='Punteggio', ax=ax3, palette='Set2', 
                    showmeans=True, meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red", "markersize": 7},
                    boxprops=dict(alpha=0.75))
        sns.stripplot(data=melted, x='Metrica', y='Punteggio', ax=ax3, color='black', alpha=0.45, jitter=0.2, size=5)
        ax3.set_title(f"Distribuzione Metriche ({n_nodes} Nodi) — Punto rosso: Media", fontsize=13, fontweight='bold')
    else:
        df_plot_m = df_metrics.set_index('Node')[metrics_to_plot].copy()
        df_plot_m.loc['Media'] = df_plot_m.mean()
        melted = df_plot_m.reset_index().melt(id_vars='Node', var_name='Metrica', value_name='Punteggio')
        hue_order_bar = node_order + ['Media']
        sns.barplot(data=melted, x='Metrica', y='Punteggio', hue='Node', hue_order=hue_order_bar, ax=ax3, palette='Set2')
        sort_legend_handles(ax3)
        ax3.set_title("Confronto Metriche Finali per Nodo e Media", fontsize=13, fontweight='bold')
        
    ax3.set_ylim(0.4, 1.0)
    ax3.set_ylabel("Score (0 - 1)")
    ax3.grid(True, alpha=0.3)
    
    # 4. Tempi
    ax4 = fig.add_subplot(gs[1, 1])
    time_df = df_metrics[['Node', 'Node_Num', 'Avg_Calc_Time_s', 'Avg_Net_Time_s']].dropna().sort_values('Node_Num')
    time_melted = time_df.melt(id_vars=['Node', 'Node_Num'], value_vars=['Avg_Calc_Time_s', 'Avg_Net_Time_s'], 
                               var_name='Tipo_Tempo', value_name='Secondi')
    time_melted['Tipo_Tempo'] = time_melted['Tipo_Tempo'].replace({
        'Avg_Calc_Time_s': 'Calcolo Locale',
        'Avg_Net_Time_s': 'Rete / Attesa Peer'
    })
    
    if n_nodes > 12:
        sns.boxplot(data=time_melted, x='Tipo_Tempo', y='Secondi', ax=ax4, palette=['#2b5c8f', '#d95f02'],
                    showmeans=True, meanprops={"marker":"D", "markerfacecolor":"yellow", "markeredgecolor":"black", "markersize": 8})
        sns.stripplot(data=time_melted, x='Tipo_Tempo', y='Secondi', ax=ax4, color='black', alpha=0.5, jitter=0.15, size=6)
        ax4.set_title(f"Distribuzione Tempi Medi per Round ({n_nodes} Nodi)", fontsize=13, fontweight='bold')
        ax4.set_xlabel("")
    else:
        plot_pivot = time_df.set_index('Node')[['Avg_Calc_Time_s', 'Avg_Net_Time_s']]
        plot_pivot.columns = ['Calcolo Locale', 'Rete / Attesa Peer']
        plot_pivot.plot(kind='bar', ax=ax4, color=['#2b5c8f', '#d95f02'], alpha=0.85)
        ax4.set_title("Tempi Medi per Round per Nodo", fontsize=13, fontweight='bold')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        ax4.legend(loc='upper right')
        
    ax4.set_ylabel("Secondi (s)")
    ax4.grid(True, alpha=0.3)
    
    title_main = (
        f"Federated Learning [{arch_label}] — Rete di {n_nodes} Nodi\n"
        f"Parametri Addestramento: {rounds_str} Round Totali | {epochs_str} Epoche Locali per Round"
    )
    plt.suptitle(title_main, fontsize=15, fontweight='bold')
    
    plot_path = os.path.join(output_dir, 'fl_dashboard_grafici.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return plot_path

def main():
    root = tk.Tk()
    root.withdraw()
    
    folder_selected = filedialog.askdirectory(title="Seleziona la cartella con i file CSV dei nodi")
    if not folder_selected:
        print("Operazione annullata.")
        sys.exit()
        
    csv_files = glob.glob(os.path.join(folder_selected, "*.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('fl_report')]
    
    if not csv_files:
        messagebox.showerror("Errore", f"Nessun file CSV trovato in:\n{folder_selected}")
        sys.exit(1)
        
    df_metrics, df_prog, file_errors = load_and_resolve_all_nodes(csv_files)
    
    if df_metrics.empty:
        messagebox.showwarning("Attenzione", "Nessuna metrica valida trovata nei file CSV selezionati.")
        sys.exit()
        
    arch_label, rounds_str, epochs_str = detect_experiment_metadata(csv_files, df_prog)
    
    # Controllo integrità dati ed epoche mancanti
    integrity_report = check_missing_data_per_node(df_metrics, df_prog, file_errors, rounds_str, epochs_str)
    
    # Generazione report e grafici
    _, csv_rep, txt_rep = generate_report_files(df_metrics, df_prog, folder_selected, arch_label, rounds_str, epochs_str, integrity_report)
    plot_path = generate_visualizations(df_metrics, df_prog, folder_selected, arch_label, rounds_str, epochs_str)
    
    # Notifica all'utente
    msg = (
        f"Elaborati con successo {len(df_metrics)} nodi!\n\n"
        f"Configurazione:\n"
        f"• Architettura: {arch_label}\n"
        f"• Round totali: {rounds_str}\n"
        f"• Epoche per round: {epochs_str}\n\n"
    )
    if integrity_report:
        msg += f"ATTENZIONE: {len(integrity_report)} file hanno dati mancanti o troncati!\n"
        msg += f"Consulta la sezione iniziale di '{os.path.basename(txt_rep)}' per i dettagli dei file da riesportare.\n\n"
    else:
        msg += "Tutti i file sono completi al 100%.\n\n"

    msg += f"File generati in cartella:\n• {os.path.basename(txt_rep)}\n• {os.path.basename(csv_rep)}\n• {os.path.basename(plot_path)}"

    if integrity_report:
        messagebox.showwarning("Analisi Completata con Dati Mancanti", msg)
    else:
        messagebox.showinfo("Analisi Completata", msg)

if __name__ == "__main__":
    main()