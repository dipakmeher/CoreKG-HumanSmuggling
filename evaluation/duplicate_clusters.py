import pandas as pd
import itertools
import re
import argparse
import networkx as nx
from rapidfuzz import fuzz

def cluster_raw(df, threshold=95):
    total_entities = len(df)
    duplicate_clusters = 0
    duplicate_nodes = 0
    per_type_duplicate_nodes = {}
    all_clusters = []

    for entity_type, group in df.groupby('type'):
        names = group['name_clean'].tolist()
        if len(names) < 2:
            continue

        G = nx.Graph()
        G.add_nodes_from(names)

        for i, j in itertools.combinations(range(len(names)), 2):
            name1, name2 = names[i], names[j]
            score = fuzz.partial_ratio(name1, name2)
            if score >= threshold:
                G.add_edge(name1, name2)

        clusters = [list(c) for c in nx.connected_components(G) if len(c) > 1]
        if clusters:
            per_type_duplicate_nodes[entity_type.upper()] = sum(len(c) - 1 for c in clusters)
        all_clusters.extend([(entity_type.upper(), cluster) for cluster in clusters])
        duplicate_clusters += len(clusters)
        duplicate_nodes += sum(len(c) - 1 for c in clusters)

    duplication_rate = round(100 * duplicate_nodes / total_entities, 2)

    return {
        "total_entities": total_entities,
        "duplicate_clusters": duplicate_clusters,
        "duplicate_nodes": duplicate_nodes,
        "duplication_rate": duplication_rate,
        "per_type_duplicate_nodes": per_type_duplicate_nodes,
        "clusters": all_clusters
    }

def save_raw_results(stats, output_path):
    with open(output_path, "w") as f:
        f.write("RAW DUPLICATION STATS\n")
        f.write(f"{'Total Entities':<30} {stats['total_entities']}\n")
        f.write(f"{'Duplicate Clusters':<30} {stats['duplicate_clusters']}\n")
        f.write(f"{'Duplicate Nodes':<30} {stats['duplicate_nodes']}\n")
        f.write(f"{'Duplication Rate (%)':<30} {stats['duplication_rate']}\n\n")

        f.write("Per-Type Duplicate Node Counts:\n")
        for etype, count in stats["per_type_duplicate_nodes"].items():
            f.write(f"  {etype}: {count}\n")

        f.write("\nDuplicate Clusters:\n")
        for etype, cluster in stats["clusters"]:
            f.write(f"  [{etype}] {cluster}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw Duplicate Entity Detection")
    parser.add_argument("--file", required=True, help="Path to CreateFinalEntities CSV file.")
    parser.add_argument("--threshold", type=int, default=95, help="Fuzzy matching threshold (default: 95)")
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    df['type'] = df['type'].fillna('UNKNOWN').replace('', 'UNKNOWN')
    df['name_clean'] = df['name'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()

    print("\nRunning RAW duplicate detection...")
    raw_stats = cluster_raw(df, threshold=args.threshold)

    print("\nRAW DUPLICATION SUMMARY")
    print(f"{'Total Entities':<30} {raw_stats['total_entities']}")
    print(f"{'Duplicate Clusters':<30} {raw_stats['duplicate_clusters']}")
    print(f"{'Duplicate Nodes':<30} {raw_stats['duplicate_nodes']}")
    print(f"{'Duplication Rate (%)':<30} {raw_stats['duplication_rate']}")

    print("\nSaving results to 'raw_deduplication_report.txt'...")
    save_raw_results(raw_stats, "raw_deduplication_report.txt")
