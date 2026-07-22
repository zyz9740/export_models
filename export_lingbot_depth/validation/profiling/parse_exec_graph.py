"""Parse a benchmark_app -exec_graph_path XML dump and rank layers by execTimeMcs.

Usage:
    python parse_exec_graph.py <exec_graph.xml> [--top N]
"""
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict


def load_layers(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    layers = []
    for layer in root.iter("layer"):
        data = layer.find("data")
        if data is None:
            continue
        exec_time_raw = data.get("execTimeMcs")
        if exec_time_raw is None or exec_time_raw == "not_executed":
            continue
        layers.append({
            "name": layer.get("name"),
            "type": layer.get("type"),
            "execTimeMcs": int(exec_time_raw),
            "execOrder": data.get("execOrder"),
            "primitiveType": data.get("primitiveType"),
            "outputLayouts": data.get("outputLayouts"),
            "outputPrecisions": data.get("outputPrecisions"),
        })
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml_path")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    layers = load_layers(args.xml_path)
    total_time = sum(l["execTimeMcs"] for l in layers)
    print(f"Executed layers: {len(layers)}  Total instrumented time: {total_time/1000:.3f} ms")

    print(f"\n--- Top {args.top} layers by execTimeMcs ---")
    for l in sorted(layers, key=lambda x: -x["execTimeMcs"])[: args.top]:
        pct = 100 * l["execTimeMcs"] / total_time
        print(f"{l['execTimeMcs']:>8} us ({pct:5.2f}%)  type={l['type']:<28} prim={str(l['primitiveType']):<25} name={l['name']}")

    print("\n--- Aggregate by layer type ---")
    by_type = defaultdict(int)
    count_type = defaultdict(int)
    for l in layers:
        by_type[l["type"]] += l["execTimeMcs"]
        count_type[l["type"]] += 1
    for t, tm in sorted(by_type.items(), key=lambda x: -x[1]):
        pct = 100 * tm / total_time
        print(f"{tm:>8} us ({pct:5.2f}%)  count={count_type[t]:<5} type={t}")

    print("\n--- Reference/fallback kernels (primitiveType containing '_ref') ---")
    ref_total = 0
    ref_layers = [l for l in layers if "_ref" in (l["primitiveType"] or "")]
    for l in sorted(ref_layers, key=lambda x: -x["execTimeMcs"]):
        ref_total += l["execTimeMcs"]
    print(f"Total ref-kernel time: {ref_total/1000:.3f} ms ({100*ref_total/total_time:.2f}% of instrumented total), {len(ref_layers)} layers")
    for l in sorted(ref_layers, key=lambda x: -x["execTimeMcs"])[: args.top]:
        print(f"{l['execTimeMcs']:>8} us  type={l['type']:<12} prim={l['primitiveType']:<20} {l['name']}")


if __name__ == "__main__":
    main()
