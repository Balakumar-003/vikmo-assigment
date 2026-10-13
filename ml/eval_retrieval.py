import os
import sys
import json
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dealerdesk.settings")
django.setup()

from core.services.retrieval import get_retriever

EVAL_QUERIES = [
    # 1-5: Standard exact & vehicle queries
    {"query": "brake pads for Bajaj Pulsar 150", "expected_skus": ["BRK-1044", "BRK-1063"], "type": "standard"},
    {"query": "KTM Duke 390 chain sprocket kit", "expected_skus": ["CHN-1040", "CHN-1076"], "type": "standard"},
    {"query": "battery for Bajaj Dominar 400", "expected_skus": ["ELE-1002"], "type": "standard"},
    {"query": "LED indicator set TVS Apache RTR 160", "expected_skus": ["ELE-1001"], "type": "standard"},
    {"query": "Brake Shoe Set Suzuki Gixxer SF", "expected_skus": ["BRK-1001"], "type": "standard"},

    # 6-10: Vehicle shorthand / nicknames
    {"query": "Duke 390 brake pads", "expected_skus": ["BRK-1032"], "type": "vehicle_variant"},
    {"query": "Pulsar 150 chain kit", "expected_skus": ["CHN-1001", "CHN-1002"], "type": "vehicle_variant"},
    {"query": "Dominar side mirror set", "expected_skus": ["BDY-1001", "BDY-1086"], "type": "vehicle_variant"},
    {"query": "Apache 200 tank pad", "expected_skus": ["BDY-1002"], "type": "vehicle_variant"},
    {"query": "Gixxer foot peg set", "expected_skus": ["GRP-1001"], "type": "vehicle_variant"},

    # 11-15: Misspellings & typo tolerance
    {"query": "braek pad pulsar 150", "expected_skus": ["BRK-1044", "BRK-1063"], "type": "misspelling"},
    {"query": "chen sproket ktm duke", "expected_skus": ["CHN-1040", "CHN-1076"], "type": "misspelling"},
    {"query": "indicater led apache rtr", "expected_skus": ["ELE-1001", "ELE-1075"], "type": "misspelling"},
    {"query": "domnar 400 batry", "expected_skus": ["ELE-1002"], "type": "misspelling"},
    {"query": "gixxr brake shue", "expected_skus": ["BRK-1001"], "type": "misspelling"},

    # 16-20: Hindi/English (Hinglish) mix
    {"query": "pulsar 150 ke liye brake pad", "expected_skus": ["BRK-1044", "BRK-1063"], "type": "hinglish"},
    {"query": "dominar 400 ka mirror set", "expected_skus": ["BDY-1001", "BDY-1086"], "type": "hinglish"},
    {"query": "duke 390 ke liye chain kit chahiye", "expected_skus": ["CHN-1040", "CHN-1076"], "type": "hinglish"},
    {"query": "apache rtr 160 ka indicator dikhao", "expected_skus": ["ELE-1001"], "type": "hinglish"},
    {"query": "bike cover universal chahiye", "expected_skus": ["ACC-1001"], "type": "hinglish"},

    # 21-23: Category / Generic queries
    {"query": "coolant 1L universal", "expected_skus": ["OIL-1001"], "type": "generic"},
    {"query": "universal bike body cover", "expected_skus": ["ACC-1001"], "type": "generic"},
    {"query": "Motul coolant 1L", "expected_skus": ["OIL-1001"], "type": "generic"},

    # 24-26: Tricky / Zero-match (out of scope queries where no match is expected)
    {"query": "Ferrari carbon ceramic brake rotor", "expected_skus": [], "type": "unmatched"},
    {"query": "Harley Davidson road glide windshield", "expected_skus": [], "type": "unmatched"},
    {"query": "Apple iPhone 15 fast charger", "expected_skus": [], "type": "unmatched"},
]


def run_retrieval_evaluation():
    retriever = get_retriever()
    total_evaluable = 0
    hits_at_5 = 0
    mrr_sum = 0.0

    print("Running Retrieval Evaluation on 26 labelled test queries...")
    print("-" * 80)
    print(f"{'Query':<45} | {'Type':<15} | {'Recall@5':<8} | {'Top Match SKU'}")
    print("-" * 80)

    category_stats = {}

    for item in EVAL_QUERIES:
        q = item["query"]
        expected = item["expected_skus"]
        q_type = item["type"]

        results = retriever.search(query=q, top_k=5)
        top_skus = [p.sku for p in results]

        if q_type not in category_stats:
            category_stats[q_type] = {"total": 0, "hits": 0}

        if expected:
            total_evaluable += 1
            category_stats[q_type]["total"] += 1

            hit = any(e in top_skus for e in expected)
            if hit:
                hits_at_5 += 1
                category_stats[q_type]["hits"] += 1
                # Find reciprocal rank
                for rank, s in enumerate(top_skus, start=1):
                    if s in expected:
                        mrr_sum += 1.0 / rank
                        break
            status_str = "PASS" if hit else "FAIL"
        else:
            # Unmatched query: passes if top results have low confidence or zero results
            status_str = "N/A (unmatched)"

        top_match = top_skus[0] if top_skus else "None"
        print(f"{q[:45]:<45} | {q_type:<15} | {status_str:<8} | {top_match}")

    recall_at_5 = (hits_at_5 / total_evaluable) * 100 if total_evaluable > 0 else 0
    mrr = (mrr_sum / total_evaluable) if total_evaluable > 0 else 0

    print("-" * 80)
    print(f"Overall Recall@5: {hits_at_5}/{total_evaluable} ({recall_at_5:.1f}%)")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
    print("-" * 80)
    print("Breakdown by Query Category:")
    for cat, stats in category_stats.items():
        if stats["total"] > 0:
            rec = (stats["hits"] / stats["total"]) * 100
            print(f"  - {cat:<18}: {stats['hits']}/{stats['total']} ({rec:.1f}%)")

    # Save queries and metrics to json
    out_file = os.path.join(os.path.dirname(__file__), "retrieval_eval_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "total_queries": len(EVAL_QUERIES),
            "evaluable_queries": total_evaluable,
            "hits_at_5": hits_at_5,
            "recall_at_5": round(recall_at_5, 2),
            "mrr": round(mrr, 3),
            "breakdown": category_stats,
            "queries": EVAL_QUERIES,
        }, f, indent=2)
    print(f"Results written to {out_file}")


if __name__ == "__main__":
    run_retrieval_evaluation()
