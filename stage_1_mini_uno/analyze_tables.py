import csv
import sys
import os
from collections import defaultdict

def analyze_tables():
    oracle_path = "stage_1_mini_uno/generated_tables/oracle_table_10.csv"
    pomdp_path = "stage_1_mini_uno/generated_tables/pomdp_table_10.csv"
    
    print(f"Analyzing {oracle_path} and {pomdp_path}...")
    
    with open(oracle_path, 'r') as f_oracle, open(pomdp_path, 'r') as f_pomdp:
        oracle_reader = csv.DictReader(f_oracle)
        pomdp_reader = csv.DictReader(f_pomdp)
        
        total_states = 0
        matches = 0
        
        # Metrics by Value Bucket
        # Buckets: "Winning" (V > 0), "Losing" (V < 0), "Neutral" (V = 0)
        by_value = defaultdict(lambda: {"total": 0, "matches": 0})
        
        # Mismatch examples
        mismatches = []
        
        for row_o, row_p in zip(oracle_reader, pomdp_reader):
            # Verify alignment (simple check)
            if row_o["H1"] != row_p["H1"] or row_o["Pt"] != row_p["Pt"]:
                print(f"Error: Row misalignment at index {total_states}")
                print(f"Oracle: {row_o}")
                print(f"POMDP:  {row_p}")
                break
                
            total_states += 1
            
            # Get Value
            val = float(row_o["Value"])
            bucket = "Neutral"
            if val > 0.001: bucket = "Winning"
            elif val < -0.001: bucket = "Losing"
            
            # Check Match
            # Note: pomdp_table has a 'Match' column, but let's recompute to be safe/flexible
            # The actions are strings like "Action(X_1=('B', 2))"
            act_o = row_o["Action"]
            act_p = row_p["POMDP Action"]
            
            is_match = (act_o == act_p)
            
            by_value[bucket]["total"] += 1
            if is_match:
                matches += 1
                by_value[bucket]["matches"] += 1
            else:
                if len(mismatches) < 10: # Store first 10 mismatches
                    mismatches.append({
                        "State": f"H1={row_o['H1']}, Pt={row_o['Pt']}",
                        "Value": val,
                        "Oracle": act_o,
                        "POMDP": act_p
                    })

    print("-" * 30)
    print(f"Total States Analyzed: {total_states}")
    print(f"Overall Match Rate:    {matches / total_states * 100:.2f}% ({matches}/{total_states})")
    print("-" * 30)
    print("Match Rate by State Value (Oracle's Evaluation):")
    for bucket in ["Winning", "Neutral", "Losing"]:
        stats = by_value[bucket]
        if stats["total"] > 0:
            rate = stats["matches"] / stats["total"] * 100
            print(f"  {bucket:8}: {rate:6.2f}% ({stats['matches']}/{stats['total']})")
    print("-" * 30)
    print("Sample Mismatches:")
    for m in mismatches:
        print(f"  State: {m['State']}")
        print(f"    Value: {m['Value']:.4f}")
        print(f"    Oracle: {m['Oracle']}")
        print(f"    POMDP:  {m['POMDP']}")
        print()

if __name__ == "__main__":
    analyze_tables()
