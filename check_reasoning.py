import csv
p = "outputs/top100_test.csv"
with open(p, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
truncated = [r for r in rows if r["reasoning"].endswith("...")]
lens = [len(r["reasoning"]) for r in rows]
print(f"Truncated strings: {len(truncated)}")
print(f"Reasoning lengths: min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.0f}")
print()
print("Sample reasoning (rank 1):")
print(rows[0]["reasoning"])
print()
print("Sample reasoning (rank 50):")
print(rows[49]["reasoning"])
