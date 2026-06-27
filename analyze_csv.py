import csv
from pathlib import Path

path = Path('submission.csv')
with open(path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

scores = [float(r['score']) for r in rows]
reasoning_lens = [len(r['reasoning']) for r in rows]

print(f'Total rows: {len(rows)}')
print(f'Score range: {min(scores):.4f} - {max(scores):.4f}')
print(f'Score mean: {sum(scores)/len(scores):.4f}')
print(f'Top 5 scores: {[round(s,4) for s in scores[:5]]}')
print(f'Bottom 5 scores: {[round(s,4) for s in scores[-5:]]}')
print()
print(f'Reasoning string lengths:')
print(f'  Min: {min(reasoning_lens)} chars')
print(f'  Max: {max(reasoning_lens)} chars')
print(f'  Avg: {sum(reasoning_lens)/len(reasoning_lens):.0f} chars')
print()

# Check monotonicity
for i in range(len(scores)-1):
    if scores[i] < scores[i+1]:
        print(f'MONOTONICITY VIOLATION at rank {i+1} -> {i+2}: {scores[i]:.6f} < {scores[i+1]:.6f}')
        break
else:
    print('PASS: Scores are non-increasing (monotonic)')

# Check score bunching at 0.15
cap_count = sum(1 for s in scores if abs(s - 0.15) < 0.001)
print(f'Candidates at 0.15 cap: {cap_count}')

# Score distribution bands
bands = {'>=0.7': 0, '0.5-0.7': 0, '0.3-0.5': 0, '0.15-0.3': 0, '<0.15': 0}
for s in scores:
    if s >= 0.7: bands['>=0.7'] += 1
    elif s >= 0.5: bands['0.5-0.7'] += 1
    elif s >= 0.3: bands['0.3-0.5'] += 1
    elif s >= 0.15: bands['0.15-0.3'] += 1
    else: bands['<0.15'] += 1
print()
print('Score bands:')
for b, c in bands.items():
    print(f'  {b}: {c}')

print()
print('--- Top-5 Reasoning Samples ---')
for r in rows[:5]:
    print(f"Rank {r['rank']} ({r['candidate_id']}): {r['reasoning'][:250]}")
    print()
print('--- Bottom-3 Reasoning Samples ---')
for r in rows[-3:]:
    print(f"Rank {r['rank']} ({r['candidate_id']}): {r['reasoning'][:250]}")
    print()
