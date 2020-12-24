import sys

hit, tot = 0, 0
for pred, ans in zip(*[open(f).read().split('\n')[1:] for f in [sys.argv[1], sys.argv[2]]]):
    if pred == ans:
        hit += 1
    tot += 1

print(hit, tot, hit / tot)
