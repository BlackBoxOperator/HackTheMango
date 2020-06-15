import sys
from pprint import pprint

pipes = []
signle_pipes = 82
totle_pipes = 93

for fn in sys.argv[1:]:
    ctx = open(fn, 'r').read().split('\n')
    pipe = None; val = []
    for line in ctx:
        if 'pipeline' in line:
            if pipe and val: # prev data need be store
                pipes.append((pipe, tuple(val)))
                val = []
            pipe = eval(''.join(line.split()[1:]))
            if not isinstance(pipe, list):
                print("error: pipe is", pipe)
            pipe = tuple(pipe)
        elif 'Validation' in line:
            val.append(float(line.split()[-1]))
    if pipe and val: # prev data need be store
        pipes.append((pipe, tuple(val)))

pipes = list(set(pipes))
avg = sorted([(p, sum(v) / len(v)) for p, v in pipes], key = lambda x: x[1])
bst = sorted([(p, max(v)) for p, v in pipes], key = lambda x: x[1])
lst = sorted([(p, v[-1]) for p, v in pipes], key = lambda x: x[1])

bestAvgNum = 10
bestBstNum = 10
bestLstNum = 10

good_pipes = set()
out = open("good_pipes.txt", "w")
print("avg:", file = out)
#pprint(avg)
print('\n'.join([str(p) for p in avg[-bestAvgNum:]]), file = out)
print("best:", file = out)
#pprint(bst)
print('\n'.join([str(p) for p in bst[-bestBstNum:]]), file = out)
print("last:", file = out)
#pprint(lst)
print('\n'.join([str(p) for p in lst[-bestLstNum:]]), file = out)

print("all by best:", file = out)
print('\n'.join([str(p) for p in bst]), file = out)

good_pipes.update([p for pl, v in avg[-bestAvgNum:] for p in pl])
good_pipes.update([p for pl, v in bst[-bestBstNum:] for p in pl])
good_pipes.update([p for pl, v in lst[-bestLstNum:] for p in pl])
good_pipes = [p for p in good_pipes if p >= signle_pipes or p % 2 == 0]
print(len(good_pipes), good_pipes)
