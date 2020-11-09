import sys
import pandas as pd
from sklearn.model_selection import train_test_split

if len(sys.argv) < 2:
    print("usage: {} [train.csv] [dev.csv]".format(sys.argv[0])), exit(0)

def info(data):
    label = 'grade' if 'grade' in data else 'label'
    classes = data[label].value_counts()
    print(classes) # min

def balance(data):
    label = 'grade' if 'grade' in data else 'label'
    g = data.groupby(label)
    return g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])

print('train:'); info(train)
print('test:'); info(test)

train, valid0 = train_test_split(train, test_size=0.2)
train, valid1 = train_test_split(train, test_size=0.25)

print('new train:'); info(train)

train.to_csv('new_train.csv')

print('new valid 0:'); info(valid0)
print('new valid 1:'); info(valid1)

valid0.to_csv('new_valid0.csv')
valid1.to_csv('new_valid1.csv')

bvalid0 = balance(valid0)
print('new balanced valid 0:'); info(bvalid0)
bvalid1 = balance(valid1)
print('new balanced valid 1:'); info(bvalid1)

bvalid0.to_csv('new_balanced_valid0.csv')
bvalid1.to_csv('new_balanced_valid1.csv')
