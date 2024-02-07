from datasets import load_dataset
import csv
import os
from tqdm import tqdm

'''
train_ratio_gen = 75000 / 1207222 
val_ratio_gen = 4400 / 67068 
test_ratio_gen = 4400 / 2500
'''

train_ratio_gen = 100 / 1207222 
val_ratio_gen = 20 / 2500
test_ratio_gen = 20 / 2500

cache_dir = "/data/james/.cache"
output_dir = "."

cpc_codes = {'a': "Human Necessities",
             'b': "Performing Operations; Transporting",
             'c': "Chemistry; Metallurgy",
             'd': "Textiles; Paper",
             'e': "Fixed Constructions",
             'f': "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
             'g': "Physics",
             'h': "Electricity",
             'y': "General tagging of new or cross-sectional technology"}

train_splits = {'a': 174134, 'b': 161520, 'c': 101042, 
                'd': 10164, 'e': 34443, 'f': 85568,
                'g': 258935, 'h': 257019, "y": 124397}

val_splits = {'a': 9674, 'b': 8973, 'c': 5613, 
                'd': 565, 'e': 1914, 'f': 4754,
                'g': 14385, 'h': 14279, "y": 14279}

test_splits = {'a': 9675, 'b': 8974, 'c': 5614, 
                'd': 565, 'e': 1914, 'f': 4754,
                'g': 14386, 'h': 14279, "y": 14279}

total_train_ds = []
total_val_ds = []
total_test_ds = []

title = ["text", "label"]

for code, name in tqdm(cpc_codes.items(), desc="Processing through all code types"):
    train_split = f'train[:{round(train_splits[code] * train_ratio_gen)}]'
    val_split = f'validation[:{round(val_splits[code] * val_ratio_gen)}]'
    test_split = f'test[:{round(test_splits[code] * test_ratio_gen)}]'

    train_ds, val_ds, test_ds = load_dataset("big_patent", code,
                                              split=[train_split, val_split, test_split],
                                              cache_dir=cache_dir)
    for train in train_ds:
        total_train_ds.append([train['abstract'], name]) 
    for val in val_ds:
        total_val_ds.append([val['abstract'], name]) 
    for test in test_ds:
        total_test_ds.append([test['abstract'], name]) 

train_path = os.path.join(output_dir, "train.csv") 

with open(train_path, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(title)
    for train in total_test_ds:
        csv_writer.writerow(train)