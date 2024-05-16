from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split

label_dict = {
            0: "Company",
            1: "School",
            2: "Artist",
            3: "Athlete",
            4: "Polite",
            5: "Transportation",
            6: "Building",
            7: "Nature",
            8: "Village",
            9: "Animal",
            10: "Plant",
            11: "Album",
            12: "Film",
            13: "Book",
        }

cache_dir = "/data/james/.cache"
output_dir = '/data/james/dbpedia_14_data'

dataset = load_dataset("dbpedia_14")
dataset.set_format("pandas")

df = dataset['train'][:].sample(frac=(200./560.))
df = df[['content', 'label']]
df = df.rename(columns={'content': 'text'})
df['label'] = df['label'].transform(lambda x: label_dict[x])

df_train, df_val = train_test_split(df, test_size=(5. / 200.))

df_test = dataset['test'][:].sample(frac=(5. / 70.))
df_test = df_test[['content', 'label']]
df_test = df_test.rename(columns={'content': 'text'})
df_test['label'] = df_test['label'].transform(lambda x: label_dict[x])

train_path = os.path.join(output_dir, 'train.csv')
val_path = os.path.join(output_dir, 'val.csv')
test_path = os.path.join(output_dir, 'test.csv')

df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path, index=False)
df_test.to_csv(test_path, index=False)