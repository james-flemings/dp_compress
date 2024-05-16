from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split

cache_dir = "/data/james/.cache"
output_dir = '/data/james/reddit_data'

dataset = load_dataset('webis/tldr-17', cache_dir=cache_dir)
dataset.set_format("pandas")

df = dataset['train'][:900000]
unique_subreddit = df[:].subreddit.value_counts()
top_subreddit = unique_subreddit[unique_subreddit > 2000]
print(top_subreddit.index.size)

df_filtered = df[df.subreddit.isin(top_subreddit.index)]
print(len(df_filtered.index))

df_filtered = df_filtered[['content', 'subreddit']]
df_filtered = df_filtered.rename(columns={'content': 'text', 'subreddit': 'label'})

df_train, df_test = train_test_split(df_filtered, test_size=0.02)
df_train, df_val = train_test_split(df_train, test_size=0.02)

train_path = os.path.join(output_dir, "train.csv")
val_path = os.path.join(output_dir, "val.csv")
test_path = os.path.join(output_dir, "test.csv")

df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path, index=False)
df_test.to_csv(test_path, index=False)