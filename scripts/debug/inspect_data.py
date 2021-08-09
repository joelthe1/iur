import json
import tqdm
import pandas as pd

from datetime import datetime
from operator import itemgetter
from collections import Counter

from absl import flags
from absl import app
from absl import logging

flags.DEFINE_string('df_path', None, 'Path to DataFrame')
#flags.mark_flags_as_required(['df_path'])

args = flags.FLAGS

def inspect_df():
  df = pd.read_pickle(args.df_path)
  logging.info(df)

def inspect_jsonl(jsonl_filename):
  counter = 0
  with open(jsonl_filename) as s0:
    for line in s0:
      if len(json.loads(line)['action_type']) < 16:
        counter += 1
        # print(len(json.loads(line)['action_type']))

  print(counter)

def inspect_splits():
  df = pd.read_pickle(args.df_path)
  logging.info(df)

  # print(df.iloc[303744])
  # return

  with open('data/avacado/exp_20210809_1/sender_history_split_0.ids') as s0, open('data/avacado/exp_20210809_1/sender_history_split_1.ids') as s1:
    counter = 0
    for line0, line1 in zip(s0, s1):
      counter += 1
      from_set = set()
      results1 = Counter()
      results2 = Counter()
      for s in line0.split():
        from_set.add(df.iloc[int(s)]['from'])
        results1[df.iloc[int(s)]['subject']] += 1

      for s in line1.split():
        from_set.add(df.iloc[int(s)]['from'])
        results2[df.iloc[int(s)]['subject']] += 1

      # if len(line0.split()) < 16 or len(line1.split()) < 16:
      if counter == 1000:
        print(counter)
        print(from_set)
        print(results1)
        print(results2)
        print('\n\n')
        break

      assert len(from_set) == 1, f'Error in grouping {from_set}'
      assert len(set(results1.keys())&set(results2.keys())) == 0, f'Found same subjects in both splits: {results1} \n and \n {results2}'
      #assert len(results1) == len(results2), f'Error in grouping {results1} and {results2}'

      if counter%100 == 0:
        print(results1)
        print('\n')
        print(results2)
        print('-'*20)
        print()


def inspect_subjects():
  df = pd.read_pickle(args.df_path)
  logging.info(df)

  #print(df['subject'].unique())
  print(df[ (df['subject'].isnull()) | (df['subject']=='') ].index)
  # print(df['subject'].unique().shape)


def main(argv):
  #logging.info(f"Inspecting DF {args.df_path}")
  #inspect_jsonl('/usr/local/src/iur/data/avacado/avacado-dataset-split-0.jsonl')
  #inspect_jsonl('/usr/local/src/iur/data/avacado/avacado-dataset-split-1.jsonl')
  #inspect_df()
  
  inspect_splits()
  # inspect_subjects()

if __name__ == "__main__":
  app.run(main)
