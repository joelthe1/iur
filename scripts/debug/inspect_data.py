import json
import tqdm
import pandas as pd

from datetime import datetime
from operator import itemgetter
from collections import Counter

import pickle
import random

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
    fw_counter = 0
    for line0, line1 in zip(s0, s1):
      from_set = set()
      results1 = Counter()
      results2 = Counter()
      for s in line0.split():
        counter += 1
        from_set.add(df.iloc[int(s)]['from'])
        results1[df.iloc[int(s)]['subject']] += 1

      for s in line1.split():
        counter += 1
        from_set.add(df.iloc[int(s)]['from'])
        results2[df.iloc[int(s)]['subject']] += 1

      # if len(line0.split()) < 16 or len(line1.split()) < 16:

      assert len(from_set) == 1, f'Error in grouping {from_set}'
      # assert len(set(results1.keys())&set(results2.keys())) == 0, f'Found same subjects in both splits: {results1} \n and \n {results2}'
      #assert len(results1) == len(results2), f'Error in grouping {results1} and {results2}'

      # if counter%100 == 0:
      #   print(results1)
      #   print('\n')
      #   print(results2)
      #   print('-'*20)
      #   print()

      for key, value in results1.items():
        if key.strip() and key.strip().lower().startswith('fw'):
          fw_counter += value

      for key, value in results2.items():
        if key.strip() and key.strip().lower().startswith('fw'):
          fw_counter += value

    print(f'Found {fw_counter} messages starting with `fw` out of {counter} total messages.')

def inspect_subjects():
  df = pd.read_pickle(args.df_path)
  logging.info(df)

  #print(df['subject'].unique())
  print(df[ (df['subject'].isnull()) | (df['subject']=='') ].index)
  # print(df['subject'].unique().shape)


def inspect_authors():
  df = pd.read_pickle(args.df_path)
  logging.info(df)
  # print(df.iloc[352790])

  # with open('/usr/local/src/iur/data/avacado/exp_20210809_2/authors.pickle', 'rb') as f:
  #   authors_map = pickle.load(f)
  #   #print(authors_map)

  with open('/usr/local/src/iur/data/avacado/exp_20210809_2/sender_history_split_0.ids') as split_ids_0_f, open('/usr/local/src/iur/data/avacado/exp_20210809_2/sender_history_split_1.ids') as split_ids_1_f:
    for author_idx, (line0, line1) in enumerate(zip(split_ids_0_f, split_ids_1_f)):

      if author_idx != 6:
        continue
      
      line0 = [int(x) for x in line0.split()]
      line1 = [int(x) for x in line1.split()]
      
      for idx0, idx1 in zip(random.sample(line0, 3), random.sample(line1, 3)):
        print(df.iloc[idx0]['body'])
        print()
        print('-'*100)
        input()
        print(df.iloc[idx1]['body'])
        print()
        print('-'*100)
        input()

      print()
      print('='*100)      

def main(argv):
  #logging.info(f"Inspecting DF {args.df_path}")
  #inspect_jsonl('/usr/local/src/iur/data/avacado/avacado-dataset-split-0.jsonl')
  #inspect_jsonl('/usr/local/src/iur/data/avacado/avacado-dataset-split-1.jsonl')
  #inspect_df()
  
  # inspect_splits()
  # inspect_subjects()
  inspect_authors()

if __name__ == "__main__":
  app.run(main)
