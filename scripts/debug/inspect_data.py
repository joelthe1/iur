import json
import pandas as pd
from collections import Counter

from absl import logging
from absl import flags
from absl import app


flags.DEFINE_string('df_path', None, 'Path to DataFrame')
#flags.mark_flags_as_required(['df_path'])

args = flags.FLAGS

def inspect_df():
  df = pd.read_pickle(args.df_path)
  logging.info(df)

  # with open('/usr/local/src/iur/data/output/sender_history_split_0.ids') as ids_0, \
  #      open('/usr/local/src/iur/data/output/sender_history_split_1.ids') as ids_1:
  #   for query_id_line, target_id_line in zip(ids_0, ids_1):
  #     for query, target in zip(query_id_line.split(), target_id_line.split()):
  #       logging.info(f"{df.loc[int(query)]['from']} ******** {df.loc[int(query)]['from']}")

  #print(df[ (df['date'].isnull()) & (df['date']=='') ].index)

  idxs = []

  with open('data/ta3/sender_history_split_0.ids') as s0:
    for line in s0:
      idxs += [int(idx) for idx in line.strip().split()]

  with open('data/ta3/sender_history_split_0.ids') as s1:
    for line in s1:
      idxs += [int(idx) for idx in line.strip().split()]
  
  new_df = df.iloc[idxs]
  print(new_df['subject'].unique().shape)


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

  with open('data/avacado/exp_20210804_1/temp_0.ids') as s0, open('data/avacado/exp_20210804_1/temp_1.ids') as s1:
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

      if len(line0.split()) < 16 or len(line1.split()) < 16:
        print(counter)
        print(from_set)
        print(results1)
        print(results2)
        print('\n\n')
        break

      assert len(from_set) == 1, f'Error in grouping {from_set}'
      #assert len(results1) == len(results2), f'Error in grouping {results1} and {results2}'

      # for key, value in results1:
      #   if abs(results1[key] - results2[key]) > 2:
      #     print(results
          


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
