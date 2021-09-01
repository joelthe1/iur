import json
import tqdm
import pandas as pd

from datetime import datetime
from operator import itemgetter
from collections import Counter, defaultdict
from tqdm import tqdm

from polyleven import levenshtein

from email.utils import parsedate_to_datetime
from pytz import utc

import pickle
import random
import re

from absl import flags
from absl import app
from absl import logging

flags.DEFINE_string('df_path', None, 'Path to DataFrame')
flags.DEFINE_string('output_prefix', '.', 'Prefix to use for the split out ID files')
flags.DEFINE_integer('min_episode_length', 2, 'The minimum number of messages (aka episodes) present per author per split so as to be considered for splitting.')
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


def shuffle_splits():
  df = pd.read_pickle(args.df_path)
  logging.info(df)

  # print(df.loc[303744])
  # return

  with open('data/avacado/exp_20210825_1/curated_history_0.ids') as s0, open('data/avacado/exp_20210825_1/curated_history_1.ids') as s1, open('data/avacado/exp_20210825_1/shuffled_history_0.ids', 'w') as w0, open('data/avacado/exp_20210825_1/shuffled_history_1.ids', 'w') as w1:
    for line0, line1 in zip(s0, s1):
      all_author_ids = line0.strip().split() + line1.strip().split()
      random.seed(99)
      random.shuffle(all_author_ids)
      w0.write(' '.join(all_author_ids[:16]) + '\n')
      w1.write(' '.join(all_author_ids[16:]) + '\n')


def inspect_splits():
  df = pd.read_pickle(args.df_path)
  # logging.info(df)

  # print(df.loc[303744])
  # return

  with open('data/avacado/exp_20210825_1/curated_history_0.ids') as s0, open('data/avacado/exp_20210825_1/curated_history_1.ids') as s1:
    counter = 0

    fw_counter = 0
    for line0, line1 in zip(s0, s1):
      from_set = set()
      results1 = Counter()
      results2 = Counter()
      len_map = {}
      for s in line0.split():
        counter += 1
        from_set.add(df.loc[int(s)]['from'])
        results1[df.loc[int(s)]['subject']] += 1
        len_map[s] = len(df.loc[int(s)]['body'].split())

      for s in line1.split():
        counter += 1
        from_set.add(df.loc[int(s)]['from'])
        results2[df.loc[int(s)]['subject']] += 1
        len_map[s] = len(df.loc[int(s)]['body'].split())
      
      # print(len_map)
      # print(from_set)
      print(f'{from_set.pop()}: {len([v for v in len_map.values() if v<64])}/{len(len_map)}')

      # print(from_set)
      # print(results1.keys())
      # print(results2.keys())
      #print()
      # assert len(from_set) == 1, f'Error in grouping {from_set}'
      assert len(set(results1.keys())&set(results2.keys())) == 0, f'Found same subjects in both splits: {results1} \n and \n {results2}'
      # assert len(results1) == len(results2), f'Error in grouping {results1} and {results2}'

      # if counter%100 == 0:
      #   print(results1)
      #   print('\n')
      #   print(results2)
      #   print('-'*20)
      #   print()

    #   for key, value in results1.items():
    #     if key.strip() and key.strip().lower().startswith('fw'):
    #       fw_counter += value

    #   for key, value in results2.items():
    #     if key.strip() and key.strip().lower().startswith('fw'):
    #       fw_counter += value

    # print(f'Found {fw_counter} messages starting with `fw` out of {counter} total messages.')

def inspect_subjects():
  df = pd.read_pickle(args.df_path)
  logging.info(df)

  #print(df['subject'].unique())
  print(df[ (df['subject'].isnull()) | (df['subject']=='') ].index)
  # print(df['subject'].unique().shape)


def inspect_authors():
  df = pd.read_pickle(args.df_path)
  logging.info(df)
  # print(df.loc[352790])

  # with open('/usr/local/src/iur/data/avacado/exp_20210809_2/authors.pickle', 'rb') as f:
  #   authors_map = pickle.load(f)
  #   #print(authors_map)

  with open('/usr/local/src/iur/data/avacado/exp_20210823_2/sender_history_split_0.ids') as split_ids_0_f, open('/usr/local/src/iur/data/avacado/exp_20210823_2/sender_history_split_1.ids') as split_ids_1_f:
    for author_idx, (line0, line1) in enumerate(zip(split_ids_0_f, split_ids_1_f)):

      if author_idx != 329:
        continue
      
      line0 = [int(x) for x in line0.split()]
      line1 = [int(x) for x in line1.split()]
      
      for idx0, idx1 in zip(random.sample(line0, 3), random.sample(line1, 3)):
        print(df.loc[idx0]['body'])
        print()
        print('-'*100)
        input()
        print(df.loc[idx1]['body'])
        print()
        print('-'*100)
        input()

      print()
      print('='*100)      

def curate_splits():
  df = pd.read_pickle(args.df_path)
  logging.info(df)

  # with open('/usr/local/src/iur/data/avacado/exp_20210809_2/authors.pickle', 'rb') as f:
  #   authors_map = pickle.load(f)
  #   #print(authors_map)

  with open('/usr/local/src/iur/data/avacado/exp_20210825_1/curated_history_0.ids') as split_ids_0_f, open('/usr/local/src/iur/data/avacado/exp_20210825_1/curated_history_1.ids') as split_ids_1_f:
    for i0 in split_ids_0_f:
      print('new row\n')
      for val in i0.strip().split():
        print(df.loc[int(val)]['from'])

    for i1 in split_ids_1_f:
      print('new row\n')
      for val in i1.strip().split():
        print(df.loc[int(val)]['from'])

    # for author_id in random_authors:
    #   record_idx = int(split_0[author_id].strip().split()[0])
    #   print(df.loc[record_idx]['from'])


selected_senders = [
'arun.kalasapudi@avocadoit.com',
'haider.kazmi@avocadoit.com',
'kant.hung@avocadoit.com',
'akarwal@avocadoit.com',
'miyuki.goldman@avocadoit.com',
'asun@avocadoit.com',
]





# 'dean.fulton@avocadoit.com',
# 'rvuong@avocadoit.com',
# 'traj@avocadoit.com',
# 'nmehta@avocadoit.com',
# 'jackie.valle@avocadoit.com',
# 'kchancellor@avocadoit.com',
# 'mkadanoff@avocadoit.com',
# 'dave.truman@avocadoit.com',
# 'madhava.gullapalli@avocadoit.com',
# 'james.larrue-baulch@avocadoit.com',
# 'kkerr@avocadoit.com',
# 'piyer@avocadoit.com',
# 'sswanson@avocadoit.com',
# 'darshan.patel@avocadoit.com',
# 'john.armbruster@avocadoit.com',
# 'speshkar@avocadoit.com',
# 'sreddy@avocadoit.com',
# 'mkadanoff@avocadoit.com',
# 'dswanson@avocadoit.com',
# 'ravikumar.palanisamy@avocadoit.com',
# 'sweller@avocadoit.com',
# 'rajeev@avocadoit.com'
#]

# 'avyas@avocadoit.com',

def create_subject_based_curated_split():
  df = pd.read_pickle(args.df_path)
  logging.info(df)

  # Match `re`
  re_re = re.compile(r'\bre\b:?')

  print(selected_senders)

  history = defaultdict(list)
  for sender, split in tqdm(df.groupby('from')):
    # handle case where the
    # sender is invalid
    if sender not in selected_senders:
      continue

    # for idx, row in split.iterrows():
    #   if row['body']

      try:
        time_sent = parsedate_to_datetime(row['date'])
        if not time_sent.tzinfo:
          time_sent = time_sent.replace(tzinfo=utc)

        subject = re_re.sub(lambda x: '', row['subject'].lower()).strip()

      except Exception:
        # Skip messages with error
        # while parsing date.
        continue

      if len(subject.strip()) == 0:
        continue
      elif subject.strip().lower().startswith('fw'):
        continue

      history[sender].append((idx, time_sent, subject))

  split_ids = defaultdict(list)
  sender_count = 0
  skipped_sender_count = 0
  for sender, sender_meta in tqdm(history.items()):
    sender_count += 1

    # Skip cases where there are fewer
    # than `min_episode_length` messages
    if len(sender_meta) < 2*args.min_episode_length:
      skipped_sender_count += 1
      continue

    sender_meta = set(sender_meta)
    split1 = set()
    split2 = set()
    counter = 0
    while len(sender_meta) > 0:
      primary_item = sender_meta.pop()
      counter += 1

      if counter%2 == 0:
        split2.add(primary_item)
      else:
        split1.add(primary_item)

      for item in sender_meta:
        if (1 - levenshtein(primary_item[2], item[2])/max(len(primary_item[2]), len(item[2]))) >= 0.75:
          if counter%2 == 0:
            split2.add(item)
          else:
            split1.add(item)
      
      sender_meta -= split1
      sender_meta -= split2

    # Split 2-way the idx
    split_ids[0].append([s[0] for s in split1])
    split_ids[1].append([s[0] for s in split2])

  # check_split_ids(split_ids, df)
  assert len(split_ids[0]) == len(split_ids[1]), 'Error while splitting the data!'

  counter_out = 0
  counter_in = 0
  keep = 'n'
  with open(f'{args.output_prefix}_0.ids.run', 'w') as history_file_0, open(f'{args.output_prefix}_1.ids.run', 'w') as history_file_1:
    for ids0, ids1 in zip(split_ids[0], split_ids[1]):
      if len(ids0) >= args.min_episode_length and \
         len(ids1) >= args.min_episode_length:
        
        print('*'*20)
        print('Save query IDs')
        print('*'*20)
        counter_in = 0
        for num in ids0:

          if counter_in > 15:
            break

          print(df.loc[num]['body'])
          print('='*20)
          keep = input('Keep? (y/n)')
          print('='*20)
          print()
          print()

          if keep == 'y' or keep == '':
            history_file_0.write(str(num) + ' ')
            counter_in += 1

        history_file_0.write('\n')
        history_file_0.flush()
        
        print('*'*20)
        print('Save target IDs')
        print('*'*20)
        counter_in = 0
        for num in ids1:

          if counter_in > 15:
            break

          print(df.loc[num]['body'])
          print('='*20)
          keep = input('Keep? (y/n)')
          print('='*20)
          print()
          print()

          if keep == 'y' or keep == '':
            history_file_1.write(str(num) + ' ')
            counter_in += 1

        history_file_1.write('\n')
        history_file_1.flush()

        # history_file_1.write(' '.join([str(num) for num in ids1]) + '\n')


def main(argv):
  #logging.info(f"Inspecting DF {args.df_path}")
  #inspect_jsonl('/usr/local/src/iur/data/avacado/avacado-dataset-split-0.jsonl')
  #inspect_jsonl('/usr/local/src/iur/data/avacado/avacado-dataset-split-1.jsonl')
  #inspect_df()
  
  inspect_splits()
  # inspect_subjects()
  # inspect_authors()
  # curate_splits()

  # create_subject_based_curated_split()

  # shuffle_splits()

if __name__ == "__main__":
  app.run(main)
