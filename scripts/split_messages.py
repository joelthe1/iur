import os
import math
from collections import defaultdict
from datetime import datetime
from operator import itemgetter

from email.utils import parsedate_to_datetime
from pytz import utc

import pandas as pd

from absl import logging
from absl import flags
from absl import app

args = flags.FLAGS

flags.DEFINE_string('df', None, 'Path to pickled DataFrame')
flags.DEFINE_string('output_prefix', '.', 'Prefix to use for the split out ID files')
flags.DEFINE_integer('min_episode_length', 2, 'The minimum number of messages (aka episodes) present per author per split so as to be considered for splitting.')
flags.mark_flags_as_required(['df'])

def create_sender_history(df, num_splits=2):
  '''Read the DataFrame and create the history'''
  # Currently only supports 2-way splits
  assert num_splits == 2, 'Incompatible value. Currently only supports 2-way splits'
  
  history = defaultdict(list)
  for sender, split in df.groupby('from'):
    for idx, row in split.iterrows():
      try:
        time_sent = parsedate_to_datetime(row['date'])
        if not time_sent.tzinfo:
          time_sent = time_sent.replace(tzinfo=utc)
      except Exception:
        # Skip messages with error
        # while parsing date.
        continue

      history[sender].append((idx, time_sent))
    
  split_ids = defaultdict(list)
  sender_count = 0
  skipped_sender_count = 0
  for sender, sent_times in history.items():
    sorted_idx = []
    sender_count += 1

    # Skip cases where there are fewer
    # than `min_episode_length` messages
    if len(sent_times) < 2*args.min_episode_length:
      skipped_sender_count += 1
      continue

    for entry in sorted(sent_times, key=itemgetter(1)):
      sorted_idx.append(entry[0])

    # Split 2-way the idx
    split_idx = math.trunc(len(sorted_idx)/2)
    split_ids[0].append((sorted_idx[:split_idx]))
    split_ids[1].append((sorted_idx[split_idx:]))

  assert len(split_ids[0]) == len(split_ids[1]), 'Error while splitting the data!'

  # Write out into `num_splits` number of files
  with open(f'{args.output_prefix}_0.ids', 'w') as history_file:
    for value in split_ids[0]:
      history_file.write(' '.join([str(num) for num in value]) + '\n')

  with open(f'{args.output_prefix}_1.ids', 'w') as history_file:
    for value in split_ids[1]:
      history_file.write(' '.join([str(num) for num in value]) + '\n')

  logging.info(f'Found total {sender_count} senders and kept {sender_count - skipped_sender_count} out of them (skipped {skipped_sender_count} due to not having at least {2*args.min_episode_length} messages sent by them).')


def main(argv):
  logging.info("Starting to split out IDs...")

  if not os.path.exists(args.df):
    logging.info(f"Did not find an existing pickled DataFrame file at {args.df}")
    exit(1)
  
  logging.info(f"Reading pickled DataFrame file found at {args.df}")
  df = pd.read_pickle(args.df)

  create_sender_history(df)
  logging.info(f"Done. See files {args.output_prefix}_0.ids and {args.output_prefix}_1.ids")

if __name__ == "__main__":
  app.run(main)
