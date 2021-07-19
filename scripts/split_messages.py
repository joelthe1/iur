import os
import math
from collections import defaultdict
from datetime import datetime
from operator import itemgetter

from email.utils import parsedate_to_datetime
from pytz import timezone, utc

import pandas as pd

from absl import logging
from absl import flags
from absl import app

args = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'Path to directory with email content')
flags.DEFINE_integer('num_headers', 0, 'Number of header lines to extract in the message text')
flags.DEFINE_string('output_prefix', '.', 'Prefix to use for the split out ID files')
flags.DEFINE_string('text_key', 'body', 'Column name for text field')
flags.mark_flags_as_required(['input_dir'])

def read_ta3_messages():
  '''Read in the TA3 messages and return it as a DataFrame'''
  data = defaultdict(list)
  for root, dirs, files in os.walk(args.input_dir):
    for file in files:
      if not file.endswith('.txt'):
        continue
      
      with open(os.path.join(root, file)) as f:
        content = f.readlines()

      # Read the headers in the file
      for idx, line in enumerate(content):
        if idx < args.num_headers:
          header_name, header_value = [part.strip() for part in line.strip().split(':', 1)]
          data[header_name.lower()].append(header_value)

      # Remove headers from the rest of the content
      content = ''.join(content[args.num_headers:])
      data[args.text_key].append(content)

  return pd.DataFrame(data)


def create_sender_history(df, num_splits=2):
  '''Read the DataFrame and create the history'''
  # Currently only supports 2-way splits
  assert num_splits == 2, 'Incompatible value. Currently only supports 2-way splits'
  
  history = {}
  for sender, split in df.groupby('from'):
    for idx, row in split.iterrows():
      time_sent = parsedate_to_datetime(row['date'])
      if not time_sent.time().tzinfo:
        time_sent.replace(tzinfo=utc)

      if sender not in history:
        history[sender] = [(idx, time_sent)]
      else:
        history[sender].append((idx, time_sent))
    
  split_ids = defaultdict(list)
  for sender, sent_times in history.items():
    sorted_idx = []
    for entry in sorted(sent_times, key=itemgetter(1)):
      sorted_idx.append(entry[0])

    # Skip cases where there are fewer
    # than two messages
    if len(sorted_idx) < 2:
      continue

    # Split 2-way the idx
    split_idx = math.trunc(len(sorted_idx)/2)
    split_ids[0].append((sorted_idx[:split_idx]))
    split_ids[1].append((sorted_idx[split_idx:]))

  # Write out into `num_splits` number of files
  with open(f'{args.output_prefix}_0.ids', 'w') as history_file:
    for value in split_ids[0]:
      history_file.write(' '.join([str(num) for num in value]) + '\n')

  with open(f'{args.output_prefix}_1.ids', 'w') as history_file:
    for value in split_ids[1]:
      history_file.write(' '.join([str(num) for num in value]) + '\n')


def main(argv):
  logging.info("Starting to split out IDs...")
  #os.makedirs(args.output_dir, exist_ok=True)
  df = read_ta3_messages()
  create_sender_history(df)
  logging.info(f"Done. See files {args.output_prefix}_0.ids and {args.output_prefix}_1.ids")

if __name__ == "__main__":
  app.run(main)
