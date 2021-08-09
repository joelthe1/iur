import os
import re
import math
from collections import defaultdict
from datetime import datetime
from operator import itemgetter
from tqdm import tqdm

import cProfile
import pstats
from functools import wraps

from fuzzywuzzy import fuzz
from polyleven import levenshtein

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

def create_timebased_sender_history(df, num_splits=2):
  '''Read the DataFrame and create the history'''
  # Currently only supports 2-way splits
  assert num_splits == 2, 'Incompatible value for num_splits. Currently only supports 2-way splits'
  
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


def profile(output_file=None, sort_by='cumulative', lines_to_print=None, strip_dirs=False):
    """A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/
    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + '.prof'
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, 'w') as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner


#@profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
def create_subject_based_split(df, num_splits=2):
  '''Read the DataFrame and create the history based on subject'''
  # Currently only supports 2-way splits
  assert num_splits == 2, 'Incompatible value for num_splits. Currently only supports 2-way splits'

  # Match `re`
  re_re = re.compile(r'\bre\b:?')

  history = defaultdict(list)
  for sender, split in tqdm(df.groupby('from')):
    # handle case where the
    # sender is invalid
    if sender == '<>':
      continue

    for idx, row in split.iterrows():
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

  # TODO: Handle case when there are not enough (`min_episode_length`)
  # messages in each split. Probably randomly splitting them would be 
  # good enough. Should display a count of how many were split in such a way.
  counter_in = 0
  counter_out = 0

  with open(f'{args.output_prefix}_0.ids', 'w') as history_file_0, open(f'{args.output_prefix}_1.ids', 'w') as history_file_1:
    for ids0, ids1 in zip(split_ids[0], split_ids[1]):
      if len(ids0) >= args.min_episode_length and \
         len(ids1) >= args.min_episode_length:
        history_file_0.write(' '.join([str(num) for num in ids0]) + '\n')
        history_file_1.write(' '.join([str(num) for num in ids1]) + '\n')
        counter_in += 1
      else:
        counter_out += 1

  logging.info(f'Found {counter_in} authors with enough unique splits and {counter_out} without.')


  # # Write out into `num_splits` number of files
  # with open(f'{args.output_prefix}_0.ids', 'w') as history_file:
  #   for value in split_ids[0]:
  #     history_file.write(' '.join([str(num) for num in value]) + '\n')

  # with open(f'{args.output_prefix}_1.ids', 'w') as history_file:
  #   for value in split_ids[1]:
  #     history_file.write(' '.join([str(num) for num in value]) + '\n')

  logging.info(f'Found total {sender_count} senders and kept {sender_count - skipped_sender_count} out of them (skipped {skipped_sender_count} due to not having at least {2*args.min_episode_length} messages sent by them).')


def main(argv):
  logging.info("Starting to split out IDs...")

  if not os.path.exists(args.df):
    logging.info(f"Did not find an existing pickled DataFrame file at {args.df}")
    exit(1)
  
  logging.info(f"Reading pickled DataFrame file found at {args.df}")
  df = pd.read_pickle(args.df)

  create_subject_based_split(df)
  logging.info(f"Done. See files {args.output_prefix}_0.ids and {args.output_prefix}_1.ids")

if __name__ == "__main__":
  app.run(main)
