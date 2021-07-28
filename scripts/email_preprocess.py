"""The input consists of textual content from email messages and 
we output preprocessed data (features) to a JSON file. The JSON file 
may then be converted to TFRecords as a separate step.

Note that for simplicity all preprocessing happens in-memory. This
limits the size of datasets that may be preprocessed using this
script.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from operator import itemgetter
import os
import pickle
import tempfile
import pandas as pd
import gzip
try:
  import ujson as json
except ImportError:
  print("Install the `ujson` pip package to speed things up.")
  import json
from collections import Counter
from collections import defaultdict

from tqdm import tqdm

import sentencepiece as spm

from absl import logging
from absl import flags
from absl import app

from email import (
    policy,
    message_from_file
)
from email.utils import parsedate_to_datetime
from pytz import timezone, utc

from aid.features import F
from aid.features import FeatureConfig
from aid.features import write_tfrecords_from_generator

args = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'Path to directory with email content')
flags.DEFINE_string('df', None, 'Path to pickled DataFrame. Uses it if it exists, else creates and saves on here from the data in the `input_dir`.')
flags.DEFINE_string('ids', None, 'Path to user history file')
flags.DEFINE_string('output_dir', '.', 'Output directory')
flags.DEFINE_string('model_dir', '.', 'Model directory')
flags.DEFINE_integer('num_headers', 0, 'Number of header lines to extract in the message text')
flags.DEFINE_string('json_filename', 'examples.json', 'Output JSON file name.')
flags.DEFINE_string('config', 'reddit.json', 'Experiment configuration')
flags.DEFINE_string('unk_subject', '<unk>', 'Name of unknown subject')
flags.DEFINE_string('model_prefix', 'model', 'Prefix for subword model files')
flags.DEFINE_string('model_type', 'unigram', 'Model type')
flags.DEFINE_string('subreddit_path', '.', 'Path to subreddit pickle')
flags.DEFINE_float('character_coverage', 1.0, 'Character coverage')
flags.DEFINE_integer('input_sentence_size', 1000,
                     'Number of sentences used to fit subword model')
flags.DEFINE_integer('pad_id', 0, 'Padding ID')
flags.DEFINE_integer('bos_id', -1, 'BOS ID')
flags.DEFINE_integer('eos_id', 1, 'EOS ID')
flags.DEFINE_integer('unk_id', 2, 'Unk ID')
flags.DEFINE_float('min_ascii_fraction', 0.75,
                   'Filter messages with less than this fraction of ASCII')
flags.DEFINE_integer('min_chars', 1, 'Minimum comment length')
flags.DEFINE_integer('min_subwords', 10, 'Minimum number of subwords')
flags.DEFINE_string('text_key', 'body', 'Column name for text field')
flags.DEFINE_string('subjects_key', 'subject', 'Column name for message subject')
flags.DEFINE_integer('n_to_print', 1, 'Number of comments to print to console')
flags.DEFINE_string('sample_file_path', None, 'Path to JSON lines file with sample indices')
flags.DEFINE_integer('min_episode_length', 1, 'The minimum number of messages (aka episodes) present per author to be kept')

flags.mark_flags_as_required(['input_dir', 'df', 'ids'])


def get_hour_from_timestamp(timestamp):
  timestamp = parsedate_to_datetime(timestamp)
  if not timestamp.tzinfo:
    timestamp = timestamp.replace(tzinfo=utc)

  return timestamp.hour


def keep_message(text, min_ascii_fraction=0.75, min_length=1):
  """ For purposes of vocabulary creation ignore non-ascii documents """
  len_total = len(text)
  if len_total < min_length:
    return False
  len_ascii = sum(c.isalpha() for c in text)
  frac_ascii = float(len_ascii) / float(len_total)
  if frac_ascii < min_ascii_fraction:
    return False
  return True


def fit_subword_vocabulary(df):
  """ Fit subword vocabulary, see https://arxiv.org/abs/1808.06226"""
  config = FeatureConfig.from_json(args.config)
  
  model_prefix = os.path.join(
    args.model_dir,
    f"{config.num_symbols}_{args.model_type}")

  if os.path.exists(model_prefix + ".model"):
    logging.info(f"Subword model {model_prefix}.model already exists; using that")
    return
  else:
    logging.info(f"Fitting new subword vocabulary of size {config.num_symbols}")
  
  with tempfile.NamedTemporaryFile(mode='w+t') as temp:
    index = 0
    logging.info(f"Writing text content to {temp.name}")
    for key, row in df.iterrows():
      text = row[args.text_key]
      if keep_message(
          text,
          min_ascii_fraction=args.min_ascii_fraction,
          min_length=args.min_chars):
        text = text.replace('\n', ' ')
        temp.write(text + '\n')
      if index > args.input_sentence_size:
        break
      index += 1

    temp.flush()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Creating Vocabulary with SentencePiece...")

    trainer_args = [
      f'--input={temp.name}',
      f'--model_prefix={model_prefix}',
      f'--vocab_size={config.num_symbols}',
      f'--hard_vocab_limit=false',
      f'--model_type={args.model_type}',
      f'--character_coverage={args.character_coverage}',
      f'--input_sentence_size={args.input_sentence_size}',
      f'--shuffle_input_sentence=true',
      f'--pad_id={args.pad_id}',
      f'--eos_id={args.eos_id}',
      f'--bos_id={args.bos_id}',
      f'--unk_id={args.unk_id}'
    ]
    logging.info("Fitting SentencePiece model")
    spm.SentencePieceTrainer.Train(' '.join(trainer_args))


def fit_subject_vocab(df):
  config = FeatureConfig.from_json(args.config)
  message_subject_map_path = os.path.join(
    args.model_dir,
    f'{config.num_action_types}_subjects.pickle')
  if os.path.exists(message_subject_map_path):
    logging.info(f"Using existing messages map: {message_subject_map_path}")
    return
  logging.info("Creating subjects map")
  logging.info("Obtaining unique subjects")
  subjects = df[args.subjects_key]
  counts = Counter(subjects)
  most_common = counts.most_common()
  logging.info("Most common subjects:")
  for sr, count in most_common[:10]:
    logging.info(f"  {sr} {count}")
  output_map = {}
  for i, sr in enumerate([x for x, _ in most_common[:config.num_action_types-1]]):
    output_map[sr] = i
  assert args.unk_subject not in output_map
  output_map[args.unk_subject] = len(output_map)
  assert len(output_map) == config.num_action_types
  logging.info(f"Kept {len(output_map)} subjects")
  logging.info(f"Saving subjects map to: {message_subject_map_path}")
  with open(message_subject_map_path, 'wb') as f:
    pickle.dump(output_map, f)


def fit_author_vocab(df):
  """ Keep track of author IDs """
  author_map_path = os.path.join(
    args.output_dir, 'authors.pickle')
  if os.path.exists(author_map_path):
    logging.info(f"Using existing author map: {author_map_path}")
    return
  author_map = {}
  for i, a in enumerate(set(df['from'])):
    author_map[a] = i
  logging.info(f"{len(author_map)} authors")
  with open(author_map_path, 'wb') as f:
    pickle.dump(author_map, f)


def print_examples(df, print_if_less_than=15):
  config = FeatureConfig.from_json(args.config)
  model_path = os.path.join(
    args.model_dir,
    f"{config.num_symbols}_{args.model_type}.model")
  sp = spm.SentencePieceProcessor()
  sp.Load(model_path)
  logging.info(f"Piece size: {sp.GetPieceSize()}")
  n_printed = 0
  for index, row in df.iterrows():
    raw_text = row[args.text_key]
    if len(raw_text.split()) < print_if_less_than:
      logging.info(raw_text)
      pieces = sp.EncodeAsPieces(raw_text)
      logging.info(" ".join(
        [f"{piece}:{sp.PieceToId(piece)}" for piece in pieces]))
      n_printed += 1
    if n_printed > args.n_to_print:
      break


def maybe_load_sample_file():
  if args.sample_file_path is None:
    return None
  samples = {}
  logging.info(f"Loading sample file: {args.sample_file_path}")
  with open(args.sample_file_path) as fh:
    for line in fh:
      sample = json.loads(line)
      samples[sample['from']] = sample
  assert samples
  return samples


def write_json(df):
  json_path = os.path.join(args.output_dir, args.json_filename)
  if os.path.exists(json_path):
    logging.info(f'{json_path} exists; delete to remake')
    return
  samples = maybe_load_sample_file()
  config = FeatureConfig.from_json(args.config)
  model_path = os.path.join(
    args.model_dir,
    f"{config.num_symbols}_{args.model_type}.model")
  sp = spm.SentencePieceProcessor()
  sp.Load(model_path)
  subjects_map_path = os.path.join(
    args.model_dir,
    f'{config.num_action_types}_subjects.pickle')
  with open(subjects_map_path, 'rb') as fh:
    logging.info(f"Loading subjects map: {subjects_map_path}")
    subjects_map = pickle.load(fh)
  author_map_path = os.path.join(
    args.output_dir, 'authors.pickle')
  with open(author_map_path, 'rb') as fh:
    logging.info(f"Loading author map: {author_map_path}")
    author_map = pickle.load(fh)
  logging.info(f"Writing preprocessed data to: {json_path}")
  N = len(open(args.ids).readlines())
  with open(json_path, 'w') as fout, \
       open(args.ids, 'r') as ids_file:
    for line in tqdm(ids_file, total=N):
      message_ids = line.split()
      first_id = int(message_ids[0])
      author = df.loc[first_id]['from']
      if samples:
        if author not in samples:
          continue
        sample = samples[author]
        assert len(message_ids) == sample['num_actions_total']
        start_index = sample['start_index']
        length = sample['episode_length']
        message_ids = message_ids[start_index:start_index+length]
        assert len(message_ids) == length
      history = {
        F.SYMBOLS.value: [],
        F.HOUR.value: [],
        F.ACTION_TYPE.value: [],
        F.AUTHOR_ID.value: author_map[author]
      }
      for id_ in message_ids:
        id_ = int(id_)
        message = df.loc[id_]
        history[F.SYMBOLS.value].append(sp.EncodeAsIds(message[args.text_key]))
        history[F.HOUR.value].append(
          get_hour_from_timestamp(message['date']))
        subject_index = subjects_map[args.unk_subject]
        if message['subject'] in subjects_map:
          subject_index = subjects_map[message['subject']]
        history[F.ACTION_TYPE.value].append(subject_index)

      fout.write(json.dumps(history) + '\n')

def read_email_messages():
  '''Read in the email messages and return it as a DataFrame'''
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
          data[header_name.strip().lower()].append(header_value.strip())

      # Remove headers from the rest of the content
      content = ''.join(content[args.num_headers:])
      data[args.text_key].append(content)
  
  # Expecing this sort to be consistent across runs
  return pd.DataFrame(data).sort_values(by=['date', 'from']).reset_index(drop=True)


def create_sender_history(df):
  '''Read the DataFrame and create the history'''
  history = defaultdict(list)
  for sender, split in df.groupby('from'):
    for idx, row in split.iterrows():
      try:
        time_sent = parsedate_to_datetime(row['date'])
        if not time_sent.tzinfo:
          time_sent = time_sent.replace(tzinfo=utc)
      except Exception as e:
        # Skip messages with error
        # while parsing date
        continue

      history[sender].append((idx, time_sent))
    
  skipped_authors_count = 0
  with open(args.ids, 'w') as history_file:
    for sender, sent_times in history.items():

      # Only keep authors with `min_episode_length`
      if len(sent_times) < args.min_episode_length:
        skipped_authors_count += 1
        continue

      sorted_idx = []
      for entry in sorted(sent_times, key=itemgetter(1)):
        sorted_idx.append(entry[0])

      history_file.write(' '.join([str(num) for num in sorted_idx]) + '\n')

  if skipped_authors_count > 0:
    logging.info(f'Skipped {skipped_authors_count} authors since not having at least {args.min_episode_length} messages sent by them')


def main(argv):
  logging.info(f"Output directory: {args.output_dir}")
  os.makedirs(args.output_dir, exist_ok=True)

  # If `df` given, then use it
  # else create from `input_dir`
  if not os.path.exists(args.df):
    logging.info(f"Did not find an existing pickled DataFrame file. Creating one at {args.df}")    
    df = read_email_messages()
    df.to_pickle(args.df)
  else:
    logging.info(f"Using existing pickled DataFrame file found at {args.df}")
    df = pd.read_pickle(args.df)

  logging.info(f'\n{df}')
  
  # Use existing history file 
  # if it exists, else create one
  if not os.path.exists(args.ids):
    logging.info(f"Did not find existing history file. Creating one at {args.ids}")    
    create_sender_history(df);
  else:
    logging.info(f"Using existing history file found at {args.ids}")

  fit_subword_vocabulary(df)
  print_examples(df)
  fit_subject_vocab(df)
  fit_author_vocab(df)
  write_json(df)


if __name__ == "__main__":
  app.run(main)
