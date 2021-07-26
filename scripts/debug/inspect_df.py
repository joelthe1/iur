import json
import pandas as pd

from absl import logging
from absl import flags
from absl import app


flags.DEFINE_string('df_path', None, 'Path to DataFrame')
flags.mark_flags_as_required(['df_path'])

args = flags.FLAGS

def inspect_df():
  df = pd.read_pickle(args.df_path)
  logging.info(df)

  # with open('/usr/local/src/iur/data/output/sender_history_split_0.ids') as ids_0, \
  #      open('/usr/local/src/iur/data/output/sender_history_split_1.ids') as ids_1:
  #   for query_id_line, target_id_line in zip(ids_0, ids_1):
  #     for query, target in zip(query_id_line.split(), target_id_line.split()):
  #       logging.info(f"{df.loc[int(query)]['from']} ******** {df.loc[int(query)]['from']}")

  print(df[ (df['date'].isnull()) & (df['date']=='') ].index)
    


def main(argv):
  logging.info(f"Inspecting DF {args.df_path}")
  inspect_df()

if __name__ == "__main__":
  app.run(main)
