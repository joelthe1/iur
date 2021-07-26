import tensorflow as tf 
import json
from google.protobuf.json_format import MessageToJson

from absl import logging
from absl import flags
from absl import app


flags.DEFINE_string('tfrecord_path', None, 'Path to TFRecord')
flags.mark_flags_as_required(['tfrecord_path'])

args = flags.FLAGS

def inspect_tfrecord():
  raw_dataset = tf.data.TFRecordDataset(args.tfrecord_path)
  
  for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    logging.info(example)
    #json_data = json.loads(MessageToJson(example))
    #logging.info(json_data['features'])


def main(argv):
  logging.info(f"Inspecting TFRecord {args.tfrecord_path}")
  inspect_tfrecord()

if __name__ == "__main__":
  app.run(main)
