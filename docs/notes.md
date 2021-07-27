## Notes

# Setup
- I copied over the pre-trained sentencepiece model from /usr/local/share/data/reddit/iur-master/data/reddit/data/output/train/65536_unigram.model to the location data/reddit/pretrained/model mainly due to permissions issues for writting out to the directory under 'iur-master'- which is needed when encoding a new dataset.

#### Commands
- Convert full TA3 data to JSON. This step is mainly for creating and pickling a DataFrame of the data at `--df`. You should remove all other generated files from this step.
```sh
python scripts/email_preprocess.py --input_dir data/ta3/sender-based-ta3-subject-weeks1-13 --df data/ta3/sender-based-ta3-subject-weeks1-13-df-pickle.zip --num_headers 3 --ids data/ta3/sender_history.ids --config data/ta3/sender-based-ta3-subject-weeks1-13/config.json --output_dir data/ta3 --json_filename ta3-dataset.jsonl --model_dir data/reddit/pretrained/model
```
```sh
python scripts/email_preprocess.py --input_dir data/avacado/from_text_3 --df data/avacado/avacado-df-pickle.zip --num_headers 3 --ids data/avacado/sender_history.ids --config data/avacado/from_text_3/config.json --output_dir data/avacado --json_filename avacado-dataset.jsonl --model_dir data/reddit/pretrained/model
```

- Split out messages IDs by date sent. This is for creating queries and target datasets.
```sh
python scripts/split_messages.py --df data/ta3/sender-based-ta3-subject-weeks1-13-df-pickle.zip --min_episode_length 16 --output_prefix data/ta3/sender_history_split
```
```sh
python scripts/split_messages.py --df data/avacado/avacado-df-pickle.zip --min_episode_length 16 --output_prefix data/avacado/sender_history_split
```

- Convert the query and target IDs related data to JSON
  It is important to use the same author vocabulary across the two invocations here.
```sh
# For queries
python scripts/email_preprocess.py --input_dir data/ta3/sender-based-ta3-subject-weeks1-13 --num_headers 3 --df data/ta3/sender-based-ta3-subject-weeks1-13-df-pickle.zip --ids data/ta3/sender_history_split_0.ids --config data/ta3/sender-based-ta3-subject-weeks1-13/config.json --output_dir data/ta3 --json_filename ta3-dataset-split-0.jsonl --model_dir data/reddit/pretrained/model

# For targets
python scripts/email_preprocess.py --input_dir data/ta3/sender-based-ta3-subject-weeks1-13 --num_headers 3 --df data/ta3/sender-based-ta3-subject-weeks1-13-df-pickle.zip --ids data/ta3/sender_history_split_1.ids --config data/ta3/sender-based-ta3-subject-weeks1-13/config.json --output_dir data/ta3 --json_filename ta3-dataset-split-1.jsonl --model_dir data/reddit/pretrained/model
```
```sh
# For queries
python scripts/email_preprocess.py --input_dir data/avacado/from_text_3 --df data/avacado/avacado-df-pickle.zip --num_headers 3 --ids data/avacado/sender_history_split_0.ids --config data/avacado/from_text_3/config.json --output_dir data/avacado --json_filename avacado-dataset-split-0.jsonl --model_dir data/reddit/pretrained/model

# For targets
python scripts/email_preprocess.py --input_dir data/avacado/from_text_3 --df data/avacado/avacado-df-pickle.zip --num_headers 3 --ids data/avacado/sender_history_split_1.ids --config data/avacado/from_text_3/config.json --output_dir data/avacado --json_filename avacado-dataset-split-1.jsonl --model_dir data/reddit/pretrained/model
```

- Convert the JSONs to TFRecords
```sh
# For queries
python scripts/json2tf.py --json data/ta3/ta3-dataset-split-0.jsonl --tf data/ta3/split-0 --config data/ta3/sender-based-ta3-subject-weeks1-13/config.json  --shard_size 5000 --max_length 32
# For targets
python scripts/json2tf.py --json data/ta3/ta3-dataset-split-1.jsonl --tf data/ta3/split-1 --config data/ta3/sender-based-ta3-subject-weeks1-13/config.json  --shard_size 5000 --max_length 32
```
```sh
# For queries
python scripts/json2tf.py --json data/avacado/avacado-dataset-split-1.jsonl --tf data/avacado/split-0 --config data/avacado/from_text_3/config.json --shard_size 5000 --max_length 32

# For targets
python scripts/json2tf.py --json data/avacado/avacado-dataset-split-1.jsonl --tf data/avacado/split-1 --config data/avacado/from_text_3/config.json --shard_size 5000 --max_length 32
```

- Run `rank`
```sh
source .venv/bin/activate
python scripts/fit.py --mode rank --expt_config_path /usr/local/share/data/reddit/iur-master/data/reddit/config.json --expt_dir /usr/local/share/data/reddit/iur-master/experiment --train_tfrecord_path "data/ta3/split-0.*.tf" --valid_tfrecord_path "data/ta3/split-1.*.tf" --results_filename data/ta3/results.txt
```
```sh
source .venv/bin/activate
python scripts/fit.py --mode rank --expt_config_path /usr/local/share/data/reddit/iur-master/data/reddit/config.json --expt_dir /usr/local/share/data/reddit/iur-master/experiment --train_tfrecord_path "data/avacado/split-0.*.tf" --valid_tfrecord_path "data/avacado/split-1.*.tf" --results_filename data/avacado/results.txt
```