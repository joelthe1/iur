## Notes

# Setup
- I copied over the pre-trained sentencepiece model from /usr/local/share/data/reddit/iur-master/data/reddit/data/output/train/65536_unigram.model to the location data/reddit/pretrained/model mainly due to permissions issues for writting out to the directory under 'iur-master'- which is needed when encoding a new dataset.

#### Commands
- Convert full TA3 data to JSON. This step also creates and pickles a DataFrame of the data at `--df` 
```sh
python scripts/ta3_preprocess.py --input_dir data/sender-based-ta3-subject-weeks1-13 --df data/sender-based-ta3-subject-weeks1-13-df-pickle.zip --num_headers 3 --ids data/output/sender_history.ids --config data/sender-based-ta3-subject-weeks1-13/config.json --output_dir data/output/ta3_json --model_dir data/reddit/pretrained/model
```

- Split out messages IDs by date sent. This is for creating queries and target datasets.
```sh
python scripts/split_messages.py --df data/sender-based-ta3-subject-weeks1-13-df-pickle.zip --min_episode_length 16 --output_prefix data/output/sender_history_split
```

- Convert the query and target IDs related data to JSON
```sh
# For queries
python scripts/ta3_preprocess.py --input_dir data/sender-based-ta3-subject-weeks1-13 --num_headers 3 --df data/sender-based-ta3-subject-weeks1-13-df-pickle.zip --ids data/output/sender_history_split_0.ids --config data/sender-based-ta3-subject-weeks1-13/config.json --output_dir data/output/ta3_json/queries --model_dir data/reddit/pretrained/model

# For targets
python scripts/ta3_preprocess.py --input_dir data/sender-based-ta3-subject-weeks1-13 --num_headers 3 --df data/sender-based-ta3-subject-weeks1-13-df-pickle.zip --ids data/output/sender_history_split_1.ids --config data/sender-based-ta3-subject-weeks1-13/config.json --output_dir data/output/ta3_json/targets --model_dir data/reddit/pretrained/model
```

- Convert the JSONs to TFRecords
```sh
# For queries
python scripts/json2tf.py --json data/output/ta3_json/queries/examples.json --tf data/output/ta3_tf/queries --config data/sender-based-ta3-subject-weeks1-13/config.json  --shard_size 5000 --max_length 32

# For targets
python scripts/json2tf.py --json data/output/ta3_json/targets/examples.json --tf data/output/ta3_tf/targets --config data/sender-based-ta3-subject-weeks1-13/config.json  --shard_size 5000 --max_length 32
```

- Run `rank`
```sh
python 
```