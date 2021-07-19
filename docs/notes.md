## Notes

#### Commands
- Convert TA3 data to JSON
```sh
python scripts/ta3_preprocess.py --input_dir data/sender-based-ta3-subject-weeks1-13 --num_headers 3 --ids data/output/sender_history.ids --config data/sender-based-ta3-subject-weeks1-13/config.json --output_dir data/output/ta3_json --model_dir data/output/models
```

- Convert TA3 JSON to TFRecords
```sh
python scripts/json2tf.py --json data/output/ta3_json/examples.json --tf data/output/ta3_tf/ta3 --config data/sender-based-ta3-subject-weeks1-13/config.json  --shard_size 5000 --max_length 32
```

- Split out messages IDs by date sent
```sh
python scripts/split_messages.py --input_dir data/sender-based-ta3-subject-weeks1-13 --num_headers 3 --output_prefix data/output/sender_history_split
```