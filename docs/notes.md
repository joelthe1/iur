## Notes

#### Commands
- Convert TA3 data to JSON
```
python scripts/ta3_preprocess.py --input_dir data/sender-based-ta3-subject-weeks1-13 --num_headers 3 --ids data/output/sender_history.ids --config data/sender-based-ta3-subject-weeks1-13/config.json --output_dir data/output/ta3_json --model_dir data/output/models
```