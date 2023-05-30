# Datadir
--data_dir param is folder with competition data
Example:
`"../data/"`

# Install requirements

`pip install -r requirements.txt`

## Next steps execute from folder `pipeline` so:

`cd ./pipeline`

# Train model for embeddings ~ 1h on gpu

`python minilm_finetuning.py --data_dir="../data/"`

# Preprocess dataset

`python preprocess.py` --data_dir="../data/"`

# Inference


`python inference.py`