# Data folder

* DATA_DIR="Path_to_folder"

# Install requirements

`pip install -r requirements.txt`

## Next steps execute from folder `pipeline` so:

`cd ./pipeline`

# Prepare embeddings

`python minilm_finetuning.py --data_dir="../data/"`

# Preprocess dataset


`python preprocess.py`

# Inference


`python inference.py`