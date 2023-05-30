import warnings

import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from utils import preprocess_text, embed_model, labse_encode


tqdm.pandas()
warnings.filterwarnings("ignore")
pd.set_option('use_inf_as_na', True)

LaBSEtokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
LaBSE_model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

RuBERTtokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
RuBERT_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
minilm = SentenceTransformer('/home/ailab_user/hackathon-all-MiniLM-L6-v2-2023-05-20_17-54-44', device=0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Current device:', device)
train = pd.read_parquet("../../data/hackathon_files_for_participants_ozon/train_data.parquet")  # kate_hackathon_data/train_data_trained_minilm.parquet
test = pd.read_parquet("../../data/hackathon_files_for_participants_ozon/test_data.parquet")  # kate_hackathon_data/test_data_trained_minilm.parquet
train.dropna(subset=['name'], inplace=True)
train['Train'] = True
test['Train'] = False

etl = pd.concat([train, test])

etl['clean_name'] = etl['name'].progress_apply(preprocess_text)
etl['name_labse'] = etl['name'].progress_apply(lambda text: labse_encode(text, LaBSE_model, LaBSEtokenizer, 128, device=device)[0])
etl['name_rubert_tiny'] = etl['name'].progress_apply(lambda text: embed_model(text, RuBERT_model , RuBERTtokenizer, 128, device=device)[0])
etl['name_minilm'] = etl['name'].progress_apply(lambda text: minilm.encode(text))

etl[['variantid', 
     'clean_name', 
     'name_rubert_tiny', 
     'name_minilm', 
     'name_labse', 
     'name_bert_64', 
     'main_pic_embeddings_resnet_v1', 
     'Train'
     ]].to_parquet('etl_embeds.parquet', index=False)
