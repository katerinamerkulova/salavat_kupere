import os
import pandas as pd
import warnings
from glob import glob

ozon_files = glob('../data/ozon/*.xlsx')
warnings.filterwarnings('ignore')

from transformers import (AutoModel,AutoModelForMaskedLM, 
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
os.chdir('/home/ailab_user/hakathon')
META_FOLDER = '../data/'
gisp_df = pd.read_parquet(META_FOLDER+'gisp_products.parquet')
gisp_df['text'] = gisp_df['name'].astype(str) + gisp_df['description'].astype(str)
ozon_names=[]
for f in ozon_files:
	try:
		temp = pd.read_excel(f)
		# print('Shape of df:',temp['Название товара'].shape)
		(ozon_names.extend(temp['Название товара'].tolist()))
	except Exception as e:
		print('current file:',f)
		print(e)
text  = '\n'.join(gisp_df['text'].tolist())
ozon_text = '\n'.join(str(i) for i in ozon_names if len(str(i)) > 10)
print('len(text)',len(text))
print('len(ozon_text)',len(ozon_text))
with open(META_FOLDER+'mlm_text.txt','w') as f:
    f.write(text)
with open(META_FOLDER+'mlm_text.txt','a') as f:
    f.write(ozon_text)
model_name = 'cointegrated/LaBSE-en-ru'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(META_FOLDER +'models/LaBSE-en-ru/LaBSE-en-ru-mlm-gisp');


train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=META_FOLDER+"mlm_text.txt", #mention train text file here
    block_size=256)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=META_FOLDER+"mlm_text.txt", #mention valid text file here
    block_size=256)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=META_FOLDER +"models/LaBSE-en-ru/LaBSE-en-ru-mlm-gisp-chkpt", #select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy= 'steps',
    save_total_limit=2,
    eval_steps=500,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end =True,
    prediction_loss_only=True,
	fp16=True,
    report_to = "none")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)

trainer.train()
trainer.save_model(META_FOLDER +'models/LaBSE-en-ru/LaBSE-en-ru-mlm-gisp')