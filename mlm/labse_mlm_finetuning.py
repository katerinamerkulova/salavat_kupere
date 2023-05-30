import os
import warnings

import pandas as pd
from transformers import (
    AutoModelForMaskedLM, AutoTokenizer,
    DataCollatorForLanguageModeling, LineByLineTextDataset,
    Trainer, TrainingArguments
)

warnings.filterwarnings('ignore')

os.chdir('/home/ailab_user/hakathon')
META_FOLDER = '../data/hackathon_files_for_participants_ozon/'

if not os.path.exists(META_FOLDER+'hakaton_mlm_text.txt'):    
    ozon_products_part1=pd.read_parquet(META_FOLDER+'test_data.parquet')
    ozon_products_part2=pd.read_parquet(META_FOLDER+'train_data.parquet')
    ozon_products = pd.concat([ozon_products_part1, ozon_products_part2])
    text  = '\n'.join(ozon_products['name'].tolist())

    print('len(text)',len(text))

    with open(META_FOLDER+'hakaton_mlm_text.txt','w') as f:
        f.write(text)

model_name = 'cointegrated/LaBSE-en-ru'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(META_FOLDER + 'models/LaBSE-en-ru/LaBSE-en-ru-mlm-ozon');


train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=META_FOLDER + "hakaton_mlm_text.txt", #mention train text file here
    block_size=256)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=META_FOLDER +" hakaton_mlm_text.txt", #mention valid text file here
    block_size=256)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=META_FOLDER + "models/LaBSE-en-ru/LaBSE-en-ru-mlm-gisp-chkpt", #select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=3,
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
    report_to = "none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)

trainer.train()
trainer.save_model(META_FOLDER + 'models/LaBSE-en-ru/LaBSE-en-ru-mlm-ozon')