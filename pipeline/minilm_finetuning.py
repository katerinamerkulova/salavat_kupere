import argparse
import json
import logging
import math
import os

import pandas as pd
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def run_training(
        DATA_DIR="../../data/hackathon_files_for_participants_ozon",
        MODEL_DIR="../models",
        mode="FULL"  # NAME
):
    
    print(os.path.exists(MODEL_DIR))
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model_name = os.path.join(MODEL_DIR, "all-MiniLM-L6-v2")
    try:
        model = SentenceTransformer(model_name)
    except:
        logging.error(f'No model {model_name}, Load from internet:')
        model = SentenceTransformer("all-MiniLM-L6-v2")
    train_batch_size = 8
    num_epochs = 1

    model_save_path = os.path.join(MODEL_DIR, f"{mode.lower()}_minilm")

    logging.info("Read STSbenchmark train dataset")
    dataset = pd.read_parquet(os.path.join(DATA_DIR, "train_pairs.parquet"))#salavat_kupere/data/train_pairs.parquet
    etl = pd.read_parquet(os.path.join(DATA_DIR, "train_data.parquet"))

    samples = []
    for i, row in dataset.iterrows():
        chars1 = etl.loc[etl["variantid"] == row["variantid1"], "characteristic_attributes_mapping"].item()
        chars2 = etl.loc[etl["variantid"] == row["variantid1"], "characteristic_attributes_mapping"].item()
        char1 = " ".join(sorted([f"[CHAR]{char}[VAL]{val}" for char, val in json.loads(chars1).items()])) if chars1 else ""
        char2 = " ".join(sorted([f"[CHAR]{char}[VAL]{val}" for char, val in json.loads(chars2).items()])) if chars2 else ""
        name1 = etl.loc[etl["variantid"] == row["variantid1"], "name"].item()
        name2 = etl.loc[etl["variantid"] == row["variantid2"], "name"].item()

        if mode == "NAME":
            text1, text2 = name1, name2

        if mode == "FULL":
            text1, text2 = f"{name1} {char1}", f"{name2} {char2}"

        inp_example = InputExample(texts=[text1, text2], label=float(row["target"]))
        samples.append(inp_example)

    train_samples, test_samples = train_test_split(samples, test_size=0.3)
    dev_samples, test_samples = train_test_split(test_samples, test_size=0.5)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    logging.info("Read STSbenchmark dev dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="sts-dev")

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)

    model = SentenceTransformer(model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")
    test_evaluator(model, output_path=model_save_path)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--data_dir",
                           default=None,
                           type=str,
                           required=True,
                           help="Path to dir with data")
    argParser.add_argument("--model_dir",
                           default=None,
                           type=str,
                           required=True,
                           help="Path to dir with model")

    args = argParser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    if data_dir is not None:
        run_training(DATA_DIR=data_dir, MODEL_DIR=model_dir, mode="FULL")
        run_training(DATA_DIR=data_dir, MODEL_DIR=model_dir, mode="NAME")
    else:
        logging.error("no --data_dir param as input")
