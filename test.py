import os, time
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer

from dataset import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--logfile", type=str, default="LOG")
    parser.add_argument("--outfile", type=str, default="pred.tsv")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--large", action="store_true", help="Whether use Bert Large")
    parser.add_argument("--do_lower_case", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    formatter = logging.Formatter(
        "[ %(levelname)s: %(asctime)s ] - %(message)s"
    )
    logging.basicConfig(level=logging.DEBUG,
                        format="[ %(levelname)s: %(asctime)s ] - %(message)s")
    logger = logging.getLogger("QNLI")
    fh = logging.FileHandler(args.logfile)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(str(args))

    processor = QnliProcessor()
    model_size = "large" if args.large else "base"
    model_cased = "uncased" if args.do_lower_case else "cased"
    bert_model = f"bert-{model_size}-{model_cased}"

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=args.do_lower_case)

    label_list = processor.get_labels()
    num_labels = len(label_list)
    output_mode = "classification"

    eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)

    eval_all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(eval_all_input_ids, eval_all_input_mask, eval_all_segment_ids)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = BertForSequenceClassification.from_pretrained(args.model_dir, num_labels=num_labels)
    model.to(device)

    # Doing evaluation
    model.eval()
    preds = []
    for eval_batch in eval_dataloader:
        eval_batch = tuple(t.to(device) for t in eval_batch)
        eval_input_ids, eval_input_mask, eval_segment_ids = eval_batch
        with torch.no_grad():
            eval_logits = model(eval_input_ids, eval_segment_ids, eval_input_mask, labels=None)
            preds.extend(np.argmax(eval_logits.detach().cpu().numpy(), axis=1))
    preds = np.array(preds)

    with open(args.outfile, 'w') as outfile:
        outfile.write('index\tprediction\n')
        for idx, pred in enumerate(preds):
            outfile.write(str(idx)+'\t'+label_list[pred]+'\n')
    

if __name__ == '__main__':
    main()