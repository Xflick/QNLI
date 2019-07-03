import os, time
import argparse
import logging
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from dataset import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--logfile", type=str, default="LOG")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="For GPU memory issues")
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

    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    processor = QnliProcessor()
    model_size = "large" if args.large else "base"
    model_cased = "uncased" if args.do_lower_case else "cased"
    bert_model = f"bert-{model_size}-{model_cased}"

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=args.do_lower_case)

    label_list = processor.get_labels()
    num_labels = len(label_list)
    output_mode = "classification"

    train_examples = processor.get_train_examples(args.data_dir)
    train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, output_mode)

    train_all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    train_all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(train_all_input_ids, train_all_input_mask, train_all_segment_ids, train_all_label_ids)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)

    eval_all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(eval_all_input_ids, eval_all_input_mask, eval_all_segment_ids, eval_all_label_ids)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, drop_last=False)

    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_-1')    
    model = BertForSequenceClassification.from_pretrained(bert_model, cache_dir=cache_dir, num_labels=num_labels)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=args.lr,
                        warmup=args.warmup_proportion,
                        t_total=num_train_optimization_steps)
    criterion = nn.CrossEntropyLoss().to(device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model.train()

    for epoch in trange(args.epochs, desc="Epoch"):
        tr_loss = 0
        nb_tr_examples = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = model(input_ids, segment_ids, input_mask, labels=None)
            loss = criterion(logits.view(-1, num_labels), label_ids.view(-1))
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (step+1) % (len(train_dataloader)//5) == 0:
                # Doing evaluation
                model.eval()
                preds = []
                for eval_batch in eval_dataloader:
                    eval_batch = tuple(t.to(device) for t in eval_batch)
                    eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids = eval_batch
                    with torch.no_grad():
                        eval_logits = model(eval_input_ids, eval_segment_ids, eval_input_mask, labels=None)
                        preds.extend(np.argmax(eval_logits.detach().cpu().numpy(), axis=1))
                preds = np.array(preds)
                targets = eval_all_label_ids.numpy()
                assert len(preds) == len(targets)
                logger.info(f"Step: {step+1:4d}/{len(train_dataloader):4d}, Epoch: {epoch+1:2d}/{args.epochs:2d}, "
                    f"Training Loss: {tr_loss/nb_tr_examples:.8f}, Evaluation Acc: {(preds==targets).mean()*100:.2f}%.")
                tr_loss = 0
                nb_tr_examples = 0
                model.train()

        torch.save(model.state_dict(), os.path.join(args.model_dir, f"pytorch_model_{epoch+1}.bin"))
    model.config.to_json_file(os.path.join(args.model_dir, f"bert_config.json"))

    logger.info("***** Training Finished *****")


if __name__ == '__main__':
    main()