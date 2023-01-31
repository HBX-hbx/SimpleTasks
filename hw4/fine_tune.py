import torch
import argparse
import evaluate
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification

import warnings
warnings.filterwarnings("ignore")

"""
label 0: Begin
label 1: Mid
label 2: End
label 3: Single
"""
B = 0
M = 1
E = 2
S = 3
label_list = ["B", "M", "E", "S"]
id2label = {0: "B", 1: "M", 2: "E", 3: "S"}
label2id = {"B": 0, "M": 1, "E": 2, "S": 3}

def read_file_data(fpath):
    fp = open(fpath, encoding='utf-8')
    texts = []
    labels = []
    for line in tqdm(fp.readlines()):
        tokens = line.strip().split(' ')
        label = []
        for token in tokens:
            if len(token) == 1:
                label.append(S)
            else:
                label.append(B) # Begin
                label.extend([M] * (len(token) - 2))
                label.append(E)
        labels.append(label)
        texts.append(''.join(tokens))
    fp.close()
    return texts, labels

class TaggingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.dataset.items()}
        return item
    
    def __len__(self):
        return len(self.dataset['labels'])


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--output_dir', type=str, default='./results')
parser.add_argument('--lr', type=float, default=2e-5, help='learning_rate, default to 2e-5')
parser.add_argument('--bsz', type=int, default=16)
parser.add_argument('--wd', type=float, default=0.01)
args = parser.parse_args()

seqeval = evaluate.load('seqeval')

def compute_metrics(eval_pred):
    """_summary_
    Args:
        preds : [[0, 1, 2], [0, 1, 2]]
        labels: [[0, 1, 1], [0, 1, 1]]
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

print('<---------------- loading data ------------------->')

data_dir = Path(args.data_dir)
train_texts, train_labels = read_file_data(data_dir.joinpath('train.txt'))
dev_texts, dev_labels = read_file_data(data_dir.joinpath('dev.txt'))
test_texts, test_labels = read_file_data(data_dir.joinpath('test.txt'))

print('<---------------- tokenizing ------------------->')

def tokenize_and_align_labels(texts, labels):
    """
    Args:
        texts  : ['迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）',]
        labels : [[0, 2 ......],]
    """
    tokenized_inputs = tokenizer(texts, truncation=True, padding=True) # [CLS or SEP]
    new_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i) # [None, 0, 1, ... , None]
        pre_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None: # [CLS or SEP]
                label_ids.append(-100)
            elif word_id != pre_word_id:
                label_ids.append(label[word_id])
            else:
                label_ids.append(-100)
            pre_word_id = word_id
        new_labels.append(label_ids)
    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs

tokenizer = AutoTokenizer.from_pretrained('uer/chinese_roberta_L-8_H-512')
train_dataset = tokenize_and_align_labels(train_texts, train_labels)
dev_dataset = tokenize_and_align_labels(dev_texts, dev_labels)
test_dataset = tokenize_and_align_labels(test_texts, test_labels)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

train_dataset = TaggingDataset(train_dataset)
dev_dataset = TaggingDataset(dev_dataset)
test_dataset = TaggingDataset(test_dataset)

test_dataloader = DataLoader(test_dataset, batch_size=16)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=args.bsz,
    per_device_eval_batch_size=16,
    weight_decay=args.wd,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    load_best_model_at_end=True,
    learning_rate=args.lr, # changing
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print('<---------------- training ------------------->')

model = AutoModelForTokenClassification.from_pretrained('uer/chinese_roberta_L-8_H-512', num_labels=4, id2label=id2label, label2id=label2id)
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

print('<---------------- testing ------------------->')
metric = evaluate.load("seqeval")
model.eval()

tokenized_test_texts = []

for batch in tqdm(test_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
    
    predictions = torch.argmax(outputs.logits, dim=-1)
    labels = batch["labels"]
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    for example, preds in zip(batch['input_ids'], true_predictions):
        tokens = tokenizer.decode(example).split(' ')
        tokens = [token for token in tokens if token != "[PAD]" and token != "[CLS]" and token != "[SEP]"]
        merged_tokens = []
        tmp = ''
        for (token, pred) in zip(tokens, preds):
            tmp += token
            if pred == 'S' or pred == 'E':
                merged_tokens.append(tmp)
                tmp = ''
        tokenized_test_texts.append(' '.join(merged_tokens))
    
    metric.add_batch(predictions=true_predictions, references=true_labels)

with open('./cls_reports/acc_lr_%s_bsz_%s_wd_%s.txt' % (str(args.lr), str(args.bsz), str(args.wd)), 'w') as f:
    results = metric.compute()
    results = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    f.write(str(results))
    f.write('\n\n')
    f.write('\n'.join(tokenized_test_texts))