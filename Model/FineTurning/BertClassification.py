from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from Static.Config import MODEL, train_validate_test_split, get_device


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        tensor = torch.zeros(10)
        tensor[int(self.labels[idx]) - 1] = 1.0
        item["labels"] = tensor
        return item

    def __len__(self):
        return len(self.labels)


data = pd.read_csv(
    'https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Cluster/IntentClassification/Positive'
    '/learn_data.csv',
    header=0)
num_class = len(data['cluster_index'].value_counts())
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_class)
device = get_device()
model.to(device)

train, valid = train_validate_test_split(data)

# tokenize the dataset, truncate when passed `max_length`,
# and pad with 0's when less than `max_length`
train_encodings = tokenizer(train['sentence'].tolist()[0:3], truncation=True, padding=True, max_length=512)
valid_encodings = tokenizer(train['sentence'].tolist()[0:3], truncation=True, padding=True, max_length=512)

train_dataset = ClassificationDataset(train_encodings, train['cluster_index'].tolist()[0:3])
valid_dataset = ClassificationDataset(valid_encodings, valid['cluster_index'].tolist()[0:3])


def compute_metrics(pred):
    labels = pred.label_ids.argmax(-1)
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


training_args = TrainingArguments(
    output_dir='/content/drive/MyDrive/',  # output directory
    num_train_epochs=30,  # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,  # batch size for evaluation
    warmup_steps=100,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    load_best_model_at_end=True,
    save_strategy='epoch',  # log & save weights each logging_steps
    evaluation_strategy="epoch",  # evaluate each `logging_steps`
)
trainer = Trainer(
    model=model,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=valid_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,  # the callback that computes metrics of interest
)

trainer.train()
trainer.save_model()
