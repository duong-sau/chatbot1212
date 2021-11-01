from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd


from Model.Common import compute_metrics
from Model.FineTurning.DataSet import ClassificationDataset
from Static.Config import MODEL, train_validate_test_split, get_device

tokenizer = BertTokenizerFast.from_pretrained(MODEL['classification_name'], do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(MODEL['classification_name'], num_labels=MODEL['num_class'])
device = get_device()
model.to(device)

data = pd.read_csv(
    'https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/A/learn_data.csv',
    header=0)
target_names = 2

train, test = train_validate_test_split(data)
train_texts = train['source'].tolist()
train_labels = train['target'].tolist()
train_labels = [int(i) - 1 for i in train_labels]
valid_texts = test['source'].tolist()
valid_labels = test['target'].tolist()
valid_labels = [int(i) - 1 for i in valid_labels]

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MODEL['max_length'])
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=MODEL['max_length'])

# convert our tokenized data into a torch Dataset
train_dataset = ClassificationDataset(train_encodings, train_labels)
valid_dataset = ClassificationDataset(valid_encodings, valid_labels)

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
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

trainer.train()
trainer.save_model()