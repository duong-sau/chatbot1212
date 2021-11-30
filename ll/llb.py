from transformers import BertTokenizerFast, BertForSequenceClassification, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
import pandas as pd


from Model.Common import compute_metrics
from Model.FineTurning.DataSet import ClassificationDataset
from Static.Config import MODEL, train_validate_test_split, get_device

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model2 = T5ForConditionalGeneration.from_pretrained('t5-base')
model.baseline = model2.encoder
device = get_device()
model.to(device)

data = pd.read_csv(
    'https://raw.githubusercontent.com/duong-sau/chatbot1212/1ddc5fe6616c08920a725a491a4cb7f8f93c64b5/Model/Data'
    '/IntentClassification/LabelClassification/learn_data.csv',
    header=0)
target_names = 2

train, valid = train_validate_test_split(data)

train_dataset = ClassificationDataset(df=train, tokenizer=tokenizer)
valid_dataset = ClassificationDataset(df=valid, tokenizer=tokenizer)

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