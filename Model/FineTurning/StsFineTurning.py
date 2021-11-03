import torch
import pandas as pd

from transformers import Trainer, AutoTokenizer, T5ForConditionalGeneration, T5Config


from Model.FineTurning.DataSet import SiameseDataset
from Static.Config import tokenizer_config, MODEL, freeze_layer, training_args, train_validate_test_split

tokenizer = AutoTokenizer.from_pretrained(MODEL['name'])
tokenizer_config(tokenizer=tokenizer)
assert tokenizer

config = T5Config.from_pretrained(MODEL['name'])
# config.num_decoder_layers = MODEL['num_decoder_layers']
model = T5ForConditionalGeneration.from_pretrained(MODEL['name'], config=config)
freeze_layer(model, MODEL['num_freeze'])
# optimizer = getOptimizer(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.cuda()
assert model

data = pd.read_csv(MODEL['data_link'], header=0)
data = data.astype(str)

train_data, val_data = train_validate_test_split(data)
train_dataset = SiameseDataset(df=train_data, tokenizer=tokenizer)
val_dataset = SiameseDataset(df=val_data, tokenizer=tokenizer)

assert_data = train_dataset.__getitem__(121)
assert_inputs = assert_data['input_ids']
assert assert_inputs[-1] == 1
assert_label = assert_data['label']
assert assert_label[-1] == 1

# lr_scheduler = AdafactorSchedule(optimizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model()
