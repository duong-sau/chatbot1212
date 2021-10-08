import pandas as pd
import random
import time
import numpy as np
from tqdm.notebook import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import Adafactor, get_linear_schedule_with_warmup, MT5ForConditionalGeneration, T5Tokenizer
from transformers.utils.notebook import format_time

from Static.Define import path_common

tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
print(tokenizer)
model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small', return_dict=True)

GPU_NUM = 4
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
print('Current cuda device ', torch.cuda.current_device())  # check

train_data = pd.read_csv(path_common.train.value, error_bad_lines=False)
test_data = pd.read_csv(path_common.test.value, error_bad_lines=False)
dev_data = pd.read_csv(path_common.dev.value, error_bad_lines=False)

train_data.target = round(train_data.target * 5) / 5
train_data = train_data.applymap(str)
train_target = train_data.target

dev_data.target = round(dev_data.target * 5) / 5
dev_data = dev_data.applymap(str)
dev_target = dev_data.target

test_data.target = round(test_data.target * 5) / 5
test_data = test_data.applymap(str)
test_target = test_data.target

train_inputs, train_targets, dev_inputs, dev_targets, test_inputs, test_targets = [], [], [], [], [], []

for input in train_data.source:
    tokenized_inputs = tokenizer.encode_plus(input, max_length=283, padding='max_length', return_tensors="pt").input_ids
    train_inputs.append(tokenized_inputs)

for target in train_target:
    tokenized_targets = tokenizer.encode_plus(target, max_length=2, padding='max_length', return_tensors="pt").input_ids
    train_targets.append(tokenized_targets)

for input in dev_data.source:
    tokenized_inputs = tokenizer.encode_plus(input, max_length=283, padding='max_length', return_tensors="pt").input_ids
    dev_inputs.append(tokenized_inputs)

for target in dev_target:
    tokenized_targets = tokenizer.encode_plus(target, max_length=2, padding='max_length', return_tensors="pt").input_ids
    dev_targets.append(tokenized_targets)

for input in test_data.source:
    tokenized_inputs = tokenizer.encode_plus(input, max_length=283, padding='max_length', return_tensors="pt").input_ids
    test_inputs.append(tokenized_inputs)

for target in test_target:
    tokenized_targets = tokenizer.encode_plus(target, max_length=2, padding='max_length', return_tensors="pt").input_ids
    test_targets.append(tokenized_targets)

train_input_ids = torch.cat(train_inputs, dim=0)
train_labels = torch.cat(train_targets, dim=0)

dev_input_ids = torch.cat(dev_inputs, dim=0)
dev_labels = torch.cat(dev_targets, dim=0)

test_input_ids = torch.cat(test_inputs, dim=0)
test_labels = torch.cat(test_targets, dim=0)

train_dataset = TensorDataset(train_input_ids, train_labels)
dev_dataset = TensorDataset(dev_input_ids, dev_labels)
test_dataset = TensorDataset(test_input_ids, test_labels)

batch_size = 16
train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset),  # Select batches randomly
    batch_size=batch_size  # Trains with this batch size.
)
dev_dataloader = DataLoader(
    dev_dataset,  # The validation samples.
    sampler=SequentialSampler(dev_dataset),  # Pull out batches sequentially.
    batch_size=batch_size  # Evaluate with this batch size.
)
test_dataloader = DataLoader(
    test_dataset,  # The validation samples.
    sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
    batch_size=batch_size  # Evaluate with this batch size.
)

model.cuda()

params = list(model.named_parameters())

optimizer = Adafactor(model.parameters(),
                      lr=1e-3,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=(1e-30, 1e-3),
                      relative_step=False
                      )

epochs = 30
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

predictions_all = []
seed_val = 0

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []
total_t0 = time.time()

for epoch_i in tqdm(range(0, epochs)):
    #               Training
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0

    model.train()

    for step, batch in tqdm(enumerate(train_dataloader)):

        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)

        model.zero_grad()

        output = model(input_ids=b_input_ids, labels=b_labels, return_dict=True)
        loss = output.loss
        logits = output.logits

        total_train_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    #               Validation
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in tqdm(dev_dataloader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)

        with torch.no_grad():
            output = model(input_ids=b_input_ids, labels=b_labels, return_dict=True)
            loss = output.loss
            logits = output.logits

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

    avg_val_loss = total_eval_loss / len(dev_dataloader)
    validation_time = format_time(time.time() - t0)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    # test
    print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))
    model.eval()
    predictions = []

    for batch in tqdm(test_dataloader):
        b_input_ids = batch[0].to(device)

        with torch.no_grad():
            outputs = model.generate(b_input_ids)
            predictions.append(outputs)
    print('DONE.')

    predictions_all.append(predictions)

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

for i in range(10):
    output = model.generate(test_input_ids[i].cuda().reshape(1, -1))
    print(tokenizer.decode(output[0]))
