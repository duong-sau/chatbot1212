
# load packages
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import datetime
import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import re

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler


torch.cuda.amp.autocast(enabled=True)
SEED = 15
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
## <torch._C.Generator object at 0x000000001F53E050>
torch.backends.cudnn.deterministic = True

# tell pytorch to use cuda
device = torch.device("cuda")
# prepare and load data
def prepare_df(pkl_location):
    # read pkl as pandas
    df = pd.read_pickle(pkl_location)
    # just keep us/kabul labels
    df = df.loc[(df['target'] == 'US') | (df['target'] == 'Kabul')]
    # mask DV to recode
    us = df['target'] == 'US'
    kabul = df['target'] == 'Kabul'
    # reset index
    df = df.reset_index(drop=True)
    return df

# load df
df = prepare_df('C:\\Users\\Andrew\\Desktop\\df.pkl')


# prepare data
def clean_df(df):
    # strip dash but keep a space
    df['body'] = df['body'].str.replace('-', ' ')
    # lower case the data
    df['body'] = df['body'].apply(lambda x: x.lower())
    # remove excess spaces near punctuation
    df['body'] = df['body'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
    # generate a word count for body
    df['word_count'] = df['body'].apply(lambda x: len(x.split()))
    # generate a word count for summary
    df['word_count_summary'] = df['title_osc'].apply(lambda x: len(x.split()))
    # remove excess white spaces
    df['body'] = df['body'].apply(lambda x: " ".join(x.split()))
    # lower case to body
    df['body'] = df['body'].apply(lambda x: x.lower())
    # lower case to summary
    df['title_osc'] = df['title_osc'].apply(lambda x: x.lower())
    # add " </s>" to end of body
    df['body'] = df['body'] + " </s>"
    # add " </s>" to end of target
    df['target'] = df['target'] + " </s>"
    return df


# clean df
df = clean_df(df)
# 1.2 Instantiate Tokenizer
# Next, we instantiate the T5 tokenizer from transformers and check some special token IDs.

# instantiate T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# check token ids
tokenizer.eos_token_id
## 1
tokenizer.bos_token_id
tokenizer.unk_token_id
## 2
tokenizer.pad_token_id
## 0
# 1.3 Tokenize the Corpus
# Then, we proceed to tokenize our corpus like usual. Notice that we effectively do this process twice as we tokenize our corpus and also tokenize our targets.

# tokenize the main text
def tokenize_corpus(df, tokenizer, max_len):
    # token ID storage
    input_ids = []
    # attension mask storage
    attention_masks = []
    # max len -- 512 is max
    max_len = max_len
    # for every document:
    for doc in df:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            doc,  # document to encode.
            add_special_tokens=True,  # add tokens relative to model
            max_length=max_len,  # set max length
            truncation=True,  # truncate longer messages
            pad_to_max_length=True,  # add padding
            return_attention_mask=True,  # create attn. masks
            return_tensors='pt'  # return pytorch tensors
        )

        # add the tokenized sentence to the list
        input_ids.append(encoded_dict['input_ids'])

        # and its attention mask (differentiates padding from non-padding)
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


# create tokenized data
body_input_ids, body_attention_masks = tokenize_corpus(df['body'].values, tokenizer, 512)



# how long are tokenized targets
ls = []
for i in range(df.shape[0]):
    ls.append(len(tokenizer.tokenize(df.iloc[i]['target'])))

temp_df = pd.DataFrame({'len_tokens': ls})
temp_df['len_tokens'].mean()  # 2.7
## 2.772822299651568
temp_df['len_tokens'].median()  # 3
## 3.0
temp_df['len_tokens'].max()  # 3

# create tokenized targets
## 3
target_input_ids, target_attention_masks = tokenize_corpus(df['target'].values, tokenizer, 3)
# 1.4 Prepare and Split Data
# Next, we split our data into train, validation, and test sets.

# prepare tensor data sets
def prepare_dataset(body_tokens, body_masks, target_token, target_masks):
    # create tensor data sets
    tensor_df = TensorDataset(body_tokens, body_masks, target_token, target_masks)
    # 80% of df
    train_size = int(0.8 * len(df))
    # 20% of df
    val_size = len(df) - train_size
    # 50% of validation
    test_size = int(val_size - 0.5*val_size)
    # divide the dataset by randomly selecting samples
    train_dataset, val_dataset = random_split(tensor_df, [train_size, val_size])
    # divide validation by randomly selecting samples
    val_dataset, test_dataset = random_split(val_dataset, [test_size, test_size+1])

    return train_dataset, val_dataset, test_dataset


# create tensor data sets
train_dataset, val_dataset, test_dataset = prepare_dataset(body_input_ids,
                                                           body_attention_masks,
                                                           target_input_ids,
                                                           target_attention_masks
                                                           )
# 1.5 Instantiate Training Models
# Now we are ready to prepare our training scripts which follow the other guides closely. T5ForConditionalGeneration asks that we supply four inputs into the model’s forward function: (1) corpus token ids, (2) corpus attention masks, and (3) our label ids, and (4) our label attention masks.

def train(model, dataloader, optimizer):

    # capture time
    total_t0 = time.time()

    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')

    # reset total loss for epoch
    train_total_loss = 0
    total_train_f1 = 0

    # put model into traning mode
    model.train()

    # for each batch of training data...
    for step, batch in enumerate(dataloader):

        # progress update every 40 batches.
        if step % 40 == 0 and not step == 0:

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

        # Unpack this training batch from our dataloader:
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input tokens
        #   [1]: attention masks
        #   [2]: target tokens
        #   [3]: target attenion masks
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_target_ids = batch[2].cuda()
        b_target_mask = batch[3].cuda()

        # clear previously calculated gradients
        optimizer.zero_grad()

        # runs the forward pass with autocasting.
        with autocast():
            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_target_ids,
                            decoder_attention_mask=b_target_mask)

            loss, prediction_scores = outputs[:2]

            # sum the training loss over all batches for average loss at end
            # loss is a tensor containing a single value
            train_total_loss += loss.item()

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # update the learning rate
        scheduler.step()

    # calculate the average loss over all of the batches
    avg_train_loss = train_total_loss / len(dataloader)

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'Train Loss': avg_train_loss
        }
    )

    # training time end
    training_time = format_time(time.time() - total_t0)

    # print result summaries
    print("")
    print("summary results")
    print("epoch | trn loss | trn time ")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {training_time:}")

    return training_stats


def validating(model, dataloader):

    # capture validation time
    total_t0 = time.time()

    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")

    # put the model in evaluation mode
    model.eval()

    # track variables
    total_valid_loss = 0

    # evaluate data for one epoch
    for batch in dataloader:

        # Unpack this training batch from our dataloader:
        # `batch` contains three pytorch tensors:
        #   [0]: input tokens
        #   [1]: attention masks
        #   [2]: target tokens
        #   [3]: target attenion masks
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_target_ids = batch[2].cuda()
        b_target_mask = batch[3].cuda()

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_target_ids,
                            decoder_attention_mask=b_target_mask)

            loss, prediction_scores = outputs[:2]

            # sum the training loss over all batches for average loss at end
            # loss is a tensor containing a single value
            total_valid_loss += loss.item()

    # calculate the average loss over all of the batches.
    global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)

    # Record all statistics from this epoch.
    valid_stats.append(
        {
            'Val Loss': avg_val_loss,
            'Val PPL.': np.exp(avg_val_loss)
        }
    )

    # capture end validation time
    training_time = format_time(time.time() - total_t0)

    # print result summaries
    print("")
    print("summary results")
    print("epoch | val loss | val ppl | val time")
    print(f"{epoch+1:5d} | {avg_val_loss:.5f} | {np.exp(avg_val_loss):.3f} | {training_time:}")

    return valid_stats


def testing(model, dataloader):

    print("")
    print("Running Testing...")

    # measure training time
    t0 = time.time()

    # put the model in evaluation mode
    model.eval()

    # track variables
    total_test_loss = 0
    total_test_acc = 0
    total_test_f1 = 0
    predictions = []
    actuals = []

    # evaluate data for one epoch
    for step, batch in enumerate(dataloader):
        # progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

        # Unpack this training batch from our dataloader:
        # `batch` contains three pytorch tensors:
        #   [0]: input tokens
        #   [1]: attention masks
        #   [2]: target tokens
        #   [3]: target attenion masks
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_target_ids = batch[2].cuda()
        b_target_mask = batch[3].cuda()

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_target_ids,
                            decoder_attention_mask=b_target_mask)

            loss, prediction_scores = outputs[:2]

            total_test_loss += loss.item()

            generated_ids = model.generate(
                input_ids=b_input_ids,
                attention_mask=b_input_mask,
                max_length=3
            )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in b_target_ids]

            total_test_acc += accuracy_score(target, preds)
            total_test_f1 += f1_score(preds, target,
                                      average='weighted',
                                      labels=np.unique(preds))
            predictions.extend(preds)
            actuals.extend(target)

    # calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(dataloader)

    avg_test_acc = total_test_acc / len(test_dataloader)

    avg_test_f1 = total_test_f1 / len(test_dataloader)

    # Record all statistics from this epoch.
    test_stats.append(
        {
            'Test Loss': avg_test_loss,
            'Test PPL.': np.exp(avg_test_loss),
            'Test Acc.': avg_test_acc,
            'Test F1': avg_test_f1
        }
    )
    global df2
    temp_data = pd.DataFrame({'predicted': predictions, 'actual': actuals})
    df2 = df2.append(temp_data)

    return test_stats


# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
# 1.6 Dealing with Imbalanced Classifcation: Data Loaders
# Since my corpus is imbalanced, we produce weighted samplers to help balance the distribution of data as it is fed via the data loaders.

# helper function to count target distribution inside tensor data sets
def target_count(tensor_dataset):
    # set empty count containers
    count0 = 0
    count1 = 0
    # set total container to turn into torch tensor
    total = []
    for i in tensor_dataset:
        # for kabul tensor
        if torch.all(torch.eq(i[2], torch.tensor([20716, 83, 1]))):
            count0 += 1
        # for us tensor
        elif torch.all(torch.eq(i[2], torch.tensor([837, 1, 0]))):
            count1 += 1
    total.append(count0)
    total.append(count1)
    return torch.tensor(total)


# prepare weighted sampling for imbalanced classification
def create_sampler(target_tensor, tensor_dataset):
    # generate class distributions [x, y]
    class_sample_count = target_count(tensor_dataset)
    # weight
    weight = 1. / class_sample_count.float()
    # produce weights for each observation in the data set
    new_batch = []
    # for each obs
    for i in tensor_dataset:
        # if i is equal to kabul
        if torch.all(torch.eq(i[2], torch.tensor([20716, 83, 1]))):
            # append 0
            new_batch.append(0)
            # elif equal to US
        elif torch.all(torch.eq(i[2], torch.tensor([837, 1, 0]))):
            # append 1
            new_batch.append(1)
    samples_weight = torch.tensor([weight[t] for t in new_batch])
    # prepare sampler
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    return sampler


# need to make them numeric now
train_sampler = create_sampler(target_count(train_dataset), train_dataset)


# check balancer
train_dataloader = DataLoader(train_dataset,
                              batch_size=24,
                              sampler=train_sampler,
                              shuffle=False)

# lets check class balance for each batch to see how the sampler is working
for i, (input_ids, input_masks, target_ids, target_masks) in enumerate(train_dataloader):
    count_kabul = 0
    count_us = 0
    if i in range(0, 10):
        for j in target_ids:
            if (torch.all(torch.eq(j, torch.tensor([20716, 83, 1])))):
                count_kabul += 1
            else:
                count_us += 1
        print("batch index {}, 0/1: {}/{}".format(i, count_kabul, count_us))
## batch index 0, 0/1: 12/12
## batch index 1, 0/1: 13/11
## batch index 2, 0/1: 9/15
## batch index 3, 0/1: 15/9
## batch index 4, 0/1: 17/7
## batch index 5, 0/1: 9/15
## batch index 6, 0/1: 4/20
## batch index 7, 0/1: 14/10
## batch index 8, 0/1: 10/14
## batch index 9, 0/1: 12/12
# Before training, several prepatory objects are instantiated like the model, data loaders, and the optimizer.
#
# 1.7 Prepare for Training
# instantiate model T5 transformer with a language modeling head on top
model = T5ForConditionalGeneration.from_pretrained('t5-small').cuda()  # to GPU


# create DataLoaders with samplers
## Some weights of T5ForConditionalGeneration were not initialized from the model checkpoint at t5-small and are newly initialized: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight']
## You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
train_dataloader = DataLoader(train_dataset,
                              batch_size=24,
                              sampler=train_sampler,
                              shuffle=False)

valid_dataloader = DataLoader(val_dataset,
                              batch_size=24,
                              shuffle=True)

test_dataloader = DataLoader(test_dataset,
                             batch_size=24,
                             shuffle=True)


# Adam w/ Weight Decay Fix
# set to optimizer_grouped_parameters or model.parameters()
optimizer = AdamW(model.parameters(),
                  lr = 3e-5
                  )

# epochs
epochs = 6

# lr scheduler
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

# create gradient scaler for mixed precision
scaler = GradScaler()


# create training result storage
training_stats = []
valid_stats = []
best_valid_loss = float('inf')

# for each epoch
for epoch in range(epochs):
    # train
    train(model, train_dataloader, optimizer)
    # validate
    validating(model, valid_dataloader)
    # check validation loss
    if valid_stats[epoch]['Val Loss'] < best_valid_loss:
        best_valid_loss = valid_stats[epoch]['Val Loss']
        # save best model for use later
        torch.save(model.state_dict(), 't5-classification.pt')  # torch save
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained('./model_save/t5-classification/')  # transformers save
        tokenizer.save_pretrained('./model_save/t5-classification/')  # transformers save
##
## ======== Epoch 1 / 6 ========
## Training...
##   Batch    40  of    335.
##   Batch    80  of    335.
##   Batch   120  of    335.
##   Batch   160  of    335.
##   Batch   200  of    335.
##   Batch   240  of    335.
##   Batch   280  of    335.
##   Batch   320  of    335.
##
## summary results
## epoch | trn loss | trn time
##     1 | 1.86750 | 0:01:28
## [{'Train Loss': 1.867504431307316}]
##
## Running Validation...
##
## summary results
## epoch | val loss | val ppl | val time
##     1 | 0.14962 | 1.161 | 0:00:05
## [{'Val Loss': 0.14961695591253893, 'Val PPL.': 1.1613892942138855}]
## ('./model_save/t5-classification/spiece.model', './model_save/t5-classification/special_tokens_map.json', './model_save/t5-classification/added_tokens.json')
##
## ======== Epoch 2 / 6 ========
## Training...
##   Batch    40  of    335.
##   Batch    80  of    335.
##   Batch   120  of    335.
##   Batch   160  of    335.
##   Batch   200  of    335.
##   Batch   240  of    335.
##   Batch   280  of    335.
##   Batch   320  of    335.
##
## summary results
## epoch | trn loss | trn time
##     2 | 0.18869 | 0:01:40
## [{'Train Loss': 1.867504431307316}, {'Train Loss': 0.18868607740793655}]
##
## Running Validation...
##
## summary results
## epoch | val loss | val ppl | val time
##     2 | 0.13648 | 1.146 | 0:00:05
## [{'Val Loss': 0.14961695591253893, 'Val PPL.': 1.1613892942138855}, {'Val Loss': 0.1364755311182567, 'Val PPL.': 1.146226830543859}]
## ('./model_save/t5-classification/spiece.model', './model_save/t5-classification/special_tokens_map.json', './model_save/t5-classification/added_tokens.json')
##
## ======== Epoch 3 / 6 ========
## Training...
##   Batch    40  of    335.
##   Batch    80  of    335.
##   Batch   120  of    335.
##   Batch   160  of    335.
##   Batch   200  of    335.
##   Batch   240  of    335.
##   Batch   280  of    335.
##   Batch   320  of    335.
##
## summary results
## epoch | trn loss | trn time
##     3 | 0.15885 | 0:01:27
## [{'Train Loss': 1.867504431307316}, {'Train Loss': 0.18868607740793655}, {'Train Loss': 0.15885204529361938}]
##
## Running Validation...
##
## summary results
## epoch | val loss | val ppl | val time
##     3 | 0.12382 | 1.132 | 0:00:05
## [{'Val Loss': 0.14961695591253893, 'Val PPL.': 1.1613892942138855}, {'Val Loss': 0.1364755311182567, 'Val PPL.': 1.146226830543859}, {'Val Loss': 0.12382005155086517, 'Val PPL.': 1.131812184825848}]
## ('./model_save/t5-classification/spiece.model', './model_save/t5-classification/special_tokens_map.json', './model_save/t5-classification/added_tokens.json')
##
## ======== Epoch 4 / 6 ========
## Training...
##   Batch    40  of    335.
##   Batch    80  of    335.
##   Batch   120  of    335.
##   Batch   160  of    335.
##   Batch   200  of    335.
##   Batch   240  of    335.
##   Batch   280  of    335.
##   Batch   320  of    335.
##
## summary results
## epoch | trn loss | trn time
##     4 | 0.15560 | 0:01:33
## [{'Train Loss': 1.867504431307316}, {'Train Loss': 0.18868607740793655}, {'Train Loss': 0.15885204529361938}, {'Train Loss': 0.15560255393163483}]
##
## Running Validation...
##
## summary results
## epoch | val loss | val ppl | val time
##     4 | 0.11525 | 1.122 | 0:00:05
## [{'Val Loss': 0.14961695591253893, 'Val PPL.': 1.1613892942138855}, {'Val Loss': 0.1364755311182567, 'Val PPL.': 1.146226830543859}, {'Val Loss': 0.12382005155086517, 'Val PPL.': 1.131812184825848}, {'Val Loss': 0.1152467570666756, 'Val PPL.': 1.1221503019282884}]
## ('./model_save/t5-classification/spiece.model', './model_save/t5-classification/special_tokens_map.json', './model_save/t5-classification/added_tokens.json')
##
## ======== Epoch 5 / 6 ========
## Training...
##   Batch    40  of    335.
##   Batch    80  of    335.
##   Batch   120  of    335.
##   Batch   160  of    335.
##   Batch   200  of    335.
##   Batch   240  of    335.
##   Batch   280  of    335.
##   Batch   320  of    335.
##
## summary results
## epoch | trn loss | trn time
##     5 | 0.14363 | 0:01:34
## [{'Train Loss': 1.867504431307316}, {'Train Loss': 0.18868607740793655}, {'Train Loss': 0.15885204529361938}, {'Train Loss': 0.15560255393163483}, {'Train Loss': 0.14363485894986053}]
##
## Running Validation...
##
## summary results
## epoch | val loss | val ppl | val time
##     5 | 0.10829 | 1.114 | 0:00:05
## [{'Val Loss': 0.14961695591253893, 'Val PPL.': 1.1613892942138855}, {'Val Loss': 0.1364755311182567, 'Val PPL.': 1.146226830543859}, {'Val Loss': 0.12382005155086517, 'Val PPL.': 1.131812184825848}, {'Val Loss': 0.1152467570666756, 'Val PPL.': 1.1221503019282884}, {'Val Loss': 0.10829435022813934, 'Val PPL.': 1.1143757138609098}]
## ('./model_save/t5-classification/spiece.model', './model_save/t5-classification/special_tokens_map.json', './model_save/t5-classification/added_tokens.json')
##
## ======== Epoch 6 / 6 ========
## Training...
##   Batch    40  of    335.
##   Batch    80  of    335.
##   Batch   120  of    335.
##   Batch   160  of    335.
##   Batch   200  of    335.
##   Batch   240  of    335.
##   Batch   280  of    335.
##   Batch   320  of    335.
##
## summary results
## epoch | trn loss | trn time
##     6 | 0.13688 | 0:01:33
## [{'Train Loss': 1.867504431307316}, {'Train Loss': 0.18868607740793655}, {'Train Loss': 0.15885204529361938}, {'Train Loss': 0.15560255393163483}, {'Train Loss': 0.14363485894986053}, {'Train Loss': 0.13688372016064268}]
##
## Running Validation...
##
## summary results
## epoch | val loss | val ppl | val time
##     6 | 0.11283 | 1.119 | 0:00:05
## [{'Val Loss': 0.14961695591253893, 'Val PPL.': 1.1613892942138855}, {'Val Loss': 0.1364755311182567, 'Val PPL.': 1.146226830543859}, {'Val Loss': 0.12382005155086517, 'Val PPL.': 1.131812184825848}, {'Val Loss': 0.1152467570666756, 'Val PPL.': 1.1221503019282884}, {'Val Loss': 0.10829435022813934, 'Val PPL.': 1.1143757138609098}, {'Val Loss': 0.11282750214671805, 'Val PPL.': 1.1194388155003379}]
##
## C:\Users\Andrew\Anaconda3\envs\my_ml\lib\site-packages\torch\optim\lr_scheduler.py:123: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
##   "https://pytorch.
# organize results
pd.set_option('precision', 3)
df_train_stats = pd.DataFrame(data=training_stats)
df_valid_stats = pd.DataFrame(data=valid_stats)
df_stats = pd.concat([df_train_stats, df_valid_stats], axis=1)
df_stats.insert(0, 'Epoch', range(1, len(df_stats)+1))
df_stats = df_stats.set_index('Epoch')
print(df_stats)
##        Train Loss  Val Loss  Val PPL.
## Epoch
## 1           1.868     0.150     1.161
## 2           0.189     0.136     1.146
## 3           0.159     0.124     1.132
## 4           0.156     0.115     1.122
## 5           0.144     0.108     1.114
## 6           0.137     0.113     1.119


# test the model
df2 = pd.DataFrame({'predicted': [], 'actual': []})
test_stats = []
model.load_state_dict(torch.load('t5-summary.pt'))
## <All keys matched successfully>
testing(model, test_dataloader)
##
## Running Testing...
##   Batch    40  of     42.    Elapsed: 0:00:15.
## [{'Test Loss': 0.13512379383402212, 'Test PPL.': 1.144678479718354, 'Test Acc.': 0.8438208616780045, 'Test F1': 0.8333258198269518}]
##
## C:\Users\Andrew\Anaconda3\envs\my_ml\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
##   'precision', 'predicted', average, warn_for)
df_test_stats = pd.DataFrame(data=test_stats)
print(df_test_stats)
##    Test Loss  Test PPL.  Test Acc.  Test F1
## 0      0.135      1.145      0.844    0.833

print(df2.head(5))
##   predicted actual
## 0     Kabul  Kabul
## 1     Kabul  Kabul
## 2     Kabul     US
## 3     Kabul  Kabul
## 4     Kabul  Kabul
