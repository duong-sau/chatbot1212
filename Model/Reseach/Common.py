import time
import pandas as pd
from tqdm import tqdm

from Static.Define import path_common


def selfPipeline(fast, tokenizer, model, name, sentence1, sentence2, do_sample, padding=None) -> str:
    start_time = time.time()
    sentence = 'stsb sentence1: ' + sentence1 + '. sentence2: ' + sentence2
    inputs = tokenizer(sentence, max_length=512, padding='longest', return_tensors="pt")
    result_ids = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                do_sample=do_sample)
    similarity = tokenizer.batch_decode(result_ids, skip_special_tokens=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return {'model_name': name, 'fast_token': fast, 'result': similarity[0], 'length': len(sentence),
            'elapsed_time': elapsed_time,
            'do_sample': do_sample,
            'padding': padding}


def TeZuPipeline(fast, tokenizer, model, name, do_sample):
    s = 'sentence'
    sentence2 = "Binary, morphological and SNP data"
    sentence_df = pd.read_csv(path_common.test_pos.value, header=0)
    time_to_run_df = pd.DataFrame()
    for index, row in tqdm(sentence_df.iterrows(),desc='Tedzukiri'):
        sentence1 = row[s]
        result = selfPipeline(fast=fast, tokenizer=tokenizer, name=name, model=model, sentence1=sentence1,
                              sentence2=sentence2,
                              do_sample=True)
        time_to_run_df = time_to_run_df.append(result, ignore_index=True)
        time_to_run_df.to_csv("elapsed_time.csv", index=False, mode='a', header=False)


def huggingFacePipeline(pipeline):
    s = 'sentence'
    sentence2 = "Binary, morphological and SNP data"
    sentence_df = pd.read_csv(path_common.test_pos.value, header=0)
    time_to_run_df = pd.DataFrame()
    for index, row in tqdm(sentence_df.iterrows(), desc='huggingFace'):
        start_time = time.time()
        sentence1 = row[s]
        sentence = 'stsb sentence1: ' + sentence1 + '. sentence2: ' + sentence2
        r = pipeline(sentence)
        end_time = time.time()
        elapsed_time = end_time - start_time
        new = {'model_name': 'pipeline', 'fast_token': 'pipeline', 'result': r, 'length': len(sentence),
               'elapsed_time': elapsed_time,
               'do_sample': 'pipeline',
               'padding': 'pipeline'}
        time_to_run_df = time_to_run_df.append(new, ignore_index=True)
    time_to_run_df.to_csv("elapsed_time.csv", index=False, mode='a', header=False)
