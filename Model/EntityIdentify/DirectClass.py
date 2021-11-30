import torch


def get_labels(labels_str_ids):
    r = []
    sssss = labels_str_ids.split()
    for s in sssss:
        k = 0
        try:
            k = float(s)
        except ValueError:
            print("value error")
            k = -1
        r.append(k)
    while len(r) < 3:
        r.append(-1)
    return r[0], r[1], r[2]


if __name__ == '__main__':
    import pandas as pd
    from tqdm.auto import tqdm
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from os import path, mkdir

    from Static.Config import tokenizer_config

    names = ['3Class0.82']
    tqdm.pandas()
    for name in names:
        model_path = 'D:\\chatbot1212\\Model\\CheckPoint/' + name + "/"
        result_dir = 'D:\\chatbot1212\\Model\\Result/' + name
        result_path = result_dir + '/result.csv'
        if not path.exists(result_dir):
            mkdir(result_dir)

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        tokenizer_config(tokenizer=tokenizer)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.cpu()

        test_link = "D:\\chatbot1212\\Model\\Cluster\\IntentClassification\\test.csv"

        test_df = pd.read_csv(test_link, header=0)
        columns = ["test_id", "expected", "actual", "max2", "max3"]
        result_df = pd.DataFrame(columns=columns)

        for index, row in tqdm(test_df.iterrows(), leave=False, total=len(test_df)):
            test_sentence = row["sentence"]
            classify_sentence = "multilabel classification: " + test_sentence
            inputs = tokenizer(classify_sentence, return_tensors="pt", padding=True)
            output_sequences = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                              do_sample=False)
            ss = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            max1, max2, max3, = get_labels(ss[0])
            new_row = {'test_id': row["sentence_index"], 'expected': row["cluster_index"], 'actual': max1, "max2": max2,
                       "max3": max3}
            result_df = result_df.append(new_row, ignore_index=True)
        result_df.to_csv(path_or_buf=result_path, mode='w', index=False)
