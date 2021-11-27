
if __name__ == '__main__':
    import pandas as pd
    from tqdm.auto import tqdm
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from os import path, mkdir

    from Static.Config import tokenizer_config

    names = ['DirectClass']
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

        test_link = "D:\\chatbot1212\\Model\\Data\\IntentClassification\\sentence_list.csv"

        test_df = pd.read_csv(test_link, header=0)
        columns = ["test_id", "expected", "actual", "max2", "max3"]
        result_df = pd.DataFrame(columns=columns)

        for index, row in tqdm(test_df.iterrows(), leave=False, total=len(test_df)):
            test_sentence = row["sentence"]
            classify_sentence = "classification: " + test_sentence
            inputs = tokenizer(classify_sentence, return_tensors="pt", padding=True)
            output_sequences = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                              do_sample=False)
            ss = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            try:
                max1 = float(ss[0])
            except ValueError:
                max1 = -1
                print("not found")
            new_row = {'test_id': row["sentence_index"], 'expected': row["label_index"], 'actual': max1}
            result_df = result_df.append(new_row, ignore_index=True)
        result_df.to_csv(path_or_buf=result_path, mode='w', index=False)
