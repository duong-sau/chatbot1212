import socket

if __name__ == '__main__':
    import pandas as pd
    from tqdm.auto import tqdm
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from os import path, mkdir

    from Model.Common import get_similarity
    from Static.Config import MODEL, tokenizer_config

    HOST = '127.0.0.1'
    PORT = 8000

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (HOST, PORT)
    print('connecting to %s port ' + str(server_address))
    s.connect(server_address)

    # names = []
    # for i in range(64):
    #     if i == 24:
    #         continue
    #     bString = bin(i)[2:].zfill(6)
    #     names.append(bString)
    names = ['NhuHoa']
    tqdm.pandas()
    for name in names:
        model_path = '../CheckPoint/' + name + "/"
        result_dir = '../Result/' + name
        result_path = result_dir + '/result.csv'
        if not path.exists(result_dir):
            mkdir(result_dir)
        # if path.exists(result_path):
        #     print("result exist -> duplicate run time: ", str(int(name, 2)))
        #     continue
        # else:
        #     print('start')
        #     print('start run on runtime:               ', str(int(name, 2)))
        # if not path.exists(model_path):
        #     print('Model not found in runtime:         ', str(int(name, 2)))
        #     continue

        tokenizer = T5Tokenizer.from_pretrained(model_path)
        tokenizer_config(tokenizer=tokenizer)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.cpu()

        test_link = "D:\\chatbot1212\\Model\\Data\\IntentClassification\\test.csv"

        test_df = pd.read_csv(test_link, header=0)
        columns = ["test_id", "expected", "actual", "max2", "max3"]
        result_df = pd.DataFrame(columns=columns)

        for index, row in tqdm(test_df.iterrows(), leave=False, total=len(test_df)):
            s.sendall(bytes('clr-t5', "utf8"))
            s.sendall(bytes('clr-ln', "utf8"))
            temp_df = pd.read_csv(
                "D:\\chatbot1212\\Model\\Data\\IntentClassification\\sentence_list.csv",
                header=0)
            test_sentence = row["sentence"]
            for i, r in temp_df.iterrows():
                compare_sentences = r["sentence"]
                similarity = get_similarity(tokenizer=tokenizer, model=model, test_sentence=test_sentence,
                                            compare_sentences=compare_sentences)
                s.sendall(bytes('t5_'+str(similarity), "utf8"))
                temp_df.loc[i, "similarity"] = similarity
            temp_df['similarity'] = pd.to_numeric(temp_df['similarity'], errors='coerce')
            mean_df = temp_df.groupby(["label_index"])["similarity"].mean().reset_index().sort_values("similarity")
            max1 = mean_df.iloc[-1]
            max2 = mean_df.iloc[-2]
            max3 = mean_df.iloc[-3]
            ls = temp_df[temp_df['label_index'] == max1['label_index']]['sentence_index'].tolist()
            new_row = {'test_id': row["sentence_index"], 'expected': row["label_index"], 'actual': max1["label_index"],
                       'max2': max2["label_index"], 'max3': max3["label_index"]}
            result_df = result_df.append(new_row, ignore_index=True)
        result_df.to_csv(path_or_buf=result_path, mode='w', index=False)
