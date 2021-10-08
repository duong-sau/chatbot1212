from transformers import T5ForConditionalGeneration, T5Tokenizer
if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    input_ids = tokenizer("stsb sentence1: hello. sentence2: hi",  truncation=True, padding='max_length', max_length=300, return_tensors="pt")['input_ids']
    label_ids = model(**input_ids)
    label = tokenizer.decode(label_ids)
