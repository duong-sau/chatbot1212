from transformers import AutoModel, AutoTokenizer

from Model.FineTurn.Define import MODEL

tokenizer = AutoTokenizer.from_pretrained(MODEL['name'])
model = AutoModel.from_pretrained(MODEL['name'])
model.cpu()

index = 0
for layer in model.base_model.encoder.layer:
    index = index +1
print(index)
