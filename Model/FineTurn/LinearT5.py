import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
print('start')
model = T5ForConditionalGeneration.from_pretrained('../Save/t5-small/google')
print(model)
tokenizer = T5Tokenizer.from_pretrained('t5-small')
print(tokenizer)
d_model = 512
tensors = []
linear = model.lm_head
for i in range(10000):
    tensor = torch.rand(d_model)
    output = linear(tensor)
    simi = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print(simi)