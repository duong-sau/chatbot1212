from transformers import T5Tokenizer, T5TokenizerFast, T5ForConditionalGeneration, pipeline
import time
from Model.Reseach.Common import TeZuPipeline, huggingFacePipeline

name = 't5-small'
model = T5ForConditionalGeneration.from_pretrained(name)
token = T5Tokenizer.from_pretrained(name)
tme = time.time()
print(tme)
batch_input_str = (("Mary spends $20 on pizza"), ("She likes eating it"), ("The pizza was great"))
tme = time.time()
print(tme)
print(token.batch_encode_plus(batch_input_str, pad_to_max_length=True))
print(token.encode_plus("Mary spends $20 on pizza"))
print(token.encode_plus("She likes eating it"))
print(token.encode_plus("The pizza was great"))
tme = time.time()
print(tme)
