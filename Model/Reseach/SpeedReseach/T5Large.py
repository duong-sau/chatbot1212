from transformers import T5Tokenizer, T5TokenizerFast, T5ForConditionalGeneration, pipeline

from Model.Reseach.SpeedReseach.Common import TeZuPipeline, huggingFacePipeline

name = 't5-large'


def large_start():
    model = T5ForConditionalGeneration.from_pretrained(name)
    token = T5Tokenizer.from_pretrained(name)
    fast = T5TokenizerFast.from_pretrained(name)
    ppl = pipeline("text2text-generation", model=model, tokenizer=token)

    TeZuPipeline(fast=False, tokenizer=token, model=model, name=name, do_sample=True)
    TeZuPipeline(fast=False, tokenizer=token, model=model, name=name, do_sample=False)
    TeZuPipeline(fast=True, tokenizer=fast, model=model, name=name, do_sample=False)
    huggingFacePipeline(ppl)
