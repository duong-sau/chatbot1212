import nltk
from sklearn.metrics import accuracy_score
from statistics import mean


def to_sts_sentence(sentence1, sentence2):
    prefix = 'stsb '
    s1 = 'sentence1: '
    s2 = '. sentence2: '
    sts_format_sentence = prefix + s1 + sentence1 + s2 + sentence2
    return sts_format_sentence


# def get_similarity(tokenizer, model, test_sentence, compare_sentences):
#     sentences = nltk.sent_tokenize(test_sentence)
#     s = []
#     for sentence in sentences:
#         sts_format_sentence = to_sts_sentence(sentence1=sentence, sentence2=compare_sentences)
#         inputs = tokenizer(sts_format_sentence, return_tensors="pt", padding=True)
#         output_sequences = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
#                                           do_sample=False)
#         ss = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
#         try:
#             s.append(float(ss[0]))
#         except ValueError:
#             s.append(0)
#     return mean(s)
def get_similarity(tokenizer, model, test_sentence, compare_sentences):
    sts_format_sentence = to_sts_sentence(sentence1=test_sentence, sentence2=compare_sentences)
    inputs = tokenizer(sts_format_sentence, return_tensors="pt", padding=True)
    output_sequences = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                      do_sample=False)
    ss = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    try:
        return float(ss[0])
    except ValueError:
        return 0


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }
