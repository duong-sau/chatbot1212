def toT5sentence(sentence1, sentence2):
    prefix = 'stsb '
    s1 = 'sentence1: '
    s2 = '. sentence2: '
    T5Sentence = prefix + s1 + sentence1 + s2 + sentence2
    return T5Sentence
