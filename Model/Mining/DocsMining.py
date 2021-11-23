import os

import nltk
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import re

from Model.Mining.Common import group_by_tag
from Static.Define import PathCommon, Tag

page_links = ['', 'Advanced-Tutorial', 'Assessing-Phylogenetic-Assumptions', 'Tutorial', 'Command-Reference',
              'Compilation-Guide', 'Complex-Models', 'Concordance-Factor', 'Developer-Guide',
              'Frequently-Asked-Questions', 'Quickstart', 'Home', 'Dating', 'Polymorphism-Aware-Models', 'Rootstrap',
              'Substitution-Models', 'Web-Server-Tutorial']


def clean_text(sentence):
    s1 = re.sub('[!@#$:\n\t]', '', sentence)
    s2 = re.sub('[ +]', ' ', s1)
    s3 = re.sub('IQ-TREE', ' ', s2)
    return s3


def sentence_combine2(sentences):
    result = []
    es = nltk.sent_tokenize(sentences)
    for s1 in es:
        for s2 in es:
            sentence = clean_text(s1 + s2)
            if '---' in sentence:
                continue
            result.append(sentence)
    return result


def sentence_combine3(sentences):
    result = []
    es = nltk.sent_tokenize(sentences)
    for s1 in es:
        for s2 in es:
            for s3 in es:
                sentence = clean_text(s1 + s2 + s3)
                if '---' in sentence:
                    continue
                result.append(sentence)
    return result


def sentence_make(sentences):
    result = []
    es = nltk.sent_tokenize(sentences)
    for i in range(len(es)):
        list1 = es[:i]
        list2 = es[i:]
        m1 = ""
        for s in list1:
            m1 = m1 + s
        sentence = clean_text(m1)
        if '---' in sentence:
            continue
        result.append(sentence)
        m2 = ""
        for s2 in list2:
            m2 = m2 + s2
        sentence = clean_text(m2)
        if '---' in sentence:
            continue
        result.append(sentence)
    return result


def sentence_concat(sentences, depth):
    result = []
    es = nltk.sent_tokenize(sentences)
    count = 0
    sentence = ""
    for index, value in enumerate(es):
        if not value == "":
            count += 1
            sentence = sentence + value
            if count == depth or index == len(es) - 1:
                sentence = clean_text(sentence)
                if '---' in sentence:
                    continue
                result.append(sentence)
                count = 0
                sentence = ""
    return result


def train_mining():
    sentence_df = pd.DataFrame()
    label_df = pd.DataFrame()
    cluster_df = pd.DataFrame()
    answer_df = pd.DataFrame()
    sentence_index = 0
    label_index = 0
    cluster_index = 0
    for root, dirs, files in os.walk(PathCommon.data + "\\Document", topdown=True):
        for file in tqdm(files):
            path = root + "\\" + file
            with open(file=path, mode='r', encoding='utf-8') as f:
                content = f.read()
                html = BeautifulSoup(content.strip("\n"), 'html.parser')
                cluster_index = cluster_index + 1
                cluster = html.h1.text
                cluster_new = {"cluster_index": cluster_index, "cluster": cluster}
                cluster_df = cluster_df.append(cluster_new, ignore_index=True)
                group_by_tag(html, Tag.h2, Tag.classify1)
                head = html.find_all(Tag.classify1)
                for h in head:
                    elements = h.find_all(recursive=False)
                    h2 = h.h2
                    label_index = label_index + 1
                    label = h2.text
                    label_new = {'label_index': label_index, 'label': label, "cluster_index": cluster_index,
                                 "cluster": cluster}
                    label_df = label_df.append(label_new, ignore_index=True)
                    answer = 'http://www.iqtree.org/doc/' + page_links[cluster_index] + '#' + elements[0]['id']
                    new = {'answer_index': label_index, 'answer': answer, 'first': elements[2].text,
                           'label_index': label_index, 'label': label,
                           'cluster_index': cluster_index, "cluster": cluster, }
                    answer_df = answer_df.append(new, ignore_index=True)
                    for e in elements:
                        value = e.text
                        if not value == "":
                            sentence_index = sentence_index + 1
                            sentence = clean_text(e.text)
                            if '---' in sentence:
                                continue
                            sentence_new = {'sentence_index': sentence_index, 'sentence': sentence,
                                            'label_index': label_index, 'label': label, "cluster_index": cluster_index,
                                            "cluster": cluster}
                            sentence_df = sentence_df.append(sentence_new, ignore_index=True)
                    text = h.text
                    for deep in range(1, len(nltk.sent_tokenize(text))):
                        ss = sentence_concat(text, deep)
                        for s in ss:
                            sentence_index += 1
                            sentence_new = {'sentence_index': sentence_index, 'sentence': s,
                                            'label_index': label_index, 'label': label, "cluster_index": cluster_index,
                                            "cluster": cluster}
                            sentence_df = sentence_df.append(sentence_new, ignore_index=True)
                    if len(nltk.sent_tokenize(text)) < 4:
                        ss1 = sentence_concat(text, deep)
                        ss2 = sentence_combine2(text)
                        ss3 = sentence_combine3(text)
                        ss4 = sentence_make(text)
                        ss = ss1 + ss2 + ss3 + ss4
                        for s in ss:
                            sentence_index += 1
                            sentence_new = {'sentence_index': sentence_index, 'sentence': s,
                                            'label_index': label_index, 'label': label, "cluster_index": cluster_index,
                                            "cluster": cluster}
                            sentence_df = sentence_df.append(sentence_new, ignore_index=True)
                    if len(nltk.sent_tokenize(text)) < 5:
                        ss1 = sentence_concat(text, deep)
                        ss2 = sentence_combine2(text)
                        ss3 = sentence_combine3(text)
                        ss = ss1 + ss2 + ss3
                        for s in ss:
                            sentence_index += 1
                            sentence_new = {'sentence_index': sentence_index, 'sentence': s,
                                            'label_index': label_index, 'label': label, "cluster_index": cluster_index,
                                            "cluster": cluster}
                            sentence_df = sentence_df.append(sentence_new, ignore_index=True)
                    if len(nltk.sent_tokenize(text)) < 6:
                        ss1 = sentence_concat(text, deep)
                        ss2 = sentence_combine2(text)
                        ss = ss1 + ss2
                        for s in ss:
                            sentence_index += 1
                            sentence_new = {'sentence_index': sentence_index, 'sentence': s,
                                            'label_index': label_index, 'label': label, "cluster_index": cluster_index,
                                            "cluster": cluster}
                            sentence_df = sentence_df.append(sentence_new, ignore_index=True)
        break
    label_df.to_csv(PathCommon.label_list, index=False, mode='w')
    sentence_df.to_csv(PathCommon.sentence_list, index=False, mode='w')
    cluster_df.to_csv(PathCommon.cluster_list, index=False, mode='w')
    answer_df.to_csv(PathCommon.answer_list, index=False, mode='w')


if __name__ == '__main__':
    train_mining()
