import os
import sys
import logging
#WordNet
import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

from scipy.stats import spearmanr

#dealing with japanese
mpl.rcParams['font.family'] = 'AppleGothic'

import numpy as np
import codecs
import csv

import util

# ログ
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(message)s')
logger = logging.getLogger(__name__)

class WSLObject:
    def __init__(self, name, attribute, lang = None):
        self.name = name
        self.attribute = attribute
        self.lang = lang


class MultilingualWordVector:
    def __init__(self, folder, langs, generality=False):
        self.folder = folder
        self.langs = langs
        self.map = ['word', 'lemma']
        self.N = 30
        self.generality = generality

        self.get = {}

        self.get[('synset', None)] = util.Shared()
        self.get[('synset', None)].loadTxtModel(self.folder+"/synsets.txt")

        for lang in langs:
            self.get[('word', lang)] = util.Shared()
            self.get[('word', lang)].loadTxtModel(self.folder+"/"+lang+"/words.txt")
            self.get[('lemma', lang)] = util.Shared()
            self.get[('lemma', lang)].loadTxtModel(self.folder+"/"+lang+"/lemmas.txt")
        print("MultilingualWordVector READY!")

        if self.generality:
            self.generality = {}

            self.generality[('synset', None)] = util.Shared()
            self.generality[('synset', None)].loadGenerality(self.folder+"/generality/synsets.txt")

            for lang in langs:
                self.generality[('word', lang)] = util.Shared()
                self.generality[('word', lang)].loadGenerality(self.folder+"/"+lang+"/generality/words.txt")
                self.generality[('lemma', lang)] = util.Shared()
                self.generality[('lemma', lang)].loadLemmaGenerality(self.folder+"/"+lang+"/generality/lemmas.txt")
            print("MultilingualLemmaGenerality READY!")
        else:
            print("MultilingualLemmaGenerality DON'T USE!")


    #e = [lang, attr(word or synset or lemma), id]
    def relatedness(self, e1, e2):
        try:
            v1 = np.array(self.get[(e1.attribute, e1.lang)].model[e1.name])
            v2 = np.array(self.get[(e2.attribute, e2.lang)].model[e2.name])
            return sum(v1*v2)/np.sqrt(sum(v1*v1)*sum(v2*v2))
        except:
            return -1

    # generality in paticular language
    def generality(self, e):
        try:
            return self.generality[(e.attribute, e.lang)].G[e.name]
        except:
            return -1

    # word generality for some synset or synset generality for some word
    def _generality(self, e, lang='eng'):
        if e.attribute is 'lemma':
            lemma = e.name
            synset = lemma.split(':')[0]
            word = lemma.split(':')[1]
            return self.generality[('lemma', e.lang)].G_s[synset], self.generality[('lemma', e.lang)].G_w[word]
        if e.attribute is 'synset':
            return self.generality[('lemma', lang)].G_s[e.name]
        if e.attribute is 'word':
            return self.generality[('lemma', e.lang)].G_w[e.name]

    # unambiguity as var of generality list
    def unambiguity(self, e):
        if e.attribute is 'lemma':
            lemma = e.name
            synset = lemma.split(':')[0]
            word = lemma.split(':')[1]
            n_s =  len(self.generality[('lemma', lang)].G_s[synset])
            n_w = len(self.generality[('lemma', lang)].G_w[word])
            var_s = np.var(list(self.generality[('lemma', lang)].G_s[synset].values()))*n_s*n_s/(n_s-1)
            var_w = np.var(list(self.generality[('lemma', lang)].G_w[word].values()))*n_w*n_w/(n_w-1)
            return np.sqrt(self.generality[('lemma', lang)].G_s[synset][word]*self.generality[('lemma', lang)].G_w[word][synset]*np.sqrt(var_s)*np.sqrt(var_w))

        if e.attribute is 'synset':
            n = len(self.generality[('lemma', lang)].G_s[e.name])
            return np.var(list(self.generality[('lemma', lang)].G_s[e.name].values()))*n*n/(n-1)
        if e.attribute is 'word':
            n = len(self.generality[('lemma', e.lang)].G_w[e.name])
            return np.var(list(self.generality[('lemma', e.lang)].G_w[e.name].values()))*n*n/(n-1)

    def shortest_path(self, e_in, e_out):
        langs = self.langs
        relation_map = ["@", "&", "$"] #hypernym, similar, verbGroup
        shortest_path = 0
        E_all = []
        E = []
        E_all.append(e_in)
        E.append(e_in)

        #maiximum depth searching for is N
        for i in range(self.N):
            if e_out not in E:
                E_temp = []
                for e in E:
                    if e.attribute is 'synset':
                        synset = wn.synset(e.name)
                        for relation_symbol in relation_map:
                            for s in synset._related(relation_symbol):
                                e_temp = WSLObject(s.name(),'synset')
                                if e_temp not in E_all:
                                    E_temp.append(e_temp)
                                    E_all.append(e_temp)
                        for lang in langs:
                            for lemma in synset.lemmas(lang = lang):
                                word = lemma.name().lower()
                                e_temp = WSLObject(synset.name()+':'+word, 'lemma', e.lang)
                                if e_temp not in E_all:
                                    E_temp.append(e_temp)
                                    E_all.append(e_temp)

                    if e.attribute is 'word':
                        for s in wn.synsets(e.name, lang = e.lang):
                            e_temp = WSLObject(s.name()+':'+e.name ,'lemma', e.lang)
                            if e_temp not in E_all:
                                E_temp.append(e_temp)
                                E_all.append(e_temp)

                    if e.attribute is 'lemma':
                        synset_name = e.name.split(":")[0]
                        word_name = e.name.split(":")[1]
                        e_temp = WSLObject(synset_name, 'synset')
                        if e_temp not in E_all:
                            E_temp.append(e_temp)
                            E_all.append(e_temp)
                        e_temp = WSLObject(word_name, 'word', e.lang)
                        if e_temp not in E_all:
                            E_temp.append(e_temp)
                            E_all.append(e_temp)

                    E = E_temp
                    shortest_path += 1

        return shortest_path


    def plot2D(self, file_name, descriptors):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for d in descriptors:
            try:
                v = self.get[(d.attribute, d.lang)].model[d.name]
                #ベクトルの1,2次元に射影
                x = v[0]
                y = v[1]
                w = d.name+'@'+d.lang
                ax.plot(x,y, ".")
                ax.annotate(w, xy=(x, y), size=10)
            except:
                continue
        plt.savefig(file_name+'.png' )
        plt.show()

    def output_relatedness_matrix(self, file_name, descriptors):
        f = codecs.open(file_name+'.txt', 'w', 'utf-8')
        f.write(",")
        for d in descriptors:
            f.write(d.name+"@"+d.lang+"@"+d.attribute+",")
        f.write("\n")
        for d1 in descriptors:
            f.write(d1.name+"@"+d1.lang+"@"+d1.attribute+",")
            for d2 in descriptors:
                    f.write(str(self.relatedness(d1,d2)))
                    f.write(",")
            f.write("\n")
        f.close()
        print("OUTPUT FILE DONE!")

    # ************TEST************

    def scws_test(self, progress_per = 100):
        N = 999 #datanum
        lang = 'eng'
        df = pd.read_csv('Test/input/ratings1.csv')

        Lists = {}
        Lists['s1'] = []
        Lists['s2'] = []
        Lists['result'] = []

        logger.info('SCWS TEST : START')

        for i in range(N):
            if i % progress_per == 0:
                logger.info('SCWS TEST : %i/%i DONE', i ,N )

            # word1の文脈ベクトル算出
            c1 = re.sub(r'[^a-zA-Z ]', '', df['Context1'][i]).lower().split(' ')
            c1_list = []
            for c in c1:
                if c in self.get[('word', lang)].model:
                    c1_list.append(self.get[('word', lang)].model[c])
            c1_vec = sum(c1_list)

            # word2の文脈ベクトル算出
            c2 = re.sub(r'[^a-zA-Z ]', '', df['Context2'][i]).lower().split(' ')
            c2_list = []
            for c in c2:
                if c in self.get[('word', lang)].model:
                    c2_list.append(self.get[('word', lang)].model[c])
            c2_vec = sum(c2_list)

            sim1 = 0

            for s1 in wn.synsets(df['Word1'][i], pos = df['POS1'][i]):
                if s1.name() in self.get[('synset', None)].model:
                    syn_vec = self.get[('synset', None)].model[s1.name()]
                    if syn_vec is not 0:
                        temp1 = sum(np.array(syn_vec)*c1_vec)/np.sqrt(sum(np.array(syn_vec)*np.array(syn_vec))*sum(c1_vec*c1_vec))
                    else:
                        temp1 = 0
                else:
                    temp1 = 0

                if temp1 > sim1:
                    s1c = s1.name()
                    s1vec = syn_vec
                    sim1 = temp1
            if sim1 is 0:
                Lists['s1'].append('None')
            else:
                Lists['s1'].append(s1c)

            sim2 = 0
            # for s2 in WN17.synsets(df['Word2'][i]):
            for s2 in wn.synsets(df['Word2'][i], pos = df['POS2'][i]):
                if s2.name() in self.get[('synset', None)].model:
                    syn_vec = self.get[('synset', None)].model[s1.name()]
                    if syn_vec is not 0:
                        temp2 = sum(np.array(syn_vec)*c2_vec)/np.sqrt(sum(syn_vec*syn_vec)*sum(c2_vec*c2_vec))
                    else:
                        temp2 = 0
                else:
                    temp2 = 0

                if temp2 > sim2:
                    s2c = s2.name()
                    s2vec = syn_vec
                    sim2 = temp2
            if sim2 is 0:
                Lists['s2'].append('None')
            else:
                Lists['s2'].append(s2c)

            if (sim1 == 0) or (sim2 == 0):
                Lists['result'].append(-1)
            else:
                Lists['result'].append(sum(s1vec*s2vec)/np.sqrt(sum(s1vec*s1vec)*sum(s2vec*s2vec)))

        df['s1'] = Lists['s1']
        df['s2'] = Lists['s2']
        df['result'] = Lists['result']

        df.to_csv('Test/output/scws_raiting_synset.csv')

        df = pd.read_csv('Test/output/scws_raiting_synset.csv')

        mean = df['mean']
        r = spearmanr(mean, df['result'])
        print(r)

    def scws_test_lemma(self, progress_per = 100):
        N = 999 #datanum
        lang = 'eng'
        df = pd.read_csv('Test/input/ratings1.csv')

        Lists = {}
        Lists['l1'] = []
        Lists['l2'] = []
        Lists['result'] = []

        logger.info('SCWS TEST@LEMMA : START')

        for i in range(N):
            if i % progress_per == 0:
                logger.info('SCWS TEST : %i/%i DONE', i ,N )

            # word1の文脈ベクトル算出
            c1 = re.sub(r'[^a-zA-Z ]', '', df['Context1'][i]).lower().split(' ')
            c1_list = []
            for c in c1:
                if c in self.get[('word', lang)].model:
                    c1_list.append(self.get[('word', lang)].model[c])
            c1_vec = sum(c1_list)

            # word2の文脈ベクトル算出
            c2 = re.sub(r'[^a-zA-Z ]', '', df['Context2'][i]).lower().split(' ')
            c2_list = []
            for c in c2:
                if c in self.get[('word', lang)].model:
                    c2_list.append(self.get[('word', lang)].model[c])
            c2_vec = sum(c2_list)

            sim1 = 0

            for s1 in wn.synsets(df['Word1'][i], pos = df['POS1'][i]):
                if s1.name()+':'+df['Word1'][i] in self.get[('lemma', lang)].model:
                    lemma_vec = self.get[('lemma', lang)].model[s1.name()+':'+df['Word1'][i]]
                    if lemma_vec is not 0:
                        temp1 = sum(np.array(lemma_vec)*c1_vec)/np.sqrt(sum(np.array(lemma_vec)*np.array(lemma_vec))*sum(c1_vec*c1_vec))
                    else:
                        temp1 = 0
                else:
                    temp1 = 0

                if temp1 > sim1:
                    l1c = s1.name()+':'+df['Word1'][i]
                    l1vec = lemma_vec
                    sim1 = temp1
            if sim1 is 0:
                Lists['l1'].append('None')
            else:
                Lists['l1'].append(l1c)

            sim2 = 0
            # for s2 in WN17.synsets(df['Word2'][i]):
            for s2 in wn.synsets(df['Word2'][i], pos = df['POS2'][i]):
                if s2.name()+':'+df['Word2'][i] in self.get[('lemma', lang)].model:
                    lemma_vec = self.get[('lemma', lang)].model[s2.name()+':'+df['Word2'][i]]
                    if lemma_vec is not 0:
                        temp2 = sum(np.array(lemma_vec)*c2_vec)/np.sqrt(sum(lemma_vec*lemma_vec)*sum(c2_vec*c2_vec))
                    else:
                        temp2 = 0
                else:
                    temp2 = 0

                if temp2 > sim2:
                    l2c = s2.name()+':'+df['Word2'][i]
                    l2vec = lemma_vec
                    sim2 = temp2
            if sim2 is 0:
                Lists['l2'].append('None')
            else:
                Lists['l2'].append(l2c)

            if (sim1 == 0) or (sim2 == 0):
                Lists['result'].append(-1)
            else:
                Lists['result'].append(sum(l1vec*l2vec)/np.sqrt(sum(l1vec*l1vec)*sum(l2vec*l2vec)))

        df['l1'] = Lists['l1']
        df['l2'] = Lists['l2']
        df['result'] = Lists['result']

        df.to_csv('Test/output/scws_raiting_lemma.csv')

        df = pd.read_csv('Test/output/scws_raiting_lemma.csv')

        mean = df['mean']
        r = spearmanr(mean, df['result'])
        print(r)

    def sim353_test(self, progress_per=100):
        N = 353
        df = pd.read_csv('Test/input/combined.csv')

        Lists = {}
        Lists['s1'] = []
        Lists['s2'] = []
        Lists['result'] = []

        for i in range(N):
            if i % progress_per == 0:
                logger.info('SIM353 TEST : %i/%i DONE', i ,N )

            Temps = []
            S1 = []
            S2 = []

            for s1 in wn.synsets(df['Word 1'][i]):
                if s1.name() in self.get[('synset', None)].model:
                    syn1vec = self.get[('synset', None)].model[s1.name()]
                S1.append(s1.name())

                for s2 in wn.synsets(df['Word 2'][i]):
                    if s2.name() in self.get[('synset', None)].model:
                        syn2vec = self.get[('synset', None)].model[s2.name()]
                    S2.append(s2.name())

                    try:
                        temp = sum(syn1vec*syn2vec)/np.sqrt(sum(syn1vec*syn1vec)*sum(syn2vec*syn2vec))
                        Temps.append(temp)
                    except:
                        Temps.append(-1)

            max_index = Temps.index(max(Temps))[0]
            Lists['s1'].append(S1[max_index])
            Lists['s2'].append(S2[max_index])
            Lists['result'].append(Temps[max_index])

        df['s1'] = Lists['s1']
        df['s2'] = Lists['s2']
        df['result'] = Lists['result']

        df.to_csv('Test/output/sim353_test.csv')

        df = pd.read_csv('Test/output/sim353_test.csv')

        mean = df['Human (mean)']
        r = spearmanr(mean, df['result'])
        print(r)

    def sim353_test_lemma(self, progress_per=100):
        N = 353
        lang = 'eng'
        df = pd.read_csv('Test/input/combined.csv')

        Lists = {}
        Lists['l1'] = []
        Lists['l2'] = []
        Lists['result'] = []

        for i in range(N):
            if i % progress_per == 0:
                logger.info('SIM353 TEST@LEMMA : %i/%i DONE', i ,N )

            Temps = []
            L1 = []
            L2 = []

            for s1 in wn.synsets(df['Word 1'][i]):
                if s1.name()+':'+df['Word 1'][i] in self.get[('lemma', lang)].model:
                    lemma1vec = self.get[('lemma', lang)].model[s1.name()+':'+df['Word 1'][i]]
                L1.append(s1.name()+':'+df['Word 1'][i])

                for s2 in wn.synsets(df['Word 2'][i]):
                    if s2.name() in self.get[('lemma', lang)].model:
                        lemma2vec = self.get[('lemma', lang)].model[s1.name()+':'+df['Word 2'][i]]
                    L2.append(s1.name()+':'+df['Word 2'][i])

                    try:
                        temp = sum(lemma1vec*lemma2vec)/np.sqrt(sum(lemma1vec*lemma1vec)*sum(lemma2vec*lemma2vec))
                        Temps.append(temp)
                    except:
                        Temps.append(-1)

            max_index = Temps.index(max(Temps))[0]
            Lists['l1'].append(L1[max_index])
            Lists['l2'].append(L2[max_index])
            Lists['result'].append(Temps[max_index])

        df['l1'] = Lists['l1']
        df['l2'] = Lists['l2']
        df['result'] = Lists['result']

        df.to_csv('Test/output/sim353_test_lemma.csv')

        df = pd.read_csv('Test/output/sim353_test_lemma.csv')

        mean = df['Human (mean)']
        r = spearmanr(mean, df['result'])
        print(r)


def load_csv(file_name):
    result = []
    with codecs.open(file_name, 'r', 'utf-8') as f:
        reader = csv.reader(f)
        header = next(reader) #skip header

        for row in reader:
            name = row[0]
            attribute = row[1]
            lang = row[2]
            obj = WSLObject(name, attribute, lang)
            result.append(obj)
    print("LOADING CSV DONE!!")
    return result






#****************************************************Test****************************************************

# #folder for 'ae'
# folder = '/Users/arai9814/Desktop/ae'
# # folder = 'ae'
# langs = ['eng', 'jpn', 'fra']
#
# #descriptor file_name
# file_name = 'descriptors_utf8.csv'
#
# mwv = MultilingualWordVector(folder, langs)
#
# descriptors = load_csv(file_name)
#
# #generating relatedness matrix
# mwv.output_relatedness_matrix('geneva_positive_list', descriptors)
# #generating 2d plot
# mwv.plot2D('geneva_positive_list', descriptors)
