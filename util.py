import os
import sys

import numpy as np
from gensim.models import word2vec
# from gensim.models import KeyedVectors

import codecs

class Shared:
    def __init__(self):
        self.G = {}
        self.G_w = {}
        self.G_s = {}

    def loadModel(self, file_name):
        self.model = word2vec.Word2Vec.load(file_name)
        self.is_w2v = True

    def loadGoogleModel(self, file_name):
        self.model = KeyedVectors.load_word2vec_format(file_name, binary=True)
        self.is_w2v = True

    def loadTxtModel(self, file_name):
        f = codecs.open(file_name, 'r', 'utf-8')
        line = f.readline()
        lines = []

        while line:
            lines.append(line)
            line = f.readline()
        f.close()
        model = {}

        for idx, l in enumerate(lines):
            #最後の改行を除いてスペースでスプリット
            temp = l.replace("\n", "").split(" ")
            word = temp[0]
            # word = temp[0].split("-")[2:]
            # synset = wnm.wn._synset_from_pos_and_offset(word[1], int(word[0]))
            embedding = [float(x) for x in temp[1:]]
            model[word] = embedding
            # dic[synset.name()] = embedding
        self.model = model
        self.is_w2v = False

    def loadLemmaGenerality(self, file_name):
        f = codecs.open(file_name, 'r', 'utf-8')
        lines = f.readlines()
        G_w = {}
        G_s = {}
        for line in lines:
            temp = line.strip("\n").split(" ")
            lemma = temp[0]
            synset = lemma.split(":")[0]
            word = lemma.split(":")[1]
            generality = float(temp[1])

            self.G[lemma] = generality

            if word not in self.G_w:
                self.G_w[word] = {}
            self.G_w[word][synset] = generality

            if synset not in self.G_s:
                self.G_s[synset] = {}
            self.G_w[word][synset] = generality
            self.G_s[synset][word] = generality

            if word not in G_w:
                G_w[word] = 0
            G_w[word] += generality
            if synset not in G_s:
                G_s[synset] = 0
            G_s[synset] += generality

        for word in self.G_w.keys():
            for synset in self.G_w[word].keys():
                if G_w[word] is not 0:
                    self.G_w[word][synset] /= G_w[word] #/len(self.G_w[word])
        for synset in self.G_s.keys():
            for word in self.G_s[synset].keys():
                if G_s[synset] is not 0:
                    self.G_s[synset][word] /= G_s[synset] #/len(self.G_s[synset])

    def loadGenerality(self, file_name):
        f = codecs.open(file_name, 'r', 'utf-8')
        lines = f.readlines()
        for line in lines:
            temp = line.strip("\n").split(" ")
            name = temp[0]
            generality = float(temp[1])

            self.G[name] = generality

    def in_vocab(self, key):
        if self.is_w2v:
            if key in self.model.vocab:
                return True
            else:
                return False
        else:
            if key in self.model:
                return True
            else:
                return False


    def getVectorAsString(self, vector):
        sb = ""
        for i in range(len(vector)):
            sb += str(vector[i])
            sb += " "
        return sb.strip()
