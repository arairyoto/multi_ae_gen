import os
import sys

import numpy as np
import codecs

import logging
from numpy import dtype, float32 as REAL

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(message)s')
logger = logging.getLogger(__name__)

class AutoExtend:
    def __init__(self, folder=None,iter_num=1000,dimention=None):
        self.folder = folder #folder name
        self.iter = iter_num
        self.dimention =  dimention #dimention of vector space
        self.theta = 200.0
        self.alpha = 0.3
        self.beta = 0.3
        self.gamma = 0.4
        self.max = 0

        # recording learning error
        self.error_w = 0.0
        self.error_l = 0.0
        self.error_r = 0.0

        self.w = {}
        self.s = {}
        self.l = {}
        self.E = {}
        self.D = {}
        self.G = {}
        self.G_w = {}
        self.G_s = {}
        self.R = []

        self.words = []
        self.synsets = []
        self.lemmas = []


    def loadWordVector(self, file_name):
        f = codecs.open(file_name, 'r', 'utf-8')
        lines = f.readlines()
        for line in lines:
            temp = line.strip("\n").split(" ")
            word = temp[0]
            vector = [float(x) for x in temp[1:]]

            if self.dimention is None:
                self.dimention = len(vector)

            self.w[word] = np.array(vector, dtype=REAL)
            if np.max(self.w[word]) > self.max:
                self.max = np.max(self.w[word])
            self.words.append(word)

    def normalizeWordVector(self):
        print("MAXIMUM LENGTH: %f" % self.max)
        for word in self.w.keys():
            self.w[word] /= self.max

    def loadFiles(self, file_name):
        f = codecs.open(file_name, 'r', 'utf-8')
        lines = f.readlines()
        for line in lines:
            temp = line.strip("\n").strip("\r").strip(",").split(" ")
            synset = temp[0]
            words = temp[1].split(",")
            # self.s[synset] = np.zeros(self.dimention)
            self.synsets.append(synset)
            self.D[synset] = {}
            for word in words:
                if word in self.words:
                    self.D[synset][word] = np.zeros(self.dimention, dtype=REAL)
                    if word not in self.E:
                        self.E[word] = {}
                    self.E[word][synset] = np.zeros(self.dimention, dtype=REAL)
                    self.lemmas.append([word,synset])
            for word in words:
                if word in self.words:
                    # self.D[synset][word] = np.ones(self.dimention)/len(self.D[synset])
                    # self.E[word][synset] = np.ones(self.dimention)/len(self.E[word])
                    self.D[synset][word] = np.random.rand(self.dimention).astype(REAL)-np.ones(self.dimention, dtype=REAL)/2
                    self.E[word][synset] = np.random.rand(self.dimention).astype(REAL)-np.ones(self.dimention, dtype=REAL)/2

    def loadGenerality(self, file_name):
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
            self.G[lemma] = generality+0.01
            if word not in self.G_w:
                self.G_w[word] = {}
            # self.G_w[word][synset] = generality+0.01
            if synset not in self.G_s:
                self.G_s[synset] = {}
            self.G_w[word][synset] = generality+0.01
            self.G_s[synset][word] = generality+0.01
            if word not in G_w:
                G_w[word] = 0.0
            G_w[word] += generality+0.01
            if synset not in G_s:
                G_s[synset] = 0.0
            G_s[synset] += generality+0.01

        for word in self.G_w.keys():
            for synset in self.G_w[word].keys():
                self.G_w[word][synset] /= G_w[word]
        for synset in self.G_s.keys():
            for word in self.G_s[synset].keys():
                self.G_s[synset][word] /= G_s[synset]


    def loadRelation(self, file_name):
        f = codecs.open(file_name, 'r', 'utf-8')
        lines = f.readlines()
        for line in lines:
            temp = line.strip("\n").split(" ")
            synset_in = temp[0]
            synset_out = temp[1]
            # initialize R
            self.R.append([synset_in, synset_out])

    def word_pair_training(self, word):
        # **********FORWARD**********
        # predict word vector
        w_ = np.zeros(self.dimention, dtype=REAL)

        s = {}
        for synset in self.E[word].keys():
            # initialize synset vector
            s[synset] = np.zeros(self.dimention, dtype=REAL)
            # calculating synset vector
            for _word in self.D[synset].keys():
                s[synset] += self.G_w[_word][synset]*self.E[_word][synset]*self.w[_word]
            # clipping
            s[synset] = np.clip(s[synset], -1,1)
            # predict word vector
            w_ += self.G_s[synset][word]*self.D[synset][word]*s[synset]
        # calculating delta
        dw = w_ - self.w[word]
        dw = np.clip(dw, -1, 1)
        # calculating square error
        self.error_w += sum(np.absolute(dw))

        # **********BACKWARD**********

        ds = {}
        for synset in self.E[word].keys():
            ds[synset] = self.G_s[synset][word]*self.D[synset][word]*dw
            ds[synset] = np.clip(ds[synset], -1, 1)
            # backprop
            dD = self.theta*self.alpha*self.G_s[synset][word]*s[synset]*dw
            self.D[synset][word] -= np.clip(dD, -1, 1)
            for _word in self.D[synset].keys():
                dE = self.theta*self.alpha*self.G_w[_word][synset]*self.w[_word]*ds[synset]
                self.E[_word][synset] -= np.clip(dE, -1, 1)

    def lemma_pair_training(self, word, synset):

        l = self.E[word][synset]*self.w[word]

        s = np.zeros(self.dimention, dtype=REAL)
        for _word in self.D[synset].keys():
            s += self.G_w[_word][synset]*self.E[_word][synset]*self.w[_word]

        s = np.clip(s, -1, 1)

        _l = self.D[synset][word]*s

        dl = l - _l
        dl = np.clip(dl, -1, 1)

        self.error_l += sum(np.absolute(dl))

        dE = self.theta*self.beta*self.w[word]*dl
        self.E[word][synset] -= np.clip(dE, -1, 1)
        ds = self.D[synset][word]*dl
        ds = np.clip(ds, -1, 1)
        dD = self.theta*self.beta*s*dl
        self.D[synset][word] += np.clip(dD, -1, 1)
        for _word in self.D[synset].keys():
            dE = self.theta*self.beta*self.G_w[_word][synset]*self.w[_word]*ds
            self.E[_word][synset] += np.clip(dE, -1, 1)

    def relation_pair_training(self, synset_in, synset_out):
        s_in = np.zeros(self.dimention, dtype=REAL)
        s_out = np.zeros(self.dimention, dtype=REAL)

        for _word in self.D[synset_in].keys():
            s_in += self.G_w[_word][synset_in]*self.E[_word][synset_in]*self.w[_word]
        for _word in self.D[synset_out].keys():
            s_out += self.G_w[_word][synset_out]*self.E[_word][synset_out]*self.w[_word]

        s_in = np.clip(s_in, -1, 1)
        s_out = np.clip(s_out, -1, 1)

        dr = s_in - s_out

        self.error_r += sum(np.absolute(dr))

        for word in self.D[synset_in].keys():
            self.E[word][synset_in] -= self.theta*self.gamma*self.G_w[word][synset_in]*self.w[word]*dr
        for word in self.D[synset_out].keys():
            self.E[word][synset_out] += self.theta*self.gamma*self.G_w[word][synset_out]*self.w[word]*dr

    def word_batch_training(self):
        # initialize error
        self.error_w = 0
        for word in self.words:
            self.word_pair_training(word)
        self.error_w /= len(self.words)

    def lemma_batch_training(self):
        # initialize error
        self.error_l = 0
        for lemma in self.lemmas:
            self.lemma_pair_training(lemma[0], lemma[1])
        self.error_l /= len(self.lemmas)

    def relation_batch_training(self):
        # initialize error
        self.error_r = 0
        for r in self.R:
            self.relation_pair_training(r[0], r[1])
        self.error_r /= len(self.R)

    def train(self):
        print("Loading files ...")
        self.loadWordVector(self.folder+'words.txt')
        print("  Normalize word vector ...")
        self.normalizeWordVector()
        print("      DONE!!")
        self.loadFiles(self.folder+'synset.txt')
        self.loadGenerality(self.folder+'generality.txt')
        self.loadRelation(self.folder+'relations.txt')
        print("    DONE!!")

        print("Training model ...")
        for i in range(self.iter):
            if (i+1)%5==0:
                logger.info("PROGRESS : %i / %i", (i+1), self.iter)
                print("Error word : %f" % self.error_w)
                print("Error lemma : %f" % self.error_l)
                print("Error relation : %f" % self.error_r)
                print("Error all told : %f" % (self.error_w+self.error_l+self.error_r))
            self.word_batch_training()
            self.lemma_batch_training()
            self.relation_batch_training()

        print("    DONE!!")

    def save_model(self):
        print("Saving model ...")
        fSynsets = codecs.open(self.folder+'naive/synsets.txt','w','utf-8')
        fLemmas = codecs.open(self.folder+'naive/lemmas.txt','w','utf-8')

        for synset in self.synsets:
            self.s[synset] = np.zeros(self.dimention)
        for lemma in self.lemmas:
            word = lemma[0]
            synset = lemma[1]
            fLemmas.write(synset+":"+word+" ")
            vecText = ""
            for d in range(self.dimention):
                vecText += str(self.E[word][synset][d]*self.w[word][d])+" "
                self.s[synset][d] += self.G_w[word][synset]*self.E[word][synset][d]*self.w[word][d]
            fLemmas.write(vecText.strip(" ")+"\n")

        for synset, vector in self.s.items():
            fSynsets.write(synset+" ")
            vecText = ""
            for d in range(self.dimention):
                vecText += str(vector[d])+" "
            fSynsets.write(vecText.strip(" ")+"\n")
        print("    DONE!!")

    def load_model(self, folder):
        f = codecs.open(folder+'words.txt', 'r', 'utf-8')
        lines = f.readlines()
        for line in lines:
            temp = line.strip("\n").split(" ")
            word = temp[0]
            vector = [float(x) for x in temp[1:]]
            self.w[word] = vector
        f = codecs.open(folder+'synsets.txt', 'r', 'utf-8')
        lines = f.readlines()
        for line in lines:
            temp = line.strip("\n").split(" ")
            synset = temp[0]
            vector = [float(x) for x in temp[1:]]
            self.s[synset] = vector
        f = codecs.open(folder+'lemmas.txt', 'r', 'utf-8')
        lines = f.readlines()
        for line in lines:
            temp = line.strip("\n").split(" ")
            lemma = temp[0]
            synset = lemma.split(":")[0]
            word = lemma.split(":")[1]
            vector = [float(x) for x in temp[1:]]
            if synset not in self.l:
                self.l[synset] = {}
            self.l[synset][word] = vector
        self.loadGenerality(folder+'generality.txt')

ae = AutoExtend(folder="/Users/arai9814/Desktop/ae_generality/eng/",iter_num = 10000)
ae.train()
ae.save_model()
#
# ae = AutoExtend()
# ae.load_model("test_folder/naive/")
# print(ae.w["bbb"])
# print(ae.l["hhh.a.01"]["bbb"])
# print(ae.l["ggg.v.03"]["bbb"])
# print(ae.G_w["bbb"])
#
# print(ae.G_s["hhh.a.01"])
# n = len(ae.G_s["hhh.a.01"])
# result = np.var(list(ae.G_s["hhh.a.01"].values()))*n*n/(n-1)
# print(result)
