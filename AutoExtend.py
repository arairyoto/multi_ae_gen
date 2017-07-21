import os
import sys

import numpy as np
import codecs

class AutoExtend:
    def __init__(self, folder=None,iter_num=1000,dimention=None):
        self.folder = folder #folder name
        self.iter = iter_num
        self.dimention =  dimention #dimention of vector space
        self.theta = 0.1
        # for s -s
        self.alpha = 0.4 *self.theta
        # for l -l
        self.beta = 0.4 *self.theta
        # for relation
        self.gamma = (1- self.alpha -self.beta)*self.theta
        # recording learning error
        self.error_w = 0
        self.error_l = 0
        self.error_r = 0

        self.w = {}
        self.s = {}
        self.l = {}
        self.E = {}
        self.D = {}
        self.G = {}
        self.G_w = {}
        self.G_s = {}
        self.R = []
        self._w = {}

        self.delta_w = {}
        self.delta_l = {}
        self.delta_r = {}

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

            self.w[word] = np.array(vector)
            self.words.append(word)
            # initialize delta_w
            self.delta_w[word] =np.zeros(self.dimention)

    def normalizeWordVector(self):
        for word in self.w.keys():
            if sum(self.w[word]*self.w[word]) != 0.0:
                self.w[word] /= np.sqrt(sum(self.w[word]*self.w[word]))

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
                    self.D[synset][word] = np.zeros(self.dimention)
                    if word not in self.E:
                        self.E[word] = {}
                    self.E[word][synset] = np.zeros(self.dimention)
                    self.lemmas.append([word,synset])
                    if word not in self.delta_l:
                        self.delta_l[word] = {}
                    self.delta_l[word][synset] = np.zeros(self.dimention)
            for word in words:
                if word in self.words:
                    self.D[synset][word] = np.ones(self.dimention)/len(self.D[synset])
                    self.E[word][synset] = np.ones(self.dimention)/len(self.E[word])

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
                if G_w[word] != 0.0:
                    self.G_w[word][synset] /= G_w[word]
        for synset in self.G_s.keys():
            for word in self.G_s[synset].keys():
                if G_s[synset] != 0.0:
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
            #initialize delta_r
            if synset_in not in self.delta_r:
                self.delta_r[synset_in] = {}
            self.delta_r[synset_in][synset_out] = np.zeros(self.dimention)

    def forward(self):
        #initialize
        self.error_w = 0
        self.error_l = 0
        self.error_r = 0

        for word in self.words:
            self._w[word] = np.zeros(self.dimention)
        for synset in self.synsets:
            self.s[synset] = np.zeros(self.dimention)

        for lemma in self.lemmas:
            word = lemma[0]
            synset = lemma[1]
            for d in range(self.dimention):
                self.s[synset][d] += self.G_w[word][synset]*self.E[word][synset][d]*self.w[word][d]
        for lemma in self.lemmas:
            word = lemma[0]
            synset = lemma[1]
            for d in range(self.dimention):
                self._w[word][d] += self.G_s[synset][word]*self.D[synset][word][d]*self.s[synset][d]

        for word in self.words:
            self.delta_w[word] = self._w[word] - self.w[word]
            self.error_w += sum(self.delta_w[word]*self.delta_w[word])

        for lemma in self.lemmas:
            word = lemma[0]
            synset = lemma[1]
            for d in range(self.dimention):
                self.delta_l[word][synset][d] = self.E[word][synset][d]*self.w[word][d] - self.D[synset][word][d]*self.s[synset][d]
                self.error_l += self.delta_l[word][synset][d]**2

        for r in self.R:
            synset_in = r[0]
            synset_out = r[1]
            self.delta_r[synset_in][synset_out] = self.s[synset_in] - self.s[synset_out]
            self.error_r += sum(self.delta_r[synset_in][synset_out]*self.delta_r[synset_in][synset_out])

        # print(self.E)

    def backward(self):

        E_temp = self.E
        D_temp = self.D

        for lemma in self.lemmas:
            word = lemma[0]
            synset = lemma[1]
            for d in range(self.dimention):
                self.D[synset][word][d] -= self.alpha*self.G_s[synset][word]*self.delta_w[word][d]*self.s[synset][d]
                self.E[word][synset][d] -= self.alpha*self.G_s[synset][word]*self.G_w[word][synset]*self.delta_w[word][d]*self.w[word][d]*D_temp[synset][word][d]

                self.D[synset][word][d] += self.beta*self.delta_l[word][synset][d]*self.s[synset][d]
                self.E[word][synset][d] -= self.beta*self.delta_l[word][synset][d]*self.w[word][d]*(1-D_temp[synset][word][d]*self.G_w[word][synset])

        for r in self.R:
            synset_in = r[0]
            synset_out = r[1]
            for d in range(self.dimention):
                for word in self.D[synset_in].keys():
                    self.E[word][synset_in][d] -= self.gamma*self.delta_r[synset_in][synset_out][d]*self.G_w[word][synset_in]*self.w[word][d]
                for word in self.D[synset_out].keys():
                    self.E[word][synset_out][d] += self.gamma*self.delta_r[synset_in][synset_out][d]*self.G_w[word][synset_out]*self.w[word][d]


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
            if (i+1)%1000==0:
                print("%d / %d ended" % (i+1,self.iter) )
                print("Error word : %f" % self.error_w)
                print("Error lemma : %f" % self.error_l)
                print("Error relation : %f" % self.error_r)
                print("Error all told : %f" % (self.error_w+self.error_l+self.error_r))

            self.forward()
            self.backward()
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

ae = AutoExtend(folder="/Users/arai9814/Desktop/ae_generality/eng/",iter_num = 1000)
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
