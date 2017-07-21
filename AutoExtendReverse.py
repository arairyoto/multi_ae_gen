import os
import sys

import numpy as np
import codecs

class AutoExtendReverse:
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
        self.error_s = 0
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
        self._s = {}

        self.delta_s = {}
        self.delta_l = {}
        self.delta_r = {}

        self.words = []
        self.synsets = []
        self.lemmas = []


    def loadSynsetVector(self, file_name):
        f = codecs.open(file_name, 'r', 'utf-8')
        lines = f.readlines()
        for line in lines:
            temp = line.strip("\n").split(" ")
            synset = temp[0]
            vector = [float(x) for x in temp[1:]]

            if self.dimention is None:
                self.dimention = len(vector)

            self.s[synset] = vector
            self.synsets.append(synset)
            # initialize delta_w
            self.delta_s[synset] =np.zeros(self.dimention)

    def normalizeSynsetVector(self):
        for synset in self.s.keys():
            if sum(self.s[synset]*self.s[synset]) != 0.0:
                self.s[synset] /= np.sqrt(sum(self.s[synset]*self.s[synset]))

    def loadFiles(self, file_name):
        f = codecs.open(file_name, 'r', 'utf-8')
        lines = f.readlines()
        for line in lines:
            temp = line.strip("\n").strip("\r").strip(",").split(" ")
            word = temp[0]
            synsets = temp[1].split(",")
            # self.s[synset] = np.zeros(self.dimention)
            self.words.append(word)
            self.E[word] = {}
            for synset in synsets:
                if synset in self.synsets:
                    self.E[word][synset] = np.zeros(self.dimention)
                    if synset not in self.D:
                        self.D[synset] = {}
                    self.D[synset][word] = np.zeros(self.dimention)
                    self.lemmas.append([synset,word])
                    if synset not in self.delta_l:
                        self.delta_l[synset] = {}
                    self.delta_l[synset][word] = np.zeros(self.dimention)
            for synset in synsets:
                if synset in self.synsets:
                    self.E[word][synset] = np.ones(self.dimention)/len(self.E[word])
                    self.D[synset][word] = np.ones(self.dimention)/len(self.D[synset])

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

        for lemma in self.lemmas:
            synset = lemma[0]
            word = lemma[1]
            if word not in self.G_w:
                self.G_w[word] = {}
                self.G_w[word][synset] = 0
                G_w[word] = 0
            else:
                if synset not in self.G_w[word]:
                    self.G_w[word][synset] = 0
            if synset not in self.G_s:
                self.G_s[synset] = {}
                self.G_s[synset][word] = 0
                G_s[synset] = 0
            else:
                if word not in self.G_s[synset]:
                    self.G_s[synset][word] = 0

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
            word_in = temp[0]
            word_out = temp[1]
            # initialize R
            self.R.append([word_in, word_out])
            #initialize delta_r
            if word_in not in self.delta_r:
                self.delta_r[word_in] = {}
            self.delta_r[word_in][word_out] = np.zeros(self.dimention)

    def forward(self):
        #initialize
        self.error_s = 0
        self.error_l = 0
        self.error_r = 0

        for synset in self.synsets:
            self._s[synset] = np.zeros(self.dimention)
        for word in self.words:
            self.w[word] = np.zeros(self.dimention)

        for lemma in self.lemmas:
            synset = lemma[0]
            word = lemma[1]
            for d in range(self.dimention):
                self.w[word][d] += self.G_s[synset][word]*self.D[synset][word][d]*self.s[synset][d]
        for lemma in self.lemmas:
            synset = lemma[0]
            word = lemma[1]
            for d in range(self.dimention):
                self._s[synset][d] += self.G_w[word][synset]*self.E[word][synset][d]*self.w[word][d]

        for synset in self.synsets:
            self.delta_s[synset] = self._s[synset] - self.s[synset]
            self.error_s += sum(self.delta_s[synset]*self.delta_s[synset])

        for lemma in self.lemmas:
            synset = lemma[0]
            word = lemma[1]
            for d in range(self.dimention):
                self.delta_l[synset][word][d] = self.D[synset][word][d]*self.s[synset][d] - self.E[word][synset][d]*self.w[word][d]
                self.error_l += self.delta_l[synset][word][d]**2

        for r in self.R:
            word_in = r[0]
            word_out = r[1]
            self.delta_r[word_in][word_out] = self.w[word_in] - self.w[word_out]
            self.error_r += sum(self.delta_r[word_in][word_out]*self.delta_r[word_in][word_out])

        # print(self.E)

    def backward(self):

        E_temp = self.E
        D_temp = self.D

        for lemma in self.lemmas:
            synset = lemma[0]
            word = lemma[1]
            for d in range(self.dimention):
                self.E[word][synset][d] -= self.alpha*self.G_w[word][synset]*self.delta_s[synset][d]*self.w[word][d]
                self.D[synset][word][d] -= self.alpha*self.G_w[word][synset]*self.G_s[synset][word]*self.delta_s[synset][d]*self.s[synset][d]*E_temp[word][synset][d]

                self.E[word][synset][d] += self.beta*self.delta_l[synset][word][d]*self.w[word][d]
                self.D[synset][word][d] -= self.beta*self.delta_l[synset][word][d]*self.s[synset][d]*(1-E_temp[word][synset][d]*self.G_s[synset][word])

        for r in self.R:
            word_in = r[0]
            word_out = r[1]
            for d in range(self.dimention):
                for synset in self.E[word_in].keys():
                    self.D[synset][word_in][d] -= self.gamma*self.delta_r[word_in][word_out][d]*self.G_s[synset][word_in]*self.s[synset][d]
                for synset in self.E[word_out].keys():
                    self.D[synset][word_out][d] += self.gamma*self.delta_r[word_in][word_out][d]*self.G_s[synset][word_out]*self.s[synset][d]


    def train(self):
        print("Loading files ...")
        self.loadSynsetVector(self.folder+'synsets.txt')
        print("  Normalize synset vector ...")
        self.normalizeSynsetVector()
        print("      DONE!!")
        self.loadFiles(self.folder+'synset.txt')
        self.loadGenerality(self.folder+'generality.txt')
        self.loadRelation(self.folder+'relations.txt')
        print("    DONE!!")


        print("Training model ...")
        for i in range(self.iter):
            if (i+1)%1==0:
                print("%d / %d ended" % (i+1,self.iter) )
                print("Error synset : %f" % self.error_s)
                print("Error lemma : %f" % self.error_l)
                print("Error relation : %f" % self.error_r)
                print("Error all told : %f" % (self.error_s+self.error_l+self.error_r))

            self.forward()
            self.backward()
        print("    DONE!!")

    def save_model(self):
        print("Saving model ...")
        fWords = codecs.open(self.folder+'naive/words.txt','w','utf-8')
        fLemmas = codecs.open(self.folder+'naive/lemmas.txt','w','utf-8')

        for word in self.words:
            self.w[word] = np.zeros(self.dimention)
        for lemma in self.lemmas:
            synset = lemma[0]
            word = lemma[1]
            fLemmas.write(synset+":"+word+" ")
            vecText = ""
            for d in range(self.dimention):
                vecText += str(self.D[synset][word][d]*self.s[synset][d])+" "
                self.w[word][d] += self.G_s[synset][word]*self.D[synset][word][d]*self.s[synset][d]
            fLemmas.write(vecText.strip(" ")+"\n")

        for word, vector in self.w.items():
            fWords.write(word+" ")
            vecText = ""
            for d in range(self.dimention):
                vecText += str(vector[d])+" "
            fWords.write(vecText.strip(" ")+"\n")
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

ae = AutoExtendReverse(folder="/Users/arai9814/Desktop/ae_generality/jpn/",iter_num = 1000)
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
