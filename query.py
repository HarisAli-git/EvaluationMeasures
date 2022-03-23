#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import re
import csv
import math
import numpy as np
from xml.etree import ElementTree
from stemming.porter2 import stem
from operator import itemgetter, attrgetter


# In[2]:


def getQueryDesc():   
    
    dom = ElementTree.parse('topics.xml')

    root = dom.getroot()

    query = []
    desc = []
    q_id = []

    for tid in root.findall('topic'):
        q_id.append(tid.get('number'))
    
    for q in root.iter('query'):
        query.append(q.text)

    for d in root.iter('description'):
        desc.append(d.text)

    return query, desc, q_id


# In[3]:


def tokenize(p_txt):
    """
    takes all of the text and concerts it into tokens
    
    """ 
    
    tokens=[]
    
    for txt in p_txt:
        token = re.findall(r'\w+', txt)
        tokens.append(token)
    
    return tokens


# In[4]:


def lowerCase(tokens):
    """
    switch all tokens to lowercase
    
    """
    return [[w.lower() for w in a] for a in tokens]


# In[5]:


def removeNextLineChar(text):
    """
    Removes '\n' character after every word in given list
    
    """
    # An dict for stopwords tokens
    lst = {}

    for word in text:
        # We have to ignore the last character from every word
        # as it contains '\n'
        word = word[:-1]
        lst[word] = 1
    
    return lst

def RemoveStopWordsfromTK(stopwords, tokens):
    
    """
    Stop Words Removal from given list of tokens
    
    """
    pool = []
    
    for token in tokens:
        for word in token:
            if word in stopwords.keys():
                token.remove(word)
        pool.append(token)
        
    return pool

def StopWordsRem(file, tokens):

    # Reading the stoplist.txt file
    text = file.readlines()
    
    stopwords = removeNextLineChar(text)

    pool = RemoveStopWordsfromTK(stopwords, tokens)
         
    return pool


# In[6]:


#------------Stemming block--------------#

def stemming(tokens):
    
    """
    Applies Stemming to tokens
    
    """
    my_dict = {}
    
    for x in tokens:
        k = stem(x)
        if k in my_dict.keys():
            my_dict[k] = my_dict[k] + 1
        else:
            my_dict[k] = 1
            
    return my_dict


# In[7]:


def open_terminfo():
    file = 'term_info.txt'

    dict = {}
    lst = []

    with open(file, newline = '') as files:                                                                                          
        lines = csv.reader(files, delimiter='\t')
        for line in lines:
            lst.append(line)

    for x in lst:
        dict[int(x[0])] = (int(x[1]), int(x[2]), int(x[3]))
    
    return dict

def open_docID():

    lst = []
    with open('docids.txt', newline = '') as files:                                                                                          
            lines = csv.reader(files, delimiter='\t')
            for line in lines:
                lst.append(line)

    dic2 = {}
    tot_docs = 0
    
    for x in lst:
        dic2[x[1]] = int(x[0])
        tot_docs += 1
    
    return tot_docs, dic2

def open_docIndex():

    lst = []
    with open('doc_index.txt', newline = '') as files:                                                                                          
            lines = csv.reader(files, delimiter='\t')
            for line in lines:
                lst.append(line)

    dic2 = {}
    
    for x in lst:
        dic2[int(x[0]), int(x[1])] = int(x[2])
        
    return dic2

def open_termID():

    lst = []
    with open('termids.txt', newline = '') as files:                                                                                          
            lines = csv.reader(files, delimiter='\t')
            for line in lines:
                lst.append(line)

    dic2 = {}

    for x in lst:
        dic2[x[1]] = int(x[0])
    
    return dic2


# In[8]:


def open_invertIndex(offset):

    file = open('term_index.txt')
    file.seek(offset)

    lst = file.readline()

    file.close()
    
    return lst


# In[9]:


def getfreq(docid, s):

    ex = 0
    n = 0
    did = 0
    freq = 0

    for i in range(len(s)):
        if (s[i].isnumeric()):
            rem = int(s[i]) % 10
            n *= 10
            n += rem

        if (s[i] == ':'):
            did = n
            n = 0

        if (s[i] == '\t'):
            freq = n
            n = 0
            if (did == docid):
                return freq
        else:
            ex = 0

    return 0


# In[10]:


def get_tot_terms_indocs(): #returns total terms in a document
    dic = open_docIndex()

    dic[1, 2]

    dic2 = {}

    for a,b in dic:
        if a not in dic2.keys():
            dic2[a] = dic[a, b]
        else:
            dic2[a] += dic[a, b]

    return dic, dic2


# In[11]:


def normalizedWeight(array,maxWeight): 
    normalisedWeight=  (array/maxWeight)
    return np.round(normalisedWeight,3)


# In[12]:


def queryIDFProcessing(lst):
    
    dict_idf = {}
    
    for x in lst:
        for key in x:
            if key in dict_idf:
                dict_idf[key] += 1
            else:
                dict_idf[key] = 1 
    
    tot_qs = len(lst)   
    
    idf_q = {}
    for x in dict_idf:
        idf_q[int(x)] = math.log10(float(tot_qs) / float(dict_idf[x]))
        
    return idf_q


# In[13]:


def queryTF_IDFProcessing(lst, idf_q):
    
    dict_tf = {}
            
    tot_qs = len(lst)   
    
    tf_idf = {}
        
    a = 0
    idf_q = {}
    for x in lst:
        a += 1
        for key in x:
            tf = 1 + math.log10(x[key])
            tf_idf[a, key] = round(float(tf * 1), 3)

    return tf_idf, a


# In[14]:


def printSimilarity_TF_IDF(lsttt, lsttt_query, q_id, my_dict2):
   
    """
    Prints similarity between the given (Docs + Query) vectors
    
    """
    t = []
    query_c = 0
    lst = []
    lst1 = []
    for x in lsttt_query:
        List1 = np.array(x)
        q_c = q_id[query_c]
        query_c += 1
        doc_c = 0
        for y in lsttt:
            doc_c += 1
            List2 = np.array(y)
            similarity_scores = List1.dot(List2)/ (np.linalg.norm(List1) * np.linalg.norm(List2))
            score = round(similarity_scores, 8)
            #print("Similarity between DocID: ", doc_c, " and QueryID: ", q_c, ":\t", score)
            t.append(q_c)
            t.append(my_dict2[doc_c])
            t.append(score)
            lst.append(t)
            t = []
        lst1.append(lst)
        lst = []

    return lst1


# In[15]:


def get_first_elem(iterable):
    return iterable[0]

def SortList23(lstx):
    
    print(lstx, "\n\nStgehdbhbsbashdbhbahdsbma\n\n")
    
    lstx.sort(key=get_first_elem)
    
    x2 = sorted(lstx, key=itemgetter(2), reverse=True)
    
    return x2


# In[16]:


def SortList(lstx):
    
    lstx1 = []
    
    for x in lstx:
        q = tuple(x)
        x1 = sorted(q, key=itemgetter(2), reverse=True)
        lstx1.append(list(x1))
        
    return lstx1


# In[17]:


def WriteToRun23(lst_q, run_c):
    
    file = open('run.txt', 'w')
    
    run_c += 1
    rank = 0
  #  <topic> 0 <docid> <rank> <score> <run>
    for y in lst_q:
        rank += 1
        string = (str(y[0]) + ' ' + str(0) + ' ' + str(y[1]) + ' ' + str(rank) + ' ' + str(y[2]) + ' ' + ('run' + str(run_c)) + '\n')
        file.write(string)
    
    file.close()


# In[18]:


def WriteToRun(lst_q, run_c):
    
    file = open('run.txt', 'w')
    
    run_c += 1
  #  <topic> 0 <docid> <rank> <score> <run>
    for x in lst_q:
        rank = 0
        for y in x:
            rank += 1
            string = (str(y[0]) + ' ' + str(0) + ' ' + str(y[1]) + ' ' + str(rank) + ' ' + str(y[2]) + ' ' + ('run' + str(run_c)) + '\n')
            file.write(string)
    
    file.close()


# In[19]:


def ProcessDocIndex(q_id, lst_query, doc_dic, tot_terms_indoc):
    qid = 0
    tot_tc = {}
    jkm = {}
    tot_words = 0 # total words in the whole corpus 
    
    for q in lst_query:
        for x, y in doc_dic:
            tot_words += doc_dic[x, y]
            if x in tot_tc.keys():
                tot_tc[x] += doc_dic[x, y]
            else:
                tot_tc[x] = doc_dic[x, y]
            if y in q.keys():
                jkm[q_id[qid], x, y] = doc_dic[x, y]
            else:
                jkm[q_id[qid], x, y] = 0.0    
        qid += 1
    
    return jkm, tot_tc, tot_words


# In[20]:


def CalProbs(jkm, tot_tc, tot_words, tif_dic): #jkm = Jelinek Mercer Dict & tot_tc = total terms in docs
    
    bck_prob = {}
    doc_prob = {}
    
    for x, y, z in jkm: #x: Qid, y: doc_id, z: tid
        a, b, c = tif_dic[y] #b: total count of term apperances in the whole corpus
        bck_prob[int(x), y] = round((float(b) / float(tot_words)), 8)
        doc_prob[int(x), y, z] = round((float(jkm[x, y, z]) / float(tot_tc[int(x)])), 8)

    return doc_prob, bck_prob


# In[21]:


def Smoothing(jkm, bck_prob, doc_prob, my_dict2):
    """
    Applies Smoothing to the calculted Probabilities
    
    """
    lam = 0.6 #lambda = 0.6 (given)
    jkm_p = {}
    
    for x, y, z in jkm: #x: Qid, y: doc_id, z: tid
        jkm_p[x, y, z] = float(doc_prob[int(x), y, z] * lam) + float(bck_prob[int(x), y] *  (1.0 - lam))
    
    dict_1 = {}
    
    for x, y, z in jkm_p:
        a = my_dict2[y]
        if (x, y) in dict_1.keys():
            dict_1[int(x), a] += jkm_p[x, y, z]
        else:  
            dict_1[int(x), a] = jkm_p[x, y, z]
            
    return dict_1


# In[22]:


def OkapiBM25(doc_dic, dic, avg_len, q_id, idf_dic, tot_tc, my_dict2):
    k1 = 1.2
    k2 = 10
    b = 0.75
    D = len(dic)
    dict_bm = {}
    qid = 0
    lst = []
    
    for q in lst_query:
        q_idd = q_id[qid]
        qid += 1
        dict_bm = {}
        for q_tid in q:
            for x in dic:
                if (x, int(q_tid)) in doc_dic.keys():
                    tfdi = doc_dic[x, int(q_tid)]
                else:
                    tfdi = 0
                                
                idf = idf_dic[int(q_tid)]
                tfqi = q[int(q_tid)]
                
                K = k1 * (((1 - b) + b) * (tot_tc[x] / avg_len))
                f1 = math.log10((D + 0.5) / (idf + 0.5))
                f2 = ((1 + k1) * tfdi) / (K + tfdi)
                f3 = (1 + k2) * tfqi / (k2 + tfqi)
                
                if (int(q_idd), my_dict2[x]) in dict_bm.keys():
                    score = round((f1 * f2 * f3), 8)
                    dict_bm[int(q_idd), my_dict2[x]] += score
                else:
                    score = round((f1 * f2 * f3), 8)
                    dict_bm[int(q_idd), my_dict2[x]] = score
            
        lst.append(dict_bm)
                
    return lst


# In[23]:


def CalculateAvgLen(tot_tc, doc_count):
    
    total_words = 0
    
    for x in tot_tc:
        total_words += tot_tc[x]
    
    return total_words // doc_count


# In[24]:


def processdic(lst111):
    
    """
    changes dict to iteratable list
    """
    
    lst1 = []
    lst = []
    
    for xa in lst111:
        for x, y in xa:
            t = []
            t.append(x)
            t.append(y)
            t.append(xa[x, y])
            lst.append(t)
            t = []
        lst1.append(lst)
        lst = []
    
    return lst1


# In[25]:


def score_TF_IDF(lst_query, max_termID, doc_dic, idf_dic, q_id, my_dict2):
    
    idf_q = queryIDFProcessing(lst_query)

    q_tf_idf, q_len = queryTF_IDFProcessing(lst_query, idf_q) #q_len: total queries, #q_tf_idf: all queries tf_idf vectors

    lsttt_query = []

    for a1 in range(q_len):
        a2 = 1 + a1
        temp = []
        for x1 in range(max_termID):
            if (a2, x1) in q_tf_idf.keys():
                temp.append(q_tf_idf[a2, x1])
            else:
                temp.append(0)
        lsttt_query.append(temp)
    
    for a, b in doc_dic: # a = docID, b = termID
        tf = 1 + math.log10(doc_dic[a, b])
        tf_idf[a, b] = round((tf * idf_dic[b]), 3)
    
    temp = []
    x = 1

    a1 = 1

    lsttt = []

    for a1 in range(len(dic)):
        x1 = 0
        a2 = 1 + a1
        temp = []
        for x1 in range(max_termID):
            if (a2, x1) in tf_idf.keys():
                temp.append(tf_idf[a2, x1])
            else:
                temp.append(0)
        lsttt.append(temp)
    
    lst = printSimilarity_TF_IDF(lsttt, lsttt_query, q_id, my_dict2) #lsttt: Doc Vectors, lsttt_query: Query Vectors 
        
    lst_q = SortList(lst)
    
    return lst_q


# In[26]:


def score_OkapiBM25(tot_tc, dic, doc_dic, my_dict2, q_id, idf_dic):
    
    avglen = CalculateAvgLen(tot_tc, len(dic))
    dict_bm = OkapiBM25(doc_dic, my_dict2, avglen, q_id, idf_dic, tot_tc, my_dict2)
    
    lst = processdic(dict_bm)
    
    lst_q = SortList(lst)
    
    return lst_q


# In[27]:


def score_JM(jkm, tot_tc, tot_words, tif_dic, my_dict2, q_id):
    
    doc_prob, bck_prob = CalProbs(jkm, tot_tc, tot_words, tif_dic)

    dict1 = Smoothing(jkm, bck_prob, doc_prob, my_dict2)
    
    lst = SliceDict(dict1, q_id)

    lst1 = processdic(lst)
    
    lst_q = SortList(lst1)
    
    return lst_q


# In[28]:


def SliceDict(dict1, q_id):

    lst = []
    prev = int(q_id[0])
    next = int(q_id[0])

    new_dict = {}

    c = 0
    for x, y in dict1:
        c += 1
        next = x

        if prev != next:
            prev = x
            lst.append(new_dict)
            new_dict = {}
            new_dict[x, y] = dict1[x, y]
        else:
            new_dict[x, y] = dict1[x, y]
            prev = x
        if (c == len(dict1) - 1):
            lst.append(new_dict)

    return lst


def PrintRes(lst_q, run_c):
    
    run_c += 1
  #  <topic> 0 <docid> <rank> <score> <run>
    for x in lst_q:
        rank = 0
        for y in x:
            rank += 1
            string = (str(y[0]) + ' ' + str(0) + ' ' + str(y[1]) + ' ' + str(rank) + ' ' + str(y[2]) + ' ' + ('run' + str(run_c)) + '\n')
            print(string)

# In[ ]:

n = len(sys.argv)

dec = []
query, desc, q_id = getQueryDesc()
tokens = lowerCase(tokenize(query))

file = open ("C:/Users/haris/Downloads/stoplist.txt","r")

stoplist = StopWordsRem(file, tokens)

lst = []

for x in stoplist:
    lst.append(stemming(x))

termids = open_termID()

lst_query = []

for x in lst:
    dict_tf = {}
    for key in x:
        if key in termids.keys():
            dict_tf[termids[key]] = x[key]
    lst_query.append(dict_tf)

max_term = max(termids, key=termids.get) #max_termID: maximum termID term in the whole corpus

max_termID = termids[max_term] 

# ------------ query end ----------------#

tif_dic = open_terminfo() # term_info dictionary

idf_dic = {}

docsN, dic = open_docID() # docsN = total docs in corpus, dic stores each docID with its name

for x in tif_dic:
    offset, tot_freq, term_docs = tif_dic[x] # term_docs = total documents in which term appeared, tot_freq = total frequency in entire corpus
    idf_dic[int(x)] = math.log10(float(docsN) / float(term_docs))

tf_idf = {}

doc_dic, tot_terms_indoc = get_tot_terms_indocs() # tot_terms_indoc = total terms in a document dic, doc_dic = doc_index dic
    
my_dict2 = {y:x for x,y in dic.items()} #changing Doc dic value to key so dict key becomes docID and values becomes Docname

jkm, tot_tc, tot_words = ProcessDocIndex(q_id, lst_query, doc_dic, tot_terms_indoc) # total words in a doc

#------------------------- general Processing Done ----------------------#
run_c = 2

if (n != 3):
    print("\nWrong CMD Args given. Please try again with correct args!\n")
elif (sys.argv[2] == 'TF-IDF'):
    lst_q = score_TF_IDF(lst_query, max_termID, doc_dic, idf_dic, q_id, my_dict2)
    WriteToRun(lst_q, run_c)
elif (sys.argv[2] == 'BM25'):
    lst_q = score_OkapiBM25(tot_tc, dic, doc_dic, my_dict2, q_id, idf_dic)
    PrintRes(lst_q, run_c)
elif (sys.argv[2] == 'JM'):
    lst_q = score_JM(jkm, tot_tc, tot_words, tif_dic, my_dict2, q_id)
    WriteToRun(lst_q, run_c)
# In[ ]:




