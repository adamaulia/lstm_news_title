#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:37:17 2017

@author: adam
"""
import gensim
import numpy as np
from keras.models import load_model
import sys

keras_model = load_model('keras_model/lstm_news_title-09-0.91.hdf5')
w2v_model = gensim.models.Word2Vec.load('word2vec_model/w2v_model')


def words_2_vec(words,length):
    vec = np.zeros((length,100))
    for i in range(len(words)):
        if words[i] in w2v_model.wv.vocab.keys() and i < length:
            vec[i,:]=w2v_model.wv[words[i]]
    return vec


def news_classification(words):
    words = words.split()
    target = ['entertainment', 'business', 'health' , 'technology']
    
    vec = words_2_vec(words,15)
    vec = vec.reshape(1,15,100)
    
    result = target[np.argmax(keras_model.predict(vec))]
    print result
    return result


if __name__=='__main__':
    words = sys.argv[1]
    news_classification(words)    
