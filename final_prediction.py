# -*- coding: utf-8 -*-
#!/usr/bin/python

from __future__ import unicode_literals
import os
import numpy as np
import sys
import getopt
import eval_util
from scipy.sparse import coo_matrix
from itertools import izip

def sort_tuples(t):
    return sorted(t, key=lambda x: (x[0],-x[2]))

def sort_coo(m):
    tuples = izip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: (x[0], -x[2]))

def create_full_matrix(input_score,mapper,n,k):
    output = np.zeros((n,k))
    c = 0
    with open(input_score, 'r') as f_pred:
        for line in f_pred:
             
            split_line = line.split(',')
            label_score = split_line[1]
            video_id = split_line[0]
            print '%d\n'%c
            if c > 0:
                split_label = label_score.split(' ')
                list_label = [int(x) for x in split_label[0::2]]
                list_score = np.array([float(x) for x in split_label[1::2]])
                
                output[mapper[video_id],list_label] = list_score 
            c +=  1

    return output

def create_sparse_matrix(input_score):
    row = []
    col = []
    data = []
    c = 0
    with open(input_score, 'r') as f_pred:
        for line in f_pred:
             
            split_line = line.split(',')
            label_score = split_line[1]
            video_id = split_line[0]

            if c%100000==0:
                print '%d\n'%c

            if c > 0:
                split_label = label_score.split(' ')
                list_label = [int(x) for x in split_label[0::2]]
                list_score = np.array([float(x) for x in split_label[1::2]])
                   
                row.extend([c-1]*len(list_label))   
                col.extend(list_label)
                data.extend(list_score)

            c +=  1
    print 'done\n'
    return data,row,col

def get_videoid(filename):
    count = 0
    mapper = {}
    mapper2 = {}
    c = 0
    with open(filename) as f_train:
        for line in f_train:
            if c > 0:
                split_line = line.split(',')
                video_id = split_line[0]
                mapper[video_id] = count
                mapper2[count] = video_id

                count += 1
            c = 1

    return (mapper,mapper2)

def rank_normalization(X):
    ind_sort = np.argsort(X)
    ranks = np.linspace(0,1,X.shape[1])
    output = np.zeros(X.shape)

    for i in range(X.shape[0]):
        output[i,ind_sort[i,:]] = ranks

    return output
def compute_labels(filename):
    count = 0
    mapper = {}
    output = np.zeros((30000,4716))

    with open(filename) as f_train:
        for line in f_train:
            split_line = line.split(',')
            video_id = split_line[0]
            mapper[video_id] = count
            labels = split_line[1]
            split_label = labels.split(' ')
            list_label = [int(x) for x in split_label]

            output[count,list_label] = 1

            count += 1

    output = output[0:count]

    return (output,mapper)

def save_score(sorted_ind, pred, output, mapper):
    
    #ind_sort = np.argsort(-pred,axis=1)

    f_output = open(output, 'w')
    f_output.write('VideoId,LabelConfidencePairs\n')

    vid_id = -1
    c = 0


    for i in range(pred.shape[0]):
        line = ''
        video_id = mapper[i]
        line = line + video_id + ','
        print vid_id
        for c in range(20):
            line += ( str(sorted_ind[c]) + ' ' + str(pred[i,sorted_ind[c]]))
            if c == 19:  
                f_output.write(line + '\n')
            else:
                line += ' '

    '''
    for t in pred:
        if vid_id != t[0]:
            c = 0
            vid_id = t[0]
            line = ''
            video_id = mapper[vid_id]
            line = line + video_id + ','
            print vid_id
        if c < 20:
            line += ( str(t[1]) + ' ' + str(t[2]))
            if c == 19:  
                f_output.write(line + '\n')
            else:
                line += ' '
        c += 1 
    '''
    f_output.close()

def main(argv):
    fn = ['']*2
    fn[0] = 'test-gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv'
    fn[1] = 'test-GRU-0002-1200-2.csv'
    '''fn[2] = 'test-gatednetfvLF-128k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv'
    fn[3] = 'test-gateddboflf-4096-1024-80-0002-300iter.csv'
    fn[4] = 'test-softdboflf-8000-1024-80-0002-300iter.csv'
    fn[5] = 'test-gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv'
    fn[6] = 'test-lstm-0002-val-150-random.csv' '''


    pred = []

    for i in range(len(fn)):
        print 'reading %s'%fn[i]
        data,row,col = create_sparse_matrix(fn[i]) 
        pred.append(coo_matrix((data,(row,col)),shape=(700641,4716)))

    print 'reading data done ...\n'
    print 'ensembling ...\n'

    ens = 0
    for i in range(len(fn)):
        if i == 0:
            ens = pred[i]
        else:
            ens = ens + pred[i]  

    print 'ensembling done ! \n'

    print 'sorting per score ...\n'
    #ens_coo = coo_matrix(ens)
    #sorted_pred = sort_coo(ens_coo) 
    sorted_ind = np.argsort(ens,axis=1)
    sorted_ind = sorted_ind[:,-1:-21:-1]
    print 'sorting done ! \n'

    print 'outputing prediction file ...\n'

    m1, m2 = get_videoid(fn[0])

    save_score(sorted_ind, ens,'submission_file_WILLOW.csv',m2) 
if __name__ == "__main__":
    main(sys.argv[1:])
