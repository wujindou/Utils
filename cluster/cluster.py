#coding:utf-8
import sys 
import json 
import jieba 
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD

corpus = []
with open('/Users/wujindou/Downloads/bq_corpus/train.tsv','r',encoding='utf-8') as lines:
    for line in lines:
        data =line.strip().split('\t')
        corpus.append(' '.join(jieba.lcut(data[0])))
vecotiazer = TfidfVectorizer(analyzer='word',ngram_range=(1,3),min_df=2)
X = vecotiazer.fit_transform(corpus)
transformer = TruncatedSVD(n_components=300,random_state=0)
X=transformer.fit_transform(X)
clustering = DBSCAN(eps=0.3,min_samples=5,n_jobs=6).fit(X)
labels =clustering.labels_
n_clusters = len(set(labels))-(1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print('聚类簇数量:%d'%(n_clusters))
print('噪声样本数量:%d'%(n_noise))
import collections 
cluster_info =collections.defaultdict(list)
for text,label in zip(corpus,labels):
    cluster_info[label].append(text)
for cluster_id,lst in cluster_info.items():
    print('-'*10 +str(cluster_id)+'\t'+str(len(lst))+'\t'+';'.join(lst[:10]))
