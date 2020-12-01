from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from Metrics import Util
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import math
import random

print("start")

text_file = open("C:\\Users\\colto\\PycharmProjects\\ClusteringAntiVax\\proplus.csv", "r", encoding="utf8").read().split("\n")

# kmeans = load("kmeans.joblib")
# tf_idf = load("tfidf.joblib")

tf_idf = TfidfVectorizer(stop_words="english")
kmeans = KMeans(17)
# kmeans = load("C:\\Users\\colto\\PycharmProjects\\ClusteringAntiVax\\anti20.joblib")
# dbscan = DBSCAN(eps=0.4, metric="cosine")
# out_file = open("AntiLabels.txt", "w+", encoding="utf8")
# for num in range(len(kmeans.labels_)):
#     out_file.write(text_file[num].replace("\n", " ") + " ~ " + str(kmeans.labels_[num]) + "\n")

tf_idf_matrix = tf_idf.fit_transform(text_file)

# svd = TruncatedSVD(n_components=100)
#
# svd_matrix = svd.fit_transform(tf_idf_matrix)
# kmeans.fit(svd_matrix)
# dbscan.fit(svd_matrix)
kmeans.fit(tf_idf_matrix)



clusters = Util.create_clusters(tf_idf_matrix, kmeans.labels_)

count_of_clusters = dict()

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]


cluster_centers = kmeans.cluster_centers_

averages = dict()
counts = dict()

new_matrix = kmeans.transform(tf_idf_matrix)

tf_idf_matrix = tf_idf_matrix.todense()
labels = kmeans.labels_

for ele in labels:
    if ele in count_of_clusters:
        count_of_clusters[ele] += 1
    else:
        count_of_clusters[ele] = 1

terms = tf_idf.get_feature_names()
for i in range(17):
    print("Cluster " + str(i) + " + count " + str(count_of_clusters[i]) , end='')
    for ind in order_centroids[i, :50]:
        print(' %s' % terms[ind], end='')
    print()

for ele in range(len(labels)):
    if labels[ele] in averages:
        averages[labels[ele]] += new_matrix[ele][labels[ele]]
        counts[labels[ele]] += 1
    else:
        averages[labels[ele]] = new_matrix[ele][labels[ele]]
        counts[labels[ele]] = 1
for ele in averages.keys():
    averages[ele] /= counts[ele]
    print(str(ele) + " " +  str(averages[ele]))

outfile = open("distancesPro.csv", "w+")
for num in range(17):
    outfile.write("," + str(num))
outfile.write("\n")
counter = 0
for ele in cluster_centers:
    outfile.write(str(counter) + ",")
    counter+=1
    for ele2 in cluster_centers:
        outfile.write(str(Util.distance_between_two_points(ele, ele2)) + ",")
    outfile.write("\n")
