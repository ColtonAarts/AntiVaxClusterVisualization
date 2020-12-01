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

text_file = open("C:\\Users\\colto\\PycharmProjects\\ClusteringAntiVax\\antiplus.csv", "r", encoding="utf8").read().split("\n")

# kmeans = load("kmeans.joblib")
# tf_idf = load("tfidf.joblib")

tf_idf = TfidfVectorizer(stop_words="english")
kmeans = KMeans(17)
# kmeans = load("C:\\Users\\colto\\PycharmProjects\\ClusteringAntiVax\\anti20.joblib")
# dbscan = DBSCAN(eps=0.4, metric="cosine")
out_file = open("AntiLabels.txt", "w+", encoding="utf8")
# for num in range(len(kmeans.labels_)):
#     out_file.write(text_file[num].replace("\n", " ") + " ~ " + str(kmeans.labels_[num]) + "\n")

tf_idf_matrix = tf_idf.fit_transform(text_file)

# svd = TruncatedSVD(n_components=100)
#
# svd_matrix = svd.fit_transform(tf_idf_matrix)
# kmeans.fit(svd_matrix)
# dbscan.fit(svd_matrix)
kmeans.fit(tf_idf_matrix)

tf_idf_matrix = tf_idf_matrix.todense()
# print("text file")
# new_file = open("antiClusterWithText.txt", "w+",  encoding="utf8")
# for ele in range(len(text_file)):
#     new_file.write(text_file[ele] + "~" + str(kmeans.labels_[ele]) + "\n")
# print("after text file")
# new_file.close()
# max = -1
# num_words = -1
#
# for ele in tf_idf_matrix:
#     count = 0
#     ele = np.asarray(ele)
#     for ele2 in ele[0]:
#         if ele2 != 0:
#             count += 1
#         if ele2 > max:
#             max = ele2
#     if count > num_words:
#         num_words = count
# print("Max = " + str(max))
# print("Count = " + str(num_words))
#
clusters = kmeans.cluster_centers_
print("after fit")
data = dict()

for num in range(len(clusters)):
    data[num] = dict()
    data[num]["distance_to_clusters"] = dict()
    data[num]["average_point"] = list()
    data[num]["average_point"].append(0)
    data[num]["count"] = list()
    data[num]["count"].append(0)

for num in range(len(clusters)):
    x = clusters[num]
    for y in clusters:
        distance = Util.distance_between_two_points(x, y)
        if distance > 0:
            data[num]["distance_to_clusters"].append(distance)
#
labels = set()
# # for num in dbscan.labels_:
# #     if num not in labels.keys():
# #         labels[num] = 1
# #     else:
# #         labels[num] += 1
#
for label in kmeans.labels_:
    if label not in labels:
        labels.add(label)
#
#
#
# # grouped_clusters = Util.create_clusters(svd_matrix, kmeans.labels_)
grouped_clusters = Util.create_clusters(tf_idf_matrix, kmeans.labels_)
# # new_matrix = kmeans.transform(svd_matrix)
new_matrix = kmeans.transform(tf_idf_matrix)
#
found_distance = set()
possible_distances = list()
#
# print(grouped_clusters.keys())
#
for ele in range(len(new_matrix)):
    data[kmeans.labels_[ele]]["average_point"][0] += new_matrix[ele][kmeans.labels_[ele]]
    data[kmeans.labels_[ele]]["count"][0] += 1
#
# print(labels)
#
attempt = dict()
for label in labels:
    attempt[label] = dict()
    for other in labels:
        if label != other:
            attempt[label][other] = dict()
            attempt[label][other]["distance"] = 100000000
            attempt[label][other]["point"] = list()
#
# print(attempt)
#
for label in range(len(kmeans.labels_)):
    distances = np.asarray(new_matrix[label])
    for ele in range(len(distances)):
        if ele != kmeans.labels_[label]:
            if distances[ele] < attempt[kmeans.labels_[label]][ele]["distance"]:
                attempt[kmeans.labels_[label]][ele]["distance"] = distances[ele]
                attempt[kmeans.labels_[label]][ele]["point"] = tf_idf_matrix[label]
#
# print(attempt)
#
for ele in attempt.keys():
    for label in attempt[ele].keys():
        data[ele]["distance_to_clusters"][label] = Util.distance_between_two_points(np.asarray(attempt[ele][label]["point"])[0], np.asarray(attempt[label][ele]["point"])[0])
#
# average = 0
# # for ele in data.keys():
# #     data[ele]["average_point"][0] /= data[ele]["count"][0]
# #     average += data[ele]["average_point"][0]
#
# print("Average")
# print(average/len(data.keys()))
#
#
angle = 2 * math.pi / (len(kmeans.cluster_centers_) - 1)
#
plt.axes()
#
# # colors = ['b', 'g', 'r', 'c', 'm', 'y', "BlanchedAlmond", "Aqua", "AntiqueWhite", "Brown", "BlueViolet", "BurlyWood"
# #           , "CadetBlue", "Chocolate", "Coral", "Crimson", "DarkGoldenRod", "DarkGrey", "DarkKhaki", "DarkOliveGreen"]
#
colors = ["#a3a3f5", "#7575f0", "#4747eb", "#d1d1fa", "#dbdbf0", "#b8b8e0", "#9494d1", "#7070c2", "#a3d5a3",
          "#47eb47", "#19e619", "#70c270", "#4db24d", "#d9b28c", "#e0b285", "#e8b27d", "#f0b275", "#cc9966", "#d6995c",
          "#e09952", "#eb9947"]
random.seed(30)
# random.seed(680)
random.shuffle(colors)
length_file = open("lengthsAnti.txt", "w+")

for ele in data.keys():
    count = 0
    plt.clf()
    circle0 = plt.Circle((0, 0), data[ele]["average_point"][0], color=colors[ele])
    print(ele)
    print(data[ele]["average_point"][0])
    length_file.write(str(ele) + ", " + str(data[ele]["average_point"][0]) + "\n")
    plt.gca().add_patch(circle0)
    plt.text(0, 0, "C " + str(ele), horizontalalignment='center', verticalalignment='center', fontsize=12)

    distances = list()
    cluster_nums = list()
    for num in data[ele]["distance_to_clusters"].keys():
        distances.append(data[ele]["distance_to_clusters"][num])
        cluster_nums.append(num)

    # sort_clusters = [x for _,x in sorted(zip(distances,cluster_nums))]
    sort_clusters = cluster_nums
    alignment = "bottom"
    for other in sort_clusters:
        if count * angle > math.pi:
            alignment = "top"
        length = data[ele]["distance_to_clusters"][other]
        length_mod = (length/0.712) ** 3 + 2
        circleX = math.cos(count * angle) * (data[ele]["average_point"][0] + length_mod+data[other]["average_point"][0])
        circleY = math.sin(count * angle) * (data[ele]["average_point"][0] + length_mod+data[other]["average_point"][0])
        circle1 = plt.Circle((math.cos(count * angle) * (data[ele]["average_point"][0] + length_mod + data[other]["average_point"][0]), math.sin(count * angle) * (data[ele]["average_point"][0] + length_mod + data[other]["average_point"][0])), data[other]["average_point"][0], color=colors[other])
        plt.plot((math.cos(count * angle) * data[ele]["average_point"][0], math.cos(count * angle) *
                  (data[ele]["average_point"][0] + length_mod)), (math.sin(count * angle) *
                                                                  data[ele]["average_point"][0],
                 math.sin(count * angle) * (data[ele]["average_point"][0] + length_mod)), 'k--', alpha=0.25,
                 linewidth=.75)
        plt.text(math.cos(count * angle) * (
                    data[ele]["average_point"][0] + length_mod + data[other]["average_point"][0]), math.sin(count * angle) * (
                                          data[ele]["average_point"][0] + length_mod + data[other]["average_point"][0]), "C " + str(other), horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.text(math.cos(count*angle)*(data[ele]["average_point"][0] + length_mod/1.5), math.sin(count*angle)*(data[ele]["average_point"][0] + length_mod/1.5), '{:.3f}'.format((round(length, 3))), fontsize=10.5, horizontalalignment='center', verticalalignment=alignment)
        count += 1

        plt.gca().add_patch(circle1)
    x = np.arange(0)
    y = np.arange(0)
    plt.xticks(x, " ")
    plt.yticks(y, " ")
    plt.xlim(-14, 14)
    plt.ylim(-14, 14)
    plt.savefig("ProClusterNewDistances" + str(ele) + ".png")

    plt.show()
#
#
# # fig, ax = plt.subplots()
# #
# # ax.add_artist(circle)
# #
# # fig.savefig("figure1.png")
# #
# # print(data)
# # #
# # out_file = open("resultstfidf.txt", "w+")
# # for key in data.keys():
# #     print(key)
# #     out_file.write(str(key) + "\n")
# #     for info in data[key].keys():
# #         print(info)
# #         out_file.write(str(info) + "\n")
# #         for ele in data[key][info]:
# #             out_file.write(str(ele) + "\n")
# #             print(ele)
# #     print()
# #     out_file.write("\n")
