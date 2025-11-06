from sklearn.datasets._twenty_newsgroups import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster._kmeans import KMeans
from sklearn.metrics.cluster._unsupervised import davies_bouldin_score
from sklearn import metrics
import numpy as np

def purity_score(y_true, y_pred):
    conMatrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(conMatrix, axis=0)) / np.sum(conMatrix) 

def evaluate_clusters(k,X,Y,centroids,labels):
    ordered_centroids = centroids.argsort()[:,::-1]
    n = 10
    for i in range(0,k):
        print("CLUSTER " + str(i) + "------------------------")
        s = ""
        for ind in ordered_centroids[i, 0:n]:
            s += terms[ind]+" "
        print(s)
    print("-----------------------------------")
    print("DB score: %3f" % davies_bouldin_score(X.toarray(), labels))
    print("Purity score: %3f" % purity_score(Y, labels))
    print("Homogeneity score: %3f" % metrics.homogeneity_score(Y, labels))
    print("Completeness score: %3f" % metrics.completeness_score(Y, labels))
    print("V measure: %3f" % metrics.v_measure_score(Y, labels))
    print("Rand Index: %3f" % metrics.adjusted_rand_score(Y, labels))

# load the text data
categories = ['sci.space', 'rec.autos', 'comp.graphics', 'rec.sport.baseball']
remove = ()#('headers', 'footers', 'quotes')
texts = fetch_20newsgroups(categories=categories, remove=remove)
X = texts.data
Y = texts.target

# vectorise the texts
vec = TfidfVectorizer(stop_words="english", max_features=10000, max_df=0.5, min_df=2)
X = vec.fit_transform(X)
terms = vec.get_feature_names_out()

# do kmeans clustering
for i in range(2,10):    
    k = i
    print("K = " + str(k) + "---------------------------------------")
    km = KMeans(n_clusters=k, random_state = 2)
    km.fit_transform(X)
    labels = km.labels_
    centroids = km.cluster_centers_
    evaluate_clusters(k,X,Y,centroids,labels)

exit()