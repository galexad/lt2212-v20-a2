import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
import numpy as np
random.seed(42)
from collections import Counter
from nltk.tokenize import word_tokenize
import string
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.random_projection import sparse_random_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



def part1(samples):
    #extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X

def extract_features(samples):
    l=[]
    stop_words = set(stopwords.words('english'))
    for sample in samples:
        sample = word_tokenize(sample)
        sample = [''.join(c.lower() for c in s if c!="") for s in sample if s not in string.punctuation if s.isalpha() is True]
        f_sample = [w for w in sample if not w in stop_words]

        my_dict = dict(Counter(f_sample))
        l.append(my_dict)

    df = pd.DataFrame(l)
    sum_column = df.sum(axis=0)
    dictionary = sum_column[1:].to_dict()

    d = {}
    for key, value in dictionary.items():
        if value > 10:
            d[key] = value

    final_dict = {}
    for data_key in df.keys():
        for k, v in d.items():
            if data_key == k:
                final_dict[data_key] = list(df[data_key])

    df2 = pd.DataFrame(final_dict)
    df2.fillna(0, inplace=True)
    arr = np.array(df2)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    arr2 = tfidf_transformer.fit_transform(arr)

    return arr2.toarray()



def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr

def reduce_dim(X, n=2):
    svd = TruncatedSVD(n)
    transformed = svd.fit_transform(X)
    return transformed

def reduce2(X, n=2):
    pca = PCA()
    X_reduced = pca.fit_transform(X)
    return X_reduced



def get_classifier(clf_id):
    if clf_id == 1:
        clf = SGDClassifier(loss='log', penalty='l2', random_state=42, max_iter=10000, tol=1e-3, average=True)
    elif clf_id == 2:
        clf = LinearSVC(random_state=0, max_iter=10000)
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf


def part3(X, y, clf_id):

    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifier(clf, X_train, y_train)


    # evalute model
    print("Evaluating classifier ...")
    evalute_classifier(clf, X_test, y_test)



def shuffle_split(X,y):
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test


def train_classifier(clf, X, y):
    assert is_classifier(clf)
    model = clf.fit(X, y)
    return model


def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    y_pred = clf.predict(X)
    precision = precision_score(y,y_pred, average = "macro")
    recall = recall_score(y,y_pred, average = "macro")
    f_measure = f1_score(y,y_pred, average = "macro")

    print("Accuracy:", accuracy_score(y, y_pred, normalize=True))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-measure:", f_measure)




def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names



def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(
            model_id=args.model_id,
            n_dim=args.number_dim_reduce
            )
