#HELPERS

import numpy as np
from node import Node
import pickle

#Log Likelihood Ratio Functions (natural log)

def llri(obs, exp, q): #score function for poisson (one element)
    return obs*np.log(q) + exp*(1-q)

def llri_pen(obs, exp, q, delta): #penalized score function for poisson (one element)
    return obs*np.log(q) + exp*(1-q) - delta

llr = np.vectorize(llri) #generate unpenalized scores for a given q value

llr_pen = np.vectorize(llri_pen) #genereate penalized scores for a given q value

poissons = np.vectorize(lambda x: np.random.poisson(x))


#DEFINED FUNCTIONS FOR USE IN ALGORITHM:

#BINARY SEARCH FUNCTIONS: ################################################
def q_min(obs, exp):

    minimum = 0.000001
    qmle = obs/exp

    while abs(qmle - minimum) > 0.00000001:
        q_mid = (minimum + qmle)/2

        if llri(obs, exp, q_mid) > 0:
            qmle = qmle - (qmle - minimum)/2
        else:
            minimum = minimum + (qmle - minimum)/2
    return (minimum + qmle)/2

def q_max(obs, exp):

    maximum = 10000000
    qmle = obs/exp

    while abs(maximum - qmle) > 0.000001:
        q_mid = (maximum + qmle)/2

        if llri(obs, exp, q_mid) < 0:
            maximum = maximum - (maximum-qmle)/2
        else:
            qmle = qmle + (maximum-qmle)/2

    return (maximum + qmle)/2

###################################################
def q_min_pen(obs, exp, delta):  #need to fix


    minimum = 0.000001
    qmle = obs/exp

    while abs(qmle - minimum) > 0.00000001:
        q_mid = (minimum + qmle)/2

        if llri_pen(obs, exp, q_mid, delta) > 0:
            qmle = qmle - (qmle - minimum)/2
        else:
            minimum = minimum + (qmle - minimum)/2
    return (minimum + qmle)/2
####################################################

def q_max_pen(obs, exp, delta):

    maximum = 10000000
    qmle = obs/exp

    while abs(maximum - qmle) > 0.000001:
        q_mid = (maximum + qmle)/2

        if llri_pen(obs, exp, q_mid, delta) < 0:
            maximum = maximum - (maximum-qmle)/2
        else:
            qmle = qmle + (maximum-qmle)/2

    return (maximum + qmle)/2
######################################################################################


#FOR FINDING Q-INTERVALS: ###################################################################################
def minmax(obs, exp):
    return (q_min(obs, exp), q_max(obs, exp))

def minmax_pen(obs, exp, delta):
    return [q_min_pen(obs, exp, delta), q_max_pen(obs, exp, delta)]

#Get q intervals
def get_q_values(obs, exp, delta):
    return np.array([q_min(obs, exp), q_min_pen(obs, exp, delta), q_max_pen(obs, exp, delta), q_max(obs, exp)])

# get_all_intervals = np.vectorize(get_q_values)
#############################################################################################################

def relu_scores(scores, delta):
    return min(np.abs(scores), delta)

def ReLU(scores):
    return (scores > 0).view('i1')


def compute_new_subset(scores, filter, delta):
    
    subset = np.zeros(len(filter))
    weights = np.zeros(len(filter))

    for i in range(len(filter)):

        if (scores[i] > delta):
            subset[i] = 1 #always include element in subset
            weights[i] = delta #w_i = delta

        elif (np.abs(scores[i]) <= delta):
            subset[i] = filter[i] #include iff in filter
            weights[i] = np.abs(scores[i]) #w_i = score_i

        else:
            subset[i] = 0 #always exclude from subset
            weights[i] = delta #w_i = delta
            
    return subset, weights

def get_all_intervals(observed, expected, delta):
    for i in range(len(observed)):
        if i == 0:
            q_values = get_q_values(observed[i], expected[i], delta)
        else:
            q_values = np.append(q_values, get_q_values(observed[i], expected[i], delta))
    
    q_values = np.unique(q_values)

    return q_values

def compute_initial_subset(scores, delta):
    initial = ReLU(scores)
    weights = [delta if np.abs(score) > delta else np.abs(score) for score in scores]

    return initial, weights

def compute_penalized_score(observed, expected, subset, filter, qmle, delta):
    return sum(subset*llr(observed, expected, qmle)) - sum(filter != subset) * delta

def jaccard(best_s, observed):
    if best_s.shape != observed.shape:
        raise ValueError("Shape mismatch: best_s and observed must have the same shape.")
    intersection = np.logical_and(best_s, observed)
    union = np.logical_or(best_s, observed)
    return intersection.sum() / float(union.sum())

def load_training_set():    
    with open('theta.pkl', 'rb') as f: 
        theta = pickle.load(f)
    return theta

def load_test_set():
    with open('test_set.pkl', 'rb') as f: 
        subset = pickle.load(f)
    return subset.to_numpy()

def preprocess(index):
    poissons = np.vectorize(lambda x: np.random.poisson(x))
    baseline = load_test_set()[index]*10 + 10
    observed = poissons(baseline)
    expected = poissons(np.ones(baseline.shape)*10) + 0.5

    return baseline, observed, expected

def build_tree(theta, max_depth, min_filters_to_split, name):
    tree = Node(theta, name = name)
    tree.build_tree(theta=theta, max_depth=max_depth, min_filters_to_split=min_filters_to_split)

    return tree