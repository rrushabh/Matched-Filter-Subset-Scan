import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from mfs_helpers import llr, minmax_pen, jaccard

def load_test_set():
    with open('test_set.pkl', 'rb') as f: 
        subset = pickle.load(f)
    return subset.to_numpy()

def load_training_set():    
    with open('theta.pkl', 'rb') as f: 
        theta = pickle.load(f)
    return theta

#Do something:
# with open('circles_filled_dataset.pkl', 'rb') as f:
#     circles = pickle.load(f)

def preprocess(index):
    poissons = np.vectorize(lambda x: np.random.poisson(x))
    baseline = load_test_set()[index]*10 + 10
    observed = poissons(baseline)
    expected = poissons(np.ones(baseline.shape)*10) + 0.5

    return baseline, observed, expected

def pfss(observed, expected, filter_set, delta, num_filters = None, verbose = False):
  
    #take a sample from the filter set of size 'num_filters' or just use the entire set
    if num_filters:
        filters = filter_set.sample(num_filters).to_numpy()
    else:
        filters = filter_set.to_numpy()

    Fmax = 0
    qmle_max = 0
    best_s = np.zeros(observed.shape)
    best_f = np.zeros(observed.shape)

    for i in range(filters.shape[0]):
        filter = filters[i]

        #compute q intervals:
        # qmin_i is first column, qmax_i is second column (qs)
        for k in range(len(observed)):
            if k == 0:
                qs = np.array([minmax_pen(observed[k], expected[k], delta)])
            else:
                qs = np.append(qs, [minmax_pen(observed[k], expected[k], delta)], axis=0)

        q_intervals = np.unique(qs)
        
        for j in range(len(q_intervals) - 1):
            qmid = (q_intervals[j] + q_intervals[j + 1])/2
            
            initial_subset = ((qmid > qs[:,0]) & (qmid < qs[:,1])*1)    

            qmle = sum(observed[initial_subset == 1])/sum(expected[initial_subset == 1], 0.000001)
            
            if verbose == True:
                print(f"QMLE of initial: {qmle}")

            all_scores = llr(observed, expected, qmle)

            F = sum(initial_subset*all_scores)

            if verbose == True:
                print(f"Score: {F}")

            if F > Fmax:
                Fmax = F
                qmle_max  = qmle
                best_s = initial_subset
                best_f = filter

        print(f"Filter = {i}")

        if verbose == True:
            print("-----------------------------------------------------------")

    return qmle_max, Fmax, best_s, best_f


#MAIN ROUTINE:
if __name__ == "__main__":
    
    theta = load_training_set()
    subset = load_test_set()

    times = []
    pfss_scores = []
    filter_jaccard = []
    subset_jaccard = []

    for i in range(1):
        
        baseline, observed, expected = preprocess(i)

        start = time.time()
        _, score, subset, filter = pfss(observed=observed, expected=expected, filter_set=theta, delta = 2, num_filters=50, verbose=False)
        end = time.time()

        times.append(end - start)
        pfss_scores.append(score)
        filter_jaccard.append(jaccard(baseline, filter))
        subset_jaccard.append(jaccard(baseline, subset))
    
    average_time = sum(times)/len(times)
    average_score = sum(pfss_scores)/len(pfss_scores)

    
    print(f"Average Time:{times}\nAverage Score: {pfss_scores}\n")

    plt.subplot(1,3,1)
    plt.imshow(baseline.reshape(28,28), cmap='gray')

    plt.subplot(1,3,2)
    plt.imshow(subset.reshape(28,28), cmap='gray')

    plt.subplot(1,3,3)
    plt.imshow(filter.reshape(28,28), cmap='gray')

    plt.show()