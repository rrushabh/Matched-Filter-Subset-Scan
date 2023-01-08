import numpy as np
import pandas as pd
import math

class Node:
    '''
    Multi-criterion decision tree to search the filter set.
    '''
    

    def __init__(self, theta:pd.DataFrame, name, depth = None):
        
        self.name = name
        self.num_filters = theta.shape[0]
        self.depth = depth if depth else 0

        # initialize left and right node to be empty
        self.left = None
        self.right = None

        # filter set is only stored for leaves
        self.theta = None
        
        # average filter
        self.avg_filter = theta.mean(axis=0).to_numpy()

    #calculate entropy on split element
    @staticmethod
    def calc_entropy(df):       
        return -df.mean(axis=0).apply(lambda x: x*np.log(x)+(1-x)*np.log(1-x) if x*(1-x)>0 else 0).sum()   

    # information gain for split element   
    def info_gains(self, data):
        size = data.shape[0]
        entropy_before = self.calc_entropy(data)
        info_gains = []
        for col in data.columns:
            left, right = data[data[col] == 0], data[data[col] == 1]
            left_size, right_size = left.shape[0], right.shape[0]
            entropy_after = (left_size/size)*self.calc_entropy(left) + (right_size/size)*self.calc_entropy(right)
            info_gains.append(entropy_before - entropy_after)
        return info_gains

    
    def build_tree(self, theta, max_depth=100, min_filters_to_split=2):

        if (self.depth < max_depth) and (self.num_filters >= min_filters_to_split):
        
            filters = theta
            all_info_gains = self.info_gains(filters)
            self.split_elem = filters.columns[np.argmax(all_info_gains)]
        
            #print("For node",self.name,", with",self.num_filters,"filters averaging",self.avg_filter,", we split on element",self.split_elem,"with information gain",all_info_gains[self.split_elem])
        
            l_split = filters[filters[self.split_elem] == 0]
            r_split = filters[filters[self.split_elem] == 1]   
            
            left = Node(l_split, self.name + "L",
                        self.depth + 1)
            
            self.left = left
            left.build_tree(l_split, max_depth, min_filters_to_split)

            right = Node(r_split, self.name + "R",
                        self.depth + 1)
             
            self.right = right
            right.build_tree(r_split, max_depth, min_filters_to_split)
            
        else:
            #print("Node",self.name,", with",self.num_filters,"filters averaging",self.avg_filter,", is a leaf node")
            self.theta = theta  # store filter set in leaf nodes only

    
    #Next three methods involve choosing correct branch based on Gaussian approach on 'n' iid filters. 
    #Filters may not be iid. Need to find a general method for choosing correct branch?
    def calc_mean_distance(self, subset, weights):
        return sum([weights[i] * (self.avg_filter[i]+(1-2*self.avg_filter[i])*subset[i]) for i in range(len(self.avg_filter))])

    def calc_variance(self, weights):
        return sum([(weights[i]**2) * self.avg_filter[i]*(1-self.avg_filter[i]) for i in range(len(self.avg_filter))])

    def choose_branch(self, subset, weights):

        #distance to average filter
        left_avg = self.left.calc_mean_distance(subset=subset, weights=weights)
        right_avg = self.right.calc_mean_distance(subset=subset, weights=weights)

        #calculate variance of distance
        left_var = self.left.calc_variance(weights=weights)
        right_var = self.right.calc_variance(weights=weights)

        #calculate minimum expected distance to average filter
        ex_distance_left = left_avg - math.sqrt(2*left_var*np.log(self.left.num_filters))
        ex_distance_right = right_avg - math.sqrt(2*right_var*np.log(self.right.num_filters))
        print("Comparing",self.left.name,"(distance",ex_distance_left,") and",self.right.name,"(distance",ex_distance_right,")")

        if ex_distance_left <= ex_distance_right:
            return self.left
        else:
            return self.right
        
    def traverse(self, subset, weights):
        if self.theta is None: # traverse the tree until we hit a leaf node
            return self.choose_branch(subset, weights).traverse(subset, weights)

        print("Searching node",self.name,"with",self.num_filters,"filters")
        #print(self.theta)
            
        temp_min = 1000000000
        temp_filter = np.zeros(self.theta.shape[1])

        for i in range(self.theta.shape[0]):
            dist = sum(weights*abs(self.theta.iloc[i] - subset))
            if dist < temp_min:
                temp_min = dist
                temp_filter = self.theta.iloc[i]

        return temp_filter, temp_min
