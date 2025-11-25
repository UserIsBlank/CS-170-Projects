# PROJECT 2 - Feature Search
import random
import math

class Classifier():
    def __init__(self, class_label, features):
        self.class_label = class_label
        self.features = features

        #for training
        self.training_ids = [] #set of training instances (IDs)
        self.training_class_label = [] #set of all training class labels (all labels except test one)
        self.normalized_training_features = [] #set of all normalized training feature sets
        self.means = []
        self.stds = []

    #training function - save normalized training data
    def training(self, training_ids):
        self.training_ids = training_ids
        self.training_class_label = [self.class_label[i] for i in training_ids]
        training_features = [self.features[i] for i in training_ids] #raw features
        self.normalized_training_features = []

        #find mean of each feature column
        self.means = [0.0] * len(training_features[0])
        for i in range(len(training_features[0])): #features within feature set
            total = 0.0
            for j in training_features: #feature set 0...n
                total += j[i]
            self.means[i] = total / len(training_ids)

        #find std of each feature column
        self.stds = [0.0] * len(training_features[0])
        for i in range(len(training_features[0])): #features within feature set
            var = 0.0
            for j in training_features: #feature set 0...n
                var += pow((j[i] - self.means[i]), 2)
            self.stds[i] = math.sqrt(var / len(training_ids))

        #normalize
        for f in training_features: #feature set 0...n
            norm_f = [] #normalized features for 1 feature set
            for j in range(len(training_features[0])): #features within the feature set
                if self.stds[j] == 0:
                    norm_f.append(0.0) #avoid having 0 as denominator
                else:
                    z_score = (f[j] - self.means[j]) / self.stds[j]
                    norm_f.append(z_score)
            self.normalized_training_features.append(norm_f) #set of all normalized feature sets

    #testing function - predict class label
    def test(self, test_id, feature_subset=None):
        test_features = self.features[test_id] #set of raw features in this test instance
        normalized_test_features = [] #set of normalized features in this test instance
        #normalize test features
        for i in range(len(test_features)):
            if self.stds[i] == 0:
                normalized_test_features.append(0.0)
            else:
                z_score = (test_features[i] - self.means[i]) / self.stds[i]
                normalized_test_features.append(z_score)
        
        closest_dist = float('inf')
        nearest_class_label = None
        for i in range(len(self.normalized_training_features)): #loop through set of normalized training feature sets
            train_instance = self.normalized_training_features[i] 

            #Euclidean distance
            curr_dist = 0.0
            if feature_subset is None: #all features
                for j in range(len(train_instance)): #calculate distance of each feature in training instance
                    curr_dist += pow((train_instance[j] - normalized_test_features[j]), 2)
            else: #feature selection
                for j in feature_subset:
                    curr_dist += pow((train_instance[j] - normalized_test_features[j]), 2)
            curr_dist = math.sqrt(curr_dist)

            if curr_dist < closest_dist:
                closest_dist = curr_dist
                nearest_class_label = self.training_class_label[i] #get corresponding class label of training instance
        return nearest_class_label

#load dataset
def load_dataset(path):
    class_label = []
    features = []
    instance_ids = []

    #read small dataset
    with open(path, "r") as f:
        for idx, line in enumerate(f): #each line = 1 instance
            floats = line.split()
            curr_class = int(float(floats[0])) #class label is first column
            curr_features = [float(x) for x in floats[1:]] #rest of the columns are features
            instance_ids.append(idx) #store instance ID
            class_label.append(curr_class)
            features.append(curr_features)
    return class_label, features, instance_ids

#stub eval func (returns random percentage)
def eval_func(features):
    return round(random.uniform(0, 100), 1)

def forward_selection(num_features, all_features_set):
    best_features = set() #subset of best features
    best_accuracy = eval_func(best_features)
    print(f"\nUsing no features and \"random\" evaluation, I get an accuracy of {best_accuracy}%\n")
    print("Beginning Search")

    #best feature subset can be at most the total # of features
    while len(best_features) < num_features:
        #the possible new feature subsets that can be made using the last iteration's best_features
        possible_features = {} #dict w/ key as feature subset and value as accuracy
        for f in all_features_set:
            if f in best_features: continue
            #union of the subset of best features and one of the other individual features
            features_key = frozenset(best_features | {f}) #make set immutable to use as key
            possible_features[features_key] = eval_func(features_key)
        
        for key, value in possible_features.items():
            print(f"Using feature(s) {{{', '.join(map(str, key))}}} accuracy is {value}%")
        
        #find best feature subset in features (one w/ highest accuracy)
        best_key, max_value = max(possible_features.items(), key=lambda kv: kv[1])
        if max_value < best_accuracy:
            print("\n(Warning, Accuracy has decreased!)")
            break
        
        best_features = set(best_key) #convert from frozenset back to set
        best_accuracy = max_value
        print(f"\nFeature set {best_features} was best, accuracy is {best_accuracy}%\n")
    
    # if accuracy immediately decreased at first iteration (since eval is random)
    if not best_features:
        print(f"Finished search!! The best feature subset is no features, which has an accuracy of {best_accuracy}%")
    else:
        print(f"Finished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy}%")

def backward_elimination(num_features, all_features_set):
    print("\nBeginning Search")
    current_features = all_features_set.copy()
    best_accuracy = eval_func(current_features)
    global_best_features = current_features.copy()
    global_best_accuracy = best_accuracy
    
    best_accuracy = eval_func(current_features)  # Calculate actual accuracy
    print(f"Using all features {{{', '.join(map(str, current_features))}}}, I get an accuracy of {best_accuracy}%\n")
    
    # continue until we have only one feature left
    while len(current_features) > 1:
        possible_features = {}  # dict with key as feature subset and value as accuracy
        
        # try removing each feature one at a time
        for f in current_features:
            features_subset = current_features - {f}
            features_key = frozenset(features_subset)
            possible_features[features_key] = eval_func(current_features)
        
        # print all tested subsets
        for key, value in possible_features.items():
            print(f"Using feature(s) {{{', '.join(map(str, key))}}} accuracy is {value}%")

        if not possible_features:
            break
        
        # find the best feature subset (one with highest accuracy)
        best_key, max_value = max(possible_features.items(), key=lambda kv: kv[1])
        
        # update best overall if we found a better combination
        if max_value > global_best_accuracy:
            global_best_accuracy = max_value
            global_best_features = set(best_key)
            print(f"\nNew global best found!")
        
        print(f"\nFeature set {set(best_key)} was best, accuracy is {max_value}%")
        
        # check if accuracy decreased
        if max_value < best_accuracy:
            print("\n(Warning, Accuracy has decreased from global best!)")
            # Continue with the search but keep track of global best
        
        current_features = set(best_key)  # update current features for next iteration
        best_accuracy = max_value

        print()
    
    print(f"Finished search! The best feature subset is {global_best_features}, which has an accuracy of {global_best_accuracy}%")
    return global_best_features, global_best_accuracy


if __name__ == "__main__":
    print("Welcome to our Feature Selection Algorithm")
    num_features = int(input("Please enter total number of features: "))
    print("Type the number of the algorithm you want to run\n")
    print("1. Forward Selection\n")
    print("2. Backward Elimination\n")
    
    algo = int(input())

    all_features = set(range(1, num_features + 1)) #total features stored in set

    if algo == 1:
        forward_selection(num_features, all_features)
    elif algo == 2:
        backward_elimination(num_features, all_features)
    else:
        print("Invalid input")