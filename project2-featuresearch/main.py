# PROJECT 2 - Feature Search
import random
import math
import time # Added for timing requirements

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
        if not training_features:
            return

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

# Validator Class
class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def evaluate(self, feature_subset, instance_ids):
        # Convert 1-based feature IDs (from search) to 0-based indices (for list access)
        if not feature_subset:
            subset_0_based = None
        else:
            subset_0_based = [f - 1 for f in feature_subset]
        
        correct_predictions = 0
        
        # Leave-One-Out Cross Validation Loop
        for i in range(len(instance_ids)):
            # 1. Isolate the test instance ID
            test_id = instance_ids[i]
            
            # 2. Create the training set (all IDs except the test_id)
            training_ids = instance_ids[:i] + instance_ids[i+1:]
            
            # 3. Train classifier on this specific fold
            self.classifier.training(training_ids)
            
            # 4. Test on the isolated instance
            predicted_label = self.classifier.test(test_id, subset_0_based)
            
            # 5. Check if prediction is correct
            actual_label = self.classifier.class_label[test_id]
            if predicted_label == actual_label:
                correct_predictions += 1
                
        accuracy = correct_predictions / len(instance_ids)
        return accuracy * 100

#load dataset
def load_dataset(path):
    class_label = []
    features = []
    instance_ids = []
    
    try:
        with open(path, "r") as f:
            for idx, line in enumerate(f): #each line = 1 instance
                floats = line.split()
                if not floats: continue
                curr_class = int(float(floats[0])) #class label is first column
                curr_features = [float(x) for x in floats[1:]] #rest of the columns are features
                instance_ids.append(idx) #store instance ID
                class_label.append(curr_class)
                features.append(curr_features)
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        return [], [], []
        
    return class_label, features, instance_ids

def forward_selection(num_features, all_features_set, validator, instance_ids):
    best_features = set() 
    best_accuracy = validator.evaluate(best_features, instance_ids)
    
    print(f"\nUsing no features, I get an accuracy of {best_accuracy}%\n")
    print("Beginning Search")
    
    start_time = time.time() # Start Timer

    #best feature subset can be at most the total # of features
    while len(best_features) < num_features:
        possible_features = {} 
        for f in all_features_set:
            if f in best_features: continue
            
            current_subset = best_features | {f}
            # CALL VALIDATOR HERE
            accuracy = validator.evaluate(current_subset, instance_ids)
            
            print(f"Using feature(s) {{{', '.join(map(str, current_subset))}}} accuracy is {accuracy}%")
            possible_features[frozenset(current_subset)] = accuracy
        
        if not possible_features: break

        best_subset_key, max_accuracy = max(possible_features.items(), key=lambda kv: kv[1])
        
        if max_accuracy < best_accuracy:
            print("\n(Warning, Accuracy has decreased!)")
        
        best_features = set(best_subset_key)
        best_accuracy = max_accuracy
        print(f"\nFeature set {best_features} was best, accuracy is {best_accuracy}%\n")
    
    end_time = time.time() # End Timer
    print(f"Search finished in {end_time - start_time:.2f} seconds.")
    
    if not best_features:
        print(f"Finished search!! The best feature subset is no features, which has an accuracy of {best_accuracy}%")
    else:
        print(f"Finished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy}%")

def backward_elimination(num_features, all_features_set, validator, instance_ids):
    print("\nBeginning Search")
    current_features = all_features_set.copy()
    
    start_time = time.time() # Start Timer
    
    best_accuracy = validator.evaluate(current_features, instance_ids)
    print(f"Using all features {{{', '.join(map(str, current_features))}}}, I get an accuracy of {best_accuracy}%\n")
    
    global_best_features = current_features.copy()
    global_best_accuracy = best_accuracy

    while len(current_features) > 1:
        possible_features = {} 
        
        for f in current_features:
            subset = current_features - {f}
            # CALL VALIDATOR HERE
            accuracy = validator.evaluate(subset, instance_ids)
            
            print(f"Using feature(s) {{{', '.join(map(str, subset))}}} accuracy is {accuracy}%")
            possible_features[frozenset(subset)] = accuracy

        if not possible_features:
            break
        
        best_subset_key, max_accuracy = max(possible_features.items(), key=lambda kv: kv[1])
        
        if max_accuracy > global_best_accuracy:
            global_best_accuracy = max_accuracy
            global_best_features = set(best_subset_key)
            print(f"\nNew global best found!")
        
        print(f"\nFeature set {set(best_subset_key)} was best, accuracy is {max_accuracy}%")
        
        if max_accuracy < best_accuracy:
            print("\n(Warning, Accuracy has decreased from global best!)")
        
        current_features = set(best_subset_key)
        best_accuracy = max_accuracy

        print()
    
    end_time = time.time() # End Timer
    print(f"Search finished in {end_time - start_time:.2f} seconds.")

    print(f"Finished search! The best feature subset is {global_best_features}, which has an accuracy of {global_best_accuracy}%")
    return global_best_features, global_best_accuracy


if __name__ == "__main__":
    print("Welcome to our Feature Selection Algorithm")
    filename = input("Type in the name of the file to test: ") 
    
    class_labels, features, instance_ids = load_dataset(filename)
    
    if features:
        num_features = len(features[0])
        
        # Initialize Classifier and Validator
        my_classifier = Classifier(class_labels, features)
        my_validator = Validator(my_classifier)

        # --- SANITY CHECK (Required by PDF to verify Part II) ---
        print("\n--- Running Sanity Check ---")
        if "small" in filename.lower():
            print("Detected Small Dataset. Checking features {3, 5, 7}...")
            acc = my_validator.evaluate({3, 5, 7}, instance_ids)
            print(f"Accuracy for {{3, 5, 7}} is {acc}% (Expected ~89%)")
        elif "large" in filename.lower():
            print("Detected Large Dataset. Checking features {1, 15, 27}...")
            acc = my_validator.evaluate({1, 15, 27}, instance_ids)
            print(f"Accuracy for {{1, 15, 27}} is {acc}% (Expected ~94.9%)")
        else:
            print("Custom dataset detected. Skipping specific feature check.")
        print("----------------------------\n")
        # --------------------------------------------------------

        print(f"Type the number of the algorithm you want to run")
        print("1. Forward Selection")
        print("2. Backward Elimination")
        
        try:
            algo = int(input())
            all_features = set(range(1, num_features + 1)) 

            if algo == 1:
                forward_selection(num_features, all_features, my_validator, instance_ids)
            elif algo == 2:
                backward_elimination(num_features, all_features, my_validator, instance_ids)
            else:
                print("Invalid Algorithm Selection")
        except ValueError:
            print("Please enter a number.")