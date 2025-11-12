# PROJECT 2 - Feature Search
import random

#stub eval func (returns random percentage)
def eval_func():
    return round(random.uniform(0, 100), 1)

def forward_selection(num_features, all_features_set, best_accuracy):
    best_features = set() #subset of best features

    #best feature subset can be at most the total # of features
    while len(best_features) < num_features:
        #the possible new feature subsets that can be made using the last iteration's best_features
        possible_features = {} #dict w/ key as feature subset and value as accuracy
        for f in all_features_set:
            if f in best_features: continue
            #union of the subset of best features and one of the other individual features
            features_key = frozenset(best_features | {f}) #make set immutable to use as key
            possible_features[features_key] = eval_func()
        
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

def backward_elimination(num_features, all_features_set, best_accuracy):
    current_features = all_features_set.copy()  # start with all features
    best_features = current_features.copy()
    
    print(f"Initial set: {current_features} with accuracy {best_accuracy}%\n")
    
    # continue until we have only one feature left
    while len(current_features) > 1:
        possible_features = {}  # dict with key as feature subset and value as accuracy
        
        # try removing each feature one at a time
        for f in current_features:
            features_subset = current_features - {f}
            features_key = frozenset(features_subset)
            possible_features[features_key] = eval_func()
        
        # print all tested subsets
        for key, value in possible_features.items():
            print(f"Using feature(s) {{{', '.join(map(str, key))}}} accuracy is {value}%")
        
        # find the best feature subset (one with highest accuracy)
        best_key, max_value = max(possible_features.items(), key=lambda kv: kv[1])
        
        # update best overall if we found a better combination
        if max_value > best_accuracy:
            best_accuracy = max_value
            best_features = set(best_key)
            print(f"\nNew global best found!")
        
        print(f"\nFeature set {set(best_key)} was best, accuracy is {max_value}%")
        
        # check if accuracy decreased
        if max_value < best_accuracy:
            print("\n(Warning, Accuracy has decreased from global best!)")
            # Continue with the search but keep track of global best
        
        current_features = set(best_key)  # update current features for next iteration
        print()
    
    print(f"Finished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy}%")


if __name__ == "__main__":
    print("Welcome to our Feature Selection Algorithm")
    num_features = int(input("Please enter total number of features: "))
    print("Type the number of the algorithm you want to run\n")
    algo = int(input("Forward Selection\nBackward Elimination\n"))

    all_features = set(range(1, num_features + 1)) #total features stored in set
    best_accuracy = eval_func()

    print(f"\nUsing no features and \"random\" evaluation, I get an accuracy of {best_accuracy}%\n")
    print("Beginning Search")
    if algo == 1:
        forward_selection(num_features, all_features, best_accuracy)