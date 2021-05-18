import sys
sys.path.append('../')
import ravop.core as R
import metrics

class NaiveBayesClassifier:
    
    def __init__(self):
        pass

    def seperate_classes(self, X, y):

        """
        To seperate the dataset into suvbsets
        """

        seperated_classes = {}

        for i in range(len(X)):
            
            feature_values = X[i]
            class_name = y[i]

            if class_name not in seperated_classes:
                seperated_classes[class_name] = []
            seperated_classes[class_name].append(feature_values)

        return seperated_classes

    def start_info(self, X):

        """
        Calculate mean and standard deviation
        """

        for feature in zip(*X):
            yield {
                'std': R.std(feature),
                'mean': R.mean(features)
            }

    def fit(self, X, y):
        
        """
        Train the model
        """

        seperated_classes = self.seperate_classes(X, y)
        self.class_summary = {}

        for class_name, feature_values in seperated_classes.items():

            self.class_summary[class_name] = {
                'prior_proba': len(feature_values)/len(X),
                'summary': [i for i in self.start_info(feature_values)]
            }

        return self.class_summary

    def distribution(self, x, mean, std):

        """
        Gaussian Distribution Function
        """

        exponent = R.exp(-((x - mean)**2 / (2*std**2)))
        gaussian_func = exponent / (R.square_root(2*(3.1415) * std))

    def predict(self, X):

        """
        Predict the output class
        """

        MAPs = []

        for row in X:
            joint_proba = {}

            for class_name, features in self.class_summary.items():
                total_features = len(features['summary'])
                likelihood = 1

                for idx in range(total_features):
                    feature = row[idx]
                    mean = features['summary'][idx]['std']
                    stdev = features['summary'][idx]['std']
                    normal_proba = self.distribution(feature, mean, stdev)
                    likelihood = normal_proba

                prior_proba = features['prior_proba']
                joint_proba[class_name] = prior_proba * likelihood

            MAP = max(joint_proba, key= joint_proba.get)
            MAPs.append((MAP))

            return MAPs

    
    def accuracy(self, y_test, y_pred):
        
        """
        Calculate model accuracy
        """

        true_true = 0

        for y_t, y_p in zip(y_test, y_pred):
            if(y_t == y_p):
                true_true += 1
        
        return true_true / len(y_test)