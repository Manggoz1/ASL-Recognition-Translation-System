import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


hand_landmarks_data = pd.read_csv("hand_landmarks_data.csv",sep=",",header=None)
hand_landmarks_letter = pd.read_csv("hand_landmarks_letter.csv",header=None)


X_train, X_test, Y_train, Y_test = train_test_split(hand_landmarks_data,hand_landmarks_letter,test_size=.2,random_state=185)


# Random parameters
hidden_layer_sizes = (20, 25)  # Example: 3 layers with 50, 100, and 50 neurons
activation = 'relu'  # Activation function: 'identity', 'logistic', 'tanh', 'relu'
solver = 'adam'  # Solver: 'lbfgs', 'sgd', 'adam'
alpha = 0.0001  # Regularization parameter
learning_rate = 'adaptive'  # Learning rate schedule: 'constant', 'adaptive', 'invscaling'
learning_rate_init = 0.001  # Initial learning rate
max_iter = 200  # Maximum number of iterations

# Create the classifier with random parameters
clf = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    solver=solver,
    alpha=alpha,
    learning_rate=learning_rate,
    learning_rate_init=learning_rate_init,
    max_iter=max_iter,
    random_state=1  # Random state for reproducibility
)

for i in range(10):
  # Train the classifier
  clf.fit(X_train, Y_train)
  test_hand_landmarks_data = pd.read_csv("test_hand_landmarks_data.csv",sep=",",header=None)
  predictions = clf.predict(test_hand_landmarks_data)
  print(predictions)

# Train the classifier
clf_train_data_score = clf.score(X_train, Y_train)
clf_test_data_score = clf.score(X_test, Y_test)

print("The score of the Neural Network in training data is "+str(clf_train_data_score))
print("The score of the Neural Network in test data is "+str(clf_test_data_score))