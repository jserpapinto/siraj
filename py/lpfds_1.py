from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# Classifiers
tree_clf = tree.DecisionTreeClassifier()
kn_clf = KNeighborsClassifier(n_neighbors=3)
nn_clf = MLPClassifier()

# Train
tree_clf.fit(X,Y)
kn_clf.fit(X,Y)
nn_clf.fit(X,Y)

# Predict
test_case = [[175,65,41]]
tree_prediction = tree_clf.predict(test_case)
kn_prediction = kn_clf.predict(test_case)
nn_prediction = nn_clf.predict(test_case)

# Print
print ("Tree: {}\nKN: {}\nNN: {}".format(tree_prediction, kn_prediction, nn_prediction))

