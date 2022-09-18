#-------------------------------------------------------------------------
# AUTHOR: Dharam Kathiriya
# FILENAME: decision_tree.py
# SPECIFICATION: Derive a decision tree using ID3 algorithm
# FOR: CS 4210- Assignment #1
# TIME SPENT: 3 days (2-3 hours per day)
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to #work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree 
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here

# create a dictionalry to store the numbers for categorical training features and classes 
numbers = { "Young": 1, "Prepresbyopic": 2, "Presbyopic": 3,
            "Myope": 1, "Hypermetrope": 2, 
            "Yes": 1, "No": 2,
            "Reduced": 1, "Normal": 2 }

# iterate over db to read and transform features to numbers
for i in range(len(db)):
    num = []
    for j in range(len(db[i]) - 1): 
        # get correspoding number for feature from dictionary and add it to numbers list
        val = db[i][j]
        num.append(numbers[val]) 
    X.append(num) # add each instance in X


#transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
for i in range(len(db)):
    classValue = db[i][len(db[i])-1]
    Y.append(numbers[classValue]) 


#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy', )
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()

