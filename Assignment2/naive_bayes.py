#-------------------------------------------------------------------------
# AUTHOR: Dharam Kathiriya
# FILENAME: naive_bayes.py
# SPECIFICATION: input the training set and output the classfication of each instance for the weather test set
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from pipes import Template
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
#--> add your Python code here
db = []
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
Outlook = {
    "Sunny": 1,
    "Overcast": 2,
    "Rain": 3
}

Temperature = {
    "Hot": 1,
    "Mild": 2,
    "Cool": 3
}

Humidity = {
    "High": 1,
    "Normal": 2
}

Wind = {
    "Strong": 1,
    "Weak": 2
}
for data in db:
    X.append([Outlook[data[1]], Temperature[data[2]], Humidity[data[3]], Wind[data[4]]])

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
PlayTennis = {
    "Yes": 1,
    "No": 2
}
for data in db:
    Y.append(PlayTennis[data[5]])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
dbTest = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, r in enumerate(reader):
        if i > 0: #skipping the header
            dbTest.append(r)

#printing the header of the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for data in dbTest:
    prediction = clf.predict_proba([ [Outlook[data[1]], Temperature[data[2]], Humidity[data[3]], Wind[data[4]]] ])[0]
    
    if prediction[0] >= 0.75:
        output = 'Yes'
        confidence = prediction[0]
    elif prediction[1] >= 0.75:
        output = 'No'
        confidence = prediction[1]
    else:
        continue

    # print result
    print(data[0].ljust(15) + data[1].ljust(15) + data[2].ljust(15) + data[3].ljust(15) + data[4].ljust(15) + output.ljust(15) + f"{confidence:.2f}")
        

