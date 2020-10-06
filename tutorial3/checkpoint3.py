# Block 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

# Block 2: Read csv file
df = pd.read_csv('states_edu.csv')

# Block 3: Cleanup data

# Block 4: Feature selection

# Block 5: EDA visualization 1

# Block 6: EDA Visualization 2

# Block 7-9 Data Creation

X = ??
y = ??
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=??, random_state=0)

# Block 10: import your sklearn class here

# Block 11: create your model here
model = None

# Block 12-14
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# for classification:

plot_confusion_matrix(model, X_test, y_test,
                         cmap=plt.cm.Blues)

# Block 15
# for regression: (pick a single column to visualize results)

# Results from this graph _should not_ be used as a part of your results -- it is just here to help with intuition. 
# Instead, look at the error values and individual intercepts.


col_name = ??
col_index = X_train.columns.get_loc(col_name)

f = plt.figure(figsize=(12,6))
plt.scatter(X_train[col_name], y_train, color = "red")
plt.scatter(X_train[col_name], model.predict(X_train), color = "green")
plt.scatter(X_test[col_name], model.predict(X_test), color = "blue")

new_x = np.linspace(X_train[col_name].min(),X_train[col_name].max(),200)
intercept = model.predict([X_train.sort_values(col_name).iloc[0]]) - X_train[col_name].min()*model.coef_[col_index]
plt.plot(new_x, intercept+new_x*model.coef_[col_index])

plt.legend(['controlled model','true training','predicted training','predicted testing'])
plt.xlabel(col_name)
plt.ylabel(??)