import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def correlation(estimator, X, y):
    y_hat = estimator.fit(X,y).predict(X)
    score = r2_score(y, y_hat)
    return score

def accuracy(estimator, X, y):
    y_hat = estimator.fit(X,y).predict(X)
    return accuracy_score(y, y_hat)



df = pd.read_csv("movie_data.csv", index_col=0)
df.head()

df["profitable"] = df["revenue"] > df ["budget"]
df["profitable"] = df["profitable"].replace({True: 1, False : 0})


#revenue = df["revenue"]
#profitable = df["profitable"]

regression_target = "revenue"
classification_target = "profitable"

df = df.replace(to_replace = ["np.inf", "-np.inf"], value = "np.nan")

df = df.dropna(axis = 0, how = "any")

genre = []

for val in df["genres"]:
    for i in range(len(val)):
        gen_val = val.split(",")
        for gen in gen_val:
            gen = gen.strip()
            if gen not in genre:
                genre.append(gen)
    
for gen in genre:
    val_list = []
    for i in df.index:
        if gen in df.loc[i][1]:
            val_list.append(1)
        else:
            val_list.append(0)
    df[gen] = val_list
    

continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]
plotting_variables = ['budget', 'popularity', regression_target]

#axes = pd.plotting.scatter_matrix(df[plotting_variables], alpha=0.15, \
 #      color=(0,0,0), hist_kwds={"color":(0,0,0)}, facecolor=(1,0,0))
# show the plot.
#plt.show()

for val in continuous_covariates:
    df[val] = np.log10(1 + df[val])

df.to_csv("movies_clean.csv")

df = pd.read_csv('movies_clean.csv')
df = df[df["revenue"] > 0]

regression_target = 'revenue'
classification_target = 'profitable'
all_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average', 'Action', 'Adventure', 'Fantasy', 
                  'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy', 'Romance', 
                  'Horror', 'Mystery', 'War', 'History', 'Music', 'Documentary', 'TV Movie', 'Foreign']

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)

forest_regression.fit(df[all_covariates], df[regression_target])    
out1 = sorted(list(zip(all_covariates, forest_regression.feature_importances_)), key=lambda tup: tup[1])
x1 = []
y1 = []
x2 = []
y2 = []

for val in range(len(out1)):
    x1.append(out1[val][0])
    y1.append(out1[val][1])
    
    
forest_classifier.fit(df[all_covariates], df[classification_target])
out2 = sorted(list(zip(all_covariates, forest_classifier.feature_importances_)), key=lambda tup: tup[1])
for val in range(len(out2)):
    x2.append(out2[val][0])
    y2.append(out2[val][1])

plt.figure(figsize = (20, 30))
plt.subplot(221)

#plt.axes().set_aspect('equal', 'box')
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.plot((0, 1), (0, 1), 'k-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Regression Score")
plt.ylabel("Forest Regression Score")


plt.subplot(222)
#plt.axes().set_aspect('equal', 'box')
plt.scatter(logistic_regression_scores, forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Classification Score")
plt.ylabel("Forest Classification Score")

plt.subplot(223)
plt.plot(x1, y1, "ro-")
plt.ylim(0, 1)
plt.xlabel("Factors")
plt.ylabel("Effect")
plt.xticks(rotation=90)

plt.subplot(224)
plt.plot(x2, y2, "ro-")
plt.ylim(0, 1)
plt.xlabel("Factors")
plt.ylabel("Effect")
plt.xticks(rotation=90)

plt.savefig("movie_research.pdf")



with open("movie_analysis.txt", "w") as my_file:
    my_file.write("Regression\n")
    my_file.write(' '.join(map(str, out1)))
    my_file.write("\n")
    my_file.write("Classification\n")
    my_file.write(' '.join(map(str, out2)))