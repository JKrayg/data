# Jake Krayger CSC495
# Final Project code

import math
import pandas as pd
import numpy as np
import tabulate as tab
import statistics as stat
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

df_iq = pd.DataFrame(pd.read_csv('iq.csv'))
df_qol = pd.DataFrame(pd.read_csv('quality_of_life.csv'))
df_pop = pd.DataFrame(pd.read_csv('population_density.csv'))

# population data has spaces after some country names
for r in df_pop['country']:
    if r.endswith(' '):
        t = list(r)
        t[len(t) - 1] = ''
        df_pop.loc[df_pop['country'] == r, 'country'] = ''.join(t)

avr = []
# add column for above or below average IQ
for r in df_iq['iq']:
    if r > df_iq['iq'].mean():
        avr.append("above")
    else:
        avr.append("below")

df_iq.insert(1, "average", avr)

# merge files
df_merge = pd.merge(df_iq, df_qol,
                    on = 'country',
                    how = 'left')
df_merge = pd.merge(df_merge, df_pop,
                    on = 'country',
                    how = 'left')


# drop country column
df_dropFirst = df_merge.iloc[:, 2:]

# convert any strings to floats
for c in df_dropFirst.columns:
    for r in df_dropFirst[c]:
        if type(r) == str:
            s = r.replace(',', '')
            if s.__contains__(' M'):
                s = s.replace(' M', '')
                f = float(s) * 10**6
                df_merge.loc[df_merge[c] == r, c] = f
            else:
                f = float(s)
                df_merge.loc[df_merge[c] == r, c] = f

    # create dictionary with column name and mean as key and value
    df_merge[c] = df_merge[c].replace(0, np.nan)
    df_merge[c] = df_merge[c].replace(np.nan, round(df_merge[c].mean(), 2))

# write to main file
df_merge.to_csv('./data.csv')



# mean, median, mode, std ...
rows = []
for c in df_dropFirst.columns:
    col = df_merge[c]
    data = [c,
            round(col.min(), 2),
            round(col.max(), 2),
            round((stat.mean(col)), 2),
            round((stat.mode(col)), 2),
            round((col.max() - col.min()), 2),
            round(col.std(), 2)]

    rows.append(data)


cols = ['Attr', 'Min', 'Max', 'Average', 'Mode', 'Range', 'Std']
df_desc = pd.DataFrame(rows, columns = cols)

# write to decriptives file
df_desc.to_csv('./decriptives.csv')

#show descriptives
print(tab.tabulate(rows, headers=cols))


#show IQ x Education expenditure plot
plt.scatter(x = df_merge['education_expenditure_per_inhabitant'], y = df_merge['iq'], s = 10)
plt.xlabel('education expenditure', fontsize = 16)
plt.ylabel("IQ", fontsize = 16)
plt.tick_params(labelsize = 12)
plt.show()


#show IQ x health plot
plt.scatter(x = df_merge['health'], y = df_merge['iq'], s = 10)
plt.xlabel('health', fontsize = 16)
plt.ylabel("IQ", fontsize = 16)
plt.tick_params(labelsize = 12)
plt.show()


#show IQ x safety plot
plt.scatter(x = df_merge['safety'], y = df_merge['iq'], s = 10)
plt.xlabel('safety', fontsize = 16)
plt.ylabel( "IQ", fontsize = 16)
plt.tick_params(labelsize = 12)
plt.show()


# show health x safety plot
plt.scatter(x = df_merge['safety'], y = df_merge['health'], s = 10)
plt.xlabel('safety', fontsize = 16)
plt.ylabel("health", fontsize = 16)
plt.tick_params(labelsize = 12)
plt.show()


# knn algorithm to classify above and below average IQ
x = df_merge.iloc[:, [2, 6, 7]]
y = df_merge.iloc[:, 1]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2)

scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'euclidean')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


# confusion matrix from knn
acc = accuracy_score(y_test, y_pred)
print("\nClassification Report:")
print('Accuracy Score:', acc, '\n')
print(classification_report(y_test, y_pred))
conf = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf)

plt.figure(figsize = (7,5))
sns.heatmap(conf, annot=True, xticklabels = ['above', 'below'] , yticklabels = ['above', 'below'])
plt.xlabel('Predicted', fontsize = 16)
plt.ylabel('Actual', fontsize = 16)
plt.tick_params(labelsize = 14)
plt.show()