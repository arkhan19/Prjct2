import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Importing Data
dataset = pd.read_csv('Data.csv')
# Correlations
correlations = dataset.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()

# Removing all those attributes with less correlation
df = dataset.drop([col for col in ['movie_title', 'color', 'director_name', 'actor_2_name', 'actor_1_name',
                                   'actor_3_name', 'plot_keywords', 'movie_imdb_link', 'language', 'country',
                                   'content_rating', 'budget', 'title_year', 'aspect_ratio', 'genres',
                                   'num_critic_for_reviews', 'num_voted_users', 'cast_total_facebook_likes',
                                   'num_user_for_reviews', 'movie_facebook_likes'] if col in dataset], axis=1)

X = df.iloc[:, 0:7].values  # except last col
y = df.iloc[:, 7].values  # Last column array
Y = y.reshape(-1, 1)  # Needed Here

# Missing Values
imp = Imputer(missing_values='NaN', strategy='mean', axis = 1)
X = imp.fit_transform(X)
# df = df.replace(np.nan, ' ', regex=True)  # Only works on dataframe object not on ndarray.
# Splitting data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


