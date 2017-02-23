import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Importing Data
dataset = pd.read_csv('Data.csv')
# Correlations
# correlations = dataset.corr()
# plot correlation matrix
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,8,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# plt.show()

# Removing all those attributes with less correlation
df = dataset.drop([col for col in ['movie_title', 'color', 'director_name', 'actor_2_name', 'actor_1_name', 'budget',
                                   'actor_3_name', 'plot_keywords', 'movie_imdb_link', 'content_rating', 'country',
                                   'title_year', 'aspect_ratio', 'genres', 'num_critic_for_reviews', 'num_voted_users',
                                   'cast_total_facebook_likes', 'num_user_for_reviews', 'movie_facebook_likes', 'language']
                   if col in dataset], axis=1)

X = df.iloc[:, :-1].values  # except last col
# Used in asarray Y = df.iloc[:, -1:].values  # Last column array
Y = np.asarray(df.iloc[:, -1:].values, dtype="|S6") #Y was object
#Label Encoder for Language column -ZZZ
label = LabelEncoder()
X[:,6] = label.fit_transform(X[:,6])

# Missing Values
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis = 1)
X = imp.fit_transform(np.array(X))

# Dummy Variable -ZZZ
dum = OneHotEncoder()
X = dum.fit_transform(X).toarray()


# df = df.replace(np.nan, ' ', regex=True)  # Only works on dataframe object not on ndarray.
# Splitting data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


