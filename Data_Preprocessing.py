import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing Data
dataset = pd.read_csv('Data.csv')
dataset.corr(method='pearson', min_periods=1)  # Correlation among the dataset df
dataset.plot(kind='density', subplots=True, sharex=False) # Density plot of the attributes
df = dataset.drop([col for col in ['movie_title', 'color', 'director_name', 'actor_2_name', 'actor_1_name',
                                   'actor_3_name', 'plot_keywords', 'movie_imdb_link', 'language', 'country',
                                   'content_rating', 'budget', 'title_year', 'aspect_ratio', 'genres',
                                   'num_critic_for_reviews', 'num_voted_users', 'cast_total_facebook_likes',
                                   'num_user_for_reviews', 'movie_facebook_likes'] if col in dataset], axis=1,
                  inplace=True)  # drops these col


X = df.iloc[:, 0:7].values  # except last col
y = df.iloc[:, 8].values  # Last column array
Y = y.reshape(-1, 1)  # Needed Here

# Missing Values
X = X.replace(np.nan, ' ', regex=True)  # Only works on dataframe object not on ndarray.
