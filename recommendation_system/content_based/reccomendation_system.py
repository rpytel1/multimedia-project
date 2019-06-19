from scipy import spatial
import numpy as np


def recommendations(title, cosine_sim=cosine_sim):
    # initializing the empty list of recommended movies
    recommended_movies = []

    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])

    return recommended_movies


def calculate_cosine_sim_matrix(train_set, test_set):
    rows = len(train_set)
    cols = len(test_set)
    matrix = np.zeros((rows, cols))
    for rows
    return matrix


def calculate_cosine(elem1, elem2):
    return 1 - spatial.distance.cosine(elem1, elem2)
