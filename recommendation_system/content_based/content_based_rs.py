import pickle
import pandas as pd
from scipy import spatial
import time
import json

from recommendation_system.metrics.metrics_util import recall_and_precision_at_k, average_precision, reciprocal_rank

usr_to_post_test = {}
post_cos_matrix = {}

cosine_matrix = {}
metrics = {}


def get_post_to_usr_dict(user_id, user_data):
    """
    Function that generates the ground truth data for each user id by returning a dictionary with keys the user ids
    and values the posts ids that belong to him from the test set
    :param user_id: the user id
    :param user_data: the user dataset
    :return: updates the usr_to_post_test dictionary
    """
    usr_to_post_test[user_id] = user_data["test_set"].index.tolist()


def prepare_testset(all_data, feature_type):
    """
    Function that generates the test set by concatenating all test sets for each user
    :param all_data: the user dataset
    :param feature_type: the feature types to be used
    :return: the generated test set
    """
    frames = []
    for value in all_data.values():
        frames += [value['test_set']]
    all_frames = pd.concat(frames)

    if feature_type == 'category':
        return all_frames[['Category', 'Concept', 'Subcategory']]
    elif feature_type == 'image':
        return all_frames.drop(['Postdate', 'Category', 'Concept', 'Subcategory'], axis=1)
    else:
        return all_frames.drop(['Postdate'], axis=1)


def prepare_trainset_for_matrix_calculations(user_data, feature_type):
    """
    Function that generates the training set for each user
    :param all_data: the user dataset
    :param feature_type: the feature types to be used
    :return: the generated training set
    """
    if feature_type == 'category':
        return user_data['train_set'][['Category', 'Concept', 'Subcategory']]
    elif feature_type == 'image':
        return user_data['train_set'].drop(['Postdate', 'Category', 'Concept', 'Subcategory'], axis=1)
    else:
        return user_data['train_set'].drop(['Postdate'], axis=1)


def create_empty_cosine_sim_matrix(train_set):
    """
    Function that initialized the cosine similarity dictionary
    :param train_set: the training set
    :return: the initialized cosine similarity dictionary
    """
    for key_train in train_set.index.tolist():
        cosine_matrix[key_train] = {}


def find_best_concepts(tr_df, tst_df):
    """
    Function that performs the concept-related optimization
    :param tr_df: the training dataframe
    :param tst_df: the test dataframe
    :return: the part of the test set to be checked
    """

    # calculate the mean of the feature vectors in the history
    history_mean = tr_df.mean().drop('Concept')

    # calculate the mean of the concept vectors from test set
    concepts = tst_df.groupby('Concept').mean()
    print('number of concepts: ' + str(concepts.shape[0]))

    concept_scores = {}
    # calculate the cosine similarity between the history mean and each concept vector
    for concept_id in concepts.index.tolist():
        concept_scores[concept_id] = calculate_cosine(history_mean, concepts.loc[concept_id])

    # and sort the results to descending cosine similarity values
    sorted_concepts = sorted(concept_scores.items(), key=lambda kv: kv[1], reverse=True)

    selected_concepts = []
    num_posts = 0
    # select the most promising concepts until the test set contains at least as mnay posts as the history
    for c in sorted_concepts:
        if num_posts > tr_df.shape[0]:
            break
        selected_concepts += [c[0]]
        num_posts += (tst_df.Concept.values == c[0]).sum()

    return tst_df.loc[tst_df['Concept'].isin(selected_concepts)]


def calculate_cosine_sim_matrix(train_set, test_set, clustered=True, enhanced=True):
    """
    Function that calculates the cosine similarity between certain pairs of posts from the train and the test set
    :param train_set: the training set
    :param test_set: the test set
    :param clustered: boolean flag to perform initial "Subcategory" grouping
    :param enhanced: boolean flag to use the concept enhancement
    :return: updates the cosine similarity dict of the user
    """
    if clustered:
        to_use = test_set.loc[test_set['Subcategory'].isin(train_set.Subcategory.unique().tolist())]
        print('Clustered test set size: ' + str(to_use.shape[0]))
        if enhanced:
            to_use = find_best_concepts(train_set, to_use)
            print('Enhanced clustered test set size: ' + str(to_use.shape[0]))
    else:
        to_use = test_set
    for key_train in train_set.index.tolist():
        for key_test in to_use.index.tolist():
            cosine = calculate_cosine(train_set.loc[key_train], to_use.loc[key_test])
            cosine_matrix[key_train][key_test] = cosine


def calculate_cosine(elem1, elem2):
    """
    Function that calculates the cosine similarity between to vectors
    :param elem1: vector 1
    :param elem2: vector 2
    :return: the cosine similarity
    """
    return 1 - spatial.distance.cosine(elem1, elem2)


def get_recommendations(top_k):
    """
    Function that adds the cosine similarities of each test post and keeps the greates top_k values
    :param top_k: the number of values to keep
    :return: the recommendations made
    """
    df = pd.DataFrame.from_dict(cosine_matrix)
    df = df.sum(axis=1)
    df = df.nlargest(top_k)
    return df.index.tolist()


def calculate_metrics(user_id, recs):
    """
    Function for calculating the evaluation metrics for the recommendation task
    :param user_id: the user id to be checked
    :param recs: the recommendations made
    :return: update the metrics dictionary with the results for the user with user id user_id
    """
    metrics[user_id] = {}
    ground_truth = usr_to_post_test[user_id]

    recall5, precision5 = recall_and_precision_at_k(recs, 5, ground_truth)
    metrics[user_id]["precision@5"] = precision5
    metrics[user_id]["recall@5"] = recall5

    recall10, precision10 = recall_and_precision_at_k(recs, 10, ground_truth)
    metrics[user_id]["precision@10"] = precision10
    metrics[user_id]["recall@10"] = recall10

    recall50, precision50 = recall_and_precision_at_k(recs, 50, ground_truth)
    metrics[user_id]["precision@50"] = precision50
    metrics[user_id]["recall@50"] = recall50

    metrics[user_id]["average_precision"] = average_precision(recs, ground_truth)
    metrics[user_id]["reciprocal_rank"] = reciprocal_rank(recs, ground_truth)


def overall_metrics():
    """
    Convert the metrics dictionary to dataframe for easier visualization
    :return: the generated dataframe
    """
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df = metrics_df.T
    return metrics_df


if __name__ == '__main__':
    with open("../../data/our_jsons/final_dataset.pickle", "rb") as input_file:
        complete_data = pickle.load(input_file)

    print('Preparing test set...')
    test_set = prepare_testset(complete_data, "all")
    print(test_set.shape[0])

    print('Testing on each user...')
    j = 0
    time_needed = {}
    start = time.time()
    final_keys = []
    # run the recommendation procedure for each user
    for key, value in complete_data.items():
        if value['train_set'].shape[0]:  # in case that the user does not have history
            final_keys += [key]
            start_per_user = time.time()
            print('Recommending on user ' + str(key) + ' with order ' + str(j))
            get_post_to_usr_dict(key, value)  # get the ground truth for each user
            train_set = prepare_trainset_for_matrix_calculations(value, "all")  # prepare the training set
            print('User\'s history length: ' + str(train_set.shape[0])) 
            create_empty_cosine_sim_matrix(train_set)  # initialize the cosine dict
            calculate_cosine_sim_matrix(train_set, test_set)  # calculate the similarities
            recommendations = get_recommendations(len(usr_to_post_test[key]))  # make the recommendations
            end_per_user = time.time()
            time_needed[key] = end_per_user - start_per_user
            calculate_metrics(key, recommendations)  # and calculate the metrics
            cosine_matrix.clear()
            print(metrics[key])
            j += 1

    end = time.time()
    time_needed['total'] = end - start
    with open('../../data/our_jsons/time_content_based.json', 'w') as outfile:
        json.dump(time_needed, outfile)

    print('Testing completed -> Let\'s see the metrics')
    metrics_df = overall_metrics()
    print(metrics_df.describe())
    with open('../../data/our_jsons/init_results.pickle', 'wb') as handle:
        pickle.dump(metrics_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('../../data/our_jsons/final_keys.pickle', 'wb') as handle:
    #     pickle.dump(final_keys, handle, protocol=pickle.HIGHEST_PROTOCOL)
