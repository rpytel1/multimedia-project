import json
from statistics import mean, median, stdev
import pickle
import pandas as pd
import datetime


def find_user_posts(user_data):
    """
    Function that creates a dictionary with the user ids as keys and the number of their posts as values
    :param user_data: the initial user dataset
    :return: the created dict
    """
    posts = {}
    for user in user_data.keys():
        posts[user] = len(user_data[user].keys())
    return posts


def remove_users(user_posts, keys_to_keep):
    """
    Function that retains only the ids specified in keys_to_keep from the dict with the user ids as keys
    :param user_posts: a user to number of posts dictionary
    :param keys_to_keep: the list of users ids to keep
    :return: the updated dictionary
    """
    fin_data = {}
    for keep in keys_to_keep:
        fin_data[keep] = user_posts[keep]
    return fin_data


def print_time_statistics(data, name):
    """
    Function that prints time-related statistics provided in the data
    :param data: a dictionary with time measurements for each user
    :param name: the name of the evaluated method (for printing purposes)
    :return: prints the time statistics
    """
    print('-------------------------- {} Time Statistics --------------------------'.format(name))
    print('Total time needed: ' + str(datetime.timedelta(seconds=data['total'])))
    del data['total']
    print('Mean value of time needed per user: ' + str(datetime.timedelta(seconds=mean(list(data.values())))))
    print('Std value of time needed per user: ' + str(datetime.timedelta(seconds=stdev(list(data.values())))))
    print('Median value of time needed per user: ' + str(datetime.timedelta(seconds=median(list(data.values())))))
    print('Max value of time needed per user ' + str(datetime.timedelta(seconds=max(list(data.values())))))
    print('Min value of time needed per user: ' + str(datetime.timedelta(seconds=min(list(data.values())))))


if __name__ == '__main__':

    # open the data and the final user ids files
    with open('data/our_jsons/user_dataset.json') as json_file:
        user_data = json.load(json_file)

    with open("data/our_jsons/final_keys.pickle", "rb") as input_file:
        final_keys = pickle.load(input_file)

    # generate the main statistics of the dataset
    posts_per_user = find_user_posts(user_data)
    posts_per_user = remove_users(posts_per_user, final_keys)
    print('-------------------------- Final Dataset Statistics --------------------------')
    print('Mean value of posts per user: ' + str(mean(list(posts_per_user.values()))))
    print('Std value of posts per user: ' + str(stdev(list(posts_per_user.values()))))
    print('Median value of posts per user: ' + str(median(list(posts_per_user.values()))))
    print('Max value of posts per user: ' + str(max(list(posts_per_user.values()))))
    print('Min value of posts per user: ' + str(min(list(posts_per_user.values()))))
    print('Total number of users: ' + str(len(posts_per_user.values())))
    print('Total number of posts: ' + str(sum(posts_per_user.values())))

    # Results printing region - Open the result pickles and display the results from the provided dataframes
    print('-------------------------- Initial results --------------------------')
    with open("data/our_jsons/results/init_results.pickle", "rb") as input_file:
        first_results = pickle.load(input_file)

    with pd.option_context('display.max_columns', 100):
        print(first_results.describe())

    print('-------------------------- Date results --------------------------')
    with open("data/our_jsons/results/date_init_results.pickle", "rb") as input_file:
        date_results = pickle.load(input_file)

    with pd.option_context('display.max_columns', 100):
        print(date_results.describe())

    print('-------------------------- Enhanced results --------------------------')
    with open("data/our_jsons/results/enhanced_results.pickle", "rb") as input_file:
        enhanced_results = pickle.load(input_file)

    with pd.option_context('display.max_columns', 100):
        print(enhanced_results.describe())

    print('-------------------------- Enhanced results with tags --------------------------')
    with open("data/our_jsons/results/enhanced_results_with_tags.pickle", "rb") as input_file:
        enhanced_results_tags = pickle.load(input_file)

    with pd.option_context('display.max_columns', 100):
        print(enhanced_results_tags.describe())

    print('-------------------------- Enhanced results with date --------------------------')
    with open("data/our_jsons/results/date_enhanced_results.pickle", "rb") as input_file:
        date_enhanced_results = pickle.load(input_file)

    with pd.option_context('display.max_columns', 100):
        print(date_enhanced_results.describe())

    print('-------------------------- Enhanced results with date and tags --------------------------')
    with open("data/our_jsons/results/date_enhanced_results_with_tags.pickle", "rb") as input_file:
        date_enhanced_results_with_tags = pickle.load(input_file)

    with pd.option_context('display.max_columns', 100):
        print(date_enhanced_results_with_tags.describe())

    print('-------------------------- ML results --------------------------')
    with open("data/our_jsons/results/ml_results.pickle", "rb") as input_file:
        ml_results = pickle.load(input_file)

    with pd.option_context('display.max_columns', 100):
        print(ml_results.describe())

    print('-------------------------- ML results with tags --------------------------')
    with open("data/our_jsons/results/ml_results_with_tags.pickle", "rb") as input_file:
        ml_results_with_tags = pickle.load(input_file)

    with pd.option_context('display.max_columns', 100):
        print(ml_results_with_tags.describe())

    # the time statistics region - again open the time results json files and print them in a readable fashion
    print('-------------------------- TIME --------------------------')
    with open('data/our_jsons/results/time_results/time_content_based.json') as json_file:
        time_data = json.load(json_file)
    print_time_statistics(time_data, 'initial content based')

    with open('data/our_jsons/results/time_results/time_content_based_enhanced.json') as json_file:
        time_data = json.load(json_file)
    print_time_statistics(time_data, 'enhanced content based')

    with open('data/our_jsons/results/time_results/time_content_based_with_tags.json') as json_file:
        time_data = json.load(json_file)
    print_time_statistics(time_data, 'enhanced content based with tags')

    with open('data/our_jsons/results/time_results/time_date_based_enhanced.json') as json_file:
        time_data = json.load(json_file)
    print_time_statistics(time_data, 'enhanced date based')

    with open('data/our_jsons/results/time_results/time_date_based_enhanced_with_tags.json') as json_file:
        time_data = json.load(json_file)
    print_time_statistics(time_data, 'enhanced date based with tags')

    with open('data/our_jsons/results/time_results/time_classify.json') as json_file:
        time_data = json.load(json_file)
    print_time_statistics(time_data, 'ml based')

    with open('data/our_jsons/results/time_results/time_classify_with_tags.json') as json_file:
        time_data = json.load(json_file)
    print_time_statistics(time_data, 'ml based with tags')
