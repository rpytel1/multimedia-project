import json

import pandas as pd
from scipy import spatial

usr_to_post_dict_train = {}
post_to_usr_test = {}
post_cos_matrix = {}

train_set = {}
test_set = {}
cosine_matrix = {}


def get__usr_to_post_dict(data):
    for key, value in data.items():
        usr_to_post_dict_train[key] = list(value["train_set"].keys())


def get_post_to_usr_dict(data):
    for key, value in data.items():
        for post_key, post_value in value["test_set"].items():
            post_to_usr_test[post_key] = key


def prepare_for_matrix_calculations(data, feature_type):
    for key, value in data.items():
        for post_key, post_value in value["train_set"].items():
            train_set[post_key] = post_value[feature_type]
        for post_key, post_value in value["test_set"].items():
            test_set[post_key] = post_value[feature_type]


def create_empty_cosine_sim_matrix():
    for key_train, value_train in train_set.items():
        cosine_matrix[key_train] = {}


def calculate_cosine_sim_matrix():
    for key_train, value_train in train_set.items():
        for key_test, value_test in test_set.items():
            cosine = calculate_cosine(value_train, value_test)
            cosine_matrix[key_train][key_test] = cosine


def calculate_cosine(elem1, elem2):
    return 1 - spatial.distance.cosine(elem1, elem2)


def get_recommendations(top_k):
    df = pd.DataFrame.from_dict(cosine_matrix)
    df = df.sum(axis=1)
    df = df.nlargest(top_k)
    return df.index.tolist()


if __name__ == '__main__':
    with open('../../data/our_jsons/test.json') as json_file:
        data_json = json.load(json_file)
    get__usr_to_post_dict(data_json)
    get_post_to_usr_dict(data_json)
    prepare_for_matrix_calculations(data_json, "all")
    create_empty_cosine_sim_matrix()
    calculate_cosine_sim_matrix()
    get_recommendations(1)