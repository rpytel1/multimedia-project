import json

import pandas as pd
from scipy import spatial

from recommendation_system.metrics.metrics_util import recall_and_precision_at_k, average_precision, reciprocal_rank

usr_to_post_test = {}
post_cos_matrix = {}

train_set = {}
test_set = {}
cosine_matrix = {}
metrics = {}


def get_post_to_usr_dict(user_id, user_data):
    usr_to_post_test[user_id] = user_data["test_set"].keys()


def prepare_testset(all_data, feature_type):
    for key, value in all_data.items():
        for post_key, post_value in value["test_set"].items():
            test_set[post_key] = post_value[feature_type]


def prepare_trainset_for_matrix_calculations(user_data, feature_type):
    for post_key, post_value in user_data["train_set"].items():
        train_set[post_key] = post_value[feature_type]


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


def clear_all():
    cosine_matrix.clear()
    train_set.clear()


def calculate_metrics(user_id, recs):
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
    precision5 = 0
    recall5 = 0
    precision10 = 0
    recall10 = 0
    precision50 = 0
    recall50 = 0
    mean_average_precision = 0
    mrr = 0
    for key, value in metrics.items():
        precision5 += value["precision@5"]
        recall5 += value["recall@5"]
        precision10 += value["precision@10"]
        recall10 += value["recall@10"]
        precision50 += value["precision@50"]
        recall50 += value["recall@50"]
        mean_average_precision += value["average_precision"]
        mrr += value["reciprocal_rank"]

    metrics["precision@5"] = precision5 / len(metrics.keys())
    metrics["recall@5"] = recall5 / len(metrics.keys())
    metrics["precision@10"] = precision10 / len(metrics.keys())
    metrics["recall@10"] = recall10 / len(metrics.keys())
    metrics["precision@50"] = precision50 / len(metrics.keys())
    metrics["recall@50"] = recall50 / len(metrics.keys())
    metrics["map"] = mean_average_precision / len(metrics.keys())
    metrics["mrr"] = mrr / len(metrics.keys())


if __name__ == '__main__':
    with open('../../data/our_jsons/test.json') as json_file:
        data_json = json.load(json_file)

    prepare_testset(data_json, "all")

    for key, value in data_json.items():
        get_post_to_usr_dict(key, value)
        prepare_trainset_for_matrix_calculations(value, "all")
        create_empty_cosine_sim_matrix()
        calculate_cosine_sim_matrix()
        recommendations = get_recommendations(1)
        calculate_metrics(key, recommendations)
        clear_all()

    overall_metrics()
    print(metrics)
