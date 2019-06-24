import pickle
import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from recommendation_system.metrics.metrics_util import recall_and_precision_at_k, average_precision, reciprocal_rank
import time
import json

metrics = {}


def prepare_testset(all_data, feature_type):
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


def prepare_trainset(user_id, all_data, feature_type):
    frames = []
    for key, value in all_data.items():
        if key == user_id:
            temp_df = value['train_set'].copy()
            temp_df['label'] = 1
        else:
            temp_df = value['train_set'].copy()
            temp_df['label'] = 0
        frames += [temp_df]
    all_frames = pd.concat(frames)

    if feature_type == 'category':
        return all_frames[['Category', 'Concept', 'Subcategory', 'Label']]
    elif feature_type == 'image':
        return all_frames.drop(['Postdate', 'Category', 'Concept', 'Subcategory'], axis=1)
    else:
        return all_frames.drop(['Postdate'], axis=1)


def calculate_metrics(user_id, recs, ground_truth):
    metrics[user_id] = {}

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
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df = metrics_df.T
    return metrics_df


if __name__ == '__main__':
    with open("../../data/our_jsons/final_dataset.pickle", "rb") as input_file:
        complete_data = pickle.load(input_file)

    print('Preparing test set...')
    test_set = prepare_testset(complete_data, "all")

    print('Testing on each user...')
    j = 0
    time_needed = {}
    final_keys = []
    start = time.time()
    for key, value in complete_data.items():
        if value['train_set'].shape[0]:  # in case that the user does not have history
            final_keys += [key]
            print('Recommending on user ' + str(key) + ' with order ' + str(j))
            start_per_user = time.time()
            train_set = prepare_trainset(key, complete_data, 'all')
            y_train = train_set['label'].values
            x_train = train_set.drop(['label'], axis=1).values
            x_train, y_train = EditedNearestNeighbours().fit_resample(x_train, y_train)
            clf = RandomForestClassifier(n_estimators=50, class_weight='balanced')
            clf.fit(x_train, y_train)
            y_proba = clf.predict_proba(test_set.values)
            y_test = pd.DataFrame(data=y_proba, index=test_set.index.tolist(), columns=['non relevant', 'relevant'])
            y_test.sort_values('relevant', ascending=False, inplace=True)
            recommendations = y_test.iloc[0:value['test_set'].shape[0]].index.tolist()
            end_per_user = time.time()
            time_needed[key] = end_per_user - start_per_user
            calculate_metrics(key, recommendations, value['test_set'].index.tolist())
            j += 1

    end = time.time()
    time_needed['total'] = end - start
    with open('../../data/our_jsons/time_classify.json', 'w') as outfile:
        json.dump(time_needed, outfile)

    print('Testing completed -> Let\'s see the metrics')
    metrics_df = overall_metrics()
    print(metrics_df.describe())
    with open('../../data/our_jsons/ml_results.pickle', 'wb') as handle:
        pickle.dump(metrics_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../../data/our_jsons/final_keys.pickle', 'wb') as handle:
        pickle.dump(final_keys, handle, protocol=pickle.HIGHEST_PROTOCOL)
