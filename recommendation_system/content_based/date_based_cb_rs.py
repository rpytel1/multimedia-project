import pickle

from recommendation_system.content_based.content_based_rs import get_post_to_usr_dict, \
    create_empty_cosine_sim_matrix, get_recommendations, usr_to_post_test, cosine_matrix, metrics, calculate_cosine, \
    calculate_metrics, prepare_testset, \
    overall_metrics


def get_min_max_date(train_set):
    min_date = train_set["Postdate"].min()
    max_date = train_set["Postdate"].max()
    length = (max_date - min_date).days
    return train_set["Postdate"].min(), length


def get_weight(min_date, length, date):
    to_min_days = (date - min_date).days
    return 1 - 1 / 2 * (to_min_days / length)


def date_based_cosine_sim_matrix(train_set, test_set):
    to_use = test_set.loc[test_set['Subcategory'].isin(train_set.Subcategory.unique().tolist())]
    print('Clustered test set size: ' + str(to_use.shape[0]))
    min_date, length = get_min_max_date(train_set)
    for key_train in train_set.index.tolist():
        for key_test in to_use.index.tolist():
            cosine = calculate_cosine(train_set.loc[key_train].drop(['Postdate']), to_use.loc[key_test])
            cosine *= get_weight(min_date, length, train_set.loc[key_train]['Postdate'])
            cosine_matrix[key_train][key_test] = cosine


def prepare_datebased_trainset_for_matrix_calculations(user_data, feature_type):
    if feature_type == 'category':
        return user_data['train_set'][['Category', 'Concept', 'Subcategory']]
    elif feature_type == 'image':
        return user_data['train_set'].drop(['Category', 'Concept', 'Subcategory'], axis=1)
    else:
        return user_data['train_set']


if __name__ == '__main__':
    with open("../../data/our_jsons/final_dataset.pickle", "rb") as input_file:
        complete_data = pickle.load(input_file)

    print('Preparing test set...')
    test_set = prepare_testset(complete_data, "all")
    print(test_set.shape[0])

    print('Testing on each user...')
    for key, value in list(complete_data.items())[:1]:
        if value['train_set'].shape[0]:  # in case that the user does not have history
            print('Recommending on user ' + str(key))
            get_post_to_usr_dict(key, value)
            train_set = prepare_datebased_trainset_for_matrix_calculations(value, "all")
            print('User\'s history length: ' + str(train_set.shape[0]))
            create_empty_cosine_sim_matrix(train_set)
            date_based_cosine_sim_matrix(train_set, test_set)
            recommendations = get_recommendations(len(usr_to_post_test[key]))
            calculate_metrics(key, recommendations)
            cosine_matrix.clear()
            print(metrics[key])

    print('Testing completed -> Let\'s see the metrics')
    metrics_df = overall_metrics()
    print(metrics_df.describe())
    with open('../../data/our_jsons/date_init_results.pickle', 'wb') as handle:
        pickle.dump(metrics_df, handle, protocol=pickle.HIGHEST_PROTOCOL)