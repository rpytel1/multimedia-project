import pandas as pd
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler


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


if __name__ == '__main__':
    with open("../../data/our_jsons/final_dataset.pickle", "rb") as input_file:
        complete_data = pickle.load(input_file)

    test_set = prepare_testset(complete_data, "all")

    mms = MinMaxScaler()
    mms.fit(test_set)
    transformed = mms.transform(test_set)
    Sum_of_squared_distances = []
    K = range(1, 51)
    for k in K:
        print('Trying with ' + str(k))
        km = KMeans(n_clusters=k)
        km = km.fit(transformed)
        Sum_of_squared_distances.append(km.inertia_)

    plt.figure()
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.grid()
    plt.savefig('elbow_training.png', bbox_inches='tight')
