import json
from statistics import mean, median, mode, stdev


def find_user_posts(user_data):
    posts = {}
    for user in user_data.keys():
        posts[user] = len(user_data[user].keys())
    return posts


def remove_users(user_data, binary_users):
    fin_data = {}
    for data, keep in zip(list(user_data.values()), binary_users):
        if keep != 'remove':
            fin_data[keep] = data
    return fin_data


def find_number_of_videos(user_data):
    videos = {}
    for user in user_data.keys():
        videos[user] = sum([val['Mediatype'] == 'video' for val in user_data[user].values()])
    return videos


if __name__ == '__main__':
    with open('data/our_jsons/user_dataset.json') as json_file:
        user_data = json.load(json_file)

    posts_per_user = find_user_posts(user_data)
    print('Mean value of posts per user: ' + str(mean(list(posts_per_user.values()))))
    print('Std value of posts per user: ' + str(stdev(list(posts_per_user.values()))))
    print('Median value of posts per user: ' + str(median(list(posts_per_user.values()))))
    print('Mode value of posts per user: ' + str(mode(list(posts_per_user.values()))))
    print('Number of users with 1 post: ' + str(list(posts_per_user.values()).count(1)))

    at_least_10 = list(map(lambda x: x >= 10, list(posts_per_user.values())))
    posts_at_least_10 = [a * b for a, b in zip(at_least_10, list(posts_per_user.values()))]

    print('Number of users with at least 10 posts: ' + str(sum(at_least_10)))
    print('Number of posts of uses with at least 10 posts: ' + str(sum(posts_at_least_10)))

    at_least_50 = list(map(lambda x: x >= 50, list(posts_per_user.values())))
    posts_at_least_50 = [a * b for a, b in zip(at_least_50, list(posts_per_user.values()))]

    print('Number of users with at least 50 posts: ' + str(sum(at_least_50)))
    print('Number of posts of uses with at least 50 posts: ' + str(sum(posts_at_least_50)))
    print('Total number of users: ' + str(len(posts_per_user.values())))

    user_removed = remove_users(user_data, ['remove' if not a else b
                                            for a, b in zip(at_least_50, list(posts_per_user.keys()))])

    posts_per_user_v2 = find_user_posts(user_removed)
    print('-------------------------- New Statistics --------------------------')
    print('Mean value of posts per user: ' + str(mean(list(posts_per_user_v2.values()))))
    print('Std value of posts per user: ' + str(stdev(list(posts_per_user_v2.values()))))
    print('Median value of posts per user: ' + str(median(list(posts_per_user_v2.values()))))
    print('Max value of posts per user: ' + str(max(list(posts_per_user_v2.values()))))
    print('Min value of posts per user: ' + str(min(list(posts_per_user_v2.values()))))
    print('Total number of users: ' + str(len(posts_per_user_v2.values())))
    print('Total number of posts: ' + str(sum(posts_per_user_v2.values())))

    # with open('data/our_jsons/user_dataset_updated.json', 'w') as outfile:
    #     json.dump(user_removed, outfile)

    videos_per_user = find_number_of_videos(user_data)
    print('Total number of videos: ' + str(sum(videos_per_user.values())))
    # print('Number of videos for user 3175@N73: ' + str(videos_per_user['3175@N73']))



