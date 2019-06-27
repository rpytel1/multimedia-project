def recall_and_precision_at_k(recommendations, k, ground_truth):
    """
    Function to calculate two of the most popular retrieval task metrics: recall and precision.
    :param recommendations: list of recommendations
    :param k: number of positions to take into account for calculating metrics
    :param ground_truth: list of ground truth posts
    :return:
    """
    k = min(len(recommendations), k)
    recommendations_at_k = recommendations[:k]
    relevant_recommendation = [elem for elem in recommendations_at_k if elem in ground_truth]

    return recall(relevant_recommendation, ground_truth), precision(relevant_recommendation, k)


def precision(recommendations, k):
    """
    Metric indicating fraction of relevant post in retrieved list
    :param recommendations: list of relevant recommendations
    :param k: number of retrieved posts
    :return: fraction of relevant post in retrieved list
    """
    return len(recommendations) / k


def recall(recommendations, ground_truth):
    """
    Metric indicating fraction of relevant post compared to all relevant posts available in ground truth
    :param recommendations: list of relevant recommendations
    :param ground_truth: list of ground truth
    :return: fraction of relevant post compared to all relevant posts available in ground truth
    """
    return len(recommendations) / len(ground_truth)


def average_precision(recommendations, ground_truth):
    """
    Method calculating average precision for retrieved list of recommendations. More details on how to calculate
     average precision here (https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision)
    :param recommendations: list of retrieved recommendations
    :param ground_truth: list of ground truth relevant posts
    :return: average precision for provided list of recommendations
    """
    ap = 0
    relevant = [elem in ground_truth for elem in recommendations]
    for k in range(1, len(recommendations) + 1):
        r, p = recall_and_precision_at_k(recommendations, k, ground_truth)
        ap += p*int(relevant[k-1])

    return ap / len(ground_truth)


def reciprocal_rank(recommendations, ground_truth):
    """
    Metric calculating inverse of first relevant item retrieved
    :param recommendations: list of retrieved recommendations
    :param ground_truth: list of ground truth
    :return: inverse of the index of first relevant post
    """
    relevant = [elem in ground_truth for elem in recommendations]
    if True in relevant:
        return 1 / (1 + relevant.index(True))
    return 0
