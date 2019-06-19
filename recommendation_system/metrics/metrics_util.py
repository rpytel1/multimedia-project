def recall_and_precision_at_k(recommendations, k, ground_truth):
    k = min(len(recommendations), k)
    recommendations_at_k = recommendations[:k]
    relevant_recommendation = [elem for elem in recommendations_at_k if elem in ground_truth]

    return recall(relevant_recommendation, ground_truth), precision(relevant_recommendation, k)


def precision(recommendations, k):
    return len(recommendations) / k


def recall(recommendations, ground_truth):
    return len(recommendations) / len(ground_truth)


def average_precision(recommendations, ground_truth):
    ap = 0
    for k in range(1, len(recommendations) + 1):
        r, p = recall_and_precision_at_k(recommendations, k, ground_truth)
        ap += r * p

    return ap / len(ground_truth)


def reciprocal_rank(recommendations, ground_truth):
    relevant = [elem in ground_truth for elem in recommendations]
    if True in relevant:
        return 1 / (1 + relevant.index(True))
    return 0
