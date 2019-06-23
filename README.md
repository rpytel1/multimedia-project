# Multimedia project
This project transforms Social media prediction challenge to become recommendation tasks. 

As dataset does not provide ratings of each user per post we decided to target this as an 
content based task focusing on item to item predictions.

We distinguish 3 approaches (all in module ```recommendation_system/content_based```):

| File name                 | Recommender System description                                                                                       |
|---------------------------|----------------------------------------------------------------------------------------------------------------------|
| ```content_based_rs.py``` | Similarity based approach calculating cosine similarity between users history posts and potential recommended posts. |
| ```date_based_cb_rs.py``` | Similarity based approach similar to the one above with added weighting by user's historical post date.              |
| ```classify_rs.py```      | Machine learning approach using Random Forest classifier in order to assign recommendation for a user.               |
## Evaluation
In order to evaluate our results we decided to calculate following metrics:

| Metric                 | Explanation                                                              |
|------------------------|--------------------------------------------------------------------------|
| Precision@5            | Precision for first 5 retrieved posts.                                    |
| Precision@10           | Precision for first 10 retrieved posts.                                   |
| Precision@50           | Precision for first 50 retrieved posts.                                   |
| Recall@5               | Recall for first 5 retrieved posts.                                       |
| Recall@10              | Recall for first 10 retrieved posts.                                      |
| Recall@50              | Recall for first 50 retrieved posts.                                      |
| Mean Average Precision | Mean value of average precision. More information [here](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision).                                          |
| Mean Reciprocal Rank   | Mean value of position of first relevant post retrieved over all queries. More precise information [here](https://en.wikipedia.org/wiki/Mean_reciprocal_rank). |

## Obtain data
We used data from [Social Media Prediction Challenge](http://smp-challenge.com/). It should be stored in directory _data/train_all_json/_.

## Cofiguration
In order to install all required dependencies and reformat data for recommendation task run following command.
```bash
./configure.sh
```
## Organization of data for training
```bash
{"user_id1": {"train_set": dataframe with post ids as index and the features as columns,
              "test_set": same as train_set
             },
 "user_id2": {...},
 ...
}

```

