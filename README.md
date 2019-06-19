# Multimedia project

##TODO:
1. change features to numerical ones(VSD)
2. prepare features in batches to evaluate most helpful features -omit geospatial feats (VSD)
3. script divide train/test set (VSD) 
4. prepare Recommendation System (RPY) -> cosine similarity with history of the user, for each user train set (posts with feats)
5. prepare evaluation metrics (RPY)

## Organization of data for training
```bash
{"user_id1":{"post_id1":{
                        "train_set":{ "category":[],
                                      "tags":[],
                                      "image":[],
                                      "all":[]
                        
                        
                        },
                        "test_set":{same as train_set}
                        
           },
           "post_id2":{...}

    }
 "user_id2"{...}
 ...
}

```
## Obtain data
We used data from [Social Media Prediction Challenge](http://smp-challenge.com/). It should be stored in directory _data/train_all_json/_.

## Cofiguration
In order to install all required dependencies and reformat data for recommendation task run following command.
```bash
./configure.sh
```
