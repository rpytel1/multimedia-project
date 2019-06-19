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
(outer key- UID=> dict of posts=>a)keys of feats: "category", "tags", "image" and "all")

## Cofiguration
In order to install all required dependencies reformat data for recommendation task run following command.
```bash
./configure.sh
```
