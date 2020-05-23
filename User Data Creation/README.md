**Ensure you have changed the paths to the datasets according to your preference**

```sh
$ python User_Data.py
```

# Data Creation
The data format which we need to provide as input to the Reinforcement Learning Model has to contain the histories of users with separate excel sheets for each user. The articles that a particular user has read will have **1** and the rest will have **0** assigned beside the article index for every user's excel sheet.
Also, users who have read less than **K** books are removed from our dataset.
The code **User_Data.py** does the same from **Train Data.csv**.
For the sake of simplicity only 20 articles and 610 users are considered for data preparation. After the code is run, separate csv files are created for every user accroding to his history.

**Train.csv**
| UserID | ArticleID | Action |
|----------|----------|----------|
| 1  | 2  | 1  |
| 1  | 5  | 1  |
| 1  | 6  | 1  |
| 1  | 18  | 1  |
| 2  | 1  | 1  |
| 2  | 3 | 1  |
| 2  | 4  | 1  |
| 2  | 15  | 1  |
| 2  | 17  | 1  |
| 3  | 12  | 1  |

**DummyUser.csv**
| UserID | ArticleID | Action |
|----------|----------|----------|
|  | 1 | 0  |
|  | 2 | 0  |
|  | 3 | 0  |
|  | 4 | 0  |
|  | 5 | 0  |
|  | 6 | 0  |
|  | 7 | 0  |
|  | 8 | 0  |
|  | 9 | 0  |
|  | 10 | 0 |
|  | 11 | 0  |
|  | 12 | 0  |
|  | 13 | 0  |
|  | 14 | 0  |
|  | 15 | 0  |
|  | 16 | 0  |
|  | 17 | 0  |
|  | 18 | 0  |
|  | 19 | 0  |
|  | 20 | 0 |

**Sample Output for UserID-1**
| UserID | ArticleID | Action |
|----------|----------|----------|
| 1 | 1 | 0  |
| 1 | 2 | 1  |
| 1 | 3 | 0  |
| 1 | 4 | 0  |
| 1 | 5 | 1  |
| 1 | 6 | 1  |
| 1 | 7 | 0  |
| 1 | 8 | 0  |
| 1 | 9 | 0  |
| 1 | 10 | 0 |
| 1 | 11 | 0  |
| 1 | 12 | 0  |
| 1 | 13 | 0  |
| 1 | 14 | 0  |
| 1 | 15 | 0  |
| 1 | 16 | 0  |
| 1 | 17 | 0  |
| 1 | 18 | 1  |
| 1 | 19 | 0  |
| 1 | 20 | 0 |

Similarly all the users who have read more than **K** Articles will have separate excel sheets.
We will use these files to train our **DQN / DDQN Model**.
