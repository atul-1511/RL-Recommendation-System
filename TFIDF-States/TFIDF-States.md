**Ensure you have changed the paths to the datasets according to your preference**

# Extracting Article Features using TFIDF
Here the main goal is to construct state space of users having read **2** articles as their history and make a TFIDF Vector for each of the states to be input to the Deep-Q Network.

```sh
$ python TFIDF_States.py
```

**Dataset**
| Article Name | Genre |
| ------ | ------ |
| Cheapest Ways to Do Festival Cleaning | Bathroom Cleaning |
| Most Efficient Ways for a Quick Clean-up! | Bathroom Cleaning |
| How to wash a suit | Clothing care |
| Most Efficient Ways for a Quick Clean-up! | floor and surface cleaning |
| How To Make Compost At Home | In the home |

**Combine Article Name and Genre**
| Combined Name | 
| ------ | 
| Cheapest Ways to Do Festival Cleaning Bathroom Cleaning |
| Most Efficient Ways for a Quick Clean-up! Bathroom Cleaning |
| How to wash a suit Clothing care |
| Most Efficient Ways for a Quick Clean-up! floor and surface cleaning |
| How To Make Compost At Home In the home |

**State Space**
After **NC2** combinations, our states look like
| State Space |
| ------ | 
| Cheapest Ways to Do Festival Cleaning Bathroom Cleaning  Most Efficient Ways for a QuickClean-up! Bathroom Cleaning |
| Most Efficient Ways for a Quick Clean-up! Bathroom Cleaning How to wash a suit Clothing care |
| How to wash a suit Clothing care Most Efficient Ways for a Quick Clean-up! floor and surface cleaning |
| Most Efficient Ways for a Quick Clean-up! floor and surface cleaning How To Make Compost At Home In the home |

**Apply TFIDF Vectorization** on the **State Space** to creaet the input for Deep-Q Network.

