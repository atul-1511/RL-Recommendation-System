# Installation
Install the dependencies and devDependencies.

```sh
 pip install --upgrade -r DQN_DDQN_requirements.txt
```
# Recommendation System Pipeline
 ### Step 1
Go to the folder - **User Data Creation**
Run the code
```sh
python User_Data.py
```
Input - **User History Data.csv** & **Dummy User.csv**
Outup - **Train Data.csv** & separate excel sheets for every user. For example - **User608.csv**

 ### Step 2
Go to the folder - **TFIDF-States**
Run the code
```sh
python TFIDF_States.py
```
Input - **Articles.csv**
Outup - **TFIDF-States.csv**

 ### Step 3
Copy **Train Data.csv** , **TFIDF-States.csv** & separate excel sheets for every user. For example - **User608.csv** to the folder - **DQN-DDQN Code/Final Code**
Run the code
```sh
python DQN_DDQN.py
```
Input - **Articles.csv** , **Train Data.csv** , **TFIDF-States.csv** & separate excel sheets for every user. For example - **User608.csv**

