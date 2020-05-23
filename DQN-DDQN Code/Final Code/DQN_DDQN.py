# Importing Libraries
import random
import argparse
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import losses
import itertools as it
import pandas as pd
import keras
from keras import backend as K
from collections import Counter
import heapq

parser = argparse.ArgumentParser()

parser.add_argument('--Model_Type', type=int, default=1, help='0 = Initial Model & 1 = Target Model')
parser.add_argument('--State_Size', type=int, default=2, help='number of articles previously read by a user')
parser.add_argument('--batch', type=int, default=30, help='input batch size')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for per user')
parser.add_argument('--LR', type=float, default=0.001, help='learning rate')
parser.add_argument('--deque_length', type=int, default=2000, help='memory to remember')
parser.add_argument('--discount', type=float, default=0.95, help='gamma')
parser.add_argument('--eps', type=float, default=1.0, help='epsilon')
parser.add_argument('--eps_decay', type=float, default=0.995, help='epsilon decay')
parser.add_argument('--eps_min', type=float, default=0.01, help='minimum epsilon')
parser.add_argument('--TFIDF', type=str, default=r'C:\Users\visanand\Downloads\Atul\\', help='path to TFIDF-States')

optim = parser.parse_args()

# The input is the TFIDF of the state space consisting of 2 Articles in one state
States = pd.read_csv(r'C:\Users\visanand\Downloads\Atul\DQN - Final\DQN-DDQN Code\Double DQN\TFIDF-States.csv',header = None)

# Preprocessing the states for passing through the Neural Network
State_Space = []
Temp_List = []
for i in range(0,len(States)):
    Temp_List.append(States.iloc[i:i+1].values.tolist())    
Temp_List.pop(0)
for i in range(len(Temp_List)):
    State_Space.append(Temp_List[i][0])
Encoded = [list(map(float, x)) for x in State_Space]
Encoded_States = []
Encoded_States.append(Encoded)

# The input is the Article Names along with the Genre
file=pd.read_csv(r'C:\Users\visanand\Downloads\Atul\DQN - Final\DQN-DDQN Code\Double DQN\Articles.csv',header = None)

Data = file[1]
Data = Data.values.tolist()

Original_States = []
Original_States.append(list(it.combinations(Data, optim.State_Size)))

# Hyperparameters
batch_size = optim.batch
state_size = len(Encoded[0])
action_size = len(Data)

# Function to create an array of desired length
def CreateArray(n):
    """ 
    Creates a list of size n with numbers from 1->n.  
  
    Parameters: 
    arg1 (int): Size of list
  
    Returns: 
    list: Returns a list. 
  
    """
    A=[]
    for i in range(1,n+1):
        A.append(i)    
    return A

# Function to create a new state
def NEW_STATE(S,action_idx):
    """ 
    Creates a new state from existing state and action.  
  
    Parameters: 
    arg1 (list): Current State
    arg2 (int): Action
    Returns: 
    list: Returns the new state. 
  
    """
    New_State=(S[1],Data[action_idx])
    return New_State

Index = []

# Function to find the Index of a given state
def _STATE_INDEX(S):
    """ 
    Finds out the intex of the current state.  
  
    Parameters: 
    arg1 (list): State
  
    Returns: 
    int: Returns the index. 
  
    """
    Secondary_State=[]
    Secondary_State=list(it.permutations(S, len(S)))  
    for i in range(0,len(Secondary_State)):
        if Secondary_State[i] in Original_States[0]:
            Index.append(Original_States[0].index(Secondary_State[i]))
            break 
    return(Index[-1])  

# Creating the DQN & Double DQN Class
class DQNAgent:
    
    # Function that contains all the hyperparameters
    def __init__(self, state_size, action_size):
        """ 
        Initialize Hyperparameters.  
      
        Parameters: 
        arg1 (int): State Size
        arg2 (int): Action Size
      
        """
        self.state_size = state_size # Total number of states ( Combinations of all the Books )
        self.action_size = action_size # Number of Recommendations
        self.memory = deque(maxlen=optim.deque_length) # A double Q to store the states, actions and rewards
        self.gamma = optim.discount # Discount Factor
        self.epsilon = optim.eps  # Exploration Factor
        self.epsilon_decay = optim.eps_decay # Decay for Exploration Factor
        self.epsilon_min = optim.eps_min 
        self.learning_rate = optim.LR
        self.model = self._build_model()
    
    # Function to build a model
    def _build_model(self):
        """ 
        Constructs a Neural Network Model.  
      
        
        Returns: 
        list: Returns the Model. 
      
        """
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu')) 
        model.add(Dense(256, activation='relu')) 
        model.add(Dense(self.action_size, activation='linear')) 
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
    
    # Fuction to store the initial state, the action ( recommendation in this case ) , reward, next state and the termination condition
    def remember(self, state, action, reward, next_state, done):
        """ 
        Makes a Double Queue to store Variables.  
      
        Parameters: 
        arg1 (list): Current State
        arg1 (int): Action
        arg1 (int): Reward
        arg1 (list): Next State
        arg1 (bool): Done
      
        """
        self.memory.append((state, action, reward, next_state, done))
    
    # Function deciding the Recommendation. Either takes a random value or takes the maximum from a list of Q-Values
    def act(self, state):
        """ 
        Takes action/recommendation based on largest Q-Value.  
      
        Parameters: 
        arg1 (list): Current State
      
        Returns: 
        int: Returns the action with maximum Q-Value. 
      
        """
        if np.random.rand() <= self.epsilon: # Random Exploration
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) # Exploitation
        return np.argmax(act_values[0]) 
    
    # Implementation of Deep Q-Learning
    # Function where actual training occurs and states are passed in batches for training
    def replay(self, batch_size):
        """ 
        Trains the Model.  
      
        Parameters: 
        arg1 (int): Batch Size
      
        """
        minibatch = random.sample(self.memory, batch_size) 
        for state, action, reward, next_state,done  in minibatch: 
            target = reward
            if not done: 
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0])) 
            target_f = self.model.predict(state) 
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # Decrease the value of Exploration Factor
   
    # Implementation of Double Deep Q-Learning
    # After every 'N' iterations, the Target Model is used for predicting the next state 
    def replay2(self,batch_size):
        """ 
        Trains the Target Model.  
      
        Parameters: 
        arg1 (int): Batch Size
      
        """
        minibatch = random.sample(self.memory, batch_size) 
        for state, action, reward, next_state,done  in minibatch: 
            target = reward
            if not done: 
                target = (reward + self.gamma * np.amax(target_model.predict(next_state)[0])) 
            target_f = target_model.predict(state) 
            target_f[0][action] = target
            target_model.fit(state, target_f, epochs=1, verbose=0) 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# This function is used to provide recommendations to the new users after the model has been trained    
def Recommend(model_type,Test_States):
    """ 
    Recommends an article for every possible state.  
  
    Parameters: 
    arg1 (model): Type of Model - Target Model / Initial Model
  
    Returns: 
    list: Returns Recommendation Score for each article for a state. 
  
    """
    Recommendation_Score = (model_type.predict(S1))
    Scores = heapq.nlargest(3, range(len(Recommendation_Score[0])), key=Recommendation_Score[0].__getitem__)
    Scores = [x+1 for x in Scores]
    if Index_Original_States[0][i][0] in Scores:
        Scores.remove(Test_States[0][i][0])        
    if Index_Original_States[0][i][1] in Scores:
        Scores.remove(Test_States[0][i][1])
    return Scores,Recommendation_Score

# Declare the class
agent = DQNAgent(state_size, action_size) 

# Function to save the model and load as Target Model after every 'N' iterations
def copy_model(model):
    """ 
    Saves and Loads the Model as Target Model.  
  
    Parameters: 
    arg1 (model): Initial Model
  
    Returns: 
    model: Returns the Target Model. 
  
    """
    model.save('tmp_model')
    target_model = keras.models.load_model('tmp_model')
    return target_model

# The input contains the articles that are read by several users. This is used for training the Model
Train_Data = pd.read_csv(r'C:\Users\visanand\Downloads\Atul\DQN - Final\DQN-DDQN Code\Double DQN\Train Data.csv')
# Find all the unique userIDs
l2 = []
for i in range(0,len(Train_Data['userId'])):
    l2.append(Train_Data['userId'][i])

freq = Counter(l2)
# l2 is the list containing all the unique user ids of users who have read more than 'K' Articles
l2 = np.unique(l2)
# Iterate over every user
for i in range(len(l2)):
    print("=============================================================================")
    print("For User", l2[i])
    # Iterate to train the model for every state of the current user
    for j in range(len(Train_Data['userId'])):
        if  l2[i] == Train_Data['userId'][j]:
            S = [Data[Train_Data['article_id'][j]-1],Data[Train_Data['article_id'][j+1]-1]]
            print(S)
            x = (l2[i]).__str__()
            user = pd.read_csv(r'C:\Users\visanand\Downloads\Atul\DQN - Final\DQN-DDQN Code\Double DQN\user_' + x + '.csv')
            # Create a list -> Read having all the books the user has ignored/read in a particular state
            read = S
            # Initially the user has not Clicked, hence done = False
            done = False
            flag = 0

            # Create a list -> Read having all the books the user has ignored/read in a particular state
            read = S
            e=0
            while e<optim.niter:
                state_id = _STATE_INDEX(S) # Find the index of the initial state
                S1 = Encoded[state_id] # Find the index of the one-hot encoded state
                S1 = np.reshape(S1, [1, state_size])
                action = agent.act(S1) # Perform a recommendation
                  
                if user['action'][action] == 1:
                    user_action = 1
                else:
                    user_action = 0

                if(Data[action] in read):
                    continue

                # Assign Rewards for each action and change the user state accordingly
                # The state of the user only changes if he has clicked on the recommendation
                
                if user_action == 1:
                    reward = 15
                    print("Epoch" , e)
                    print("User Clicked")
                    print("Article ID", action+1)
                    print("-----------------------------------------------------------------------------")
                    done= True
                    next_state=NEW_STATE(S,action)
                    read = list(next_state)
                elif user_action == 0:
                    reward = -15 
                    read.append(Data[action])
                    next_state=Original_States[0][state_id]
                next_state_index = _STATE_INDEX(next_state)
                Encoded_Next_State = Encoded[next_state_index]
                Encoded_Next_State = np.reshape(Encoded_Next_State, [1, state_size])

                # Store the states, actions and rewards in a double queue
                agent.remember(S1, action, reward, Encoded_Next_State, done)
      
                S = list(next_state)

                # Start training the model once the memory has more actions than the batch-size
                if len(agent.memory) > batch_size:
                    # For every 'N' save the model and load it as Target-Model keeping the Targeted Q-Value constant for 'N' iterations
                    if e%100 == 0:
                        print("Target Model Saved")
                        target_model = copy_model(agent.model)
                        flag = 1
                    if flag == 1:
                        # Train using Target-Model
                        agent.replay2(batch_size) 
                    # Train using Regular Model
                    agent.replay(batch_size)

                # Stop the Training Process if length of read books is equal to the number of books
                if len(read) == len(Data)-2:
                    break
                e=e+1
            break

Test = CreateArray(20)

Index_Original_States = []
Index_Original_States.append(list(it.combinations(Test, optim.State_Size)))
 
if optim.Model_Type == 0:
    model_type = agent.model
    print("Model Used for Predicting - DQN")
else:
    model_type = target_model
    print("Model Used for Predicting - DDQN")


    
# Get the recommendation using Double Deep Q-Learning
for i in range(0,len(Index_Original_States[0])):
    S1 = Encoded[i]
    S1 = np.reshape(S1, [1, state_size])    
    Score_List,Recommendation_Score = Recommend(model_type, Index_Original_States)
    print("Article Recommended ->",Score_List[0],"To User With History" , Index_Original_States[0][i], " With a Score of", np.amax(Recommendation_Score))
