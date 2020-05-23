**Ensure you have changed the paths to the datasets according to your preference**

Run this command to check all the arguments you need to provide.
```sh
$ python DQN_DDQN.py -h
```
Even if you don't provide any argument, the default values will be taken automatically.
```sh
$ python DQN_DDQN.py 
```

# Double Deep Q Learning Code Structure
**Input**
In this code, we are considering the previous **2** articles that a user has read to be its history. The input to the Model is the TFIDF Vectors of the states.

**Class DQNAgent**
This class contains the Model Architecture for our Reinforcement Learning problem
 - **def __ inint __()**  
 Initialize the Hyperparameters
 - **def build_model()**  
 Construct the Neural Network Architecture.
The input shape of the neural network will be the size of the TFIDF Vector for one state.
The output shape of the neural network will be equal to the number of recommendations to be made.

 - **def remember()**
 Takes the State,Action,Reward,Next_State,Done as inputs and stores them in a double queue with a fixed maximum length.
 - **def act()**
 Performs recommendations based on Epsilon Greedy Policy.
 - **def replay()**
 A miniBatch is selected at random from the double queue in remember function and training Deep Q Network occurs on that batch. This is for **DEEP Q LEARNING**
- **def replay2()**.
 A miniBatch is selected at random from the double queue in remember function and training of the Target Network & Deep Q Network occurs on that batch. This is for **DOUBLE DEEP Q LEARNING**.

**Training the Model**
For every user the recommendation pipeline is executed for **N** Epochs and the input is the excel sheet for every user that was created earlier in Data Creation. 
Every 100th Epoch, the Target Model is generated to fix the Optimal Q Value.

