**Ensure you have changed the paths to the datasets according to your preference**

# Deep Q Network
Neural network approach turns out to be a yet better way to estimate Q-Values. The main objective stays the same. It is the minimization of the distance between Q-Valuea and TD-Target. 
![image](https://www.novatec-gmbh.de/wp-content/uploads/reinforcement_learning_loop-650x294.png)
**For Further Reading**
https://www.novatec-gmbh.de/en/blog/deep-q-networks/
# Problem with Deep Q Network
Deep Q-learning is known to sometimes learn unrealistically high action values because it includes a maximization step over estimated action values, which tends to prefer overestimated to underestimated values

# Double Deep Q Network
The idea of Double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation.
The action selection and action evaluation are coupled. We are using the Target-Network to select the action and at the same time to estimate the quality of the action.
The Target-Network calculates Optimal Q-Value for each possible action in state . The greedy policy decides upon the highest values Q-Value which action. This means that the Target-Network selects the action and at the same time evaluates its quality by calculating Optimal Q-Value. Double Q-Learning tries to decouple this both procedures from each other.

![image](https://cdn-images-1.medium.com/max/880/1*Vd1kcpLoQDnM5vrKnvzxbw.png)

**For Further Reading**
https://towardsdatascience.com/deep-double-q-learning-7fca410b193a
