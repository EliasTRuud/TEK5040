### Report: Questions answered

#### 3.1 Actor critic:
A: For actor critic network consists of 2 parts, the first part takes action (e.g go left). This is the policy_network (line 274). The goal is to optimize the rewards. The critic evaluates the actions. Which it uses to improve the policy. For this code the critic is the value network (line 284).
An advantage of the critic is e.g improved sample efficiency, meaning less interaction and training time for our policy to become good. Also faster convergence since we can update both policy and value network concurrently.  (https://www.geeksforgeeks.org/actor-critic-algorithm-in-reinforcement-learning/)

#### 3.2 Observation and action sampling
A: Not 100% sure on these answers.
- the action probabilities are calculated:
    - logits = policy_network.policy(observation) (lin 37)
- an action is sampled based on the probabilities above
    - action = tf.random.categorical(logits, 1)[0][0] (line 39)
- new state (same as an observation in this task) is generated based on the action
    - observation, r, ter, trunc, info = env.step(action) (line 50, we take action in env which returns new observation)

#### 3.3 Linear vs non-linear policy
A: Where the tasks are very basic a linear policy might work sufficiently. Like driving from A to B in a straight line. However for carracing we have turning angle (although not used for this example), speed and a track that is constantly changing as we drive along. So i'd imagine linear policy will perform badly in this example with the carracing.

#### 3.4 Value function example:
A: So for a state where we value how close to goal we are and how much time is remaining, a high value state is where are close to the finish line with a lot of time left. A state with low value would be for example if the car used basically close to all the time available and is driving in circles close to the starting line or is stuck driving into a wall.

#### 3.5 Policy and value network architectures
A: Architecture of the networks, layers and dimensions.

Both networks uses the feature extractor (interpret the image/pixels) which consists of two Conv2D layers.Sizes 16 and 32. Then a dense fully connected layer.

Policy network:
Uses the feature extractor.
Then it has a dense layer, which returns the logits. This layer output has dimension:[batch_size, num_actions].

Value network:
Also uses the feature extractor.
Then it has a dense layer set equal to "hidden units". Then a final dense layer with 1 output, being the value. So output is equal [bath_size, 1].

#### 3.6 Visualization
A: I ran program once which took a long time and it looks like it saved to high_score_model, but after implmentation i get error: "Value in checkpoint could not be found in the restored object" when running ppo.py. So i just tried to visualize the weights for ppo_linear.

The results is the image: https://ibb.co/27w5X03
Its quite hard to interpret this image, however it looks more dense in the center meaning its preference to go straight which makes sense. Also quite low density on braking.



#### 3.7 Eval policy
A: Tried to run the linear basic model with different n values for action repeat paramter.

n = 1:
min, max : (-10.5956, 6.93603)
median, mean : (-6.18105, -5.09069)
Poor results, and see visualized its barely moving. Only a few get positive scores.

n=2:
min, max : (-12.8826, 10.9598)
median, mean : (-4.32602, -2.76928)
Bit better, seemingly as just by driving forward a bit more means better results as we reached further.

n=4
min, max : (-10.5063, 30.5051)
median, mean : (4.72545, 5.4124)

n=8
min, max : (-14.0299, 54.7331)
median, mean : (3.94603, 6.59386)

It seems just like we keep reaching further since we repeat each action more times. At least it stays on the track, however it shows for quite short time. I saw this when i tried seeing longer time:
python eval_policy.py --num_episodes 16 --num_steps 999 --policy train_out\ppo_linear\high_score_model --action_repeat 4. The car seemed ok at the first straight stretch, but when it reaches the turn it loses its way and drives off.


I figured out too late to change paramter name of folder to save to. As the model takes too long to train, ill reupload second version with results assuming it works.
----------------------------------------------------------------

### Notes:

When debugging reduce paramters: 
num_episodes = 2
maxlen_environment = 12


2.2 Implementation hints

When compared to the pseudo code given in Algorithm 1, the implementation
task corresponds to lines 9, 10 and 11. That is you just need to compose the
loss function and optimize it using the usual tf.tape.gradient. The core of
implementation task is therefore to form the loss function.
Lines 5-8 in the pseudo code creates the data samples which you can use in
calculation of the loss. This part (creation of data samples) is implemented in
lines 328-335 in ppo.py. Each data sample consists of the following components:
    •observation ot. This is also the same as state in this task
    •action at
    •advantage ˆdt. This is evaluated as the difference between the return
    gt and the value function vt (i.e. ˆdt = gt −vt). The value function is
    predicted using the value network which takes (ot,T −t) as input, i.e.
    vt = vη(ot,T −t) where vη is the value network.
    •The old policy πold. More specifically it gives the probability of action at
    output by the previous(old) policy network.
    •value_target yt. This is just the return gt calculated using the old policy
    •time step t


Lines:
    9: Set surrogate objective L based on the sampled data.
    10: Optimize surrogate L wrt. η and θ, for K epochs and
    11: minibatch size M ≤∑ τ(i) - from i= 1 -> N



Task 1: Implement calculate_returens()
    - Line 95
    - 

Task 2: Implement the value_loss()

Task 3: Implement policy_loss()

Task 4: Optimize surrogate loss
    - Line 347

