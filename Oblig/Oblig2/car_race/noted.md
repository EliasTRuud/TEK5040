

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