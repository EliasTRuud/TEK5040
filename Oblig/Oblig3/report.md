### Task

#### 1: 
Ran the mc file 5 times and saved plots. 
The training data is from -0.5 to 0.5, while the test data is from -1.5 to 1.5. So the variance is gonna be a lot worse outside the range of the training data (less confidence). For the trainig data there is very low variance and reflects around what the noise is (sigma = 0.1) and the line follows the sinus curve qutie well. Notice that it increases a lot outside. We dont have data points here so the uncertainty naturally increases. From plot to plot it seems like it might fluctate a lot outside the range, which i assume is the effect of dropout being random. 

#### 2:
Quite similarily to the MC plots they do quite well for the training data range. However outside the variance increases quite consistently, like a cone. This reflects that we are even further away from the training data, meaning we are even less likely to be able to predict correctly. The curve of variance is also quite smooth in its shape. Variational seems better as its more consistent in its Bayesian approach as it provides stable uncertainty estimates. However it might be more computationally expensive than the monte carlo method which is more stochastic. 

#### 3:
The trick is found in the call method at line 59 and 62 of the densevariational file. From what i understand we use the trick is used to allow differentiability for the weights and biases. We take a sample from a standard normal distribution and transform this sample with the mean and standard deviation. (sigma and mu)

#### 4:
The first and third compononents in equation 3 is referring to the variational distribution q(w) and prior distribution p(w), which is connected to the kernel(weights) and bias. If i understand correctly i think these are added in line 64:
self.add_loss(self.var_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.var_loss(bias, self.bias_mu, bias_sigma))
where we add it the loss function as described in the equation. The actual calculation happens/is defined in the two functions below: var_loss (term1) and log_prior_prob(term3).


#### 5:
Partly assuming its to do with the neg_log_likelihood in the common.py file at line 30. Which is then sent into the model in vi_train_test.py at line 40/42 when we compile the model. Referring to the posterior network parameter distribution p(w|D). (But unsure if this line is stricly term 2 or not)

#### 6:
An advantage of MCMC as stated in lecture is that its asymptotically exact . However it is computationally expensive compared to variataional inference. 


