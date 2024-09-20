# Oblig 1

## 3.1 - Write train and val method
Pulled code mostly from lecutre slides.

## 3.2 - Train the model 
Run train.py with one argument, directory where model is stored.

'''
python train.py <dir>
'''

## 3.3 - Observe the results in Tensorboard
'''
tensorboard --logdir=〈train dir〉
'''
The model might prove to have high accuracy for the first epoch as the road often dominats the image. Such that if we predict that most is road, we might get a relvativly accurate prediction. The loss might decrease fast for the few first epochs as the model might preidct some straight lines through the image which propably tracks along most roads well. As model becomes more confident the model might be more accurate as the binary classification problem values might hover around 0.5. 


## 3.4 - Epochs and train steps
'''
train_epochs = 12
train_batch_size = 4
val_batch_size = 2*train_batch_size

indices = np.arange(287)

train_data = get_train_data(indices[:272], train_batch_size)
val_data = get_val_data(indices[272:], val_batch_size)
'''
Code from program.

We have 4 images per batch with seemingly 272 number of images if i read it correctly. Meaning we have 68 steps per epoch (images/batch_size). We run for 12 epochs which we multiply with 68 to get 816 total training steps.


## 3.5 - Metrics

Q: Can you think of any cases where you can get good accuracy result without the model having
learned anything clever?
A: As mentioned earlier we can assume everything is road or bottom 2/3, this might lead us to have 60-70% accuracy, however model has not learning anything of value. 

Q: Can you think of any other metrics that might shed some additional light on the performance
of your model?
A: We might use precesion, recall or F1 which combines them both. Presicion tells us how many of the pixels we predicted as road, is actual road pixels. If we have high presicsions it means that we have few falsee positives.

## 3.6 - Implementing U-net
Image included in image folder: unet loss.png
Loss for trainng: 0.9641
Loss for validation: 0.9543


Convolution for 1D we can think of sliding a filter, e.g size 3, along the vector 


https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406

