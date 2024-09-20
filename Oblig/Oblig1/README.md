# Oblig 1

Includes the following:
- segmentation_models.py
- train.py
- dataset for roads images
- images: results for loss for deafult and UNET model
- pdf file for problem
- models which were trained: model0 (default) and unet (unet architecture)

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
Default model implementation
Accuracy for training: 0.9174
Accuracy for val: 0.9097
Loss for training: 0.1908
Loss for val: 0.1997

Unet model implementation:
Image included in image folder: unet loss.png
Accuracy for trainng: 0.9641
Accuracy for validation: 0.9543
Loss for training: 0.07463
Loss for val: 0.09502

Q: • Briefly describe transposed convolution also known as deconvolution. (Hint: It may be easier to consider 1D-case.)
A: Convolution for 1D we can think of sliding a filter, e.g size 3, along the vector and summing them up 3 and 3 which reduces the size. For example a filter applied to a vector with 4 elements we get output of 2 element vector.
In trransposed convolution we want to reverse it. For example in a 2d input we can insert zeros between each row and column then apply the filter again after some additional operations to distrubte the values in our array. Thats my general understanding of the process.

Q: How many trainable parameters do your model have?
A: Used model.summary() which resulted in the following parameters which includes weights and biases for all layers:
    Total params: 485,817
    Trainable params: 485,817
    Non-trainable params: 0

Q: Do you expect your model to behave differently under training and test modes? State the reason for your answer.
A: Im aware of dropout which only applies during training, however theres no dropout here so i'd assume the model behaves the same way during training and testing. However im not 100% sure i interpretd the question correctly.

Q: If your task was to perform segmentation with more than two classes (eg: four classes {ROAD, BUILDINGS, CARS, OTHER}), how would you change the activation function and number of output channels?
A: So for multi class problems one output with sigmoid activation is not enough (as sigmoid is just good for prediciting one class). We need 4 outputs with softmax which determines the class which is most likely. So for line
'''
probs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(c9)
'''
we would replace it with 
'''
probs = layers.Conv2D(4, kernel_size=(1, 1), activation='softmax')(c9)
'''
where each output represents a percentage for each cclass.

Q: Did you notice any improvements over the orignal model?
A: As stated earlier we an increase for accuracy and a significant decrease in loss. Meaning the UNET model performed a lot better to no surprise. It has a lot of more weights to tweak and captures the details of the images, while still maining good generealzation for the road predictions. 

https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406

