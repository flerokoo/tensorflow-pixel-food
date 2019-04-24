# Another Machine Learning Learning Project

Same as [Pixel Food Classificator](https://github.com/flerokoo/pixel-food-classificator), but, unlike it, was done with Keras module of Tensorflow.


## Model structure

* Two convolutional layers with 3x3 kernel and 64 filters and ReLU activation function
* Dense layer with 64 neurons and ReLU 
* Dense layer with 3 neurons and softmax activation


## Result

Some plots from Tensorboard:

![Tensorboard](https://raw.githubusercontent.com/flerokoo/tensorflow-pixel-food/master/pics/plots.png)


## Usage

Convert images to input format of shape (N<sub>images</sub>, 32, 32, 1):
```
python prepare_data.py
```

Train model:
```
python train_model.py
```

Test model:
```
python test_model.py
```

Check tensorboard:
```
tensorboard --logdir=logs/
```