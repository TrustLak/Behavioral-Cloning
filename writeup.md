# Behavioral-Cloning
An implementation of "End-to-end learning for self-driving cars" paper by NVIDIA team. This project is a part of Udacity's self-driving car program.
---
## Files
Important files:
* model.py contains the script to create and train the keras model
* helpers.py some functions used to load and preprocess data
* drive.py for driving the car in autonomous mode
* model.h5 contains a trained convolution neural network 
* writeup_report.md a report summarizing the results   
---
## Model architecture
We follow the model described in this [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (see section 5). The model consists of five convolutional layers and three fully connected layers. In addition to these layers, a normalization layer is included at the input side of the model, so that the normalization is accelerated with GPU processing.
The core layers are:  
1. A normalization layer (which is a fixed transformation of the input)
2. A croppying layer (to remove uppper part of the images)
3. A convolutional layer with a 5x5 filter, 2x2 strides, depth = 24
4. A convolutional layer with a 5x5 filter, 2x2 strides, depth = 36
5. A convolutional layer with a 5x5 filter, 2x2 strides, depth = 48
6. A convolutional layer with a 3x3 filter, non-strided, depth = 64
7. A convolutional layer with a 3x3 filter, non-strided, depth = 64
8. A flattening layer that connects the previous layer to 1164 neurons
9. A fully connected layer with output dimension=100x1
10. A fully connected layer with output dimension=50x1
11. A fully connected layer with output dimension=10x1
12. A fully connected layer with output dimension=1x1 (steering angle float)


---
## Generating Data
Geneerating proper data is the core of this project. 

---
## Training the model


