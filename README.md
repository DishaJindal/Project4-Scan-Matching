CUDA Character Recognition
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Disha Jindal: [Linkedin](https://www.linkedin.com/in/disha-jindal/)
* Tested on: Windows 10 Education, Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, NVIDIA Quadro P1000 @ 4GB (Moore 100B Lab)

## Description
This code implements a simple neural network (Multi-layer Perceptron) in CUDA. It is tested on XOR and Character Recognition dataset. Following architechure depicts a neural network with an input layer, hidden layer and output layer. 
<p align="center"><img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Character-Recognition/img/MLP_Architecture.png" width="600"/></p>

### Architecture
**Input Layer** : The number of nodes in the input layer are equal to the number of features in an image which is 10201 in the given dataset.

**Hidden Layer** : I have the analyzed the convergence of model with different number of hidden nodes. It converges pretty fast for any number greater than 10, refer Performance Analysis section for more details.

**Output Layer**: The number of nodes in the output layer are equal to the number of classes which is 52 in the case of character recognition and 2 in case of XOR. 

**Activation Functions**: I tried two activation functions for the hidden layer: Sigmoid and ReLU and the model converges using any of these two.
<p align="center"><img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Character-Recognition/img/Sigmoid_ReLU.png" width="600"/></p>

**Loss Function**: I have used cross entropy loss for this classfication problem setup. 

### Implementation
`main.cpp` is the orchestrator. I have written two different functions one for XOR and the other one for Character Recognition. The required fucntion can be called from the main function. 
`mlp.cu` is the file contains all MLP specific functions with `init`, `train`, `test` and `free` being the ones called from the main script. 
`funcions.cu` contains the required kernels for elementwise operations and others. 

**Training** `train` function in `mlp.cu` orchestrators this. The entire training process runs spoch number of times. The main componants of the training phase are forward pass, backward pass and loss calculation. Forward pass does the matrix multiplication to populte all buffers for hidden and output layers. Backward pass computes the gradients and updates the weight matrices. The loss calculation step uses the true and predicted outputs to calculate the cross entropy loss which is plotted in the following learning curves. The last step is to save the final model weights to a file.

**Testing**  `test` function in `mlp.cu` orchestrators this. This comes into picture in the inference phase which comes after the model is trained. In this phase, I pass the input through the forward pass, calculate the argmax to find the output node with the maximum probability. This gives the output along with the model's confidence in that prediction.

**Hyperparameter Tuning** The two main hyperparameters in this model are the number of nodes in the hidden layer and the learning rate. I have tuned both of these parameters and The Performance Analsysis section below tasks about this in detail.

## Performance Analysis
Learning curves shows the model's learning process by plotting the loss at each epoch in the training phase. Following are the learning curves of various model on different parameters.
### XOR
Following is the learning curve for XOR. 
<p align="center"><img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Character-Recognition/img/XOR_Loss.PNG" width="600"/></p>

### Character Recognition
#### Hidden Nodes
Following are the learning curves of models trained using different number of nodes in the hidden layer. It goes from 5 to 100 hidden nodes. Since, with 100 nodes the loss dropped to 0 in 3 iterations, there is no point increasing the nodes even further. This clearly shows that in learning rate of the model increases on increasing the number of hidden nodes. It takes more than 500 epochs in case of 5 or 10 hidden units. After 20, the loss goes to 0 within 100 epochs.
<p align="center"><img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Character-Recognition/img/LC_HiddenNodes.PNG" width="700"/></p>

#### Learning Rate
Following are the learning curves of models trained using learning rates. The plots values from 0.01 to 1. If you look at the curves corresponding to 0.01, 0.05 and 0.1, it shows that the learning rate of model imporves with the increase in learning rate. This is because we are making bigger updates at each step. Whereas, if we go beyond that as can be seen from the curve corresponding to 0.5 and 1, it starts to learn faster but then it does not even converge. This is because we are making very large changes in the weights at each step which makes the gradients to get stuck at various local minimas (corresponding to each character rather than learning the actual pattern).
<p align="center"><img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Character-Recognition/img/LC_LearningRate.PNG" width="700"/></p>

## Model Weights
There are two weight matrices in this model.

**W1**: Weights between input and hidden layers. The dimesnions are INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE.

**W2**: Weights between hidden and output layers. The dimesnions are HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE.

Following are the links to the weight files for both XOR and Character Recognition models

**XOR**
[W1](https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Character-Recognition/weights/xor_model_w1.txt) , 
[W2](https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Character-Recognition/weights/xor_model_w2.txt)

**Character Recognition**
[W1](https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Character-Recognition/weights/cr_model_w1.txt) , 
[W2](https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Character-Recognition/weights/cr_model_w2.txt)
## Predictions
Following are the outputs for Character Recognition and XOR from the corresponding models. This is implemented in the test function. The inputs are shuffled before testing using shuffle method from `#include<algorithm>` to make sure that the model is not just learning the order of inputs. And then by using the forward pass and argmax logic, the target predictions and model predictions are printed along with the probabilities.
### Character Recognition

<p align="center"><img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Character-Recognition/img/Predictions_CharacterRecognition.PNG" width="600"/></p>

### XOR

<p align="center"><img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Character-Recognition/img/Predictions_XOR.PNG" width="600"/></p>

