# AlexNet 
### A Convolutional Neural Network created by Alex Krizhevsky


![Architecture Diagram](https://raw.githubusercontent.com/Garvit-k/alexnet/master/architechture.png)

## Notes :
 
  Dataset : 1.2 million high-res images from ImageNet LSVRC-2010 contest
  
  Number of Classes : 1000
  
  Error rates : top1 - 37.5% , top5 - 17.0%
  
  Trained on 2 GTX 580 GPU
  
 ### References :  

 - https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf
 - https://www.learnopencv.com/understanding-alexnet/
 - https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/

  ## Input :
  256x256 sized RGB image


  ## Architecture :
    60 million parameters
    
    6,50,000 neurons
    
    8 Learned Layers : 
    
    5 Convolutional Layers
    
    3 Fully Connected Layers
    
    1000-way softmax
  
  ## Activation function :
    ReLU (Rectified Linear Units) Non-linear as it trained faster than tanh units
  
    The activation function is responsible for transforming
    the summed weighted input from the node into the 
    activation of the node or output for that input
  ```
    ReLu Formula : F(x) = max(0.0,x)    
  ```
 
## Convolutional Layers :  

     Convolution is the first layer to extract features from an input image.
     Convolution preserves the relationship between pixels by learning image features using small squares of input data.
     It is a mathematical operation that takes two inputs such as image matrix and a filter or kernel

    ReLU activation function used for every conv layers
  
  
 ### 1st Convolutional Layer :
  Kernels - 96 
  
  Size - 11x11x3
  
  Strides - 4
  
  Multiple Convolution kernels acts like a filter and extract features

  Followed by Response Normalization then Max Pooling

 ### 2nd Convolutional Layer :
  Kernals - 256
  
  Size - 5x5x48
  
  Stride - 1
  
  Response Normalization and Max Pooling
  
 ### 3rd Convolutional Layer :
  Kernels - 384
  
  Size - 3x3x256
  
  Stride - 1
  
 ### 4th Convolutional Layer :
 Kernels - 384
  
 Size - 3x3x192

 ### 5th Convolutional Layer :
 Kernels - 256
 
 Size - 3x3x192


## Max Pooling Layer :
 It is done to downsample the output which helps in reducing overfitting.
 
 This basically takes a filter and a stride of the same length. It then applies it to the 
 input volume and outputs the maximum number in every subregion that the filter convolves around.


## Fully  Connected Layer :
 Number of Neurons : 4096

 2 Layers in series after convolutional layers both followed by dropout layer
 then last fully connected layer is outputed to 1000 way softmax 
 
 This layer looks at the output of previous layer and 
 then determines which features correlates to particular class
 
## Dropout :
 This layer "drops out" a random set of activations in that layer by setting them to zero.
 This helps in alleviating overfitting problem.

 Dorpout rate 0.5

## 1000 way softmax :
 This is a tuple of 1000 elements each representing a particular class

 This is used to get the probablity of correctness of a particular class

 We select the highest probablity to determine the prediction of the neural net

