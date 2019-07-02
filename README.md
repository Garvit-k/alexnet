# AlexNet 
### A Convolutional Neural Network created by Alex Krizhevsky

## Notes :
 
  Dataset : 1.2 million high-res images from ImageNet LSVRC-2010 contest
  
  Number of Classes : 1000
  
  Error rates : top1 - 37.5% , top5 - 17.0%
  
  Trained on 2 GTX 580 GPU
  
  Refrences :
   https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf
   https://www.learnopencv.com/understanding-alexnet/

  ## Input :
  uses 256x256 sized RGB image


  ## Architecture :
    60 million parameters
    
    6,50,000 neurons
    
    8 Learned Layers : 
    
    5 Convolutional Layers
    
    3 Fully Connected Layers
    
    1000-way softmax
  
  ## Activation function :
    ReLU (Rectified Linear Units) Nonlinearity as it trained faster than tanh units
  
      The activation function is responsible for transforming the summed weighted input from the node into the 
       activation of the node or output for that input
  
    ReLu Formula : F(x) = max(0.0,x)    


  ## 1st Convolutional Layer :
  Kernals - 96 
  
  Size - 11x11x3
  
  Stride - 4
  
  Multiple Convolution kernals acts like a filter and extract features

  Followed by Response Normalization then Max Pooling

  ## 2nd Convolutional Layer :
  Kernals = 256
  
  Size - 5x5x48
  
  Stride - 1
  
  
