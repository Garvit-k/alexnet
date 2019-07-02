# AlexNet 
### A Convolutional Neural Network created by Alex Krizhevsky

## Notes :
 
  Dataset : 1.2 million high-res images from ImageNet LSVRC-2010 contest
  
  Number of Classes : 1000
  
  Error rates : top1 - 37.5% , top5 - 17.0%


  ## Input :
  uses 256x256 sized RGB image


  ## Architecture :
    60 million parameters
    
    6,50,000 neurons
    
    8 Learned Layers : 
    
    5 Convolutional Layers
    
    3 Fully Connected Layers
    
    1000-way softmax
    
 
  ## 1st Convolutional Layer :
  Kernals - 96 
  
  Size - 11x11x3
  
  Activation function :
  ReLU (Rectified Linear Units) Nonlinearity as it trained faster than tanh units
  
    The activation function is responsible for transforming the summed weighted input from the node into the 
     activation of the node or output for that input
  
