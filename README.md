# Food-Images-recognition
Keeping manual track of food consumptions in nursing home can be very time consuming. Nursing staff may use their time more productively by attending the elderly people than wasting their time for manual record of food. For this reason, an automated system for food analysis is to be developed for better knowledge and understanding of the nutritional records associated to the meals consumed by the patient. Food recognition may be considered as a special case of object recognition, being completely energetic difficulty count in computer vision lately. In this project, I proposed an approach for automated food recognition that will classify the food images into different food categories. I developed a multi-layered convolutional neural network (CNN) system that takes different food images as input and takes advantages of the features from images to recognize the food item. I implemented a 14-layers CNN model. Where first two layers are 2D convolutional layers with 32 kernels. 3rd layer is max pooling layer, 4th layer is Dropout layer. Above all layers are repeated one time where the number of kernels has been increased from 32 to 64 with stride 1. The above layers have been repeated one last time where the number of kernels has been increased from 64 to 128 with kernel size of 2x2 to get more filtered images. Next two layers are 2D convolutional layers. Last layer before fully connected layers is global average pooling layer.
Network is trained and fine-tuned using pre-processed images. The evaluation of this approach is done on public dataset Food 101 with 15 food categories. Training accuracy achieved for Food-101 with 100 epochs are 70.9% while testing accuracies achieved are 57.8%.

# Methodology
The traditional models in machine learning are weaker in practice when it comes to image classification. Convolutional Neural Networks (CNNs) show higher accuracy for food detection and recognition in this regard. In this project, I propose a new approach based on deep learning techniques. Specifically, I propose a Convolutional Neural Network (CNN) model with several major optimizations, for example, an optimized model and optimized convolution techniques. CNNs are one of the common and effective neural networks to do image processing. CNN is a neural network composed of convolutional layers, pooling layers, output layer and optionally fully connected layers. They learn patterns, take images as input and classify them according to patterns. First, they take images as input data and then categorize them. Training algorithm of convolutional neural network is divided into two stages: forward propagation and backward propagation. In forward propagation, input data propagates throughout the complete network and reaches to output layer giving corresponding output based on calculations done in hidden layers. Difference between computed output and original labels is then backpropagated through the network where that difference is differentiated with respect to weights of layers and those weights are updated accordingly. 
![image](https://user-images.githubusercontent.com/105145104/169947721-29916300-a6f8-40ce-a857-0963dc04aee3.png)

# Model Architecture 
I implemented a 14-layers CNN model. The first layer is the Convolutional 2D layer which consists of 32 kernels of size 5x5 with stride 2 and same padding taking an input of size 224x224x3 where 224x224 is the rescaled size of our images and 3 denotes the color aspect (RGB) of the image. The next layer is again Convolution 2D layer with kernel size 5x5 ¬¬with stride 2 and same padding. The next layer is the max pooling layer with a pool size of 2x2. The next layer is dropout layer with dropping probability 0.2. The above four layers are again repeated but with 3x3 kernel size where the kernels have been increased from 32 to 64 and stride 1 to get better filtered convolved images and better feature extraction by the max pooling layer. The above layers have been repeated one last time where the kernels have been increased from 64 to 128 and kernel size of 2x2 to get more filtered images. Next two layers are Convolution 2D layers with 256 kernels of size 2x2 and same padding and stride 1. Last layer before fully connected layers is global average pooling layer. Two fully connected layers are used next with 512 and 15 neurons respectively and Dropouts have been added of 0.2 in between the dense layers to prevent overfitting by making the weights of some random neurons to zero so as to prevent overfitting on some particular neurons. 

![image](https://user-images.githubusercontent.com/105145104/169947875-a9d7fdfb-ad36-4a51-91a3-6601ff83dad1.png)

All the convolutional 2D layers and the fully connected layers except output layer have an activation function of ReLu (Rectified Linear Unit). The last layer is the output layer consisting of 15 neurons equivalent to the number of categories with softmax activation function and each neuron has an output of a probability corresponding to that particular neuron. The CNN predicts the category to be the one with the highest probability.
Model summery of Food-101 dataset after applying all packages and formulas:

![image](https://user-images.githubusercontent.com/105145104/169947937-7709ff07-740d-4ed5-b927-985a5ff8848f.png)

# Datasets Used  
The dataset used is Food 101 dataset. Food 101 dataset contains a number of different subsets of the full food-101 data. For this reason, the data includes massively downscaled versions of the images to enable quick tests. The data has been reformatted as HDF5 and specifically Keras HDF5 Matrix which allows them to be easily read in. The file names indicate the contents of the file. There are 101 categories represented, with 1000 images, and most have a resolution of around 512x512x3 (RGB, uint8).
I have used 15 categories of Food-101 in my project. They are Apple Pie, Club Sandwich, Grilled Cheese Sandwich, Tacos, Hamburger, Samosa, French Fries, Pizza, Ravioli, Cupcakes, Spring Rolls, donuts, waffles, sushi, and nachos. All images are downscaled to 224x224x3. The reason of choosing Food-101 is that it contains 101 categories with 1000 images of each category, and all are of same resolution. So, there are almost 1 lac pics which we can use for training and testing. I’ve used 15 categories which roughly consist of 15000 images and for each category I’ve used 750 for training and rest 250 for testing. I picked these categories because of their often presence in our daily diet. There is no other logic of choosing only those, there can be some other categories too.

Food101 dataset sample: 

![image](https://user-images.githubusercontent.com/105145104/169948382-b4ecfb25-28ab-4660-a5c7-08c2e30fa2d8.png)

# Results
Training accuracy and test accuracy achieved for Food-101 after training for 100 epochs are 70.9% and 57.8% respectively. While training loss and validation loss are 0.911 and 1.417 respectively. Accuracy and loss plots over epochs are shown in Fig.

![image](https://user-images.githubusercontent.com/105145104/169948497-2e98c359-95a0-45c8-9bf3-0e83c84128c1.png)

Confusion matrix and Receiver Operating Characteristics (ROC) curve for predictions over test set are also shown in Fig. 

![image](https://user-images.githubusercontent.com/105145104/169948560-f54c4d3d-88b1-4d72-88ac-d6de2bd05101.png)

Receiving Operating Characteristics (ROC) curve for Food-101:

![image](https://user-images.githubusercontent.com/105145104/169948664-cc236bd0-63f6-4fa9-81f6-327d3e357b56.png)

Sample predictions over test dataset for Food-101:

![image](https://user-images.githubusercontent.com/105145104/169948739-4d5f5b9a-081f-45c7-b3ac-48cc494b1f23.png)


