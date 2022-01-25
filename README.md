# Face-Landmarks-Detection
Face landmark detection is a computer vision task where we want to detect and track keypoints from a human face.
Have you ever thought how Snapchat manage to apply amazing filters according to your face? It have been programmed to detect some marks on your face to project a filter according to those marks. In Machine Learning those marks are known as Face Landmarks.

**Now, I will simply start with importing all the libraries we need for this task. I will use PyTorch in this article to face landmarks detection with Deep Learning. Let’s import all the libraries:**


**Download the DLIB Dataset**

The dataset I will choose here to detect Face Landmarks in an official DLIB dataset which consists of over 6666 images of different dimensions. The code below will download the dataset and unzip for further exploration


**Visualize the Dataset**

Now, let’s have a look at what we are working with, to see all the data cleaning and preprocessing opportunities that we need to go through. Here is an example of an image from the dataset we have taken for this task.


![image](https://user-images.githubusercontent.com/39939833/150931863-9553d001-cbed-4f48-a8c5-f51acbfaf42b.png)

You can see that the face covers very less amount of space in the image. If we will use this image in the neural network it will take the background also. So like we prepare a text data we will prepare this image dataset for further exploration.


**Creating Dataset Classes**

Now Let’s dig deeper into the classes and labels in the dataset. The labels_ibug_300W_train.xml consists of the input images and landmarks and bounding box to crop the face. I will store all these values in the list so that we could easily access them during the training process.


**Visualize Train Transforms:**

Now let’s have a quick look at what we have done until now. I will just visualize the dataset by performing the transformation that the above classes will provide to the dataset:

![image](https://user-images.githubusercontent.com/39939833/150932101-26f6cef8-ee64-43f5-8325-6d7f23dfc0e2.png)


**Split the Dataset for Training and Prediction of Face Landmarks**

Now, to move further, I will split the dataset into a train and a valid dataset:
The length of Train set is 6000
The length of Valid set is 666


**Testing the shape of input data:**

torch.Size([64, 1, 224, 224])
torch.Size([64, 68, 2])


**Define the Face Landmarks Detection Model**

Now I will use the ResNet18 as our fundamental framework. I will modify the first and last layers so that the layers will fit easily for our purpose:
**Helper Functions:**


**Training the Neural Network for Face Landmarks Detection**

I will now use the Mean Squared Error between the true and predicted face Landmarks:


**Face Landmarks Prediction**

Now let’s use the model that we trained above on the unseen images in the dataset:
![image](https://user-images.githubusercontent.com/39939833/150932592-8ec484f4-7aa6-467d-b220-b02b3533111f.png)

![image](https://user-images.githubusercontent.com/39939833/150932632-09730e63-6a68-4a0f-9bb4-cdc4c2145613.png)

![image](https://user-images.githubusercontent.com/39939833/150932682-e377b1b6-81e6-4001-bef8-88481c4d04b1.png)

![image](https://user-images.githubusercontent.com/39939833/150932668-d3310309-70a0-42e3-bcc5-c6fd371f0c72.png)

![image](https://user-images.githubusercontent.com/39939833/150932694-b570a343-4bcf-4f9c-804b-dbec7c5cc837.png)

![image](https://user-images.githubusercontent.com/39939833/150932721-e8dbe0d4-d028-487e-adfa-6c06578a0614.png)

![image](https://user-images.githubusercontent.com/39939833/150932737-76e70cde-f6fd-4a44-afbf-be7bfa8704c8.png)

![image](https://user-images.githubusercontent.com/39939833/150932747-93f92d5f-0a70-4170-9f9d-1ab47fed636b.png)
