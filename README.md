Plant Disease Detection Using CNN Model:
A combination of technological advances has transformed plant disease diagnosis, especially in the areas of artificial intelligence (AI) and machine learning but the advent of technology has dramatically changed the way diseases are diagnosed.
This project aims to improve precision agriculture by harnessing the disease detection capabilities of CNNs (Convolutional Neural Networks). Using images of plants with multiple diseases, the neurons can accurately distinguish between healthy and infected plants



To create an  model for tomatoes using Convolutional Neural Network (CNN), follow these steps:

1. Download the Dataset:
   - Visit Kaggle and download the tomato dataset from [this link](https://www.kaggle.com/datasets/ashishmotwani/tomato).

2. Organize Dataset:
   - After downloading, create two folders: one for training the model (train dataset) and another for validating/testing the model (validation dataset).

3. Model Creation (MODEL file):
   - Utilize a Python script to develop a CNN model for image classification. You can use popular deep learning frameworks like TensorFlow  for this purpose. Ensure your model includes layers for convolution, pooling, flattening, and dense layers. Train the model using the training dataset.

4.  Create Website:
   - Develop HTML files for your website. You need at least two files - `base.html` and `index.html`. `base.html` can contain the common structure (header, footer, etc.), while `index.html` will be the main page where users can interact with the model.

5. Integrate with Flask:
   - Utilize Flask, a web framework for Python, to integrate your model with the website. In the Flask app, you'll define routes for different pages (e.g., home page, result page) and incorporate the model to classify tomato images.

6. Website Deployment:
   - Once the website is developed, deploy it on a web server. You can use platforms like Heroku or AWS for hosting Flask applications.

By following these steps, users can visit your website, upload tomato images, and receive predictions based on the CNN model you've trained. Make sure to handle errors gracefully and provide a user-friendly interface for a seamless experience.
