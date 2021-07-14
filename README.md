## Crack Detection System

This project uses Python's OpenCV library and deep learning to detect areas of crack and warn the driver about the required speed limit as per the crack percentage.


## Description

* Introductory Page: Page which describes the project
* Login Page: for logging into the website
* Register Page: for registering into the site
* Index Page: this is the main page where we take image input of road ahead
* Output Page: after entering the image on index page, the output that is how much percentage of road is covered with cracks is shown here
* A pre-trained deep learning model using ResNet50 (Convolution Neural Network) is used to predict the cracks in image
* The output shows percentage of crack and warns about the spped limit using this percentage.


## TechStack used

* HTML - Frontend
* CSS - Frontend
* BootStrap - Frontend
* Javascript - Frontend
* Django - Backend
* Python - Deep Learning model
* PostgreSQL - Database


## Steps for Installation and Setup

1. Clone the repository 
    
    `git clone https://github.com/ak2502/crack-detection.git`
 
2. install all dependancies, preferably in a virtual environment.
    
    `sudo apt-get update`
    
    `python -m pip install Django==3.2.3`
    
    `sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'`
    
    `wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -`
    
    `sudo apt-get -y install postgresql`

    
3. Run the application
    
    `python manage.py runserver`
    
4. You can visit the web app at [localhost:(http://127.0.0.1:8000/)](http://127.0.0.1:8000/) in your browser


![image](https://user-images.githubusercontent.com/56317982/125509290-d99f0279-7d6b-4112-84aa-6329f77455c5.png)
![image](https://user-images.githubusercontent.com/56317982/125509409-d2ccb338-d93f-4621-9f31-b47f88088bc3.png)
![image](https://user-images.githubusercontent.com/56317982/125509501-8fcf0452-e043-48db-ad9f-bd1a9b8670db.png)
![image](https://user-images.githubusercontent.com/56317982/125509617-8dcfce87-9d8e-4866-a0f2-6278dd02f4f1.png)
![image](https://user-images.githubusercontent.com/56317982/125509730-d2b03d7a-17cc-4477-8f23-62e2a1fc2b87.png)
![image](https://user-images.githubusercontent.com/56317982/125509766-145782d7-254b-4a01-b684-cc95da9b7840.png)


