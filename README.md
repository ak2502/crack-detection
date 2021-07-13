## Crack Detection System

This project uses Python's OpenCV library to detect areas of crack and warn the driver about the required speed limit as per the crack percentage.


## Description

* Introductory Page: Page which describes the project
* Login Page: for logging into the website
* Register Page: for registering into the site
* Index Page: this is the main page where we take image input of road ahead
* Output Page: after entering the image on index page, the output that is how much percentage of road is covered with cracks is shown here
* A pre-trained deep learning model using ResNet50 (Convolution Neural Network) is used to predict the cracks in image
* The output shows percentage of crack and warns about the spped limit using this percentage.


## TechStack

* HTML - Frontend
* CSS - Frontend
* BootStrap - Frontend
* Javascript - Frontend
* Django - Backend
* Python - Deep Learning model
* PostgresSQL - Database


## Steps for Installation and Setup

1. Clone the repository 
    `git clone https://github.com/WCoder007/FriendsCog.git`
 
2. install all dependancies, preferably in a virtual environment.
    `sudo apt-get update`
    `python -m pip install Django==3.2.3`
    `sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'`
    `wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -`
    `sudo apt-get -y install postgresql`

    
3. Run the application
    `python manage.py runserver`
    
4. You can visit the web app at [localhost:(http://127.0.0.1:8000/)](http://127.0.0.1:8000/) in your browser



