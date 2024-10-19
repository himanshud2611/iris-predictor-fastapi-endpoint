from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from pydantic import BaseModel

app = FastAPI()

# api to create predict iris species
# building a logistic regression model and create an api endpoint for prediction.

# how? after we've build the model and defined the /predict API endpoint, we should be able to make a POST request to API
# with the input features and receive the predicted species as a response.
# basically prediction is the JSON response from the API!!

# function to return a description of the app
def get_app_description():
    return(
        "Welcome to the Iris Species Prediction API!"
    	"This API allows you to predict the species of an iris flower based on its sepal and petal measurements."
    	"Use the '/predict/' endpoint with a POST request to make predictions."
    	"Example usage: POST to '/predict/' with JSON data containing sepal_length, sepal_width, petal_length, and petal_width."
    )

# root endpoint to return the app description, so sending a GET request to root endpoint returns the description
@app.get("/")
async def root():
    return {"message" : get_app_description()}

# Building a Logistic Regression Classifier
# -> defining a predictive function that receives the input features and uses ML model to make a prediction for the species


# laoding the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# training a logistic regression model
model = LogisticRegression(max_iter=400)
model.fit(X, y)

# fundtion to predict the species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    return iris.target_names[prediction[0]]

# pydantic model for input data
class irisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# create an api endpoint
@app.post("/predict")
async def predict_species_api(iris_data: irisData):
    species = predict_species(iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width)
    return {"species": species}

# run -- uvicorn main:app --reload

