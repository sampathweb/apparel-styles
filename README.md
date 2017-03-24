# Image Classification API App

Build IRIS Machine Learning Model using Scikit-Learn and deploy using Tornado Web Framework.

## Setup Environment on Local Machine

### Installation

```
cookiecutter https://github.com/sampathweb/cc-iris-api

cd <repo>  # cd iris-api

# Install Packages
python env/create_env.py
source activate env/venv  # Windows users: activate env/venv
python env/install_packages.py

# Build the Model
python ml_src/build_model.py

# Run the App
python run.py
````

### Test App


1. Open Browser:  [http://localhost:9000](http://localhost:9000)

2. Command Line:

```
curl -i http://localhost:9000/api/iris/predict -X POST -d '{ "sepal_length": 2, "sepal_width": 5, "petal_length": 3, "petal_width": 4}'
```

3. Jupyter Notebook:

Open new terminal navigate to the new folder `iris-api`.  Start `jupyter notebook`. Open ml_src -> `api_client.ipynb`.  Test the API.

Api works!



## Credits:

Template from https://github.com/sampathweb/cc-iris-api

## Dataset:

H. Chen, A. Gallagher, B. Girod, "Describing Clothing by Semantic Attributes", European Conference on Computer Vision (ECCV), 2012.


### The End.