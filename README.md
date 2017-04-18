# Apparel Classification App

Predict multiple attributes for a given apparel. This app was developed during Insight AI Fellowship in March'2017.  The app was developed using Python and uses PyTorch deep learning framework and Tordado web server.

[Slides](http://sampathweb.com/apparel-styles/)

## Setup Environment on Local Machine

### Installation

```
git clone https://github.com/sampathweb/apparel-styles

cd <repo>  # cd apparel-styles

# Install Packages
python env/create_env.py
source activate env/venv  # Windows users: activate env/venv
python env/install_packages.py

# Build the Model
python ml_src/build_models.py

# Run the App
python run_server.py
````

### Test App

Open Browser:  [http://localhost:9000](http://localhost:9000)


## Dataset:

H. Chen, A. Gallagher, B. Girod, "Describing Clothing by Semantic Attributes", European Conference on Computer Vision (ECCV), 2012.


### The End.
=======
