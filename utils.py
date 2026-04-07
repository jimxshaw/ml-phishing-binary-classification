# This file has shared helper functions used across
# all notebooks.

import os
import yaml
import joblib
import matplotlib.pyplot as plt


# Configurations

# We load the config.yml file and return it as a
# python library. All notebooks should call this at
# the top so settings stay consistent. An error is raised
# if the config file is missing. 
def load_config(config_path = "config.yml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not at '{config_path}'. "
            "Make sure to run notebooks from the project root."
        )
    
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    print(f"Config loaded from '{config_path}'")

    return config


# File System Helpers

# We create any output directories from config.yml that don't exist yet.
# This is called once at the start of notebook 01 so all following
# notebooks can safely write to these folders without an error.
def make_sure_directories_exist(config):
    directories_to_create = [
        config["paths"]["figures"],

        config["paths"]["saved_models"],

        os.path.dirname(config["paths"]["processed_data"]),
    ]
 
    for directory in directories_to_create:
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

            print(f"Created directory: '{directory}'")
        else:
            print(f"Directory already exists: '{directory}'")


# Plotting Helpers

# We save our current matplotlib figure to the ./reports/figures/ directory.
# Always call plt.show() after calling this if we want to display the figure
# inline in the notebook, too.
# The dpi argument is the resolution of the saved image and it defaults 
# # to 150 because it's a balance of quality and file size.
def save_figure(figure_name, config, dpi = 150):
    # Make sure the filename ends in .png.
    if not figure_name.endswith(".png"):
        figure_name += ".png"
 
    output_path = os.path.join(config["paths"]["figures"], figure_name)

    plt.savefig(output_path, dpi = dpi, bbox_inches = "tight")

    print(f"Figure saved to '{output_path}'")


# Model Persistence

# We use joblib to serialize a trained model to disk.
# Saved models can be loaded in notebook 04 for evaluation without 
# having to re-train. The file is saved to the path in config.yml.
def save_model(model, model_name, config):
    model_directory = config["paths"]["saved_models"]

    output_path = os.path.join(model_directory, f"{model_name}.pkl")
 
    joblib.dump(model, output_path)
    
    print(f"Model '{model_name}' saved to '{output_path}'")

# We load a previously saved model from disk. This returns a 
# model object ready for prediction. 
def load_model(model_name, config):
    model_path = os.path.join(config["paths"]["saved_models"], f"{model_name}.pkl")
 
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found at '{model_path}'. "
            "Make sure notebook 03 has been run first."
        )
 
    model = joblib.load(model_path)

    print(f"Model '{model_name}' loaded from '{model_path}'")

    return model
 
# We serialize the fitted TF-IDF vectorizer to disk.
# The vectorizer must be fitted on training data only in notebook 02.
# It's saved and re-loaded in later notebooks to transform new data.
# This prevents data leakage from fitting on the full dataset. 
def save_vectorizer(vectorizer, config):
    output_path = config["paths"]["tfidf_vectorizer"]

    joblib.dump(vectorizer, output_path)

    print(f"TF-IDF vectorizer saved to '{output_path}'")
 
# We load the previously saved TF-IDF vectorizer from disk.
# It returns a fitted TF-IDF vectorizer object that's
# ready to transform text.
def load_vectorizer(config):
    vectorizer_path = config["paths"]["tfidf_vectorizer"]
 
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"No saved vectorizer found at '{vectorizer_path}'. "
            "Make sure notebook 02 has been run first."
        )
 
    vectorizer = joblib.load(vectorizer_path)

    print(f"TF-IDF vectorizer loaded from '{vectorizer_path}'")

    return vectorizer
