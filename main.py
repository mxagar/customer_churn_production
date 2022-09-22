"""This module runs the complete customer churn library defined in

`customer_churn/churn_library.py`
 
First, the setup is carried out:
- the configuration file is loaded
- and the existence of necessary folders is checked.

Then, two pipelines are executed one after the other:
(1) model generation/training
(2) and exemplary inference.

If the models have been generated (pipeline 1), we can comment its call out
and simply run the inference (pipeline 2).

Author: Mikel Sagardia
Date: 2022-09-21
"""
import argparse
from customer_churn import churn_library as churn

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run training and exemplary inference"
    )

    parser.add_argument(
        "--produce_images", type=int, help="Whether EDA and evaluation should be carried out and images produced.", default=1
    )

    parser.add_argument(
        "--config_filename", type=str, help="Configuration file", default="config.yaml"
    )

    args = parser.parse_args()

    #config_filename = "config.yaml"
    config_filename = args.config_filename
    churn.run(config_filename, args.produce_images>0)
