FROM python:3.8

WORKDIR /usr/src/app

# Copy and install requirements
COPY requirements.txt ./
# Shap needs to be installed first, otherwise I get version conflicts
RUN pip install shap==0.40.0
RUN pip install --no-cache-dir -r requirements.txt
# The matplotlib version for Mac (3.3.4) has issues in Docker-Linux
RUN pip install matplotlib==2.2.4

# Copy necessary files
COPY customer_churn ./customer_churn
COPY setup.py ./
COPY data ./data
COPY config.yaml ./
COPY main.py ./
COPY Makefile ./

# Install package (optional)
RUN pip install .

# Producing images with matplotlib in docker
# requires a more sophisticated setup, thus, we switch it off
CMD [ "python", "./main.py", "--produce_images", "0" ]