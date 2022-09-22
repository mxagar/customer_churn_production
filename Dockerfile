FROM python:3

WORKDIR /usr/src/app

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files
COPY data ./
COPY config.yaml ./
COPY main.py ./
COPY Makefile ./

CMD [ "python", "./main.py" ]