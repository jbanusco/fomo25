FROM python:3.10-slim

# Maintainer
LABEL maintainer="Jaume Banus <jaume.banus-cobo@chuv.ch>"

# Install the ps package
RUN apt-get update && apt-get install -y procps g++ git && apt-get clean 

# Set work directory
WORKDIR /usr/src/

# Prevent python to write pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent stderr and stdout output
ENV PYTHONUNBUFFERED=1

# Copy project
COPY . /usr/src/

# For all dependencies
RUN pip install -e ".[dev,test]"