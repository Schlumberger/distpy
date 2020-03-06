# (c) Schlumberger 20200. See LICENSE

# based on
# https://docs.docker.com/get-started/part2/
# https://hub.docker.com/_/python/
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
# use the official image as a parent image
#FROM python:3
FROM tensorflow/tensorflow:latest-py3
# run the installation of distpy and its dependencies inside your image filesystem
RUN pip install --upgrade pip
RUN pip install distpy
LABEL distpy.image.version="0.0.0"

# Set the working directory
WORKDIR /distpy


# Copy the rest of your app's source code from your host to your image filesystem
COPY . .

# Run the worker
ENTRYPOINT ["python","./distpy-docker.py"]
CMD [ "--help" ]

