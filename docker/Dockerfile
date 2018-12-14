FROM debian
MAINTAINER Prabhjot Singh Rai <hello@prabhjotrai.com>

RUN apt-get -y update
RUN apt-get -y install libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-2.0.0 libsdl2-dev libglu1-mesa libglu1-mesa-dev libgles2-mesa-dev freeglut3 xvfb libav-tools

WORKDIR /usr

COPY requirements.txt /usr/requirements.txt

RUN apt-get -y install python3-pip

RUN cd /usr

RUN pip3 install -r requirements.txt --no-cache