FROM ubuntu:latest
MAINTAINER stephanegaiffas "stephane.gaiffas@lpsm.paris"
WORKDIR /AMF
VOLUME /AMF

# Next two avoids questions during install
ENV TERM xterm-256color
ENV DEBIAN_FRONTEND noninteractive

# apt-get install all the useful stuff
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
	apt-utils \
	build-essential \
	cmake \
	curl \
	git \
	swig \
	patchelf \
	unzip \
	libssl-dev \
	zlib1g-dev \
	libbz2-dev \
	libreadline-dev \
	libsqlite3-dev \
	curl \
	llvm \
	libncurses5-dev \
	libncursesw5-dev \
	xz-utils \
	tk-dev \
	python3.6 \
	python3-pip

# Installing googletest
RUN git clone https://github.com/google/googletest.git && \
	(cd googletest && mkdir -p build && cd build && cmake .. && make && make install) && \
	rm -rf googletest

COPY requirements.txt /AMF
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Installing the scikit-garden library for the MondrianForest algorithm
RUN git clone https://github.com/scikit-garden/scikit-garden.git && \
	cd scikit-garden && \
	python3 setup.py build install

# Installing the mkn tool for much faster compilation of tick
RUN git clone http://www.github.com/Dekken/maiken.git maiken/master && \
	cd maiken/master && \
	make nix && \
	cp mkn /usr/local/bin

# Clone the tick repository in the re-online branch that contains the online module
RUN git clone -b re-online --single-branch https://github.com/X-DataInitiative/tick.git && \
	cd tick && \
	git submodule update --init && \
	cp -r . /tick


# Nompile it using mkn. This will take 2 million years
RUN  cd /tick && \
	./sh/mkn.sh

# Add tick to the python path
ENV PYTHONPATH /tick:$PYTHONPATH
