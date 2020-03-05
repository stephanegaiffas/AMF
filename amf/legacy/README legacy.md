
# AMF: Aggregated Mondrian Forests for Online Learning

This `GitHub` repository contains the algorithms described in the paper

> *AMF: Aggregated Mondrian Forests for Online Learning*
> 
> by J. Mourtada, S. Gaïffas and E. Scornet
> 
> arXiv link: http://arxiv.org/abs/1906.10529

It privides mainly the `AMFClassifier` and the `AMFRegressor`, together with a weak 
baseline called `OnlineDummyClassifier`. 

The `AMF` algorithm is implemented as the class `OnlineForestClassifier` in the `tick.online` module of the `tick` library https://github.com/X-DataInitiative/tick.
This module is not yet merged into the `master` branch of `tick` and therefore is not yet easily installable via `pip`. It must be compiled and installed from source for now.
Therefore, this `GitHub` repository is here to ease this installation, through the use of a `Docker` image.

# 1. Installation

First, you need to install `Docker` on your computer https://www.docker.com/get-started, and you need to create an account on https://hub.docker.com/ in order to use the image I created for you. It will make the installation of all the tools required to run the experiments 1000 times easier, and it won't mess with your local configuration files and `Python` environments.
Then, you need to git clone this repository somewhere by typing 

```bash
git clone https://github.com/stephanegaiffas/AMF.git
```

in a terminal. Now, you need to run the image that contains everything you need. This image is called `stephanegaiffas/amf:v1`.
Once `Docker` is installed, you can simply type in a terminal

```bash
docker run -it -v <PATH>/AMF/:/AMF stephanegaiffas/amf:v1
```

where `<PATH>` is the path leading to the AMF directory corresponding to this repository. On my laptop it is `/Users/stephanegaiffas/Code`.
You must change only the `<PATH>` in this command line and nothing else, note that `stephanegaiffas/amf:v1` is the name of the image I built for you on
 `docker-hub`.

This command-line runs the image and accesses the **container** (instance of the image) in interactive mode. This means that after running this command-line, you are **inside** the container, where everything has been already installed for you.
Note that `Docker` will download automatically the image the first time you launch this command-line, which may take some time (the download is around 2GB).

# 2. Run the experiments

Once inside the container you can simply run the scripts to reproduce the experiments of the paper. For instance

```bash
python3 plot_decisions.py
```

will create a `decisions.pdf` file, which corresponds to the second figure of the paper. Note that in this image `Python` is called using the `python3` command.

[ ] Detail all the scripts and what they do

# 3. Explain the docker-serve experiment

[ ] TODO

# Appendix. Build the docker image from the `Dockerfile`

**Warning:** only for those who know what they do and need to modify the image

The `Dockerfile` can be used to build the image containing all the tools required (although a pre-built image is available on `docker hub`, see above).
In order to generate this image, simply run

```bash
docker build -t amf:v1 .
```

in the folder containing the `Dockerfile`. This will take a very long time (almost one hour !) since it configures everything from a basic Linux image, and since the compilation of `tick` is very long.
This image can be now shared through a

```bash
docker tag amf:v1 stephanegaiffas/amf:v1
docker push stephanegaiffas/amf:v1
```

(although with another user name...)

