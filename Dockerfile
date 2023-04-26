FROM pytorch/pytorch

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install git vim graphviz -y

RUN pip install imageio networkx matplotlib jupyter cellpose scikit-learn graphviz

# Ensure source code dir exists before it is mounted.
RUN mkdir /src
# Link src so that it appears as a python package. This is done so that
# the jupyter notebooks can find the code.
RUN ln -s /src /opt/conda/lib/python3.10/site-packages/this_experiment

#COPY ./src /src
#RUN ln -s /src /opt/conda/lib/python3.10/site-packages/experiment_alp

#ENV REPOS_PATH=/repos
#ENV PY_LIBS_PATH=/opt/conda/lib/python3.10/site-packages
#RUN mkdir $REPOS_PATH
#
## Install toolkit lib
#RUN git clone https://github.com/boykovdn/malaria_data_toolkit $REPOS_PATH
#RUN ln -s ${REPOS_PATH}/malaria_data_toolkit/malaria_data_toolkit ${PY_LIBS_PATH}/malaria_data_toolkit
