
## Aim

Experiment aiming to compare performance of active learning (AL) querying VS random subset selection. The expected result is that AL performance increases more rapidly. The experiment is carried out on generated Gaussian clusters and on images using Cellpose as feature extractor.

## Current state

The diagram below summarizes the input/output structure of the experiment, where rectangular nodes are function, and elliptical ones are inputs or outputs.

![Diagram of the experiment structure.](/notebooks/al_experiment.gv.svg)

The gp\_al (stands for Gaussian Process Active Learning) algorithm is at the centre of the experiment. It handles the querying process which simulates the input received from the oracle by revealing the information to the classifier in stages. For the active learning, selection is done based on the entropy of predictions. Comparing to non-informative querying is done by passing a querying function which simply selects a random subset of the data.

![Handwritten draft of the gp\_al algorithm.](/experiment_al_algo1.svg)

## Running

Change run\_container.sh to use your directories. Use docker and the run\_container.sh script. Uncomment the jupyter bit if you want to run a notebook on localhost:8008.
