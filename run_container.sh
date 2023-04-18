docker run -i -t \
    -v '/u/homes/biv20/repos/cp_activation_viz/data:/data' \
    -v '/u/homes/biv20/repos/experiment_active_learning_performance/model:/model' \
    -v '/u/homes/biv20/repos/experiment_active_learning_performance/test:/test' \
    -v '/u/homes/biv20/repos/experiment_active_learning_performance/src:/app' \
    biv20_active_learning
