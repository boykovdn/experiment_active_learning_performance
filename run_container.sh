docker run -i -t \
    -v '/home/boyko/repos/cp_activation_viz/data:/data' \
    -v '/home/boyko/repos/experiment_active_learning_performance/model:/model' \
    -v '/home/boyko/repos/experiment_active_learning_performance/test:/test' \
    -v '/home/boyko/repos/experiment_active_learning_performance/src:/src' \
    -v '/home/boyko/repos/experiment_active_learning_performance/notebooks:/notebooks' \
    -p 8008:8008 \
    experiment_al \
    /bin/bash -c "\
        jupyter notebook \
        --notebook-dir=/notebooks --ip='*' --port=8008 \
        --no-browser --allow-root"
