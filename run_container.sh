docker run -i -t \
    -v '/u/homes/biv20/repos/cp_activation_viz/data:/data' \
    -v '/u/homes/biv20/repos/experiment_active_learning_performance/model:/model' \
    -v '/u/homes/biv20/repos/experiment_active_learning_performance/test:/test' \
    -v '/u/homes/biv20/repos/experiment_active_learning_performance/src:/app' \
    -p 8008:8008 \
    biv20_active_learning \
    /bin/bash -c "\
        jupyter notebook \
        --notebook-dir=/app --ip='*' --port=8008 \
        --no-browser --allow-root"
