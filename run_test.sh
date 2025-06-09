#!/bin/bash

set -e -x  

script="features/test/main_validator.py"
proc_name="video-validator"
args=()

echo "Running $script with the following PM2 config:"

joined_args=$(printf "%s," "${args[@]}")
joined_args=${joined_args%,}

echo "module.exports = {
  apps : [{
    name   : '$proc_name',
    script : '$script',
    interpreter: 'python3',
    min_uptime: '5m',
    max_restarts: '5',
    args: [$joined_args]
  }]
}" > app.config.js

cat app.config.js
pm2 start app.config.js

# 🚀 START THE 4 PM2 PROCESSES
pm2 start "python services/scoring/server.py" --name scoring_endpoint
pm2 start "python services/video_scheduler/worker.py" --name video_scheduler_worker
pm2 start "python services/video_scheduler/server.py" --name video_scheduler_endpoint
# pm2 start "python neurons/validator.py $joined_args" --name video-validator
# pm2 start "python services/organic_gateway/server.py" --name organic-gateway

echo "run_test.sh completed"