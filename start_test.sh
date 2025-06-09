#!/bin/bash

source venv/bin/activate
./run_test.sh --wallet.name shark01 --wallet.hotkey 85_1 --subtensor.network finney --netuid 85 --axon.port 10201 --logging.debug
deactivate
