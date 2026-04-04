#!/bin/bash
cd /Users/indarkumar/Documents/Projects/Routing_research
python3 scripts/cross_city_and_tuned_eval.py > /tmp/cross_city_run.log 2>&1
echo "EXIT_CODE=$?"
