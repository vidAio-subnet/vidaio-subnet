#!/bin/bash

if pm2 status | grep -q video-validator; then
    echo "Stopping video-validator..."
    pm2 stop video-validator
fi

if pm2 status | grep -q video_scheduler_endpoint; then
    echo "Stopping video_scheduler_endpoint..."
    pm2 stop video_scheduler_endpoint
fi

if pm2 status | grep -q video_scheduler_worker; then
    echo "Stopping video_scheduler_worker..."
    pm2 stop video_scheduler_worker
fi

if pm2 status | grep -q scoring_endpoint; then
    echo "Stopping scoring_endpoint..."
    pm2 stop scoring_endpoint
fi      
