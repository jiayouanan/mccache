#!/bin/bash
# make sure this script is executable

if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

echo "port is:" $REDISPORT

if [ $# -gt 0 ]; then

   REDISPORT=$1
   sudo cp -p /etc/redis/redis.conf /etc/redis/redis_$REDISPORT.conf
   sudo echo "port $REDISPORT" | sudo tee -a /etc/redis/redis_$REDISPORT.conf
  sudo echo "pidfile /run/redis/redis-server$REDISPORT.pid" | sudo tee -a /etc/redis/redis_$REDISPORT.conf

  sudo redis-server /etc/redis/redis_$REDISPORT.conf
  ps aux |grep redis

fi