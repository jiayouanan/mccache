#!/bin/sh
echo "start ini"

echo "install Redis Server"
echo y | curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
echo y | sudo apt-get install redis
perl -E "print '*' x 100"

echo "install  Memcached Server"
sudo apt update
echo y | sudo apt install memcached
echo y | sudo apt install libmemcached-tools
perl -E "print '*' x 100"

echo "install Pip"
sudo apt update
echo y | sudo apt install python3-pip
pip3 --version
perl -E "print '*' x 100"


echo "install  python package"
pip3 install numpy
pip3 install happybase
pip3 install pymemcache
sudo apt install python3-dev libpq-dev
pip3 install psycopg2
pip3 install redis
perl -E "print '*' x 100"