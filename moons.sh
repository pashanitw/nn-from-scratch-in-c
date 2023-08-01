#!/bin/bash

# Compile the program with gcc
gcc -o moons moons.c -lm

# If the compilation was successful, run the program
if [ $? -eq 0 ]; then
    ./moons
fi
