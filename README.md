# Moons Dataset Neural Network in C from Scratch

This project is a simple neural network implementation from scratch in C to solve a binary classification problem using the moons dataset.

## Dataset Description

The "two interleaving half-circles" dataset, often referred to as the "moons" dataset, is a common toy dataset used for binary classification problems. It is used frequently to illustrate the behavior of certain algorithms because it is not linearly separable, meaning a straight line cannot effectively separate the two classes. Instead, a model must be able to find a "curvy" boundary.

The dataset consists of two features (x1 and x2), which can be interpreted as spatial coordinates on a 2-dimensional plane. The target variable y is binary, meaning it can take on two possible outcomes often coded as 0 and 1.

In this project, we create two "half-circles" or "moons". The positive examples (y=1) are the points on the upper half-circle, and the negative examples (y=0) are the points on the lower half-circle.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have a Linux or MacOS machine. Windows subsystem for Linux (WSL) should work as well.
* You have installed the GCC compiler. If not, please follow the instructions for your specific system to install it.

## Running the Project

To run this project, follow the steps below:

1. Open a terminal in the project directory.

2. Make the shell script executable:

```bash
chmod +x moons.sh
```
```bash
./moons.sh
```
