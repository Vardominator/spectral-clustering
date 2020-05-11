# Spectral Co-Clustering C++ Library

## Installation
```
wget https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
unzip eigen-master.zip
mv eigen-master/Eigen .
m -rf eigen-master*
```

## Usage
```
g++ spectral_clustering.cpp -o spectral_clustering -std=c++11 && ./spectral_clustering $CSV_DATA
```

## Tasks
- Argument parsing to include:
    - CSV File path
    - CSV delimiter (or write delimiter detector)
    - Number of clusters
- Find a CSV library to allows dynamic memory allocation automatic delimiter detection
- Figure out why Eigen's SVD solver is so damn slow
- Find a comprehensive k-means library
