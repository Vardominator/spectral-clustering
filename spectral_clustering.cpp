#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <iterator>
#include <map>
#include <math.h>
#include <numeric>

#include "Eigen/Dense"
#include "KMeans/KMeans.h"

Eigen::IOFormat CleanFmt(3, 0, " ", "\n", "[", "]");

/*
Check if input CSV file path exists before proceeding.
*/
bool FileExists(std::string& name);

// ProcessCSV class
/*
Split a string into a vector of tokens.
*/
std::vector<std::string> SplitRow(std::string row, char delimiter='\t');

/*
Print elements of a string vector.
*/
void PrintRow(std::vector<std::string> row);

/*
Trim leading and trailing whitespaces from string.
*/
void Trim(std::string &str);

void inverseSqrt(Eigen::VectorXd &vector);

int main(int argc, char** argv)
{
    // Check if the input file exists.
    std::string fileName(argv[1]);
    if(!FileExists(fileName))
    {
        printf("%s does not exist.\n", fileName.c_str());
        return 0;
    }

    // Create column mapping.
    std::vector<std::string> indexes;
    std::vector<std::string> columns;
    auto columnMap = std::map<std::string, std::vector<std::string>>{};

    // Begin reading file.
    std::ifstream file(argv[1]);
    std::string fileRow;

    // Record header.
    if(std::getline(file, fileRow))
    {
        Trim(fileRow);

        for (auto & column : SplitRow(fileRow))
        {
            columns.push_back(column);
            std::vector<std::string> newColumn;
            columnMap.insert(std::make_pair(column, newColumn));
        }
    }


    // Read remaining file and construct adjacency matrix.
    std::vector<std::vector<float>> adjacencyMatrixPlaceholder;
    while (std::getline(file, fileRow))
    {
        std::vector<std::string> values = SplitRow(fileRow);
        std::vector<std::string>::iterator it = values.begin();
        indexes.push_back(*it);
        ++it;

        std::vector<float> adjacencyRow;
        float start = stof(*it);

        float checkZeroSum = 0.0;
        while (it != values.end())
        {   
            float w = stof(*it);
            checkZeroSum += w;
            adjacencyRow.push_back(w);
            columnMap[columns[it - values.begin() - 1]].push_back(*it);
            ++it;
        }

        if (checkZeroSum > 0.0){
            adjacencyMatrixPlaceholder.push_back(adjacencyRow);
        }

    }

    Eigen::MatrixXd adjacencyMatrix;
    
    adjacencyMatrix.resize(adjacencyMatrixPlaceholder.size(), adjacencyMatrixPlaceholder[0].size());
    
    for (int i = 0; i < adjacencyMatrixPlaceholder.size(); i++)
    {
        for (int j = 0; j < adjacencyMatrixPlaceholder[i].size(); j++)
        {
            adjacencyMatrix(i, j) = adjacencyMatrixPlaceholder[i][j];
        }
    }

    // bistochastic normalize 
    // => scale normalize
    // 0. Check sparsity of matrix
    // 1. Make sure elements nonnegative
    // 2. Calculated R^(-1/2) and C^(-1/2) efficiently (DONE)


    // // normalization
    Eigen::VectorXd rowSumSqrt = adjacencyMatrix.rowwise().sum();
    inverseSqrt(rowSumSqrt);
    rowSumSqrt = (rowSumSqrt.array().isFinite()).select(rowSumSqrt, 0);
    auto RInv = rowSumSqrt.asDiagonal();

    Eigen::VectorXd colSumSqrt = adjacencyMatrix.colwise().sum();
    inverseSqrt(colSumSqrt);
    colSumSqrt = (colSumSqrt.array().isFinite()).select(colSumSqrt, 0);
    auto CInv = colSumSqrt.asDiagonal();

    Eigen::MatrixXd adjacencyMatrixNorm = RInv * adjacencyMatrix * CInv;

    // singular value decomposition
    Eigen::JacobiSVD<Eigen::MatrixXd> SVD(adjacencyMatrixNorm, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = SVD.matrixU();
    Eigen::MatrixXd V = SVD.matrixV();

    int clusters = 10;
    int clusterSize = 800;
    int nFeatures = 10;
    int nIters = 1000;
    int seed = 42;
    int nExamplesTotal = 4000;

    U = U(Eigen::all, Eigen::seq(1, clusters));
    V = V(Eigen::all, Eigen::seq(1, clusters));

    auto ZU = RInv * U;
    auto ZV = CInv * V;

    Eigen::MatrixXd Z(ZU.rows() + ZV.rows(), ZU.cols());
    Z << ZU, ZV;

    Eigen::ArrayXXd zClusterCentroids = Eigen::ArrayXXd::Zero(clusters, Z.cols());
    Eigen::ArrayXd zClusterAssigments = Eigen::ArrayXd::Zero(Z.size());

    RunKMeans(Z.data(), nExamplesTotal, nFeatures, clusters, nIters, seed, strdup("plusplus"), zClusterCentroids.data(), zClusterAssigments.data());

    // Assign row clusters
    auto rowClusterAssignments = std::map<int, int>{};
    for(int i = 0; i < adjacencyMatrixNorm.rows(); i++)
    {
        rowClusterAssignments.insert(std::make_pair(i, zClusterAssigments(i)));
    }

    // Assign column clusters
    auto colClusterAssignments = std::map<int, int>{};
    for(int i = 0; i < adjacencyMatrixNorm.cols(); i++)
    {
        colClusterAssignments.insert(std::make_pair(i, zClusterAssigments(adjacencyMatrixNorm.rows() + i)));
    }

    for (const auto &p : rowClusterAssignments) {
        std::cout << "row[" << p.first << "] = " << p.second << '\n';
    }

    for (const auto &p : colClusterAssignments) {
        std::cout << "col[" << p.first << "] = " << p.second << '\n';
    }

    return 0;
}


bool FileExists(std::string &name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

// ProcessCSV class
std::vector<std::string> SplitRow(std::string row, char delimiter)
{
    std::istringstream stringStream(row);
    std::string token;

    std::vector<std::string> rowValues;
    while(std::getline(stringStream, token, delimiter))
    {
        rowValues.push_back(token);
    }
    return rowValues;
}

void PrintRow(std::vector<std::string> row)
{
    for (auto & entry : row)
    {
        std::cout << entry << '\t';
    }
}

void Trim(std::string &str)
{
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](int ch)
    {
        return !std::isspace(ch);
    }));
    str.erase(std::find_if(str.rbegin(), str.rend(), [](int ch)
    {
        return !std::isspace(ch);
    }).base(), str.end());
}

void inverseSqrt(Eigen::VectorXd &vector)
{
    for(int i = 0; i < vector.size(); i++)
    {
        vector(i) = 1.0f / sqrt(vector(i));
    }
}