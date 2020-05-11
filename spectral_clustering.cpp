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

#include "csv.h"
#include "Eigen/Dense"


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


// SpectralClustering class
/*
Compute weight.
*/
float distance(float pointA, float pointB);
float weight(float pointA, float pointB);

/*
Compute diagonal.
*/
std::vector<float> computeDiagonal(std::vector<std::vector<float>> adjacencyMatrix);

/*
Compute Laplacian.
*/
std::vector<std::vector<float>> computeLaplacian(std::vector<float> diagonal, std::vector<std::vector<float>> adjacencyMatrix);

/*
Compute Laplacian norm.
*/
std::vector<std::vector<float>> computeLaplacianNorm(std::vector<float> diagonal, std::vector<std::vector<float>> laplacian);

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
            // float w = weight(start, stof(*it));
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
    
    // normalization
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(adjacencyMatrixPlaceholder.size(), adjacencyMatrixPlaceholder.size());
    Eigen::VectorXd rowSum = adjacencyMatrix.rowwise().sum();
    R.diagonal() = rowSum;
    std::cout << "computed R" << std::endl;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esR(R);
    Eigen::MatrixXd RInv = esR.operatorInverseSqrt();
    std::cout << "computed RInv" << std::endl;

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(adjacencyMatrixPlaceholder[0].size(), adjacencyMatrixPlaceholder[0].size());
    Eigen::VectorXd colSum = adjacencyMatrix.colwise().sum();
    C.diagonal() = colSum;
    std::cout << "computed C" << std::endl;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esC(C);
    Eigen::MatrixXd CInv = esC.operatorInverseSqrt();
    std::cout << "computed CInv" << std::endl;

    Eigen::MatrixXd adjacencyMatrixNorm = RInv * adjacencyMatrix * CInv;
    std::cout << adjacencyMatrixNorm << std::endl;

    Eigen::BDCSVD<Eigen::MatrixXd> SVD(adjacencyMatrixNorm, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = SVD.matrixU();
    Eigen::MatrixXd V = SVD.matrixV();
    
    std::cout << U << std::endl;
    std::cout << V << std::endl;

    // // Compute diagonal.
    // std::vector<float> diagonal = computeDiagonal(adjacencyMatrix);

    // // Compute Laplacian.
    // std::vector<std::vector<float>> laplacian = computeLaplacian(diagonal, adjacencyMatrix);
    
    // // Compute the Laplacian norm.
    // std::vector<std::vector<float>> laplacianNorm = computeLaplacianNorm(diagonal, laplacian);

    // // Compute eigenvectors / eigenvalues of the Laplacian norm.

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


float distance(float pointA, float pointB)
{
    return sqrt(pow(pointA - pointB, 2));
}
float weight(float pointA, float pointB)
{
    return distance(pointA, pointB);
}

std::vector<float> computeDiagonal(std::vector<std::vector<float>> adjacencyMatrix)
{
    std::vector<float> diagonal;
    for (auto & row : adjacencyMatrix)
    {
        float columnSum = 0.0;
        for (auto & weight : row)
        {
            columnSum += weight;
        }
        diagonal.push_back(columnSum);
    }
    return diagonal;
}

std::vector<std::vector<float>> computeLaplacian(std::vector<float> diagonal, std::vector<std::vector<float>> adjacencyMatrix)
{
    std::vector<std::vector<float>> laplacian;
    for (int rowIndex = 0; rowIndex < adjacencyMatrix.size(); rowIndex++)
    {
        std::vector<float> laplacianRow;
        for (int columnIndex = 0; columnIndex < adjacencyMatrix[rowIndex].size(); columnIndex++)
        {
            if (rowIndex == columnIndex)
            {
                laplacianRow.push_back(diagonal[rowIndex] - adjacencyMatrix[rowIndex][columnIndex]);
            }
            else
            {
                laplacianRow.push_back(-1.0f * adjacencyMatrix[rowIndex][columnIndex]);
            }
        }
        laplacian.push_back(laplacianRow);
    }
    return laplacian;
}

std::vector<std::vector<float>> computeLaplacianNorm(std::vector<float> diagonal, std::vector<std::vector<float>> laplacian)
{
    std::vector<std::vector<float>> laplacianNorm;

    // compute inverse square diagonal
    for (auto & d : diagonal)
    {
        if (d > 0.0)
        {
            d = pow(d, -1 * 0.5);   
        }
    }
    
    return laplacianNorm;
}