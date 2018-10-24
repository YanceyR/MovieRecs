# Boltzmann Machine

import numpy
import pandas

# importing datasets
# movieID, movieName, genre
moviesDetails = pandas.read_csv('dataset/ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# userID, Gender, Age, userJobCode, zip
users = pandas.read_csv('dataset/ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# userID, movieID, ratings, timestamp
ratings = pandas.read_csv('dataset/ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# prepare training set(80%) and test set(20%) from 100k
# userID, movieID, rating, timestamp
userCol = 0
movieCol = 1
ratingCol = 2
tsCol = 3

trainingSet = pandas.read_csv('dataset/ml-100k/u1.base', delimiter = '\t')
trainingSet = numpy.array(trainingSet, dtype = 'int')

testSet = pandas.read_csv('dataset/ml-100k/u1.test', delimiter = '\t')
testSet = numpy.array(testSet, dtype = 'int')

# MATRIX STRUCTURE | lines : users, cols : movies, cell : ratings
# put zero if user didn't rate movie
totalUsers = max(max(trainingSet[:,userCol]), max(testSet[:,userCol]))
totalMovies = max(max(trainingSet[:,movieCol]), max(testSet[:,movieCol]))
