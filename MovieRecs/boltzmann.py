# Boltzmann Machine

import numpy
import pandas
import torch

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

# userID, movieID, rating, timestamp
trainingSet = pandas.read_csv('dataset/ml-100k/u1.base', delimiter = '\t')
trainingSet = numpy.array(trainingSet, dtype = 'int')

# userID, movieID, rating, timestamp
testSet = pandas.read_csv('dataset/ml-100k/u1.test', delimiter = '\t')
testSet = numpy.array(testSet, dtype = 'int')

# Get total number of users and movies
totalUsers = max(max(trainingSet[:, userCol]), max(testSet[:, userCol]))
totalMovies = max(max(trainingSet[:, movieCol]), max(testSet[:, movieCol]))

# Converting data to matrix | lines : users, cols : movies, cell : ratings
# put zero if user didn't rate movie
def convert(data):
    converted_data = []
    for id_users in range(1, totalUsers + 1):
        
        # second [] is a conditional, ndarray
        id_movies = data[:, movieCol][data[:,0] == id_users]
        id_ratings = data[:, ratingCol][data[:,0] == id_users]
        ratings = numpy.zeros(totalMovies)
                
        # array indexing is possible because of numpy, type is ndarray
        ratings[id_movies - 1] = id_ratings
        converted_data.append(ratings)
        
    return converted_data
        
        
trainingSet = convert(trainingSet)
testSet = convert(testSet)

# convert matrix into tensors, arrays with single data type
trainingSet = torch.FloatTensor(trainingSet)
testSet = torch.FloatTensor(testSet)

# convert ratings into binary rating for boltzmann machine
trainingSet[trainingSet == 0] = -1
trainingSet[trainingSet == 1] = 0
trainingSet[trainingSet == 2] = 0
trainingSet[trainingSet >= 3] = 1

testSet[testSet == 0] = -1
testSet[testSet == 1] = 0
testSet[testSet == 2] = 0
testSet[testSet >= 3] = 1

class RBM():
    def __init__(self, visible_nodes_count, hidden_nodes_count):
        self.visible_nodes_count = visible_nodes_count
        self.hidden_nodes_count = hidden_nodes_count
        
        # weights are initialized with random values with normal distribution
        self.weights = torch.randn(visible_nodes_count, hidden_nodes_count)
        
        # tensor functions dont accept single dim tensors
        self.hidden_bias = torch.randn(1, hidden_nodes_count)
        self.visible_bias = torch.randn(1, visible_nodes_count)
        
        # get hidden neurons that were activated according to p (hidden | visible)
        def get_hidden_sample(self, visible_neurons):
            weight_x_neurons = torch.mm(visible_neurons, self.weights.t())
            
            # make sure bias is applied to each row in wn
            activation_func = weight_x_neurons + self.hidden_bias.expand_as(weight_x_neurons)
            
            # Each element corresponds to each hidden node.
            # Each element is the probability that the hidden node is activated.
            prob_hidden_given_visible = torch.sigmoid(activation_func)
            
            # sample of hidden neurons usng Bernoulli sampling,
            # generate random val between 0 and 1.
            # If val <= probability, neuron is activated i.e cell == 1, else 0
            hidden_sample = torch.bernoulli(prob_hidden_given_visible)
            
            return prob_hidden_given_visible, hidden_sample
            
            
            








