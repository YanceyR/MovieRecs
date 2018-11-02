# Boltzmann Machine

import numpy
import pandas
import torch

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
        weight_x_neurons = torch.mm(visible_neurons, self.weights)

        # make sure bias is applied to each row in weight_x_neurons
        activation_func = weight_x_neurons + self.hidden_bias.expand_as(weight_x_neurons)

        # Each element corresponds to each hidden node.
        # Each element is the probability that the hidden node is activated.
        prob_hidden_given_visible = torch.sigmoid(activation_func)

        # sample of hidden neurons usng Bernoulli sampling,
        # generate random val between 0 and 1.
        # If val <= probability, neuron is activated i.e cell == 1, else 0
        hidden_sample = torch.bernoulli(prob_hidden_given_visible)

        return prob_hidden_given_visible, hidden_sample

    # get visible neurons that were activated according to p (visible | hidden)
    def get_visible_sample(self, hidden_neurons):
        weight_x_neurons = torch.mm(hidden_neurons, self.weights.t())
        activation_func = weight_x_neurons + self.visible_bias.expand_as(weight_x_neurons)
        prob_visible_given_hidden = torch.sigmoid(activation_func)
        visible_sample = torch.bernoulli(prob_visible_given_hidden)
        return prob_visible_given_hidden, visible_sample

    # compute gradient to minimize energy
    # k-step contrastive divergence
    # k signifies after k sampling
    def train(self, ratings_per_user, ratings_per_user_k, p_hidden_activated, p_hidden_activated_k):
        self.weights += torch.mm(ratings_per_user.t(), p_hidden_activated) - torch.mm(ratings_per_user_k.t(), p_hidden_activated_k)
        self.visible_bias += torch.sum((ratings_per_user - ratings_per_user_k), 0)
        self.hidden_bias += torch.sum((p_hidden_activated - p_hidden_activated_k), 0)

class Data():
    def __init__(self, training_set, test_set, user_col_i, movie_col_i, rating_col_i):
        self.training_set = training_set
        self.test_set = test_set
        self.user_col_i = user_col_i
        self.movie_col_i = movie_col_i
        self.rating_col_i = rating_col_i
        self.total_users = max(max(training_set[:, self.user_col_i]), max(test_set[:, self.user_col_i]))
        self.total_movies = max(max(training_set[:, self.movie_col_i]), max(test_set[:, self.movie_col_i]))

    def add_movie_lookup(self, movie_lookup):
        self.movie_lookup = movie_lookup

    def convert_training_matrix(self):
        self.training_set = self.__convert_to_matrix(self.training_set)

    def convert_test_matrix(self):
        self.test_set = self.__convert_to_matrix(self.test_set)

    def __convert_to_matrix(self, data):
        converted_data = []
        for id_users in range(1, self.total_users + 1):

            # second [] is a conditional, ndarray
            id_movies = data[:, self.movie_col_i][data[:,0] == id_users]
            id_ratings = data[:, self.rating_col_i][data[:,0] == id_users]
            ratings = numpy.zeros(self.total_movies)

            # array indexing is possible because of numpy, type is ndarray
            # put zero if user didn't rate movie
            ratings[id_movies - 1] = id_ratings
            converted_data.append(ratings)

        return converted_data

    def convert_training_tensor(self):
        self.training_set = torch.FloatTensor(self.training_set)

    def convert_test_tensor(self):
        self.test_set = torch.FloatTensor(self.test_set)

def main():
    # importing datasets
    # userID, Gender, Age, userJobCode, zip
    users = pandas.read_csv('dataset/ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

    # userID, movieID, ratings, timestamp
    ratings = pandas.read_csv('dataset/ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

    # prepare training set(80%) and test set(20%) from 100k
    # userID, movieID, rating, timestamp
    training_set = pandas.read_csv('dataset/ml-100k/u1.base', delimiter = '\t')
    training_set = numpy.array(training_set, dtype = 'int')

    # userID, movieID, rating, timestamp
    test_set = pandas.read_csv('dataset/ml-100k/u1.test', delimiter = '\t')
    test_set = numpy.array(test_set, dtype = 'int')

    movies_data = Data(training_set, test_set, 0, 1, 2)
    create_movie_lookup(movies_data)

    # Converting data to matrix | lines : users, cols : movies, cell : ratings
    movies_data.convert_training_matrix()
    movies_data.convert_test_matrix()

    # convert matrix into tensors, arrays with single data type
    movies_data.convert_training_tensor()
    movies_data.convert_test_tensor()
    convert_ratings_to_binary(movies_data.training_set)
    convert_ratings_to_binary(movies_data.test_set)

    # hidden nodes corresponds to number of features that will be detected by RBM.
    rbm = RBM(visible_nodes_count=len(movies_data.training_set[0]), hidden_nodes_count=100)
    train_rbm(rbm, movies_data)
    test_rbm(rbm, movies_data)
    prompt_user_recs(rbm, movies_data)

def create_movie_lookup(movies_data):
    # movieID, movieName, genre
    movies_details = pandas.read_csv('dataset/ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
    movies_details.columns = ['movieID', 'title', 'genre']

    # convert movies_details to dict for fast lookup. Will use later don't worry 😉
    movie_id_title = dict()
    movies_details = movies_details.drop(columns='genre')
    movies_details = movies_details.values.tolist()
    movie_lookup = {key: value for (key, value) in movies_details}

    movies_data.add_movie_lookup(movie_lookup)

# convert ratings into binary rating for boltzmann machine
def convert_ratings_to_binary(ratings):
    ratings[ratings == 0] = -1
    ratings[ratings == 1] = 0
    ratings[ratings == 2] = 0
    ratings[ratings >= 3] = 1

def train_rbm(rbm, data):
    # update weights after several observations and each observation will go into a batch
    # batch_size = 1, number of samples to work through before updating internal params
    batch_size = 100

    # Training the restricted boltzmann machine
    # nuber of times that the algorithm will work through the entire dataset
    number_of_epochs = 10

    for epoch in range(1, number_of_epochs + 1):
        # loss will increase when we find differnce between predictions and actual values
        train_loss = 0

        # counter will normalize train_loss
        counter = 0.0

        # taking batches of users
        for user_id in range(0, data.total_users - batch_size, batch_size):
            ratings_per_user_k = data.training_set[user_id:user_id + batch_size]

            # constant used to find train_loss
            ratings_per_user = data.training_set[user_id:user_id + batch_size]
            p_hidden_activated = rbm.get_hidden_sample(ratings_per_user)[0]

            # for loop for k-step contrastive divergence
            for k in range(10):
                hidden_nodes_k = rbm.get_hidden_sample(ratings_per_user_k)[1]
                ratings_per_user_k = rbm.get_visible_sample(hidden_nodes_k)[1]

                # freeze movies that haven't been rated
                # don't want to train on non-existent ratings
                ratings_per_user_k[ratings_per_user<0] = ratings_per_user[ratings_per_user<0]

            # update weights
            p_hidden_activated_k = rbm.get_hidden_sample(ratings_per_user_k)[1]
            rbm.train(ratings_per_user, ratings_per_user_k, p_hidden_activated, p_hidden_activated_k)

            # only include existing rating in training, thus 'ratings_per_user>0'
            train_loss += torch.mean(torch.abs(ratings_per_user[ratings_per_user>=0] - ratings_per_user_k[ratings_per_user>=0]))
            counter += 1

        print(f"Epoch: {epoch}" + f"   loss: {train_loss/counter}")

def test_rbm(rbm, data):
    test_loss = 0
    counter = 0.0

    t_pred_t = 0
    t_pred_f = 0
    f_pred_f = 0
    f_pred_t = 0

    for user_id in range(data.total_users):
        ratings_per_user = data.training_set[user_id:user_id + 1]
        ratings_per_user_t = data.test_set[user_id:user_id + 1]

        if len(ratings_per_user_t[ratings_per_user_t>0]) > 0:
            hidden_nodes = rbm.get_hidden_sample(ratings_per_user)[1]
            ratings_per_user = rbm.get_visible_sample(hidden_nodes)[1]
            pred_ratings = ratings_per_user[ratings_per_user_t>=0]
            test_ratings = ratings_per_user_t[ratings_per_user_t>=0]
            test_loss += torch.mean(torch.abs(test_ratings - pred_ratings))

            for index, rating in enumerate(test_ratings):
                if rating == 1:
                    if pred_ratings[index] == 1:
                        t_pred_t += 1
                    else:
                        t_pred_f += 1

                else:
                    if pred_ratings[index] == 1:
                        f_pred_t += 1
                    else:
                        f_pred_f += 1

            counter += 1

    print()
    print(f"True Positives: {t_pred_t}  |  False Positives: {f_pred_t}")
    print(f"False Negative: {t_pred_f}  |  True Negative: {f_pred_f}\n")

    print(f"Precision: {t_pred_t/(t_pred_t + f_pred_t)}\n")

    print(f"test loss: {test_loss/counter}")

def prompt_user_recs(rbm, movies_data):
    user_id = int(input(f"Enter a user id between 1 - {int(movies_data.total_users)}\n> "))
    rec_movie_ids = get_user_recs_ids(user_id, rbm, movies_data)

    print(f"Here are my movie predictions for user {user_id}!")

    print_movies(movies_data.movie_lookup, rec_movie_ids[0], "Movies you liked ❤️", 10)
    print_movies(movies_data.movie_lookup, rec_movie_ids[1], "Movies you will like 😊", 10)
    print_movies(movies_data.movie_lookup, rec_movie_ids[2], "Movies you won't like 😟", 10)

def get_user_recs_ids(user_id, rbm, movies_data):
    base_ratings = movies_data.training_set[user_id:user_id + 1]

    # get hidden nodes for user ratings. Model must be trained first!
    hidden_nodes = rbm.get_hidden_sample(base_ratings)[1]
    base_ratings = base_ratings[0]

    # use hidden nodes to predict movie rating
    pred_ratings = rbm.get_visible_sample(hidden_nodes)[1]
    pred_ratings = pred_ratings[0]

    rec_movie_ids = []
    not_rec_movie_ids = []
    liked_movie_ids = []

    for index, rating in enumerate(base_ratings):
        if int(rating) == 1:
            # movie id is one plus the index
            liked_movie_ids.append(index + 1)
        elif int(rating) == -1:
            if int(pred_ratings[index]) == 1:
                rec_movie_ids.append(index + 1)
            else:
                not_rec_movie_ids.append(index + 1)

    return liked_movie_ids, rec_movie_ids, not_rec_movie_ids

def print_movies(movie_lookup, movie_ids, message, count):
    print()
    print(message)
    for index in range(count):
        print(f"{index + 1}: {movie_lookup[movie_ids[index]]}")

if __name__ == '__main__':
    main()
