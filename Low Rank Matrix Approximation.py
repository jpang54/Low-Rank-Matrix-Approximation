"""
Low Rank Matrix Approximation

Low-rank approximation of matrices is useful for finding hidden relationships that might not be apparent from high-dimensional data.


We will be approximating a partially observed (typically) large matrix using a product of low rank matrices. 

Let us recap the ingredients of low rank matrix approximation:

- Observed matrix M of size m x n with real valued entries and missing values
    
- Low rank matrices A of size m x k and B of size k x n 
    
- We would like to find A,B such that M is approx AB. We will denote Mhat = AB
    
We are going to measure the goodness of approximation using the squared error at entries (i,j) 
for which we know the value of M_{ij} (entries where the matrix is observed, denoted as O = {(i,j) s.t. M_{ij} observed}). 
Therefore the loss is,

L = 1/|O| * \sum_{(i,j) in O} [M_{ij} -  Mhat_{ij}]^2.

"""

import matplotlib.pyplot as plt
import numpy as np

# Implement average squared loss function.
# Given M, M_approx, O return the average squared loss over the observed entries.
# M is an m x n 2-D numpy array containing the observed entries (with
# arbitrary values in the unobserved entries)
# M_approx is an m x n 2-D numpy array representing the low-dimensional approximation
# O is an m x n 2-D numpy array containing 0 if the entry is unobserved and
# 1 if the entry is observed in M. O tells us which entries in M were observed

def loss(M, M_approx, O):
  
    # Set unobserved entries to 0
    observed_M = M * O
    observed_M_approx = M_approx * O

    return np.sum(np.square(observed_M - observed_M_approx)) / np.sum(O)

"""

## Normalization of M

As the initial prediction of Mhat would likely to have entries following a standard normal distribution and 
we use the squared error as the sole loss, it would be a good practice for us to do normalization for M prior to the learning process. 
This means that we will preprocess matrix M so that the average over its observed entries would be 0 and the empirical distribution of its entries would resemble a standard normal distribution.   

Mathematically speaking, let us normalize M such that,
\sum_{(i,j) \in O} M_{ij} = 0  and  1/|O| \sum_{(i,j) \in O} M^2_{ij} = 1 
This can be achieved by finding two scalar a and s such that after the transformation,
M maps to (M - a)/s the matrix is normalized.

Specifically, a and s are the mean and standard deviation of the observed entries in M:
a = 1/|O| \sum_{(i,j)\in O} M_{ij}  and  s = \sqrt{1/|O| * \sum_{(i,j)\in O} (M_{ij} - a)^2}
"""

# Return the normalized version of M. 
# M and O are in the same format as mentioned earlier
def normalize_matrix(M, O):

    # Set unobserved entries to 0
    observed = M * O

    # Number of observed entries
    total = np.sum(O)

    # Create a matrix where unobserved entries are 0 and observed are a
    a = np.sum(observed) / total
    observed_a = a * O

    s = np.sqrt(np.sum(np.square(observed - observed_a)) / total)

    return (M - a) / s


# The following functions checks if our implementation of normalize_matrix is correct
def check_normalization():
    M = np.random.rand(10, 3)
    O = (np.random.rand(10, 3) > 0.5) + 0
    Mn = normalize_matrix(M, O)
    assert(abs(np.sum(Mn * O)) < 1e-6)
    assert(abs(np.sum(Mn**2 * O) / np.sum(O) - 1) < 1e-6)
    print("Function {} is working fine!".format('normalize_matrix'))
    
check_normalization()

"""

## Low Rank Matrix Approximation (LORMA) 

We are now ready to build a low rank approximation (abbreviated as *LORMA*) of the form,

M ~ Mhat = AB 

For brevitiy we refer to this model as *LORMA*. Note that once we established the approximating matrix A we can reuse the same function **loss** from above.
"""

# Implement lorma's prediction. 
# A is a m x k numpy 2-D array
# B is a k x n numpy 2-D array
# A and B are the low-rank matrices used to calculate M_approx
def lorma(A, B):

    return A @ B

"""## Gradient of LORMA

Next we need to implement the gradient of the LORMA model. The gradient should have the same structure as the parameters of our LORMA model, which are A (shape m x k) and B (shape k x n). 

To get the gradient to A, we apply chain rule for matrix differentiation as: 

dL/dA = dL/dMhat \cdot dMhat/dA

where dL/dMhat returns a matrix of a shape of m x n with its entries:

(dL/dMhat)_{ij} = 
                \begin{cases}
                    2/|O| * (Mhat_{ij} - M_{ij}) if (i,j)\in O 
                    0 otherwise
                \end{cases}

and 

dMhat/dA = B^T

returns a matrix of shape n x k. Note "\cdot" denotes the matrix multiplication.

Likewise, to get the gradient with respect to B, we have 

dL/dB = dMhat/dB  \cdot  dL/dMhat  where  dMhat/dB = A^T

"""

# Implement lorma gradient. 

# Given M, O, A, B return the gradient of A and B
# M is an m x n 2-D numpy array containing the observed entries (with
# arbitrary values in the unobserved entries)
# O is an m x n 2-D numpy array containing 0 if the entry is unobserved and
# 1 if the entry is observed in M. O tells us which entries in M were observed
# We would need to return two matrices as the gradients: dA & dB. dA has m x k, and dB has shape k x n.

def lorma_grad(M, O, A, B):

    # Total number of observed entries
    total = np.sum(O)

    # Calculate lorma
    M_hat = lorma(A, B)

    # Set values at unobserved indices in M_hat and M to 0
    M_hat = M_hat * O
    M_observed = M * O

    dLdM_hat = 2 / total * (M_hat - M_observed)

    # Gradient wrt A
    dA = dLdM_hat @ B.T

    # Gradient wrt B
    dB = A.T @ dLdM_hat

    return dA, dB

"""## Initialization of LORMA

Before we start learning using GD, we need to setup an initial state of A and B for estimating Mhat. Recall that we normalized M such that,
\sum_{(i,j)\in O} M_{ij} = 0  and  1/|O| * \sum_{(i,j)\in O} M^2_{ij} = 1 
We would like to make sure that the similar properties hold for Mhat. 

More specifically, we adopt random initialzation of the entries in A and B followed by a normalization process to make Mhat satisfy the following properties:  
1. Zero mean, i.e., \sum_{(i,j)\in O} Mhat_{ij} = 0:
    
  We can write Mhat_{ij} = a_i \cdot b_j where a_i and b_j are the i'th row & j'th column of A and B respectively. 
  Since A and B are completely random the probability that a_i \cdot b_j is small is pretty high so we can assume that \sum_{i,j} Mhat_{ij} = \sum_{i,j} a_i \cdot b_j approx 0. 


2. Less than or equal to unit variance, i.e., 1/{mn} * \sum_{i,j} Mhat_{ij}^2 <= 1   
    
  It suffices to have |Mhat_{ij}| <= 1 to satisfy the above condition. 
  Let us normalize each a_i and b_j so they have a unit norm:

    a_i maps to a_i/||a_i||  and 
    b_j maps to b_j/||b_j||

Once the norm of all vectors a_1,..., a_k, b_1,..., b_k is 1 the inner products a_i \cdot b_j are in [-1,1], which is essentially equivalent to |Mhat_{ij}| <= 1. 
This property is known as Cauchy's inequality which is typically presented as, |a \cdot b| <= ||a|| ||b||. 

"""

# Implement initialization in accordance to the guidelines above. 
# m, n, and k refer to the shapes of A and B
def lorma_init(m, n, k):
    # initialize A, B using a zero-mean unit-variance Gaussian per entry
    A = np.random.randn(m, k)
    B = np.random.randn(k, n)

    # normalize the rows of A and columns of B
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)

    return A, B

def check_lorma_init():
    A, B = lorma_init(10, 7, 3)
    assert(np.linalg.norm(np.diag(A @ A.T) - np.ones(10), 1) < 1e-6)
    assert(np.linalg.norm(np.diag(B.T @ B) - np.ones(7), 1) < 1e-6)
    return

check_lorma_init()

"""
## Finally, it is time to use gradient descent (GD) to learn a LORMA model
    
Details for implementation:
- We provide the way parsing of params should be performed
- The first step is to create A & B by calling lorma_init
- Next create a list called apperr which will record the approximation error after each update.
  Do so by calling LORMA's loss function. We can initialize apperr with the initial loss.

## Main Loop Implementation:

- Call lorma_grad with O to get new gradients dA & dB for A and B
- Perform gradient step with the externally provide list of learning rate eta
- Every 10 epochs do:
    - Calculate loss and append it to apperr
    - Print most recent current loss:*print((iter + 1), ': ', apperr[-1].round(4))*
"""

### Implement GD for LORMA model
# k is an integer stating the rank of our LORMA model
# epochs is an integer stating the number of epochs to run
# eta is a list of floats, with the learning rate for each epoch
# len(eta) = epochs

def lorma_learn(M, O, params):
    k, epochs, eta = params

    m, n = M.shape

    A, B = lorma_init(m, n, k)

    # Stores our approximation error
    apperr = []
    apperr.append(loss(M, lorma(A, B), O))

    for e in range(epochs):
        dA, dB = lorma_grad(M, O, A, B)

        # Perform gradient step
        A = A - eta[e] * dA
        B = B - eta[e] * dB

        # Every 10 epochs, append loss to apperr and print it
        if(e % 10 == 0):
            apperr.append(loss(M, lorma(A, B), O))
            print((e + 1), ': ', apperr[-1].round(4))

    return A, B, apperr

m, n, k = 100, 40, 5
rand_seed = 10
np.random.seed(rand_seed)

def check_lorma_learn():
    from numpy.random import binomial, randn, uniform
    mockA, mockB = uniform(1, 2, (m, k)), uniform(-2, -1, (k, n))
    M = mockA @ mockB + 0.01 * np.random.randn(m, n)
    O = binomial(1, 0.5, size=M.shape)
    epochs = 100
    eta = 2.0 * np.ones(epochs)
    params = k, epochs, eta
    A, B, l = lorma_learn(M, O, params)
    plt.plot(l , '-o')
    return M, A, B

M, A, B = check_lorma_learn()

"""### Visualizing the data matrix and it low-rank approximation

For reference let us also visualize a random matrix from the same distribution used for initialization. 
"""

def show_mat(X, str, ind):
    plt.subplot(1,3,ind)
    plt.imshow(X, cmap='hot')
    plt.axis('off')
    plt.title(str)

Ar, Br = lorma_init(m, n, k)
fig = plt.figure(figsize=(10,60))
show_mat(M, 'Original Matrix', 1)
show_mat(lorma(A, B), 'Low Rank Approximation', 2)
show_mat(lorma(Ar, Br), 'Initial Matrix', 3)

"""

## Movie Recommendation

In movie recommendation, we are given data about users and their movie preferences, and we have to recommend movies that the users have not seen, and will like. 
See [Netflix Prize](https://www.netflixprize.com) which awarded 1 million dollars to a team that significantly improved their in-house movie recommendation system. 
An interesting tidbit: the team which finished second in 2008 comprised of three Princeton undergraduates! [[Article]](https://www.cs.princeton.edu/news/article/princeton-undergraduates-challenge-1000000-netflix-prize)
    
We will be using a smaller dataset called [MovieLens](https://grouplens.org/datasets/movielens/).
"""

"""
In this code cell, we will first download the MovieLens data.
We will then convert the data into a matrix form instead of a list form.

The matrix will have people as rows and movies (or Movie IDs) as columns.

We will also download the movie names corresponding to the movie IDs.
"""

# Imports for loading and splitting data
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the MovieLens dataset
# Rows are users and columns are different movies

# Load the data from GitHub
movie_ratings_path = 'https://raw.githubusercontent.com/ameet-1997/Machine-Learning/master/MovieLens/ratings.csv'
movielens_raw = pd.read_csv(movie_ratings_path)

# Process the data to get it in the right format
# The data consists of rows - (userID, movie_ID, rating, timestamp)
# We want to convert it to a matrix where the rows are users and the columns are movies
# We will fill missing values with -1, a value that is not present in the dataset
missing_value = -1
movielens_data = movielens_raw.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(missing_value).to_numpy()

# Create the observed matrix O. It should be boolean matrix with True for observed entries and false otherwise
movielens_observed = (movielens_data != -1) + 0

# We'll convert the boolean matrix to an integer matrix
movielens_observed = movielens_observed.astype(int)

# Let's create a copy of the data because we'll need it later
movielens_data_copy = movielens_data.copy()


# Load the movie names
# The data consists of rows which are movie IDs
movie_names_path = 'https://raw.githubusercontent.com/ameet-1997/Machine-Learning/master/MovieLens/movies.csv'
movie_names = pd.read_csv(movie_names_path)

"""
We'll run LORMA on the MovieLens data
"""

rand_seed = 10
np.random.seed(rand_seed)

# Normalize the MovieLens train data
movielens_normalized = normalize_matrix(movielens_data, movielens_observed)
# Define the parameters for LORMA
low_rank = 40
# For debugging run for 100 epochs then switch back to the parameters below
epochs = 2000
eta = 10.0 * np.ones(epochs)
params = low_rank, epochs, eta

# Run LORMA
A, B, losses = lorma_learn(movielens_normalized, movielens_observed, params)

# Plot the losses to make sure they are decreasing
_ = plt.plot(losses , '-o')

"""
We'll pick three users and see what movies they liked before, and what movies our model predicted.
Hopefully there is some correlation!

In this cell we first print out some movies that they have liked
"""

# Pick three users to recommend movies to
users = [1,13, 111]

# Let's see their tastes. Print the top-p movies they have rated highly
p = 10
for user in users:
    print("\nUser {} liked the following:\n".format(user))
    
    # Sort the movies for this user in descending order based on the rating
    movie_order = np.argsort(-movielens_data[user])
    top_p = movie_order[:p]
    
    # Print the movies
    for movie in top_p:
        print("\t{:<50} rated {:.1f}  genres {:<30}".format(movie_names.iloc[movie]['title'], movielens_data_copy[user, movie], movie_names.iloc[movie]['genres']))

"""
User 1 seems to like Comedy and Horror
User 13 seems to like Drama and Romance
User 111 seems to like Sci-Fi and Adventure

Now let's make our predictions on the test data and see what movies we can recommend

"""

for user in users:
    print("\nRecommend the following movies to User {}\n".format(user))
    
    # Predict the rating for these movies by performing a matrix multiplication between the user and movie vectors
    predicted_ratings = A[user,:] @ B
    
    # If the movie review was observed in the matrix, set it to (-infinity) so that we don't predict it
    # We want to predict only from a set of movies which the user has not seen
    predicted_ratings[movielens_observed[user]] = -np.inf
    
    # Choose the top_p movies
    predicted_movie_order = np.argsort(-predicted_ratings)
    top_p = predicted_movie_order[:p]
    
    # Print the recommended movies
    for movie in top_p:
        print("\t{:<60} genres {:<30}".format(movie_names.iloc[movie]['title'][:60], movie_names.iloc[movie]['genres']))

"""

Is there any correlation between the movies they liked and the predicted movies?

p.s.: The correlation is not strong, but try to see if there is some pattern!

Yes, there is a correlation between the movies they liked and the predicted movies. 
For User 1, there are often comedy, action, and adventure movies recommended, which are genres that appeared numerous times in the movies that they liked. 
For User 13, there are multiple drama and romance movies recommended, genres that appeared numerous times in the movies they liked. 
For User 111, there are often adventure, thriller, drama, sci-fi, and crime movies recommended, genres that appeared often in the movies they liked.
"""