import math

#Measures the straight-line geometric distance between two points. This is best used for
#continuous numerical features where magnitude matters, but it is sensitive to scale so
#features should be standardized before use to prevent large-range dimensions from dominating.
def Euclidean_dist(x1, x2):
    res = 0
    for i in range(len(x1)):
        res += (x1[i] - x2[i])**2

    return math.sqrt(res)

#A helper that computes the inner product of two vectors, used by cosine_similarity.
def dot_product(x1, x2):
    dot_prod = 0
    for i in range(len(x1)):
        dot_prod += x1[i] * x2[i]
    return dot_prod


#Measures the angle between two vectors regardless of their magnitude. This is best used
#for text and sparse high-dimensional data where direction matters more than size. When
#using this in KNN, convert it to a distance with (1 - cosine_similarity) so that smaller
#values correspond to more similar points.
def cosine_similarity(x1, x2):
    dot_prod = 0
    x1_len = 0
    x2_len = 0
    dot_prod = dot_product(x1, x2)
    x1_len = math.sqrt(dot_product(x1, x1))
    x2_len = math.sqrt(dot_product(x2, x2))
    return (dot_prod)/(x1_len * x2_len)

#Measures distance as the sum of absolute differences along each dimension. It is more
#robust than Euclidean in high-dimensional spaces and less sensitive to outliers since it
#does not square the differences. Best used for discrete or grid-like data where diagonal
#movement through space is not natural.
def manhattan_distance(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += abs(x1[i] - x2[i])
    return dist