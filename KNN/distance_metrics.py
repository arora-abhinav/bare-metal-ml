import math
# Measures the straight line geometric distance between two points in space.
# This is bst used for continuous numerical features where magnitude matters
# Also sensitive to scale so standardization should be applied before use
def Euclidean_dist(x1, x2):
    res = 0
    for i in range(len(x1)):
        res += (x1[i] - x2[i])**2
    
    return math.sqrt(res)

def dot_product(x1, x2):
    dot_prod = 0
    for i in range(len(x1)):
        dot_prod += x1[i] * x2[i]
    return dot_prod


# Measures the angle between two vectors regardless of their magnitude
# Best used for text and sparse high dimensional data where direction matters more than size
# Convert to cosine distance (1 - cosine similarity) when used in KNN
def cosine_similarity(x1, x2):
    dot_prod = 0
    x1_len = 0
    x2_len = 0
    dot_prod = dot_product(x1, x2)
    x1_len = math.sqrt(dot_product(x1, x1))
    x2_len = math.sqrt(dot_product(x2, x2))
    return (dot_prod)/(x1_len * x2_len)

# Measures distance as the sum of absolute differences along each dimension.
# More robust than Euclidean in high dimensional spaces and less sensitive to outliers
# Best used for discrete or grid like data where diagonal movement isn't natural
def manhattan_distance(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += abs(x1[i] - x2[i])
    return dist