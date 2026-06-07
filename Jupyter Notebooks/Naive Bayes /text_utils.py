import re

#This script contains utility functions shared across the Bernoulli and Multinomial
#Naive Bayes spam classifiers.

#Lowercasing and stripping non-alphabetic characters ensures that "Free" and "free"
#are treated as the same token and that punctuation doesn't create false unique words.
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens

#Only the top vocab_size words by frequency are kept. This reduces the feature space
#and filters out rare words that appear too infrequently to carry predictive signal.
#Each word is assigned an integer index so it can be looked up in O(1) time.
def build_vocab_list(items, x_train, train_tokens):
    counter_map = {}
    for i in range(len(x_train)):
        for token in train_tokens[i]:
            if token not in counter_map:
                counter_map[token] = 1
            else:
                counter_map[token] += 1

    sorted_map = sorted(counter_map.items(), key=lambda item:item[1], reverse=True)
    top_words = sorted_map[:items]

    final_map = {}
    counter = 0
    for element in top_words:
        word, number = element
        final_map[word] = counter
        counter += 1

    return final_map


def calculate_class_ratio(training_y):
    m = {}
    for i in range(len(training_y)):
        if training_y[i] not in m:
            m[training_y[i]] = 1
        else:
            m[training_y[i]] += 1

    return m

#The count flag controls whether each entry stores a binary presence flag (Bernoulli)
#or the actual word count (Multinomial). Both models share this function since the only
#difference is how they treat word frequency.
def vectorize(input_matrix, vocab_size, tokens, vocab_list, count):
    vectorized_inputs = [[0] * vocab_size for _ in range(len(input_matrix))]
    for i in range(len(input_matrix)):
        for token in tokens[i]:
            if token in vocab_list:
                if not count:
                    vectorized_inputs[i][vocab_list[token]] = 1
                else:
                    vectorized_inputs[i][vocab_list[token]] += 1

    return vectorized_inputs