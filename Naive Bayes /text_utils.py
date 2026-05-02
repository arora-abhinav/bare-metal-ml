import re
#This script contains all the utility functions that are common across the bernoulli naive bayes implementation 
#as well as the multinomial naive bayes implementation of a spam classifier
#Function to tokenize
def tokenize(text):
    text = text.lower()                   
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = text.split()                  
    return tokens

#Creating the vocablist using vocab_size words based on the highest occuring 
#word count
def build_vocab_list(items, x_train, train_tokens):
    counter_map = {}
    #Storing a counter of each word to keep the max vocab_size words occuring
    for i in range(len(x_train)):
        for token in train_tokens[i]:
            if token not in counter_map:
                counter_map[token] = 1
            else:
                counter_map[token] += 1
    
    #Sorts the map based on the frequency of the words occuring 
    sorted_map = sorted(counter_map.items(), key=lambda item:item[1], reverse=True)
    top_words = sorted_map[:items]

    final_map = {}
    counter = 0
    #counter is simply used to assign an index to the word so the index can be accesed
    #in O(1) time and so can the word be looked up in O(1) time
    for element in top_words:
        word, number = element
        final_map[word] = counter
        counter += 1
    
    return final_map


#Ratio of each class (for text, it is if the classes are spam or not)
def calculate_class_ratio(training_y):
    m = {}
    for i in range(len(training_y)):
        if training_y[i] not in m:
            m[training_y[i]] = 1
        else:
            m[training_y[i]] += 1
    
    return m

#Creating vectorized inputs for each training example in x_train
#Inout matrix takes in the original training examples to be vectorized
#vocab_size is the size of the vocab list
#train tokens are the tokens split using re
#vocab list is the vocab list built using the build vocab_list function
#count is a boolean that says to keep a counter of the word count or just a
#simple flag (counter for multinomial and flag for Bernoulli)
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