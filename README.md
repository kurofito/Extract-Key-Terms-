# Extract Key Terms
This is a simple program that extracts key terms from an article using machine learning.

The program does the following.

1.  Reads a xml file containing news articles

2.  Goes through text preprocessing pipeline

    2a. Stop-words, digits, and punctations are deleted and lemmatization is applied

    2b. Part-of-speech tagging (a POS-tag) to mark nouns as keywords.

3.  Use TF-IDF(term frequency-inverted document frequency)
3a. Count the TF-IDF metric on all words  

4.  Display top five words according to the TF-IDF metric.

TF-IDF is a modified word probablity technique based on two  assumptions:

1. Frequent words have more weight in the document.
2. The smaller number of documents that contain the word, the more important the word is for the document that contains it.




