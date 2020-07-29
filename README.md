# Mood-Tunes
Music Mood Classification

Music Mood Classification

Abstract

Sentiment analysis enables natural language processing techniques to classify the emotional
content of data. Sentiment analysis has a wide range of use cases including, identifying
sentiments of online conversations, customer reviews, and feedback. In this research, we will
examine the use of sentiment analysis in identifying the emotional content in musical lyrics using
unsupervised learning techniques. The result of the experiment will show the use of sentiment
analysis in classifying music based on the sentiments of the lyrics.

1 Introduction

Musical Information Retrieval (MIR) has been a topic of research in the past and continues to be a
growing topic of automatically identifying the mood dimensions of music. Several systems have been
developed in the past to evaluate musical audio content. The Audio Mood Classification (AMC) and
Musical Information Retrieval Evaluation eXchange (MIREX) have evaluated MIR’s and concluded on
several identifying issues in their results. Evaluation of results criticize the fact that mood of a specific
song is subjective to social evaluation of music. Not only audio covey's emotional emotions of a song.
Researchers have combined both audio and lyrical evaluations to find the true sentiment of music. The
scope of this project is to evaluate the emotional sentiment of musical lyrics.
Sentiment analysis enables a computer to determine the emotions associated with a corpus. These
emotions can be clustered as either positive, negative, or neutral. To analyze the sentiments of a corpus
we apply natural language processing and machine learning techniques to score the text and categorize
it based on its emotion. There are several pieces of research on sentiment analysis. The supervised
methods use a training approach to train classifiers by feeding the classification model a large volume of
data with pre-tagged labels. In the circumstances where musical lyrics scrapped from does not contain a
prelabeled list of outcomes, the supervised method may not be feasible for this project.
We use the unsupervised method for our project. The unsupervised approach to sentiment analysis
does not require pre-tagged data to train the model about the outcome of the analysis. We upload the
data to our model to learn the structure of the dataset without any previous assumptions about the
outcome. Our first goal is to learn the dataset. We use Word2vec to learn the features of data and
produce word embeddings(vector representation of data). The advantage of using Word2vec, compared
to the bag of words(BOW) model is that Word2vec captures the distance between individual words. The
idea is that words(positive or negative) are surrounded by similar words. Meaning that a positive word
in music lyrics like ‘happy’ would be surrounded by other positive words like “content”. Vise versa, a
negative word like ‘boring’ would be surrounded by words like “didn’t”, and some neutral words. These
words form a cluster, based on the similarity of their surroundings.
Next, we use the results of our Word2vec model in a K means clustering algorithm. The K means
clustering algorithm takes in the necessary inputs and outputs the coordinates of the centroids of the
calculated clusters. We use the output of our K means cluster to determine which cluster is relatively
positive or negative. Next, we apply a sentiment score or weight to each cluster based on the cluster
they belong to identify how positive, negative the words are. We then use TFidf weighting to identify
how the uniqueness of each word for every sentence increases the level of positivity and negativity
associated with words. Finally, we evaluate the model’s performance.

2 Related Work

Music mood classification has already been researched using different text classification methods. In the
research “Identification of Mood Based on Lyric Text Mining”[10] investigates the importance of text
features in the identification of mood on 5 mood categories, by using lex summarization for the
improvement of results. The lex rank algorithm is used to gauge the importance of each sentence in
lyrics based on its neighboring sentence. Kalyani et. al use TF-IDF weighting to identify word frequency
and apply scores to words[10]. They also use K-means clustering to group similar objects together while
using cosine similarity measures for better results.
Maas et al., in the research, “Learning Word Vectors for Sentiment Analysis”[11] use a combination of
supervised and unsupervised learning-based approaches to vectorize text to capture sentiment
information and rich sentiment content. Our research will use a combination of word vectorization,
clustering, and TF-IDF weighting to extract the sentiments and mood of a song.

3 Objectives

Sentiment Analysis is one of the common applications of NLP methods, where you extract the emotions
of a lyrical music and categorize it according to emotions. To analyze, there are two main approaches:
one is supervised algorithm and other is unsupervised algorithm. For supervised algorithms, we need to
collect labeled data and teach algorithms which require manual work and it is often time consuming.
Where an unsupervised learning approach doesn’t require any predefined data. Since we don't have
labeled data, unsupervised learning is taken into consideration. Unsupervised learning model is where
we insert the data and want the model to learn the structure of data itself.
The overall objective is to identify the highest sentiment in each song's lyrics. In order to reach this final
goal, we will vectorize our text to make it understandable to our model, cluster our text to group words
by its sentiment, and understand the frequency of each term to identify its uniqueness to each
sentence to increase or reduce the level of positivity or negativity of the entire song.

4 Selected Dataset

Dataset is chosen from the kaggle website. The dataset contains 4 columns namely Artist, Song Name,
Link(link to website containing song lyrics) and Lyrics which is unmodified. It contains four columns and
57650 rows of data. It has 44824 songs which are sung by different artists. The artists are more than 160
in numbers and the lyrics are placed in a separate column called “Lyrics”, which is used for text mining
and sentiment analysis. This data is acquired from the website called LyricsFreak through scraping and
did basic modification to remove inconsistent data like non-english lyrics, extremely short and extremely
long lyrics and lyrics with non-ASCII symbols. However, for data preparation, we need to still modify to
obtain consistent data.

5 Proposed System

Our system extracts data from our comma separated values text files. We then preprocess our data and
run it through our Word2Vec model for word embedding and vectorization. Vectorized data enters the
K-means cluster model for word classification, then we assign sentiment score or weight to each cluster
based on the cluster they belong to identify how positive, negative the words are. We then use TF-IDF
weighting to identify how the uniqueness of each word for every sentence increases the level of
positivity and negativity associated with words. Finally, we evaluate the model’s performance.
Figure 1. Conceptual system architecture

6 Proposed development platforms

We use Python as a platform and Python’s NLTK libraries to develop our music mood classification
system.
● NLTK Regex Tokenizer: Regex Tokenizer splits the string using a regular expression.
● NLTK corpus stopwords: Stopwords refers to commonly used words such as “the”, “an”, “a”,
“in” etc which are programmed in such a way that the search engine ignores them while
searching or retrieving as a result of a search query. Natural Language Toolkit has stopword
corpus which has a list of stopwords containing 16 different languages.
● NLTK stem: Stemming is the method of reducing a word to its word stem which is appended to
suffixes and prefixes or to the roots of words known as lemmas. Stemming is important in
understanding the natural language (NLU) and in the processing of natural language (NLP).
● NLTK WordNet Lemmatizer: Wordnet is a massive, free and publicly accessible
English-language lexical database that aims to create organized semantic relationships between
words.Lemmatization is the mechanism by which a word is transformed to its base form. The
distinction between stemming and lemmatization is that lemmatization recognizes the context
and transforms the term into its meaningful base form, while stemming merely eliminates the
last few letters, frequently leading to incorrect definitions and errors in spelling.
● Gensim’s implementation of word2vec algorithm: Word2vec is one of the most common
techniques of using a two layer neural network to learn word embedding. The input is a text
corpus, and output is a set of vectors. Word embedding via word2vec will make the natural
language computer-readable, then mathematical operations on words can be further
implemented to detect their similarities.
● sklearn.cluster K-means: It is an unsupervised learning algorithm. The concept of K-means
clustering is to partition n observations of sample x into k clusters and each defined by the mean
of the samples in a cluster called centroids.
● TF-IDF Algorithm: TF-IDF algorithm consists of two algorithms that are multiplied. Term
Frequency(TF) refers to the number of times the term appears in a document to the total
number of terms in the document. Inverse document frequency is inverse of term frequency
where how unique a word is.

7 Implementation and Evaluation

Data Preprocessing:

We begin implementation of our solution by performing one of the important steps in any machine
learning project, data preprocessing. Data processing helps us reduce any possible bias in our model.
Preprocessing our text will help us distinguish positive and negative emotions. We import seventy songs
by Bruno Mars, and combine the song title with the lyrics. Each sentence is tokenized, words are
transformed to lower cases, and stopwords are removed. The tokens are stemmed and lemmatized to
reduce inflectional forms and sometimes derivationally related forms of a word to a common base
form[12].

Word2vec:

Word embedding is the process converting a word to a vector. Word embedding allows us to capture
the context(semantic and syntactic similarity, and relations with other words) in a document. One of the
most popular techniques to learn word embeddings is Word2Vec. Word2Vec uses shallow neural
networks to learn word embeddings. It allows us to have words with similar context occupy close spatial
positions. We use gensim’s Word2Vec algorithm with the Common Bag Of Words (CBOW) to vectorize
our words to be used in our K-means cluster.

K-mean Clustering:

K-means clustering is the most suitable clustering technique for our problem. We initialize our clustering
model to two seeked clusters which outputs calculated clustered centroids. All points are assigned to
their closest centroid. The positive words are associated with centroid zero, and the negative words are
associated with centroid one. We use sklearn’s k-means clustering algorithm to implement our cluster
with a maximum iteration of a thousand and fifty repeated starting points. The result of our cluster is
two centroids. We use gensim’s most similar method to check for clusters that are positive and negative
based on cosine distance.

Term Frequency - Inverse Document Frequency Weighting:

Term Frequency - Inverse Document Frequency Weighting (TF-IDF) is one of the most common
computations used in text processing. The goal of this statistical model is to determine the importance
of a word in a corpus. Term frequency (TF) is how often a word appears in a document, divided by how
many words are there in a document. Figure 2a. shows the TF equation, “i” indicates a term in
document “j”. “ni, j is the number of occurrences of term ti in document dj. The number of occurrences
of all the terms in the document is indicated by the summarization in the denominator.

Figure 2a.

The importance of the word is determined by the inverse document frequency. This is measured by
determining and comparing the terms commonality of occurence in other sentences. Figure 2b. shows
the IDF equation, where “|D|'' indicates the number of documents (in our case, the number of
sentences in the corpus, and the denominator indicates the number of documents( sentences)
containing the term ti.

Figure 2b.

Finally, we multiply the TF by IDF to get the weighted score of each term in the document and use the
average of that scores as a threshold to indicate the most important terms in the corpus.


8 Experimental Results

Sentiment Prediction:

The sentiment prediction of each song is determined by using the TF-IDF weighting and the thresholds.
We select the terms with TF-IDF weights greater or equal to the threshold and sum each terms’s weight
grouped by each cluster. We then compare each summed score and choose the cluster with the highest
summed weight as our final sentiment for the song. If the sum of negative(“1”) clusters are greater, the
sentiment score of the song is negative, if the sum of positive(“0”) cluster weights is higher, then the
song’s sentiment is positive. If the summed weights are equal to each other, then the sentiment is
neutral. Some songs don’t have enough terms after we eliminate the weights based on the threshold, so
they are defaulted to neutral. Figure 4 shows the results of our analysis.

Figure 3.

Model Score:

The goal for our model’s performance evaluation is to use a recall, precision, and F-score. Our
data has no validation dataset to evaluate our model by. This causes a problem that we are still
evaluating.

Appendix

References

[1] https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483
[2] https://www.yourdictionary.com/void
[3] https://www.lexalytics.com/technology/sentiment-analysis#machine-learning-sentiment
[4] https://www.kaggle.com/devisangeetha/sing-a-song-lyrics-is-here
[5] https://www.kaggle.com/mousehead/songlyrics
[6] https://gist.github.com/aparrish/2f562e3737544cf29aaf1af30362f469
[7] https://github.com/rafaljanwojcik/Unsupervised-Sentiment-Analysis
[8] https://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/
[9] Hu, Xiao & Downie, J. & Ehmann, Andreas. (2009). Lyric Text Mining in Music Mood
Classification.. Proc. Int. Soc. Music Information Retrieval Conf.. 411-416.
[10] Kalyani, V, and Dr Y Sangeetha. “Identification of Mood Based on Lyric Text Mining” 6,
no. 4 (2017): 3.
[11] Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (n.d.). Learning
Word Vectors for Sentiment Analysis. 9.
[12] https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
https://www.sciencedirect.com/topics/computer-science/inverse-document-frequency
