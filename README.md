# Movie-genre-prediction
This project focuses on predicting the genre of a movie based on its plot summary or other textual information. Using natural language processing (NLP) techniques like TF-IDF (Term Frequency-Inverse Document Frequency) and the Support Vector Machine (SVM) algorithm, the goal is to classify movies into genres based on their plot descriptions.

<b>Project Overview</b>

Predicting movie genres based on plot summaries can help in automating genre classification, improving recommendations, and enhancing search functionality on movie platforms. In this project, we preprocess the movie plot data, transform the text using TF-IDF vectorization, and apply an SVM classifier to predict the genre.

<b>Data</b>

The dataset consists of movie information including the following features:

- Movie ID: A unique identifier for each movie.
- Title: The title of the movie.
- Plot Summary: A textual description of the movie's plot.
- Genres: The genres associated with the movie (e.g., Action, Comedy, Drama, etc.).

<b>Approach</b>

1. Data Preprocessing:
- Clean and preprocess the plot summaries by removing stopwords, punctuation, and lowercasing the text.
- Tokenize the text data for analysis.

2. Feature Engineering with TF-IDF:
- The textual data (plot summaries) is converted into numerical form using TF-IDF vectorization.
- The TF-IDF method transforms the text into a matrix of token counts, weighing each token based on its frequency across documents.

3. Model Training:
- The Support Vector Machine (SVM) algorithm is used for classification due to its effectiveness with high-dimensional data.
- The SVM model was trained using the TF-IDF features extracted from the plot summaries.
