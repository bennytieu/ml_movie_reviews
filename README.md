#Sentiment Analysis on Movie Reviews - Using Support Vector Machines with Linear Kernel 
This project is part of the course CZ4041 (Machine Learning) at NTU. Kaggle challenge: "Classify the sentiment of sentences from the Rotten Tomatoes dataset"

 https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews

"There's a thin line between likably old-fashioned and fuddy-duddy, and The Count of Monte Cristo ... never quite settles on either side."

The Rotten Tomatoes movie review dataset is a corpus of movie reviews used for sentiment analysis, originally collected by Pang and Lee [1]. In their work on sentiment treebanks, Socher et al. [2] used Amazon's Mechanical Turk to create fine-grained labels for all parsed phrases in the corpus. This competition presents a chance to benchmark your sentiment-analysis ideas on the Rotten Tomatoes dataset. You are asked to label phrases on a scale of five values: negative, somewhat negative, neutral, somewhat positive, positive. Obstacles like sentence negation, sarcasm, terseness, language ambiguity, and many others make this task very challenging.

---
This paper covers an implementation of a sentiment analysis on movie reviews. The implementation involved Support Vector Machine (SVM) with linear kernel, evaluating two different strategies - One-VS-One and One-VS-All, with different pre-processing methods.

The goal of this project was to implement an accurate prediction in terms of the domain for the Kaggle competition, \textit{"Sentiment Analysis on Movie Reviews"}. The best result generated using the following pre-processing methods: Only consider letters and lowercase symbols, stemming, normalized TF-IDF and removing 0.00004 \% of the least frequent words. One-VS-One generated a higher score than One-VS-All, with C set to 0.32. The final result scored the 30:th place on the leader board resulting in the top 3.48 \% of all the submissions.
(For more information, read the Project Report)

