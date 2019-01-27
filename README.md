# Community-Detection-and-Friend-Recommendation-System-on-Facebook

Paper : http://i.stanford.edu/~julian/pdfs/nips2012.pdf

Data Set : http://snap.stanford.edu/data/ego-Facebook.html


Problem statement:This project deals with ego network of Facebook users and aims at addressing the below two problems: 

Automatically detect the user's tight communities: Given a single user and his social network, the goal is to identify his close circles, which are smaller and closer subset of his friend list. 

Building a friend recommendation system: Suggest users with choices to make friends with, based on the mutual friends from the circles identified as part of the first problem. 

Approach: 

Finding close communities on Facebook data by finding out all the possible cliques on the graph using few algorithms (https://www.nature.com/articles/srep30750)

Built a recommendation system based on the mutual friend count from people identified to be in the common communities.
