## Project 3: Using Natural Language Processing (NLP) and Classification Models to predict suicide-related content in Reddit posts 

### Introduction & Problem Statement

We from Resilience.AI, leader in mental health machine learning diagonstic tecnhologies. We have entered into a partership with Reddit to improve the mental health support offered to users on Reddit.

Reddit has been a favored platform as a place for mental health community support. There are many subreddits that cater to various mental health issues, and Reddit has implemented a report feature that allows other Reddit users to flag posts that exhibit suicide or self-harm idealation. [[1]](https://www.fastcompany.com/90472072/reddit-will-now-automatically-connect-potentially-suicidal-users-with-a-hotline) Users can flag a post using this function, and Reddit will send a private message to the author of the post, links to resources where they can get mental health support.

![image.png](attachment:image.png)

However this is solution relies on the altruism of fellow Reddit users, which in reality, with any online community there would be malicious actors. There have been complaints of abuse of this button, where the report button was used to harass instead of help. [[2]](https://www.reddit.com/r/ModSupport/comments/p3kxe0/abuse_of_the_suicide_or_self_harm_report_button/)

Hence we aim to create a classification model, that we can use to filter out suicide content reports that clearly are irrelevant, and identify those that are clearly legitimate uses of the report function. We would be aiming to optimize for specificity, to reduce the likelihood of false negatives. A further application of this model would be to identify borderline cases which can be reviewed by a dedicated team.

### Description

**Data Collection**
We have selected r/SuicideWatch and r/MentalHealth and we will scrape posts using the PushShift API from these subreddits to train the classification model with. As r/SuicideWatch is a subreddit which specifically caters support for people with suicidal thoughts, and r/MentalHealth is a more general subreddit that supports Mental Health issues in general, these would be ideal for the training of this model.

**Data Library**

|Columns|Type|Description|
| --- | --- | --- |
|'is_suicidewatch'| bool | True if post is from r/SuicideWatch, False if from r/MentalHealth |
|'author'| object | Reddit username of author of post |
|'num_comments'| int64 | Number of comments |
|'score'| int64 | Net post score (an upvote is +1, downvote is -1) |
|'upvote_ratio'| float64 | Ratio of upvotes to downvotes |
|'link_flair_text'| object | Original flair of post |
|'flair_group'| object | Grouping of flair assigned for purposes of this project |
|'repost' | bool | True if identical post has been posted in both subreddits |
|'dual_user' | bool | True if user posted in both subreddits |
|'alltext'| object | Combination of 'title' and 'selftext' |
|'clean_alltext'| object | Cleaned version of 'alltext' |
|'word_count'| int | Word count of 'clean_alltext' |

**EDA**
EDA would focus on word count by flair category, and frequency of keywords in each subreddit.

**Classifier Modelling**
We will start with a Logistic regression model, Multinomial Naive Bayes, and then a Random forest model.

Firstly we would use Logistic Regression because it suits data that has lower bias but higher variance compared to Naive Bayes, and based on our understanding of our data, it has high variance given that there is quite a number of overlap between the top-20 words, and the df threshold had to be set high for significant results. Also we observed that there people who post in r/MentalHealth that are having suicidal thoughts.

Next we would use Naive Bayes as a comparison, using the multinomial version because the variable values can be integers of more than 1.

We will use Random Forest last, as Logistic Regression and Mulinomial Naive Bayes makes the assumption that variables are independent, but it is not a requirement for Random Forest, but Random Forest is prone to overfitting and takes longer to run. 

**Error Analysis & Model Tuning**
We would do a False Postive error analysis on the results model 3 by flair category, of particular interest would be the false positive rate of the `Positive` category of flairs which we would expect low false positives as these are expected to be of happier sentiment.

We will look at the posts wrongly classified as r/SuicideWatch in the `Positive` category, and compare the keywords with True Positive in r/SuicideWatch. For the closely overlapping keywords, we will add this to the stopwords list and rerun the classification model.

### Conclusion
We would recommend TfidfVectorizer with MultinomialNB as the better model for building a classification model for the purposes of building a filter to filter out incorrect or abusive flags of posts that contain thoughts of suicide.

Based on our false positive analysis, we tuned out two stop words and successfull increased specificity as intended. However, this came at the cost of other metrics. I believe that the 4 percentage point improvement is noteworthy given that the data used for stop words tuning was only 3% of the total false positive results.

### Recommendations & Next Steps
Further studies could involve Topic Modelling and Sentiment Analysis to group posts by topic to further improve tuning of the model.
