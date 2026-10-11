**Problem #1: Make sure that the discussion.md file**

A)The **goal in this problem will be to write a Naive Bayes classifier to determine the authorship of the**
**unlabeled works in the unlabeled folder**

B) **If you only had access to the prior probabilities, would you be more likely to guess that Kennedy or Johnson authored an unlabeled paper? Why?**
**Answer**: I would guess Johnson, as he authorized a larger share of the labeled papers (0.647, or about 64.7%). In comparison, Kennedy received 0.353, or 35.3%. The prior probability alone does not establish who wrote any particular unlabeled paper. 

C) **Do one author’s documents tend to be shorter in length? What are the most common words used by each author? Do either of the authors tend to start or end their works in a consistent way?**
**Answer**: 
Kennedy’s documents tend to be [shorter] than Johnson’s, based on their [mean and median] word counts. Kennedy’s most common words include [the,of,and,to,in,a,our,that,we,], while Johnson’s include [the,of,to,and,in,that,we,I,a, and is]. Most of these are common function words, although Kennedy frequently uses “our.” In contrast, Johnson frequently uses “I.”  Kennedy frequently begins by formally addressing specific officials or members of the audience, using phrases such as “Governor Stevenson,” “Vice President Johnson,” and “Mr. Speaker.” Johnson’s openings are more varied, although he also commonly addresses the audience directly with phrases such as “My fellow Americans,” “Friends and reporters,” and formal titles. Kennedy’s sampled endings frequently use inspirational, civic, or religious language. Johnson’s endings are less consistent, although some conclude with patriotic or religious language or an expression of thanks. Because only five documents from each author were displayed, these observations should be treated as patterns in the sample rather than rules that apply to every document.

Kennedy mean words: 2878.777777777778
Kennedy median words: 2291.0
Kennedy most common words:

the     6175
of      4395
and     3691
to      3258
in      2417
a       1849
our     1538
that    1436
we      1340
is      1262
Name: count, dtype: int64

Kennedy document openings:
Governor Stevenson, Senator Johnson, Mr. Butler...
Reverend Meza, Reverend Reck, I'm grateful for ...
I have welcomed this opportunity to address thi...
Vice President Johnson, Mr. Speaker, Mr. Chief ...
Mr. Speaker, Mr. Vice President, Members of the...
Kennedy document endings:
shall we not be weary. And then we shall prevai...
my ability preserve, protect, and defend the Co...
and your prayers, as I embark on this new and s...
knowing that here on earth God's work must trul...
may be worthy of the unlimited opportunities th...


Johnson mean words: 3619.9848484848485
Johnson median words: 3848.0
Johnson most common words:

the     14965
of       8325
to       8194
and      7463
in       5480
that     5177
we       4501
i        4119
a        3646
is       2965
Name: count, dtype: int64

Johnson document openings:
THE PRESIDENT. Good afternoon, ladies and gentl...
Paul Miller and my fellow Americans: Last Frida...
THE PRESIDENT. Friends and reporters--I hope yo...
President Hatcher, Governor Romney, Senators Mc...
My fellow Americans: I am about to sign into la...
Johnson document endings:
go any further today than I have gone. Mr. Pres...
name of America still sounds in this land and a...
to do and how we can help them. Thank you, Mr. ...
his genius to the full enrichment of his life. ...
who is the Father of us all. Thank you and good...

Prior probabilities (0=Kennedy, 1=Johnson): [0.35294118 0.64705882]
Likelihood matrix shape: (2, 24390)
Predictions (0=Kennedy, 1=Johnson): [1, 0, 0, 0, 0, 1, 0, 1, 0, 0]
My Naive Bayes — accuracy: 0.8 F1: 0.75
Scikit-learn — accuracy: 0.9 F1: 0.8888888888888888

D) **Look at nb.py vscode sheet for coding details**

E) **What are your two prior probability estimates? What is the shape of the matrix storing your likelihoods? What happens when you vary the smoothing hyperparameter alpha matrix storing your likelihoods?**
**answer**
The prior probabilities are approximately 0.353 for Kennedy and 0.647 for Johnson. The likelihood matrix has shape (2, 24390): one row per author and one column per vocabulary word. The smoothing parameter alpha assigns a positive likelihood to vocabulary words that did not appear in a particular author’s training documents. Increasing alpha makes the word probabilities more even; decreasing it gives greater weight to observed word frequencies. At alpha = 0, an absent word has likelihood zero for that author, causing log(0) in my prediction calculation. Therefore, the model uses a positive value of alpha.

F) **What are the predicted authors for each of the unlabeled works?**
Unlabeled work 1 -> Johnson
Unlabeled work 2 -> Kennedy
Unlabeled work 3 -> Kennedy
Unlabeled work 4 -> Kennedy
Unlabeled work 5 -> Kennedy
Unlabeled work 6 -> Johnson
Unlabeled work 7 -> Kennedy
Unlabeled work 8 -> Johnson
Unlabeled work 9 -> Kennedy
Unlabeled work 10 -> Kennedy
The model predicts that seven works were authored by Kennedy and three works were authored by Johnson.


**Problem #2:** How do your predictions in Problem 1 compare with the scikit-learn implementation of Naive Bayes?
#In this problem, we’ll repeat Problem 1, but we’ll use the **Naive Bayes classifier** from the **scikit-learn library.**
(A) Fill in the missing code to write the Naive Bayes classifier using scikit-learn. How do your predictions in compare with the scikit-learn implementation of Naive Bayes?

My Naive Bayes — accuracy: 0.8 F1: 0.75
Scikit-learn — accuracy: 0.9 F1: 0.8888888888888888
My Naive Bayes predictions: [1, 0, 0, 0, 0, 1, 0, 1, 0, 0]
Scikit-learn predictions: [1 0 0 0 0 1 0 1 1 0]

My Naive Bayes classifier achieved an accuracy of 0.80 and an F1 score of 0.75, while the scikit-learn implementation achieved an accuracy of 0.90 and an F1 score of approximately 0.89. My classifier correctly classified 8 of the 10 test documents, whereas the scikit-learn classifier correctly classified 9 of the 10 documents. Therefore, the scikit-learn implementation performed better, correctly classifying one additional document. The difference may result from variations in probability calculations, smoothing, numerical precision, or text preprocessing.


**Problem #3**
a) **Report the accuracy and the F1-score of both the Naive Bayes classifier you created and the one-off-the-shelf from scikit-learn**

My Naive Bayes — accuracy: 0.8 F1: 0.75
Scikit-learn — accuracy: 0.9 F1: 0.8888888888888888

"My Naive Bayes Classifier" level of accuracy is seen as 0.8 and for F1 score is 0.75. For " Scikit-learn Classifier", the level of accuracy is 0.9 and the F1 score is approximately 0.89. The scikit-learn classifier performed better, 10 percentage points higher in accuracy”: the scores are 90% and 80%., on both measures. It correctly classified 9 of the 10 test documents, while my naive bayes classifier correctly classified 8 of the 10 documents.


b) **Print a confusion matrix for both the Naive Bayes classifier you created and the one off-the-shelf from scikit-learn. Save the plot as conf.jpg (or other image format). For each classifier, what do you notice from the conf.**
Both classifiers correctly classified all five Kennedy documents, and neither incorrectly classified a Kennedy document as Johnson. My Naive Bayes classifier correctly classified three of the five Johnson documents but incorrectly classified two Johnson documents as Kennedy. Its confusion matrix was [[5, 0], [2, 3]].
The scikit-learn classifier correctly classified four of the five Johnson documents and incorrectly classified only one Johnson document as Kennedy. Its confusion matrix was [[5, 0], [1, 4]]. Therefore, scikit-learn correctly classified one additional Johnson document. This explains why its accuracy was 90% (compared to 80% for my classifier) and why its F1 score was also higher.
