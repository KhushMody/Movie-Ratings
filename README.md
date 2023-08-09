# Movie-Ratings
Predicting the movie ratings based on reviews. Making use of machine learning and deep learning models

## Dataset Generation
- We read the data file using read_csv
- Keep reviews and ratings
- Convert ratings in the range of 1,2,3
- Then we take 20000 example from each class and keep them for the new dataset
## Word Embeddings
- We load the pretrained GoogleNews-vectors-negative300.bin.gz from the same
folder as the present working directory(pwd). And see its working on three
examples:-
  - Excellent and outstanding similarity = 0.5567486
  - dog and cat similarity = 0.76094574
  - man is most similar = [('woman', 0.7664011716842651)]
- Now we train a word2vec using our own dataset.We set the embedding size to be
300 and the window size to be 13. And consider a minimum word count of 9. Our
results on the previous 3 examples are:-
  - excellent and outstanding similarity = 0.65382564
  - dog and cat similarity = 0.46736932
  - man is most similar = [('woman', 0.6469601988792419)]
- Overall the pre-trained GoogleNews-vectors-negative300.bin.gz performs better than
the model I have trained one of the reasons is that for the examples listed above our
model would not have that many instances of it as compared to the google one.
## Simple Models
- Initially I do some pre-processing on the data like
  - Convert the review into lowercase using the lower() function
  - Removed the http and url links using regex
  - Made use of contractions module to make use of contractions
  - Remove stop words
  - Perform lemmatization
  - Perform tokenization using sent_tokenize
- Then I get the TFIDF features
- Then using word2vec I take the average of all the words and represent it in a list of
size 300
- Perceptron using word2vec
  - Accuracy - 59%
- Perceptron using TFIDF
  - Accuracy - 63%
- SVM using word2vec
  - Accuracy - 68%
- SVM using TFIDF
  - Accuracy - 73%
- TFIDF performs slightly better than word2vec as seen by the results above.
According to me since TFIDF gives more importance to important words while
word2vec takes the average of the words and hence I believe that TFIDF performs
better
## Feed Forward Neural Networks
- Using pytorch we created a multilayer perceptron model
- The model has 2 hidden layers of dimension 100 and 10 respectively
- For part a:-
  - Use average word2vec word embeddings
  - Have used batch size 32
  - Using relu as non-linearity function
  - And using softmax as the final activation function
  - Loss = CrossEntropyLoss
  - Optimizer = Adam
  - Learning rate = 0.01
  - Trained it for 7 epochs - the loss I got is 0.728139
  - Accuarcy for FNN with AVERAGE word2vec embedding is =
0.6658333333333334
- For part b:-
  - Use the the 1st 10 words as embedding using word2vec
  - Have used batch size 32
  - Using relu as non-linearity function
  - And using softmax as the final activation function
  - Loss = CrossEntropyLoss
  - Optimizer = Adam
  - Learning rate = 0.01
  - Trained it for 10 epochs - the loss I got is 0.526469
  - Accuarcy for FNN with 1st 10 word2vec embedding is = 0.5495833333333333
- The accuracy obtained by the FNN are not as good as the ones obtained from the simple
models since they are shallow networks and if we made slightly deeper model I believe
we can get better accuracies. But the average word embeddings outperform the 1st 10
word embeddings since they take the entire review into consideration. Also they perform
better than the perceptron for word2vec and TFIDF and is almost equal to the svm model
using word2vec embeddings
## Recurrent Neural Networks
- Use Pytorch nn module for using the RNN,GRU and LSTM modules
- We use word embedding of size 20 and we pad the shorter reviews and truncate the
longer reviews
- For part a:-
  - We use the RNN module
  - Split dataset into training and testing with 80:20 split
  - Batch size used is 256
  - Train_loader size for x,y are torch.Size([256, 20, 300]) torch.Size([256])
  - The model makes use of a RNN module followed by a Linear model
  - With nonlinearity as relu
  - The hidden layer size is 20
  - Loss = CrossEntropyLoss
  - Optimizer = Adam
  - Learning rate = 0.01
  - Trained it for 7 epochs - the loss I got is 0.817666
  - Accuarcy for RNN is = 0.60225
- The accuracy of RNN is better than the FNN which takes the 1st 10 words as embedding
since I feel that here we are taking 20 words thereby getting more context and also RNN
is a more complex model as compared to the FNN. But the accuracy of FNN which has
average of the entire review as word embeddings is better as compared to the RNN
since RNN are not taking the entire review into consideration and only the 1st 20 words.
- For part b:-
  - We use the GRU module
  - Split dataset into training and testing with 80:20 split
  - Batch size used is 128
  - Train_loader size for x,y are torch.Size([128, 20, 300]) torch.Size([128])
  - The model makes use of a GRU module followed by a Linear model
  - With nonlinearity as relu
  - The hidden layer size is 20
  - Loss = CrossEntropyLoss
  - Optimizer = Adam
  - Learning rate = 0.001
  - Trained it for 7 epochs - the loss I got is 0.738251
  - Accuarcy for GRU is = 0.6520833333333333
- For part c:-
  - We use the LSTM module
  - Split dataset into training and testing with 80:20 split
  - Batch size used is 128
  - Train_loader size for x,y are torch.Size([128, 20, 300]) torch.Size([128])
  - The model makes use of a LSTM module followed by a Linear model
  - With nonlinearity as relu
  - The hidden layer size is 20
  - Loss = CrossEntropyLoss
  - Optimizer = Adam
  - Learning rate = 0.001
  - Trained it for 6 epochs - the loss I got is 0.761786
  - Accuarcy for LSTM is = 0.6438333333333334
- GRU and LSTM perform considerably better than RNN since they remember long term
dependencies which helps them remember the important words that had come before
and had a significant impact unlike the RNN’s. GRU and LSTM have pretty much the
same performance with them being 65 and 64 respectively.
