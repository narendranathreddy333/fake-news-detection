#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loading the data
import pandas as pd
df = pd.read_csv('Constraint_Train.csv')
#To check the columns present
df.columns
df.head(5)
df.info()
df.shape
#To get the count of missing values
missing_values_count = df.isnull().sum()
print(missing_values_count)
df.drop_duplicates(inplace=True)
df.shape
#Dropping duplicate rows and checking if any rows are removed


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt

def plot_class_distribution(data_frame, label_column, colors=None, figsize=(8, 6), fontsize=12):
    count = data_frame[label_column].value_counts()
    print("Showing the count of fake and real news")
    #Printing the number of real and fake news
    print(count)

    if colors is None:
        colors = ['skyblue', 'salmon']

    plt.figure(figsize=figsize)
    ax = sns.barplot(x=count.index, y=count, palette=colors)

    for index, value in enumerate(count):
        ax.text(index, value, str(value), ha='center', va='bottom', fontsize=fontsize)

    plt.xlabel("Classes of dependent variable", fontsize=fontsize+2)
    plt.ylabel("Count of each label", fontsize=fontsize+2)
    plt.title("Distribution of classes", fontsize=fontsize+4)
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
#Calling the function to draw the barplot
plot_class_distribution(df, 'label', figsize=(10, 8), fontsize=14)


# In[3]:


#Calculating the length of words in each sentence and showing some statistics regarding text length
text_lengths = df['tweet'].apply(lambda x: len(str(x)))

# Calculation of average text length in dataframe
average_length = text_lengths.mean()

# calculating the maximum and minimum lenth across the data frame 
max_length = text_lengths.max()
min_length = text_lengths.min()

#Printing the text lengths
print("Average Text Length:", average_length)
print("Maximum Text Length:", max_length)
print("Minimum Text Length:", min_length)


# In[4]:


import numpy as np
df['text_length'] = df['tweet'].apply(lambda x: len(str(x)))
plt.hist(df['text_length'], bins=np.logspace(0, np.log10(max_length), 20), edgecolor='black')
plt.xscale('log')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.title('Histogram of word count')
plt.show()


# In[5]:


#Checking the column names
df.columns


# In[6]:


import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
def preprocess_text(tweet):
    # Removing punctuations if any in the text
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # Converting all characters in the string to lowercase
    tweet = tweet.lower()
    
    #splitting inmto words
    words = tweet.split()
    
    #removing stopwords if any in the data
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    processed_tweet = ' '.join(words)
    
    return processed_tweet


# In[7]:


df['processed_tweet'] = df['tweet'].apply(preprocess_text)
df.head(5)
#we can see new column processed tweet added after performing preprocessing


# In[8]:


#Performing lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    # Splitting the text into individual words
    words = nltk.word_tokenize(text)
    # Performing lemmatization
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in words])

    return lemmatized_text
df['lemmatized_tweet'] = df['processed_tweet'].apply(lemmatize_text)


# In[9]:


df.head(5)
#we can observe that n ew column lemmatized_tweet is added and this is th columns we will be working now as we applied all our preprocessing steps


# In[10]:


text_lengths_after_preprocessing= df['lemmatized_tweet'].apply(lambda x: len(str(x)))
print("Average Text Length:",  text_lengths_after_preprocessing.mean())
print("Maximum Text Length:", text_lengths_after_preprocessing.max())
print("Minimum Text Length:", text_lengths_after_preprocessing.min())
#Printing lenth statistcis after preprocessing the data 
#We can observe that legnth has decreased after preprocessing as we have removed many characters


# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X=df['lemmatized_tweet'] #Selecting the independet variable
y=df['label'] #Selecting the target variable/dependent variable
from sklearn.model_selection import train_test_split
#Splitting data into training and test data sets using scikit learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)
train_size = len(X_train)
test_size = len(X_test)
print(f"Count of observations in training data set is {train_size}")
print(f"Count of observations in training data set is {test_size}")


# In[12]:


tfidf_vectors_train = vectorizer.fit_transform(X_train)
#print(tfidf_vectors_train)
tfidf_vectors_test = vectorizer.transform(X_test)
#print(tfidf_vectors_test)


# In[13]:


#Importing the require libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


# In[14]:


models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machines': SVC(),
    'Random Forests': RandomForestClassifier(),
    'Boosting': GradientBoostingClassifier(),
    'Naive Bayes': MultinomialNB()
}
#For storing the tesing and train accuracies
train_accuracies = {}
test_accuracies = {}


# In[15]:


def accuracy_calculation(model, train_data, train_labels, test_data, test_labels):
    model.fit(train_data, train_labels)
    train_pred = model.predict(train_data)
    test_pred = model.predict(test_data)
    train_accuracy = accuracy_score(train_labels, train_pred)
    test_accuracy = accuracy_score(test_labels, test_pred)
    return train_accuracy, test_accuracy
for model_name, model in models.items():
    train_acc, test_acc = accuracy_calculation(model, tfidf_vectors_train, y_train, tfidf_vectors_test, y_test)
    train_accuracies[model_name] = train_acc
    test_accuracies[model_name] = test_acc
   


# In[16]:


print(train_accuracies)
print(test_accuracies)


# In[17]:


import matplotlib.pyplot as plt

#Getting the model names and accuracies from test_accuracies dictionary
model_names = list(test_accuracies.keys())
testing_accuracies = list(test_accuracies.values())

#Creation of the plot
plt.figure(figsize=(10, 6))
plt.bar(model_names, testing_accuracies, color='skyblue')

#For adding accuracy values to 3 decimals at the top of the bar
for i, v in enumerate(testing_accuracies):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')

# Adding the names to axes and customizing the fontsize
plt.xlabel('Models',fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.title('Comparing Testing Accuracies',fontsize=14)


# In[ ]:




