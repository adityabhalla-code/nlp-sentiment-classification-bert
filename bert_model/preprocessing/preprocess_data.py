from pathlib import Path
import sys

file = Path(__file__).resolve()
parent , root = file.parent , file.parents[1]
sys.path.append(str(root))

from bert_model.config.core import config
from nltk.corpus import stopwords
import pandas as pd
import string
import os
import re

# Function to remove HTML/XML tags from text
def remove_html(text):
    # Create a regular expression for finding HTML tags
    html_pattern = re.compile('<.*?>')
    # Use the sub() method to replace HTML tags with an empty string
    return html_pattern.sub('', text)

# Function to remove punctuation
def remove_punctuation(text):
    # Use the string.punctuation list and replace each punctuation with an empty string
    return text.translate(str.maketrans('', '', string.punctuation))

# Function to remove stopwords
def remove_stopwords(text):
    # Tokenize the text into words
    words = text.split()
    # Filter out stopwords from the tokens
    filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
    # Join the filtered words back into a single string
    return ' '.join(filtered_words)

# Combined function to preprocess text reviews
def preprocess_text(text):
    text = remove_html(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text


# # read the dataset
data = pd.read_csv("dataset/Reviews.csv")
print(f"Total number of records in the dataset--{data.shape[0]}")
# drop the not required variables
data.drop(['Id','HelpfulnessNumerator','HelpfulnessDenominator'],axis=1,inplace=True)

# Add a sentiment column to the dataset using Score variable
data['Sentiment'] = data['Score'].apply(lambda x:'Positive' if x>3 else 'Negative')

# drop the duplicate records
data.drop_duplicates(inplace=True)

# change time to proper format
data['Time'] = pd.to_datetime(data['Time'],unit='s')


# sample the data set for 60K records
data_sample = data.sample(60000)

# preprocess the data for the sample
data_sample['preprocessed_text'] = data_sample['Text'].apply(preprocess_text)

# convert sentiment to numerical
data_sample['sentiment_score'] = data_sample['Sentiment'].apply(lambda x: 1 if 'Positive' else 0)

# save the preprocessed data
file_name = 'preprocessed_'+filename
save_path = os.path.join(dataset_dir,file_name)

# save the preprocessed file
data_sample.to_csv(save_path)
print(f"Preprocessing done--datset saved to {save_path}\nDataset name--{file_name}")
