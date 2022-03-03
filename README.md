# Sentiment Analysis on Twitter Data
### Context :

This is the sentiment 140 dataset. It contains 1,600,000 tweets extracted using the twitter api .
* The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

It contains the following 6 fields :

       1. target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
       
       2. ids: The id of the tweet.
       
       3. date: the date of the tweet.
       
       4. flag: The query. If there is no query, then this value is NO_QUERY.
       
       5. user: the user that tweeted.
       
       6. text: the text of the tweet.

### Exploratory Data Analysis

![](https://github.com/ShivankUdayawal/Sentiment-Analysis-on-Twitter-Data/blob/main/Data%20Visualization/01.jpg)

![](https://github.com/ShivankUdayawal/Sentiment-Analysis-on-Twitter-Data/blob/main/Data%20Visualization/02.jpg)

## Word Cloud

![](https://github.com/ShivankUdayawal/Sentiment-Analysis-on-Twitter-Data/blob/main/Data%20Visualization/03.jpg)

![](https://github.com/ShivankUdayawal/Sentiment-Analysis-on-Twitter-Data/blob/main/Data%20Visualization/04.jpg)

## PreProcessing of Text

### Text Normalization

       Text Preprocessing is traditionally an important step for Natural Language Processing (NLP) tasks. 
       It transforms text into a more digestible form so that machine learning algorithms can perform better.
       
    **The Preprocessing steps taken are:**

   **1. Lower Casing:** Each text is converted to lowercase. Replacing URLs: Links starting with "http" or "https" or "www" are replaced by "URL".
   
   **2. Replacing Emojis:** Replace emojis by using a pre-defined dictionary containing emojis along with their meaning. (eg: ":)" to "EMOJIsmile")

   **3. Replacing Usernames:** Replace @Usernames with word "USER". (eg: "@Kaggle" to "USER")
   
   **4. Removing Non-Alphabets:** Replacing characters except Digits and Alphabets with a space.
   
   **5. Removing Consecutive letters:** 3 or more consecutive letters are replaced by 2 letters. (eg: "Heyyyy" to "Heyy")
   
   
## Confusion Matrix

![](https://github.com/ShivankUdayawal/Sentiment-Analysis-on-Twitter-Data/blob/main/Data%20Visualization/05.jpg)
