import pandas as pd
from textblob import TextBlob

# Load dataset
df = pd.read_csv(
    r"C:\Users\SOHAM PANDAV\OneDrive\Desktop\Amazon_AI_BI_Platform\data\amazon.csv"
)

# Sentiment function
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Create sentiment column
df["sentiment"] = df["review_content"].apply(get_sentiment)

# Save file
output_file = r"C:\Users\SOHAM PANDAV\OneDrive\Desktop\Amazon_AI_BI_Platform\reports\sentiment_analysis.csv"

df.to_csv(output_file, index=False)

print("Sentiment file saved successfully!")
print(output_file)

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())