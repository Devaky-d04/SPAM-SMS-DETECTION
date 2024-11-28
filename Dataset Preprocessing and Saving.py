# **SMS Spam Classification Dataset Preprocessing and Saving**

# Load dataset
file_path = r"/content/SMSSpamCollection"
data = pd.read_csv(file_path, sep='\t', header=None, names=["label", "message"])

# Save the dataset as CSV
data.to_csv("spam.csv", index=False, encoding='utf-8')

# Print the first few rows of the dataset
print("Dataset processed and saved as 'spam.csv'.")
print(data.head())

# **Dataset Preprocessing and Label Encoding**
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data.rename(columns={"v1": "label", "v2": "message"})
data = data[['label', 'message']]
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
print(data['label'].value_counts())
