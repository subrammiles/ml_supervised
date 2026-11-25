"""Naive Bayes for text: a simple, fast spam filter.

1. Turn text into numbers: count words (bag of words) or TF-IDF weights.
2. Naive Bayes applies Bayes' theorem, assuming words are independent given the
   class ("naive"). Wrong in reality, but works surprisingly well for text.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

spam = [
    "win a free prize now", "claim your free cash reward", "urgent you won a lottery",
    "free entry to win cash", "limited offer buy now cheap", "you are selected for a prize",
    "earn money fast from home", "click here to claim free gift", "cheap loans approved today",
    "congratulations you won free tickets", "exclusive deal buy cheap now", "win money now click",
    "free vacation claim now", "lowest price guaranteed buy today", "urgent claim your reward",
]
ham = [
    "are we still meeting for lunch", "please review the attached report", "see you at the gym",
    "can you send me the slides", "dinner at my place tonight", "the meeting moved to 3pm",
    "happy birthday hope you have fun", "call me when you get home", "project deadline is friday",
    "thanks for your help yesterday", "let us catch up this weekend", "notes from today's class",
    "did you finish the assignment", "the train is running late", "lunch tomorrow at noon",
]
texts = spam + ham
labels = np.array([1] * len(spam) + [0] * len(ham))  # 1 = spam

# What the vectorizer does
vec = CountVectorizer()
counts = vec.fit_transform(["free prize now", "free lunch now now"])
print("Vocabulary:", vec.get_feature_names_out().tolist())
print("Word counts:\n", counts.toarray())

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=0, stratify=labels)

# Pipeline: raw text in, prediction out
nb = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))  # alpha = smoothing
nb.fit(X_train, y_train)
print(f"\nNaive Bayes test accuracy: {nb.score(X_test, y_test):.3f}")
print(classification_report(y_test, nb.predict(X_test), target_names=["ham", "spam"], zero_division=0))

# Words that most strongly indicate spam
model = nb[-1]
words = nb[0].get_feature_names_out()
spam_vs_ham = model.feature_log_prob_[1] - model.feature_log_prob_[0]
print("Most 'spammy' words:", words[np.argsort(spam_vs_ham)[::-1][:8]].tolist())

new_messages = ["claim your free prize today", "can we move the meeting to friday",
                "cheap tickets click now"]
for msg, p in zip(new_messages, nb.predict_proba(new_messages)[:, 1]):
    print(f"P(spam)={p:.2f}  {msg}")

# Alternative: TF-IDF features + logistic regression, a strong text baseline
tfidf_lr = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression())
tfidf_lr.fit(X_train, y_train)
print(f"\nTF-IDF + logistic regression test accuracy: {tfidf_lr.score(X_test, y_test):.3f}")

# Smoothing (alpha) avoids zero probabilities for words never seen with a class.
# Other Naive Bayes types: GaussianNB (numeric features), BernoulliNB (word present / absent).
# With this tiny dataset results are noisy; real spam filters train on thousands of messages.
