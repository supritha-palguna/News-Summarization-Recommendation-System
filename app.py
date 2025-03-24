from fastapi import FastAPI
import feedparser
import requests
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI()

# List of RSS Feeds for news collection
RSS_FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
]

# Load Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Store news articles (temporary storage)
articles_db = []

def fetch_news_articles():
    """Fetch the latest news articles from multiple RSS feeds."""
    articles = []
    
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        
        for entry in feed.entries[:5]:  # Get top 5 articles per feed
            article = {
                "id": len(articles_db) + len(articles),  # Assign unique ID
                "title": entry.title,
                "link": entry.link,
                "summary": entry.summary if hasattr(entry, "summary") else "No summary available",
            }
            articles.append(article)

    articles_db.extend(articles)  # Store articles in memory
    return articles

@app.get("/fetch-news/")
def get_news():
    """Returns the latest news articles."""
    articles = fetch_news_articles()
    return {"articles": articles}

@app.get("/summarize/{article_id}")
def summarize_news(article_id: int):
    """Summarizes a specific news article."""
    article = next((a for a in articles_db if a["id"] == article_id), None)
    
    if not article:
        return {"error": "Article not found!"}

    summary = summarizer(article["summary"], max_length=150, min_length=50, do_sample=False)
    return {"article_title": article["title"], "summary": summary[0]["summary_text"]}

def compute_recommendations():
    """Compute similarity scores for recommending similar articles."""
    if len(articles_db) < 2:
        return []  # Not enough articles

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([a["summary"] for a in articles_db])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    recommendations = {}
    for idx, article in enumerate(articles_db):
        similar_indices = similarity_matrix[idx].argsort()[-3:-1][::-1]
        recommendations[article["id"]] = [articles_db[i]["id"] for i in similar_indices]

    return recommendations

@app.get("/recommend/{article_id}")
def recommend_articles(article_id: int):
    """Recommends similar articles based on content similarity."""
    recommendations = compute_recommendations()
    
    if article_id not in recommendations:
        return {"error": "Article not found or not enough data for recommendations."}

    recommended_articles = [
        next(a for a in articles_db if a["id"] == rec_id) for rec_id in recommendations[article_id]
    ]
    
    return {"recommended_articles": [{"id": a["id"], "title": a["title"], "link": a["link"]} for a in recommended_articles]}
