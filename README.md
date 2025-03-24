# News Summarization and Recommendation System

## Overview
This project is a **News Summarization and Recommendation System** built using **FastAPI**. It fetches news articles from multiple **RSS feeds**, generates **summaries** using a Hugging Face **transformers model**, and provides **article recommendations** based on **content similarity**.

## Features
- **Fetch Latest News**: Collects news articles from RSS feeds.
- **Summarize Articles**: Generates concise summaries using the **BART transformer model**.
- **Recommend Similar Articles**: Uses **TF-IDF** and **cosine similarity** to suggest similar articles.
- **FastAPI Backend**: Provides API endpoints for accessing news, summaries, and recommendations.
- **Docker Support**: Easily deployable using Docker.

## Installation
### Prerequisites
- Python 3.9+
- Pip
- Git

### Clone the Repository
```sh
git clone https://github.com/supritha-palguna/News-Summarization-Recommendation-System.git
cd News-Summarization-Recommendation-System
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Running the Application
### Start FastAPI Server
```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fetch-news/` | `GET` | Fetches the latest news articles from RSS feeds |
| `/summarize/{article_id}` | `GET` | Summarizes a specific article based on its ID |
| `/recommend/{article_id}` | `GET` | Recommends similar articles based on content similarity |

## Running with Docker
### Build Docker Image
```sh
docker build -t news-summarization .
```

### Run Docker Container
```sh
docker run -p 8000:8000 news-summarization
```

## Future Enhancements
- Implement a **frontend UI** using Streamlit or React.
- Integrate a **database (PostgreSQL/MongoDB)** for better article storage.
- Improve article recommendations using **advanced NLP techniques**.

## License
This project is open-source under the **MIT License**.



---
### Contributors
- [Supritha Palguna](https://github.com/supritha-palguna)



