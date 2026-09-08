# 🎬 Telugu Movie Search & Recommendation Engine

A simple and intelligent **Telugu movie search and recommendation system** that helps users find movies based on what they are looking for — even when they don't search using the exact movie title.

Instead of depending only on traditional keyword matching, this project uses **Natural Language Processing (NLP), Sentence-BERT embeddings, and FAISS vector search** to understand the meaning behind movie-related information and find movies that are semantically similar.

The project also includes a **Streamlit web interface** where users can search for movies and get recommendations in an easy-to-use way.

---

## 🚀 What Problem Does This Solve?

Finding a movie can be surprisingly difficult when you don't remember its exact name.

For example, a user might search for:

> "A movie about a police officer fighting criminals"

A traditional keyword-based search may struggle because the movie description might use completely different words.

This project tries to solve that problem by converting movie information into **numerical embeddings** that capture the meaning of the text.

The system can then compare the user's search with thousands of movies and return the most relevant results.

---

## ✨ Main Features

* 🔎 Semantic movie search
* 🎬 Telugu movie recommendations
* 🧠 NLP-based text understanding
* 🤗 Sentence-BERT embeddings
* ⚡ Fast similarity search using FAISS
* 📊 Movie information such as genre, actors, director, description and rating
* 🖥️ Interactive Streamlit interface
* 🎯 Recommendations based on movie similarity
* 📌 Handles natural-language search queries

---

## 🧠 How the Project Works

The complete pipeline looks like this:

```text
                    Movie Dataset
                         │
                         ▼
                Data Cleaning
                         │
                         ▼
              Text Preprocessing
                         │
                         ▼
          Combine Important Movie Information
                         │
                         ▼
              Sentence-BERT (SBERT)
                         │
                         ▼
              Movie Embeddings
                         │
                         ▼
                   FAISS Index
                         │
              ┌──────────┴──────────┐
              │                     │
        User Search            Movie Search
              │                     │
              ▼                     ▼
        SBERT Embedding       Existing Embedding
              │                     │
              └──────────┬──────────┘
                         ▼
                 Similarity Search
                         │
                         ▼
              Top Relevant Movies
                         │
                         ▼
                  Streamlit UI
```

---

# 🔍 Search Engine

The main idea behind the search engine is **semantic similarity**.

Suppose the database contains a movie description like:

> "A young man gets involved in a fight against a powerful criminal gang."

The user doesn't necessarily need to search for those exact words.

They could search:

> "movie where a guy fights a dangerous gang"

The system converts both the movie information and the user query into embeddings and compares them.

This allows the search engine to look at the **meaning of the query**, rather than simply checking whether the same words exist.

---

# 🤖 Why Sentence-BERT?

For this project, I used **Sentence-BERT (SBERT)** to convert movie-related text into numerical vectors.

Each movie is represented by an embedding such as:

```text
Movie → [0.021, -0.134, 0.562, ...]
```

These vectors contain information about the semantic meaning of the movie text.

The project uses a **384-dimensional embedding**, which gives us a compact representation that can be efficiently searched.

---

# ⚡ Why FAISS?

After generating embeddings, we need a fast way to search through them.

If we have thousands of movies, comparing a user's query against every movie one by one can become inefficient.

That's where **FAISS (Facebook AI Similarity Search)** comes in.

FAISS stores the movie embeddings in an index and allows us to quickly retrieve the most similar vectors.

In our project, the FAISS index contains thousands of movie embeddings.

```text
User Query
     │
     ▼
SBERT Embedding
     │
     ▼
FAISS Search
     │
     ▼
Top-K Similar Movies
```

This makes the search process much faster and more scalable.

---

# 🎥 Movie Recommendation

The recommendation system works in a similar way.

When a user searches for a movie or selects a movie, the system looks at its embedding and finds other movies whose embeddings are closest to it.

For example:

```text
User selects:
"Pushpa: The Rise"

             ↓

Movie Embedding

             ↓

FAISS Similarity Search

             ↓

Recommended Movies

1. Movie A
2. Movie B
3. Movie C
4. Movie D
5. Movie E
```

The recommendations are based on the similarity between the movie representations.

---

# 🧹 Data Processing

Before generating embeddings, the movie dataset needs to be cleaned.

The dataset contains information such as:

* Movie name
* Year
* Genre
* Cast
* Director
* Description
* Rating
* Poster

One of the important challenges was cleaning the **cast/actors information**.

Movie datasets don't always store cast information in a clean format. Sometimes multiple actor names can be merged together or represented inconsistently.

So the data preprocessing stage was important before creating the final text representation.

---

# 📝 Creating the Movie Text

Instead of generating an embedding using only the movie title, multiple useful fields are combined.

For example:

```text
Movie Name
+
Genre
+
Actors
+
Director
+
Description
```

These fields are combined to create a richer representation of the movie.

Example:

```text
Movie:
Pushpa: The Rise

Genre:
Action, Drama

Actors:
Allu Arjun, Rashmika Mandanna

Director:
Sukumar

Description:
A coolie rises through the ranks of a red sandalwood smuggling syndicate...
```

This combined information is then passed to SBERT.

This gives the embedding model more context about the movie.

---

# 🧮 Embedding Generation

The combined movie text is passed through Sentence-BERT.

```python
embedding = model.encode(movie_text)
```

The resulting embedding is stored for each movie.

The embeddings are then saved so that we don't have to regenerate them every time the application starts.

---

# 🔎 Similarity Search

When a user enters a search query:

```text
"action movie with a gangster"
```

the query is converted into an embedding.

```text
Search Query
      ↓
SBERT
      ↓
Query Embedding
      ↓
FAISS
      ↓
Nearest Movie Embeddings
      ↓
Search Results
```

The closest vectors are returned as the most relevant movies.

---

# 🖥️ Streamlit Application

The project uses **Streamlit** to create the user interface.

The goal was to make the project easy to use without requiring the user to interact with Python code directly.

The application provides:

* Search input
* Movie results
* Movie details
* Recommendation results
* Movie posters
* Ratings
* Release year
* Genre
* Cast
* Director

The interface was designed to feel more like a small movie discovery platform rather than a simple ML demo.

---

# 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning / NLP

* Sentence-BERT
* Natural Language Processing
* Semantic Similarity
* Vector Embeddings

### Vector Search

* FAISS

### Data Processing

* Pandas
* NumPy

### Web Application

* Streamlit

### Other Tools

* Scikit-learn
* Jupyter Notebook
* Git & GitHub

---

# 📂 Project Structure

```text
Telugu-Movie-Search-Engine/
│
├── app.py
│
├── data/
│   └── movies.csv
│
├── models/
│   └── movie_embeddings.npy
│
├── index/
│   └── movie_faiss.index
│
├── notebooks/
│   └── movie_search_engine.ipynb
│
├── assets/
│   └── screenshots/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

> The exact file names can be changed depending on the final project folder.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Telugu-Movie-Search-Engine.git
```

Move into the project directory:

```bash
cd Telugu-Movie-Search-Engine
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📦 Requirements

A typical `requirements.txt` for the project contains:

```text
pandas
numpy
scikit-learn
sentence-transformers
faiss-cpu
streamlit
```

If additional packages are used in the final application, they should also be added to `requirements.txt`.

---

# 📊 Dataset

The project was built using movie information collected from publicly available movie data sources.

The dataset contains information such as:

```text
movie_name
year
genre
actors
director
description
poster
rating
```

The final processed dataset contains **thousands of movies**, which gives the search engine a reasonably large collection to work with.

---

# 🎯 Example Searches

The search engine can handle queries such as:

```text
action movies

movies with police officers

romantic Telugu movies

movies with Allu Arjun

movies directed by Sukumar

crime thriller movies

family drama

movies about revenge
```

The important part is that the search doesn't have to exactly match the movie description.

---

# 💡 What I Learned From This Project

This project helped me understand how different parts of an NLP application fit together.

Some of the main things I learned were:

* Cleaning real-world text data is harder than working with clean datasets.
* Good search results depend heavily on the quality of the text representation.
* Embeddings are useful when simple keyword matching isn't enough.
* SBERT can be used to represent the meaning of sentences and documents.
* FAISS makes vector similarity search much faster.
* Building an ML model is only one part of an actual application.
* Connecting the model to a usable interface is equally important.
* Data quality can have a major impact on recommendation quality.

One of the biggest lessons from this project was that **a good recommendation system isn't only about the algorithm. The data preparation and representation are just as important.**

---

# ⚠️ Current Limitations

There are still some areas where the system can be improved.

### 1. Recommendation Quality

The recommendations depend on the information available for each movie.

If a movie has a poor or incomplete description, its embedding may not represent the movie very well.

### 2. Cast Data

Cast information from different sources isn't always consistently formatted.

This can affect semantic similarity.

### 3. Context Understanding

The current system mainly relies on semantic similarity.

It doesn't fully understand complex user preferences such as:

```text
"I want a Telugu action movie released after 2020,
with a strong female lead and a rating above 7."
```

A more advanced query-processing layer could handle these filters.

### 4. Personalization

The current recommendations are not personalized to an individual user's watch history.

---

# 🚀 Future Improvements

There are several things I would like to add in future versions.

### 🔹 Better Query Understanding

Build a query-processing layer that can identify:

* Genre
* Actor
* Director
* Year
* Rating
* Language
* User intent

### 🔹 Hybrid Search

Combine:

```text
Keyword Search
        +
Semantic Search
        +
Metadata Filtering
```

This could provide better results than relying only on embeddings.

### 🔹 Personalized Recommendations

Recommendations could be based on:

* Previously watched movies
* User ratings
* Favorite actors
* Favorite genres
* Search history

### 🔹 Better Ranking

Instead of returning results only according to vector similarity, a ranking system could combine multiple signals:

```text
Semantic Similarity
+
Rating
+
Popularity
+
Genre Match
+
User Preference
```

### 🔹 Multilingual Search

Support queries in:

* Telugu
* English
* Telugu-English mixed language

For example:

```text
"oka manchi action movie suggest cheyyi"
```

### 🔹 Deploy the Application

The Streamlit application can be deployed so that anyone can access the movie search engine through a web browser.

---

# 📸 Project Screenshots

Add screenshots of the application here.

Example:

```markdown
![Home Page](assets/screenshots/home.png)

![Search Results](assets/screenshots/search.png)

![Recommendations](assets/screenshots/recommendations.png)
```

Screenshots are worth adding because they immediately show people what you actually built.

---

# 🏗️ Project Pipeline

The complete project can be summarized as:

```text
                 RAW MOVIE DATA
                       │
                       ▼
                DATA CLEANING
                       │
                       ▼
              TEXT PREPROCESSING
                       │
                       ▼
          CREATE MOVIE REPRESENTATION
                       │
                       ▼
                 SBERT MODEL
                       │
                       ▼
              384-D EMBEDDINGS
                       │
                       ▼
                 FAISS INDEX
                       │
             ┌─────────┴─────────┐
             │                   │
        SEARCH QUERY         MOVIE SELECTION
             │                   │
             ▼                   ▼
          SBERT               EMBEDDING
             │                   │
             └─────────┬─────────┘
                       ▼
                VECTOR SEARCH
                       │
                       ▼
               RANKED RESULTS
                       │
                       ▼
                STREAMLIT UI
```

---

# 🌟 Why This Project?

I didn't want to build another simple movie recommendation project based only on ratings.

The main idea was to explore how **NLP and vector databases can be used to build a search experience that understands what the user means**.

This project combines data preprocessing, NLP, embeddings, similarity search and application development into one complete pipeline.

---

# 👨‍💻 Author

**Shiva Krishna Vutukuri**

Machine Learning / AI Enthusiast

Interested in:

* Machine Learning
* Deep Learning
* Natural Language Processing
* Computer Vision
* Generative AI
* AI Applications

---

# 📄 License

This project is created for educational and learning purposes.

If you use or modify this project, please give appropriate credit to the original project.

---

## ⭐ If You Found This Project Interesting

If this project helped you understand semantic search, NLP embeddings or recommendation systems, feel free to ⭐ the repository.
