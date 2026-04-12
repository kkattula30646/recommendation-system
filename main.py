import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox, scrolledtext


class RecommendationSystem:
    """
    A simple content-based recommendation system using TF-IDF and cosine similarity.
    Expected CSV columns:
    - title
    - description
    Optional columns:
    - category
    """

    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.df = None
        self.similarity_matrix = None
        self.indices = None
        self._load_and_prepare_data()

    def _load_and_prepare_data(self) -> None:
        """Load dataset, clean text, and build similarity matrix."""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(
                f"Dataset file '{self.csv_file}' was not found.\n"
                f"Make sure the CSV is in the same folder as main.py."
            )

        self.df = pd.read_csv(self.csv_file)

        required_columns = ["title", "description"]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(
                    f"Missing required column: '{col}'. "
                    f"Your CSV must contain: {required_columns}"
                )

        # Fill missing values
        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["description"] = self.df["description"].fillna("").astype(str)

        # Optional category field
        if "category" in self.df.columns:
            self.df["category"] = self.df["category"].fillna("").astype(str)
            self.df["combined_features"] = (
                self.df["title"] + " " + self.df["description"] + " " + self.df["category"]
            )
        else:
            self.df["combined_features"] = self.df["title"] + " " + self.df["description"]

        # Remove duplicate titles to avoid mapping issues
        self.df = self.df.drop_duplicates(subset=["title"]).reset_index(drop=True)

        # Vectorize text
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(self.df["combined_features"])

        # Similarity matrix
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Map titles to indices
        self.indices = pd.Series(self.df.index, index=self.df["title"].str.lower()).drop_duplicates()

    def recommend(self, item_title: str, top_n: int = 5):
        """
        Return top N recommendations for a given item title.
        """
        if not item_title or not item_title.strip():
            raise ValueError("Please enter a valid item title.")

        item_title = item_title.strip().lower()

        if item_title not in self.indices:
            close_matches = self._find_partial_matches(item_title)
            if close_matches:
                return {
                    "found": False,
                    "message": "Exact title not found. Here are some similar titles you can try:",
                    "suggestions": close_matches
                }
            return {
                "found": False,
                "message": "Title not found in the dataset.",
                "suggestions": []
            }

        idx = self.indices[item_title]

        # Get similarity scores for this item
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity score descending
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip first one because it is the item itself
        sim_scores = sim_scores[1:top_n + 1]

        recommendations = []
        for i, score in sim_scores:
            row = self.df.iloc[i]
            recommendations.append({
                "title": row["title"],
                "description": row["description"],
                "score": round(float(score), 4)
            })

        return {
            "found": True,
            "input_title": self.df.iloc[idx]["title"],
            "recommendations": recommendations
        }

    def _find_partial_matches(self, query: str, max_results: int = 5):
        """Suggest titles that partially match the query."""
        matches = self.df[self.df["title"].str.lower().str.contains(query, na=False)]["title"].tolist()
        return matches[:max_results]


class RecommendationApp:
    """Simple Tkinter GUI for the recommendation system."""

    def __init__(self, recommender: RecommendationSystem):
        self.recommender = recommender
        self.root = tk.Tk()
        self.root.title("Recommendation System")
        self.root.geometry("700x500")

        self._build_ui()

    def _build_ui(self):
        title_label = tk.Label(
            self.root,
            text="Content-Based Recommendation System",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        instruction_label = tk.Label(
            self.root,
            text="Enter a title from the dataset:",
            font=("Arial", 11)
        )
        instruction_label.pack(pady=5)

        self.entry = tk.Entry(self.root, width=50, font=("Arial", 12))
        self.entry.pack(pady=5)

        recommend_button = tk.Button(
            self.root,
            text="Get Recommendations",
            command=self.get_recommendations,
            font=("Arial", 11),
            width=20
        )
        recommend_button.pack(pady=10)

        self.output_box = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=20)
        self.output_box.pack(padx=10, pady=10)

    def get_recommendations(self):
        user_input = self.entry.get().strip()
        self.output_box.delete("1.0", tk.END)

        try:
            result = self.recommender.recommend(user_input, top_n=5)

            if not result["found"]:
                self.output_box.insert(tk.END, result["message"] + "\n\n")
                if result["suggestions"]:
                    for suggestion in result["suggestions"]:
                        self.output_box.insert(tk.END, f"- {suggestion}\n")
                return

            self.output_box.insert(
                tk.END,
                f"Recommendations for: {result['input_title']}\n"
                + "=" * 60 + "\n\n"
            )

            for idx, rec in enumerate(result["recommendations"], start=1):
                self.output_box.insert(tk.END, f"{idx}. {rec['title']}\n")
                self.output_box.insert(tk.END, f"Similarity Score: {rec['score']}\n")
                self.output_box.insert(tk.END, f"Description: {rec['description']}\n")
                self.output_box.insert(tk.END, "-" * 60 + "\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()


def run_cli(recommender: RecommendationSystem):
    """Run a simple command-line interface."""
    print("\nContent-Based Recommendation System")
    print("=" * 40)
    print("Type a title from your dataset to get recommendations.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter a title: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        try:
            result = recommender.recommend(user_input, top_n=5)

            if not result["found"]:
                print("\n" + result["message"])
                if result["suggestions"]:
                    for suggestion in result["suggestions"]:
                        print(f"- {suggestion}")
                print()
                continue

            print(f"\nRecommendations for: {result['input_title']}")
            print("-" * 50)
            for idx, rec in enumerate(result["recommendations"], start=1):
                print(f"{idx}. {rec['title']} (Score: {rec['score']})")
                print(f"   Description: {rec['description']}")
                print()

        except Exception as e:
            print(f"Error: {e}\n")


def create_sample_dataset(csv_file: str):
    """Create a sample dataset if one does not exist."""
    sample_data = pd.DataFrame({
        "title": [
            "The Matrix",
            "Inception",
            "Interstellar",
            "The Dark Knight",
            "Avatar",
            "Titanic",
            "The Notebook",
            "John Wick",
            "Mad Max: Fury Road",
            "The Avengers"
        ],
        "description": [
            "A computer hacker learns about the true nature of reality and his role in the war against its controllers.",
            "A thief who steals corporate secrets through dream-sharing technology is given an inverse task of planting an idea.",
            "A team travels through a wormhole in space in an attempt to save humanity.",
            "Batman faces the Joker, a criminal mastermind causing chaos in Gotham City.",
            "A marine on an alien planet becomes torn between following orders and protecting the world he feels is his home.",
            "A love story unfolds aboard the ill-fated Titanic.",
            "A romantic drama about love, memory, and enduring emotion.",
            "An ex-hitman comes out of retirement to seek vengeance.",
            "In a post-apocalyptic wasteland, survivors fight for freedom and resources.",
            "Earth's mightiest heroes must come together to stop a global threat."
        ],
        "category": [
            "Sci-Fi Action",
            "Sci-Fi Thriller",
            "Sci-Fi Drama",
            "Action Crime",
            "Sci-Fi Adventure",
            "Romance Drama",
            "Romance Drama",
            "Action Thriller",
            "Action Adventure",
            "Superhero Action"
        ]
    })
    sample_data.to_csv(csv_file, index=False)
    print(f"Sample dataset created: {csv_file}")


def main():
    csv_file = "items.csv"

    # Create sample dataset if file does not exist
    if not os.path.exists(csv_file):
        print(f"'{csv_file}' not found. Creating a sample dataset...")
        create_sample_dataset(csv_file)

    try:
        recommender = RecommendationSystem(csv_file)
    except Exception as e:
        print(f"Failed to initialize recommendation system: {e}")
        sys.exit(1)

    print("Choose interface mode:")
    print("1. Command Line")
    print("2. GUI")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        app = RecommendationApp(recommender)
        app.run()
    else:
        run_cli(recommender)


if __name__ == "__main__":
    main()