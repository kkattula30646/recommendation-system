# AI-Based Adaptive Recommendation System

## Overview
This project implements an **AI-Based Adaptive Human-Computer Interaction System** using a content-based recommendation approach.

The system not only recommends items but also **learns from user interactions and adapts its behavior over time**.

---

## Features
- Content-based recommendation using TF-IDF
- Cosine similarity for matching items
- Command Line Interface (CLI)
- Graphical User Interface (GUI)
- **Adaptive behavior based on user preferences**
- Dynamic re-ranking of recommendations

---

## Adaptive Functionality
The system tracks user selections and updates internal preferences.

### How it adapts:
- When a user selects a recommendation → system stores its category
- Future recommendations are **boosted based on learned preferences**
- Over time, results become more personalized

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Tkinter

---

## How to Run

### 1. Install dependencies