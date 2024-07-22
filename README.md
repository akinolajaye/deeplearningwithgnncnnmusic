# Music Recommendation AI System

## Introduction

It all started as a simple idea back in sixth form—a concept that seemed like a dream. Music has always been a passion of mine, and one day at home thinking about what I could do with computer science, the idea struck me: Why not create an AI system that could recommend music? This was the moment I realized why I wanted to pursue computer science.

Embarking on this project was intimidating at first. The challenge seemed daunting, but I learned an important lesson: if you can envision it, you can achieve it. The only limits we face are those we set for ourselves.

## Project Overview

In this project, I leveraged the Python machine learning library PyTorch to train a convolutional neural network (CNN) on spectrogram data. This approach allowed me to group songs into clusters using a self-supervised contrastive learning method. Additionally, I developed a graph neural network (GNN) designed to perform link prediction. This system identifies potential song preferences for users based on their listening history and the song representations derived from the CNN.

### Key Components

1. **Data Preparation**:
    - Spectrogram generation from audio files.
    - Dataset organization for training the CNN.

2. **Model Training**:
    - Training a convolutional neural network on spectrogram data using PyTorch.
    - Implementing self-supervised contrastive learning to create meaningful song representations.

3. **Graph Neural Network**:
    - Developing a GNN to perform link prediction.
    - Utilizing the song representations from the CNN to predict user preferences.

4. **Integration**:
    - Combining the CNN and GNN to create a cohesive music recommendation system.
    - Testing and evaluating the system's performance.
    