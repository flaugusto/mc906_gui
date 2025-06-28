# Checkers Game with AI

A Python implementation of the classic Checkers game featuring an AI opponent powered by a neural network.

## Features

- Classic Checkers gameplay rules
- AI opponent (plays as black)
- Simple graphical interface using Pygame
- Supports both regular moves and captures
- Automatic king promotion

## Requirements

- Python 3.12+
- Pygame
- NumPy
- Keras
- TensorFlow

## Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Make sure you have the following files in the project directory:
   - `game.py` - Main game file
   - `checkers.py` - Game logic
   - `base_model.json` - AI model architecture
   - `reinforced_model_10.weights.h5` - AI model weights

## How to Play

1. Run the game:

   ```
   python game.py
   ```

2. **Game Controls**:

   - You play as **WHITE** pieces
   - The AI plays as **BLACK** pieces
   - Click on a white piece to select it (valid moves will be highlighted)
   - Click on a highlighted square to move the selected piece
   - The AI will automatically make its move after you complete yours

3. **Game Rules**:
   - Regular pieces can only move diagonally forward
   - Kings can move diagonally in any direction
   - Capture opponent's pieces by jumping over them
   - Multiple captures in one turn are allowed
   - A piece is promoted to king when it reaches the opposite end of the board
   - The game ends when a player has no valid moves left or no pieces remaining

## AI Features

- The AI uses a pre-trained neural network to evaluate board positions
- It looks ahead to find the best possible move
- The model was trained using reinforcement learning

## Files

- `game.py` - Main game loop and UI
- `checkers.py` - Game logic and AI helper functions
- `base_model.json` - Neural network architecture
- `reinforced_model_10.weights.h5` - Trained model weights
- `requirements.txt` - Python dependencies

## Notes

- The game requires the model files to be in the same directory
- The AI may take a moment to make its move as it evaluates the board position
- The game automatically ends when a winner is determined

Enjoy the game!
