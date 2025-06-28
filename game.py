import pygame
import sys
import numpy as np
from checkers import possible_moves, compress
from keras.models import model_from_json

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 800
BOARD_SIZE = 8
SQUARE_SIZE = WINDOW_SIZE // BOARD_SIZE
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
HIGHLIGHT = (124, 252, 0, 100)

class CheckersGame:
    def __init__(self, ai_model=None):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption('Checkers')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        
        # Game state
        self.board = np.zeros((8, 8), dtype=int)
        self.initialize_board()
        self.selected_piece = None
        self.valid_moves = []
        self.turn = 1  # 1 for white, -1 for black
        self.game_over = False
        self.winner = None
        self.ai_model = ai_model

    def initialize_board(self):
        # Set up the initial board state
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 1:  # Only dark squares
                    if i < 3:  # Black pieces at the top
                        self.board[i, j] = -1
                    elif i > 4:  # White pieces at the bottom
                        self.board[i, j] = 1

    def draw_board(self):
        # Draw the checkerboard
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x = col * SQUARE_SIZE
                y = row * SQUARE_SIZE
                
                # Draw square
                color = WHITE if (row + col) % 2 == 0 else GRAY
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                
                # Highlight selected piece
                if self.selected_piece and (row, col) == self.selected_piece:
                    s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    s.fill((255, 255, 0, 100))  # Yellow highlight with transparency
                    self.screen.blit(s, (x, y))
                
                # Highlight valid moves
                if (row, col) in self.valid_moves:
                    s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    s.fill(HIGHLIGHT)  # Green highlight with transparency
                    self.screen.blit(s, (x, y))
                
                # Draw pieces
                piece = self.board[row, col]
                if piece != 0:
                    center = (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2)
                    radius = SQUARE_SIZE // 2 - 10
                    
                    # Draw piece base
                    pygame.draw.circle(self.screen, 
                                    WHITE if piece > 0 else BLACK, 
                                    center, radius)
                    
                    # Draw piece border
                    pygame.draw.circle(self.screen, 
                                    BLACK if piece > 0 else WHITE, 
                                    center, radius, 2)
                    
                    # Draw king indicator
                    if abs(piece) == 3:
                        pygame.draw.circle(self.screen, 
                                        (255, 215, 0),  # Gold color for king
                                        center, radius // 2)

        # Draw game status
        if self.game_over:
            status_text = f"{'White' if self.winner == 1 else 'Black'} wins!"
            text = self.font.render(status_text, True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE//2, 30))
            self.screen.blit(text, text_rect)
        else:
            status_text = f"{'White' if self.turn == 1 else 'Black'}'s turn"
            text = self.font.render(status_text, True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE//2, 30))
            self.screen.blit(text, text_rect)

    def get_square_from_pos(self, pos):
        x, y = pos
        row = y // SQUARE_SIZE
        col = x // SQUARE_SIZE
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return (row, col)
        return None

    def get_valid_moves(self, row, col):
        piece = self.board[row, col]
        if piece == 0 or piece * self.turn <= 0:
            return []
            
        valid_moves = []
        is_king = (abs(piece) == 3)
        
        # First check for capture moves
        capture_moves = []
        for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:  # Capture moves
            new_row, new_col = row + dr, col + dc
            
            # Skip if out of bounds
            if not (0 <= new_row < 8 and 0 <= new_col < 8):
                continue
                
            # Skip if target square is not empty
            if self.board[new_row, new_col] != 0:
                continue
                
            # Check if there's an opponent's piece to capture in between
            mid_row, mid_col = (row + new_row) // 2, (col + new_col) // 2
            if (0 <= mid_row < 8 and 0 <= mid_col < 8 and 
                self.board[mid_row, mid_col] * piece < 0):  # Opponent's piece
                capture_moves.append((new_row, new_col))
        
        # If there are capture moves available, only return those
        if capture_moves:
            return capture_moves
            
        # If no captures, check for regular moves (only if no captures are available anywhere)
        if not self.has_any_captures():
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Regular moves
                new_row, new_col = row + dr, col + dc
                
                # Skip if out of bounds
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    continue
                    
                # Skip if target square is not empty
                if self.board[new_row, new_col] != 0:
                    continue
                
                # For regular pieces, check move direction
                if not is_king:
                    if (piece == 1 and new_row < row) or (piece == -1 and new_row > row):
                        valid_moves.append((new_row, new_col))
                else:  # Kings can move any direction
                    valid_moves.append((new_row, new_col))
        
        return valid_moves
        
    def has_any_captures(self):
        """Check if there are any captures available for the current player."""
        for row in range(8):
            for col in range(8):
                piece = self.board[row, col]
                if piece * self.turn > 0:  # Current player's piece
                    # Check all possible capture moves
                    for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                        new_row, new_col = row + dr, col + dc
                        if (0 <= new_row < 8 and 0 <= new_col < 8 and 
                            self.board[new_row, new_col] == 0):  # Empty target square
                            mid_row, mid_col = (row + new_row) // 2, (col + new_col) // 2
                            # Check if there's an opponent's piece to capture
                            if (0 <= mid_row < 8 and 0 <= mid_col < 8 and 
                                self.board[mid_row, mid_col] * piece < 0):
                                return True
                            # For kings, also check longer diagonal moves
                            if abs(piece) == 3:  # King piece
                                # Check for multi-square captures
                                step_r = 1 if new_row > row else -1
                                step_c = 1 if new_col > col else -1
                                r, c = row + step_r, col + step_c
                                found_opponent = False
                                while (0 <= r < 8 and 0 <= c < 8 and 
                                      (r != new_row or c != new_col)):
                                    if self.board[r, c] * piece < 0:  # Opponent's piece
                                        found_opponent = True
                                        break
                                    if self.board[r, c] * piece > 0:  # Own piece blocking
                                        break
                                    r += step_r
                                    c += step_c
                                if found_opponent:
                                    return True
        return False

    def is_valid_move(self, start_row, start_col, end_row, end_col, force_capture):
        # Create a copy of the board to avoid modifying the actual game state
        board_copy = np.copy(self.board)
        piece = board_copy[start_row, start_col]
        
        if piece == 0:
            return False
            
        # Check if it's the player's turn
        if piece * self.turn <= 0:
            return False
            
        # Check if destination is empty
        if board_copy[end_row, end_col] != 0:
            return False
            
        # Check if move is diagonal
        row_diff = abs(end_row - start_row)
        col_diff = abs(end_col - start_col)
        
        if row_diff != col_diff:
            return False
            
        # Check if it's a capture
        if row_diff == 2:
            # Check if there's an opponent's piece in between
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            
            if (board_copy[mid_row, mid_col] * piece >= 0):  # No opponent's piece to capture
                return False
                
            # Don't modify the actual board during validation
            return True
            
        # For non-capture moves
        if row_diff == 1 and not force_capture:
            # Check if it's a forward move for regular pieces
            if abs(piece) == 1:  # Only for regular pieces (not kings)
                if (piece == 1 and end_row >= start_row) or (piece == -1 and end_row <= start_row):
                    return False
            return True
            
        return False

    def make_move(self, start, end):
        start_row, start_col = start
        end_row, end_col = end
        piece = self.board[start_row, start_col]
        is_king = (abs(piece) == 3)
        
        # Make the move
        self.board[end_row, end_col] = piece
        self.board[start_row, start_col] = 0
        
        # Check for promotion to king
        if not is_king and ((end_row == 0 and piece == 1) or (end_row == 7 and piece == -1)):
            self.board[end_row, end_col] = 3 if piece > 0 else -3
            piece = self.board[end_row, end_col]  # Update piece to king
        
        # Check if this was a capture
        row_diff = abs(end_row - start_row)
        
        if row_diff >= 2:  # It was a capture (could be more than 2 for kings in some variants)
            # Remove the captured piece(s)
            step_r = 1 if end_row > start_row else -1
            step_c = 1 if end_col > start_col else -1
            
            # Check all squares along the diagonal for captured pieces
            r, c = start_row + step_r, start_col + step_c
            while r != end_row and c != end_col:
                if self.board[r, c] != 0:  # Found a captured piece
                    self.board[r, c] = 0
                    # For standard checkers, there will be only one captured piece per jump
                    break
                r += step_r
                c += step_c
            
            # Check for additional captures from the new position
            self.valid_moves = []
            for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                new_row, new_col = end_row + dr, end_col + dc
                if (0 <= new_row < 8 and 0 <= new_col < 8 and 
                    self.board[new_row, new_col] == 0):
                    # Check if this would be a valid capture
                    mid_r = (end_row + new_row) // 2
                    mid_c = (end_col + new_col) // 2
                    if (0 <= mid_r < 8 and 0 <= mid_c < 8 and 
                        self.board[mid_r, mid_c] * piece < 0):  # Opponent's piece
                        self.valid_moves.append((new_row, new_col))
            
            # If no more captures, switch turns
            if not self.valid_moves:
                self.turn *= -1
                self.check_game_over()
                self.selected_piece = None
            else:
                # Keep the same piece selected for multiple jumps
                self.selected_piece = (end_row, end_col)
        else:
            # For non-capture moves, switch turns
            self.turn *= -1
            self.check_game_over()
            self.selected_piece = None

    def check_game_over(self):
        # Check if either player has no pieces left
        white_has_pieces = np.any(self.board > 0)
        black_has_pieces = np.any(self.board < 0)
        
        if not white_has_pieces:
            self.game_over = True
            self.winner = -1  # Black wins
            return
        elif not black_has_pieces:
            self.game_over = True
            self.winner = 1  # White wins
            return
        
        # Check if current player has any valid moves
        current_player_has_moves = False
        capture_required = False
        
        # First, check if there are any captures available
        for row in range(8):
            for col in range(8):
                if self.board[row, col] * self.turn > 0:  # Current player's piece
                    # Check for captures
                    for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                        new_row, new_col = row + dr, col + dc
                        if (0 <= new_row < 8 and 0 <= new_col < 8 and 
                            self.board[new_row, new_col] == 0):
                            mid_row = (row + new_row) // 2
                            mid_col = (col + new_col) // 2
                            if self.board[mid_row, mid_col] * self.board[row, col] < 0:  # Opponent's piece
                                capture_required = True
                                current_player_has_moves = True
                                break
                    if capture_required:
                        break
                    
                    # If no captures required, check for regular moves
                    if not capture_required and not current_player_has_moves:
                        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            new_row, new_col = row + dr, col + dc
                            if (0 <= new_row < 8 and 0 <= new_col < 8 and 
                                self.board[new_row, new_col] == 0):
                                # Check if move is in the right direction for non-kings
                                piece = self.board[row, col]
                                if abs(piece) == 1:  # Regular piece
                                    if (piece == 1 and new_row < row) or (piece == -1 and new_row > row):
                                        current_player_has_moves = True
                                        break
                                else:  # King
                                    current_player_has_moves = True
                                    break
                    
                    if current_player_has_moves and not capture_required:
                        break
                
                if current_player_has_moves:
                    break
            if current_player_has_moves:
                break
        
        # If no valid moves, game over
        if not current_player_has_moves:
            self.game_over = True
            self.winner = -self.turn  # Other player wins

    def make_ai_move(self):
        """Make a move using the AI model"""
        if self.game_over or self.turn != -1 or not self.ai_model:
            return
            
        # Get all possible moves for black pieces
        all_moves = []
        for row in range(8):
            for col in range(8):
                if self.board[row, col] < 0:  # Black pieces
                    moves = self.get_valid_moves(row, col)
                    for move in moves:
                        all_moves.append(((row, col), move))
        
        if not all_moves:
            return  # No valid moves, game over will be handled in the main loop
            
        # Find the best move using the AI model
        best_move = None
        best_score = float('-inf')
        
        for (start, end) in all_moves:
            # Make a copy of the board
            board_copy = self.board.copy()
            
            # Apply the move
            piece = board_copy[start[0], start[1]]
            board_copy[end[0], end[1]] = piece
            board_copy[start[0], start[1]] = 0
            
            # Check for promotion to king
            if end[0] == 0 and board_copy[end[0], end[1]] == -1:
                board_copy[end[0], end[1]] = -3
            
            # Convert board to 1D array of 32 elements using the compress function from checkers
            board_input = compress(board_copy)
            
            # Get the model's prediction
            score = self.ai_model.predict(board_input, verbose=0)[0][0]
            
            if score > best_score:
                best_score = score
                best_move = (start, end)
        
        # Make the best move
        if best_move:
            start, end = best_move
            self.make_move(start, end)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    pos = pygame.mouse.get_pos()
                    square = self.get_square_from_pos(pos)
                    
                    if square is not None:
                        row, col = square
                        
                        # If a piece is already selected, try to move it
                        if self.selected_piece:
                            if (row, col) in self.valid_moves:
                                self.make_move(self.selected_piece, (row, col))
                                self.selected_piece = None
                                self.valid_moves = []
                            elif self.board[row, col] * self.turn > 0:  # Select a different piece
                                self.selected_piece = (row, col)
                                self.valid_moves = self.get_valid_moves(row, col)
                            else:
                                self.selected_piece = None
                                self.valid_moves = []
                        # Select a piece if it's the player's turn
                        elif self.board[row, col] * self.turn > 0:
                            self.selected_piece = (row, col)
                            self.valid_moves = self.get_valid_moves(row, col)
            
            # If it's AI's turn (black) and game is not over, make AI move
            if not self.game_over and self.turn == -1 and self.ai_model:
                self.make_ai_move()
            
            # Draw everything
            self.screen.fill(WHITE)
            self.draw_board()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Load the AI model
    board_model = None
    try:
        json_file = open('base_model.json', 'r')
        board_json = json_file.read()
        json_file.close()

        board_model = model_from_json(board_json)
        board_model.load_weights('reinforced_model_10.weights.h5')
        board_model.compile(optimizer='nadam', loss='binary_crossentropy')
        print("AI model loaded successfully!")
    except Exception as e:
        print(f"Error loading AI model: {e}")
        print("Starting game without AI...")
    
    # Initialize the game with the AI model (will play as black)
    game = CheckersGame(ai_model=board_model)
    game.run()