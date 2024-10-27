import torch
import torch.nn as nn
import numpy as np
import random
import os
from typing import List, Tuple, Dict

class TicTacToeTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        # Input channels: current player pieces, opponent pieces, and line patterns
        self.state_embedding = nn.Linear(11, d_model)  # 2 player channels + 9 line pattern channels
        self.positional_embedding = nn.Parameter(torch.randn(1, 9, d_model))
        
        # Add attention to specific game patterns
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.1,
            batch_first=True
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Separate heads for tactical and strategic moves
        self.tactical_head = nn.Sequential(
            nn.Linear(d_model * 9, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 9)
        )
        
        self.strategic_head = nn.Sequential(
            nn.Linear(d_model * 9, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 9)
        )
        
        # Value head for position evaluation
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 9, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        # Create player channels
        player_channel = (x == 1).float()
        opponent_channel = (x == -1).float()
        
        # Extract line patterns (rows, columns, diagonals)
        line_patterns = self.extract_line_patterns(x)
        
        # Combine all features
        x = torch.cat([
            player_channel.reshape(batch_size, 9, 1),
            opponent_channel.reshape(batch_size, 9, 1),
            line_patterns
        ], dim=2)
        
        x = self.state_embedding(x)
        x = x + self.positional_embedding
        
        # Apply pattern attention
        pattern_attention_out, _ = self.pattern_attention(x, x, x)
        x = x + pattern_attention_out
        
        x = self.transformer(x)
        features = x.reshape(batch_size, -1)
        
        tactical_moves = self.tactical_head(features)
        strategic_moves = self.strategic_head(features)
        value = self.value_head(features)
        
        return tactical_moves, strategic_moves, value
    
    def extract_line_patterns(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        board = x.reshape(-1, 3, 3)
        patterns = []
        
        # Extract rows
        for i in range(3):
            row = board[:, i, :].reshape(batch_size, 3)
            patterns.append(self.encode_line(row))
        
        # Extract columns
        for i in range(3):
            col = board[:, :, i].reshape(batch_size, 3)
            patterns.append(self.encode_line(col))
        
        # Extract diagonals
        diag1 = torch.diagonal(board, dim1=1, dim2=2)
        diag2 = torch.diagonal(torch.flip(board, [2]), dim1=1, dim2=2)
        patterns.append(self.encode_line(diag1))
        patterns.append(self.encode_line(diag2))
        
        # Stack all patterns
        return torch.stack(patterns, dim=2).reshape(batch_size, 9, 9)
            
    def encode_line(self, line: torch.Tensor) -> torch.Tensor:
        """Encode a line (row/column/diagonal) into a pattern feature"""
        batch_size = line.shape[0]
        pattern = torch.zeros(batch_size, 1)
        
        # Calculate sum and count of pieces
        player_sum = (line == 1).sum(dim=1, keepdim=True).float()
        opponent_sum = (line == -1).sum(dim=1, keepdim=True).float()
        empty_count = (line == 0).sum(dim=1, keepdim=True).float()
        
        return torch.cat([player_sum, opponent_sum, empty_count], dim=1)

class TicTacToeGame:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.history = []
    
    def make_move(self, position: Tuple[int, int]) -> bool:
        row, col = position
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.history.append((position, self.current_player))
            self.current_player *= -1
            return True
        return False
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
    
    def check_winner(self) -> int:
        # Check rows, columns and diagonals
        lines = (
            self.board,  # rows
            self.board.T,  # columns
            [np.diag(self.board)],  # main diagonal
            [np.diag(np.fliplr(self.board))]  # other diagonal
        )
        
        for line_set in lines:
            for line in line_set:
                if len(line) == 3 and abs(sum(line)) == 3:
                    return line[0]
        return 0
    
    def is_draw(self) -> bool:
        return len(self.get_valid_moves()) == 0 and self.check_winner() == 0
    
    def get_state(self) -> torch.Tensor:
        return torch.tensor(self.board.flatten(), dtype=torch.float32)
    
    def evaluate_position(self) -> float:
        """Evaluate the current position from perspective of current player"""
        winner = self.check_winner()
        if winner == self.current_player:
            return 1.0
        elif winner == -self.current_player:
            return -1.0
        elif self.is_draw():
            return 0.0
        
        # Count potential winning lines
        score = 0
        lines = (
            self.board,
            self.board.T,
            [np.diag(self.board)],
            [np.diag(np.fliplr(self.board))]
        )
        
        for line_set in lines:
            for line in line_set:
                # Check for two-in-a-row opportunities
                player_count = sum(line == self.current_player)
                opponent_count = sum(line == -self.current_player)
                empty_count = sum(line == 0)
                
                if opponent_count == 2 and empty_count == 1:
                    score -= 0.3  # Urgent blocking needed
                elif player_count == 2 and empty_count == 1:
                    score += 0.3  # Potential winning move
                elif opponent_count == 0 and empty_count > 0:
                    score += 0.1 * player_count  # Building potential
                elif player_count == 0 and empty_count > 0:
                    score -= 0.1 * opponent_count  # Opponent building potential
        
        return score

def train_model(model: TicTacToeTransformer, num_episodes: int = 20000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    move_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    history = {'loss': [], 'win_rate': [], 'draw_rate': []}
    
    def calculate_move_reward(game: TicTacToeGame, move: Tuple[int, int], player: int) -> float:
        """Calculate immediate reward for a move"""
        board = game.board.copy()
        row, col = move
        board[row, col] = player
        
        # Check if this move creates a win
        lines = [
            board[row, :],  # Row
            board[:, col],  # Column
            np.diagonal(board) if row == col else np.array([]),  # Main diagonal if applicable
            np.diagonal(np.fliplr(board)) if row + col == 2 else np.array([])  # Anti-diagonal if applicable
        ]
        
        reward = 0.0
        for line in lines:
            if len(line) == 3:
                player_count = np.sum(line == player)
                opponent_count = np.sum(line == -player)
                empty_count = np.sum(line == 0)
                
                if player_count == 3:  # Winning move
                    reward += 1.0
                elif player_count == 2 and empty_count == 1:  # Setting up win
                    reward += 0.3
                elif opponent_count == 2 and empty_count == 1:  # Blocking opponent
                    reward += 0.5
                elif player_count == 2 and opponent_count == 0:  # Creating threat
                    reward += 0.2
        
        return reward
    
    for episode in range(num_episodes):
        game = TicTacToeGame()
        states = []
        moves = []
        rewards = []
        
        epsilon = max(0.1, 1.0 - episode / (num_episodes * 0.8))
        
        while True:
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            valid_moves_mask = get_valid_move_mask(valid_moves)
            
            if random.random() < epsilon:
                move_idx = random.choice([i * 3 + j for i, j in valid_moves])
                move = (move_idx // 3, move_idx % 3)
            else:
                with torch.no_grad():
                    tactical_logits, strategic_logits, value = model(state.unsqueeze(0))
                    combined_logits = tactical_logits + strategic_logits
                    masked_logits = torch.where(valid_moves_mask == 1, combined_logits[0], torch.tensor(-1e9))
                    move_idx = masked_logits.argmax().item()
                    move = (move_idx // 3, move_idx % 3)
            
            # Calculate immediate reward for the move
            immediate_reward = calculate_move_reward(game, move, game.current_player)
            
            states.append(state)
            moves.append(move_idx)
            rewards.append(immediate_reward)
            
            game.make_move(move)
            
            winner = game.check_winner()
            if winner != 0:
                # Add final rewards
                final_reward = 1.0 if winner == 1 else -1.0
                rewards[-1] += final_reward
                break
            elif game.is_draw():
                rewards[-1] += 0.1  # Small reward for forcing a draw
                break
        
        if len(states) > 0:
            states_tensor = torch.stack(states)
            moves_tensor = torch.tensor(moves)
            
            # Calculate returns with reward shaping
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + 0.9 * R
                returns.insert(0, R)
            returns_tensor = torch.tensor(returns).unsqueeze(1)
            
            optimizer.zero_grad()
            tactical_logits, strategic_logits, values = model(states_tensor)
            
            # Combine tactical and strategic losses
            tactical_loss = move_criterion(tactical_logits, moves_tensor)
            strategic_loss = move_criterion(strategic_logits, moves_tensor)
            value_loss = value_criterion(values, returns_tensor)
            
            # Total loss with different weights for each component
            total_loss = tactical_loss + 0.5 * strategic_loss + 0.3 * value_loss
            
            total_loss.backward()
            optimizer.step()
            
            history['loss'].append(total_loss.item())
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {history['loss'][-1]:.4f}, Epsilon: {epsilon:.2f}")
    
    return history

def play_game(model: TicTacToeTransformer, human_player: int = 1):
    game = TicTacToeGame()
    
    def print_board():
        print("\nCurrent board:")
        for i in range(3):
            for j in range(3):
                symbol = " "
                if game.board[i, j] == 1:
                    symbol = "X"
                elif game.board[i, j] == -1:
                    symbol = "O"
                print(f" {symbol} ", end="")
                if j < 2:
                    print("|", end="")
            if i < 2:
                print("\n-----------")
        print("\n")
    
    while True:
        print_board()
        
        if game.current_player == human_player:
            while True:
                try:
                    row = int(input("Enter row (0-2): "))
                    col = int(input("Enter column (0-2): "))
                    if 0 <= row <= 2 and 0 <= col <= 2 and game.make_move((row, col)):
                        break
                    print("Invalid move, try again.")
                except ValueError:
                    print("Invalid input, try again.")
        else:
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            valid_moves_mask = get_valid_move_mask(valid_moves)
            
            with torch.no_grad():
                tactical_logits, strategic_logits, value = model(state.unsqueeze(0))
                combined_logits = tactical_logits + 0.5 * strategic_logits
                masked_logits = torch.where(valid_moves_mask == 1, combined_logits[0], torch.tensor(-1e9))
                move_idx = masked_logits.argmax().item()
                move = (move_idx // 3, move_idx % 3)
            
            print(f"AI plays: row {move[0]}, column {move[1]}")
            game.make_move(move)
        
        winner = game.check_winner()
        if winner != 0:
            print_board()
            print(f"Player {'X' if winner == 1 else 'O'} wins!")
            break
        elif game.is_draw():
            print_board()
            print("Game is a draw!")
            break

# Helper functions remain the same
def get_valid_move_mask(valid_moves: List[Tuple[int, int]], device='cpu') -> torch.Tensor:
    mask = torch.zeros(9, device=device)
    for row, col in valid_moves:
        mask[row * 3 + col] = 1
    return mask

if __name__ == "__main__":
    model = TicTacToeTransformer()
    
    if os.path.exists("tictactoe_model.pth"):
        model.load_state_dict(torch.load("tictactoe_model.pth"))
    else:
        print("Training model...")
        history = train_model(model, num_episodes=1000)
        torch.save(model.state_dict(), "tictactoe_model.pth")
    
    print("Ready to play!")
    play_game(model, human_player=1)