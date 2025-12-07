
# Generated: 2025-08-28T06:05:49.598376
# Source Brief: brief_05776.md
# Brief Index: 5776

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to select a piece or confirm a move. Shift to deselect."
    )

    game_description = (
        "A visual game of Checkers. Capture all opponent pieces to win. If a jump is available, you must take it."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.BOARD_SIZE = 8
        self.MAX_STEPS = 1000

        # Visuals
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_LIGHT_SQ = (235, 210, 180)
        self.COLOR_DARK_SQ = (130, 80, 45)
        self.COLOR_P1 = (220, 50, 50)
        self.COLOR_P1_KING = (255, 150, 150)
        self.COLOR_P2 = (50, 100, 220)
        self.COLOR_P2_KING = (150, 180, 255)
        self.COLOR_VALID_MOVE = (80, 220, 80, 150)
        self.COLOR_CURSOR = (255, 255, 0, 180)
        self.COLOR_SELECTED = (255, 200, 0, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_INFO_PANEL = (30, 30, 45)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 14)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        # --- Board Representation ---
        self.board_dim = min(self.WIDTH, self.HEIGHT) - 40
        self.square_size = self.board_dim // self.BOARD_SIZE
        self.board_offset = ((self.HEIGHT - self.board_dim) // 2, (self.HEIGHT - self.board_dim) // 2)
        self.piece_radius = int(self.square_size * 0.38)
        self.king_radius = int(self.piece_radius * 0.5)

        # --- Game State ---
        self.board = None
        self.kings = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_player = 1
        self.p1_pieces = 12
        self.p2_pieces = 12
        
        # --- UI/Selection State ---
        self.cursor_pos = (0, 0)
        self.selected_piece = None
        self.valid_moves = []
        self.must_jump = False
        self.fading_pieces = [] # For capture animations
        self.last_action_info = ""

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_player = 1 # Player 1 starts
        
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=int)
        self.kings = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=bool)
        
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if (r + c) % 2 == 1:
                    if r < 3:
                        self.board[r, c] = 2 # Player 2
                    elif r > 4:
                        self.board[r, c] = 1 # Player 1

        self.p1_pieces = 12
        self.p2_pieces = 12
        
        self.cursor_pos = (4, 3)
        self.selected_piece = None
        self.valid_moves = []
        self.fading_pieces = []
        self.last_action_info = "Player 1's Turn"
        
        self._update_jump_status()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        move_made = False

        # --- Action Handling ---
        if shift_pressed:
            if self.selected_piece:
                self.selected_piece = None
                self.valid_moves = []
                self.last_action_info = "Selection cancelled."
        
        elif movement != 0:
            dr, dc = [(0,0), (-1,0), (1,0), (0,-1), (0,1)][movement]
            r, c = self.cursor_pos
            self.cursor_pos = ((r + dr) % self.BOARD_SIZE, (c + dc) % self.BOARD_SIZE)

        elif space_pressed:
            if self.selected_piece is None:
                # --- Try to select a piece ---
                piece_owner = self.board[self.cursor_pos]
                if piece_owner == self.current_player:
                    potential_moves = self._calculate_valid_moves_for_piece(self.cursor_pos[0], self.cursor_pos[1])
                    if self.must_jump and not any(move[2] for move in potential_moves):
                        self.last_action_info = "Invalid: A jump is mandatory."
                    elif not potential_moves:
                        self.last_action_info = "This piece has no valid moves."
                    else:
                        self.selected_piece = self.cursor_pos
                        self.valid_moves = potential_moves
                        self.last_action_info = "Piece selected. Choose a move."
                else:
                    self.last_action_info = "Select one of your own pieces."
            else:
                # --- Try to make a move ---
                move = next((m for m in self.valid_moves if m[0] == self.cursor_pos[0] and m[1] == self.cursor_pos[1]), None)
                if move:
                    # --- Execute Move ---
                    from_pos = self.selected_piece
                    to_pos = (move[0], move[1])
                    is_jump = move[2]
                    
                    # Update board
                    piece = self.board[from_pos]
                    is_king = self.kings[from_pos]
                    self.board[from_pos] = 0
                    self.kings[from_pos] = False
                    self.board[to_pos] = piece
                    self.kings[to_pos] = is_king

                    # Handle capture
                    if is_jump:
                        # sound: capture.wav
                        jumped_pos = move[3]
                        self.board[jumped_pos] = 0
                        self.kings[jumped_pos] = False
                        reward += 0.1
                        
                        # Add to fading animation list
                        self.fading_pieces.append({
                            "pos": jumped_pos, 
                            "player": 3 - self.current_player,
                            "is_king": self.kings[jumped_pos],
                            "alpha": 255
                        })
                        
                        if self.current_player == 1:
                            self.p2_pieces -= 1
                        else:
                            self.p1_pieces -= 1
                    
                    # Handle kinging
                    if not is_king and ((piece == 1 and to_pos[0] == 0) or (piece == 2 and to_pos[0] == self.BOARD_SIZE - 1)):
                        # sound: king_promote.wav
                        self.kings[to_pos] = True
                        reward += 5

                    move_made = True
                    self.last_action_info = f"Player {self.current_player} moved."
                else:
                    self.last_action_info = "Invalid move location."

        # --- Post-Move Logic ---
        if move_made:
            self.score += reward
            
            # Check for win condition
            if self.p1_pieces == 0 or self.p2_pieces == 0:
                # sound: win.wav or lose.wav
                self.game_over = True
                reward += 100
                self.score += 100
                winner = 1 if self.p2_pieces == 0 else 2
                self.last_action_info = f"GAME OVER! Player {winner} wins!"
            else:
                # Switch turns
                self.current_player = 3 - self.current_player
                self.selected_piece = None
                self.valid_moves = []
                self._update_jump_status()
                
                # Check if new player has any moves
                if not self._player_has_moves(self.current_player):
                    self.game_over = True
                    reward -= 100 # Penalty for being unable to move
                    self.score -= 100
                    winner = 3 - self.current_player
                    self.last_action_info = f"GAME OVER! Player {self.current_player} has no moves. Player {winner} wins!"
                else:
                    self.last_action_info = f"Player {self.current_player}'s Turn."
                    if self.must_jump:
                        self.last_action_info += " (Must Jump)"


        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.score -= 10 # Draw penalty
            self.last_action_info = "GAME OVER! Step limit reached."
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "p1_pieces": self.p1_pieces,
            "p2_pieces": self.p2_pieces,
            "current_player": self.current_player,
        }

    def _pos_to_pixels(self, r, c):
        x = self.board_offset[1] + c * self.square_size + self.square_size // 2
        y = self.board_offset[0] + r * self.square_size + self.square_size // 2
        return int(x), int(y)

    def _render_game(self):
        # Draw board
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                color = self.COLOR_LIGHT_SQ if (r + c) % 2 == 0 else self.COLOR_DARK_SQ
                pygame.draw.rect(self.screen, color, (self.board_offset[1] + c * self.square_size, self.board_offset[0] + r * self.square_size, self.square_size, self.square_size))

        # Update and draw fading pieces
        for piece in self.fading_pieces[:]:
            piece['alpha'] -= 15
            if piece['alpha'] <= 0:
                self.fading_pieces.remove(piece)
            else:
                r, c = piece['pos']
                is_king = piece['is_king']
                player = piece['player']
                px, py = self._pos_to_pixels(r, c)
                
                if player == 1:
                    p_color = self.COLOR_P1
                    k_color = self.COLOR_P1_KING
                else:
                    p_color = self.COLOR_P2
                    k_color = self.COLOR_P2_KING
                
                # Draw using surfaces for alpha blending
                s = pygame.Surface((self.piece_radius*2, self.piece_radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.aacircle(s, self.piece_radius, self.piece_radius, self.piece_radius, p_color + (piece['alpha'],))
                pygame.gfxdraw.filled_circle(s, self.piece_radius, self.piece_radius, self.piece_radius, p_color + (piece['alpha'],))
                if is_king:
                    pygame.gfxdraw.aacircle(s, self.piece_radius, self.piece_radius, self.king_radius, k_color + (piece['alpha'],))
                    pygame.gfxdraw.filled_circle(s, self.piece_radius, self.piece_radius, self.king_radius, k_color + (piece['alpha'],))
                self.screen.blit(s, (px - self.piece_radius, py - self.piece_radius))

        # Draw pieces
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                player = self.board[r, c]
                if player != 0:
                    px, py = self._pos_to_pixels(r, c)
                    p_color = self.COLOR_P1 if player == 1 else self.COLOR_P2
                    k_color = self.COLOR_P1_KING if player == 1 else self.COLOR_P2_KING
                    
                    pygame.gfxdraw.aacircle(self.screen, px, py, self.piece_radius, p_color)
                    pygame.gfxdraw.filled_circle(self.screen, px, py, self.piece_radius, p_color)
                    if self.kings[r, c]:
                        pygame.gfxdraw.aacircle(self.screen, px, py, self.king_radius, k_color)
                        pygame.gfxdraw.filled_circle(self.screen, px, py, self.king_radius, k_color)
        
        # Draw selected piece highlight
        if self.selected_piece:
            r, c = self.selected_piece
            px, py = self._pos_to_pixels(r, c)
            s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_SELECTED, s.get_rect(), border_radius=5)
            self.screen.blit(s, (self.board_offset[1] + c * self.square_size, self.board_offset[0] + r * self.square_size))

        # Draw valid moves
        for r, c, is_jump, _ in self.valid_moves:
            px, py = self._pos_to_pixels(r, c)
            s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, self.square_size//2, self.square_size//2, self.piece_radius//2, self.COLOR_VALID_MOVE)
            self.screen.blit(s, (px - self.square_size//2, py - self.square_size//2))
        
        # Draw cursor
        r, c = self.cursor_pos
        cursor_rect = (self.board_offset[1] + c * self.square_size, self.board_offset[0] + r * self.square_size, self.square_size, self.square_size)
        s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), 4, border_radius=5)
        self.screen.blit(s, cursor_rect[:2])

    def _render_ui(self):
        panel_rect = (self.board_offset[1] + self.board_dim + 10, 0, self.WIDTH - (self.board_offset[1] + self.board_dim) - 10, self.HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_INFO_PANEL, panel_rect, border_radius=10)
        
        x_base = panel_rect[0] + 20
        y_pos = 20

        # Title
        title_surf = self.font_main.render("CHECKERS", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (x_base, y_pos))
        y_pos += 40
        
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (x_base, y_pos))
        y_pos += 30

        # Steps
        steps_surf = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (x_base, y_pos))
        y_pos += 40

        # Piece Counts
        p1_color = self.COLOR_P1
        p2_color = self.COLOR_P2
        
        p1_surf = self.font_main.render("PLAYER 1", True, p1_color)
        self.screen.blit(p1_surf, (x_base, y_pos))
        y_pos += 25
        p1_count_surf = self.font_main.render(f"Pieces: {self.p1_pieces}", True, self.COLOR_TEXT)
        self.screen.blit(p1_count_surf, (x_base, y_pos))
        y_pos += 50

        p2_surf = self.font_main.render("PLAYER 2", True, p2_color)
        self.screen.blit(p2_surf, (x_base, y_pos))
        y_pos += 25
        p2_count_surf = self.font_main.render(f"Pieces: {self.p2_pieces}", True, self.COLOR_TEXT)
        self.screen.blit(p2_count_surf, (x_base, y_pos))
        y_pos += 50

        # Status Message
        status_surf = self.font_small.render(self.last_action_info, True, self.COLOR_TEXT)
        status_rect = status_surf.get_rect(center=(panel_rect[0] + panel_rect[2]//2, y_pos + 20))
        self.screen.blit(status_surf, status_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.last_action_info.replace("GAME OVER! ", "")
            text_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _is_valid(self, r, c):
        return 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE

    def _calculate_valid_moves_for_piece(self, r, c):
        jumps = self._get_jumps(r, c)
        if self.must_jump:
            return jumps
        
        # If any jump is available on the board, this piece must take it if it can.
        # If this piece can't jump, but another can, this piece has no moves.
        # This is handled by the `must_jump` flag at a higher level.
        # So if we get here, either no jumps are available, or this piece cannot jump.
        
        moves = self._get_moves(r, c)
        return jumps + moves

    def _get_jumps(self, r, c):
        jumps = []
        player = self.board[r, c]
        opponent = 3 - player
        is_king = self.kings[r, c]

        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if not is_king:
            directions = directions[:2] if player == 1 else directions[2:]

        for dr, dc in directions:
            # King's flying jump
            if is_king:
                for i in range(1, self.BOARD_SIZE):
                    path_r, path_c = r + dr * i, c + dc * i
                    if not self._is_valid(path_r, path_c): break
                    if self.board[path_r, path_c] == player: break
                    if self.board[path_r, path_c] == opponent:
                        # Check space behind opponent
                        land_r, land_c = path_r + dr, path_c + dc
                        if self._is_valid(land_r, land_c) and self.board[land_r, land_c] == 0:
                            jumps.append((land_r, land_c, True, (path_r, path_c)))
                        break # Can't jump over more than one piece in a line
            # Normal piece jump
            else:
                jump_over_r, jump_over_c = r + dr, c + dc
                land_r, land_c = r + 2 * dr, c + 2 * dc
                if self._is_valid(land_r, land_c) and self.board[jump_over_r, jump_over_c] == opponent and self.board[land_r, land_c] == 0:
                    jumps.append((land_r, land_c, True, (jump_over_r, jump_over_c)))
        return jumps

    def _get_moves(self, r, c):
        moves = []
        player = self.board[r, c]
        is_king = self.kings[r, c]

        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if not is_king:
            directions = directions[:2] if player == 1 else directions[2:]

        for dr, dc in directions:
            # King's flying move
            if is_king:
                for i in range(1, self.BOARD_SIZE):
                    nr, nc = r + dr * i, c + dc * i
                    if not self._is_valid(nr, nc) or self.board[nr, nc] != 0:
                        break
                    moves.append((nr, nc, False, None))
            # Normal piece move
            else:
                nr, nc = r + dr, c + dc
                if self._is_valid(nr, nc) and self.board[nr, nc] == 0:
                    moves.append((nr, nc, False, None))
        return moves

    def _update_jump_status(self):
        self.must_jump = False
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r, c] == self.current_player:
                    if self._get_jumps(r, c):
                        self.must_jump = True
                        return

    def _player_has_moves(self, player):
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r, c] == player:
                    if self._calculate_valid_moves_for_piece(r, c):
                        return True
        return False
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")