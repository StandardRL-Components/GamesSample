
# Generated: 2025-08-28T03:22:11.148588
# Source Brief: brief_04908.md
# Brief Index: 4908

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to select and clear a group of same-colored squares."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the board by selecting groups of two or more adjacent, same-colored squares. Plan your moves to maximize your score before you run out of turns!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_ROWS = 5
    GRID_COLS = 5
    NUM_COLORS = 5
    MAX_MOVES = 20
    
    # Colors
    COLOR_BG = (25, 28, 32)
    COLOR_GRID = (60, 65, 75)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_CURSOR = (255, 255, 255)
    
    SQUARE_COLORS = [
        (231, 76, 60),    # Red
        (46, 204, 113),   # Green
        (52, 152, 219),   # Blue
        (241, 196, 15),   # Yellow
        (155, 89, 182),   # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # Game state variables
        self.grid = None
        self.cursor_pos = None
        self.moves_left = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.previous_space_held = False
        self.animations = [] # For particle effects
        self.np_random = None

        # Calculate grid geometry
        self.grid_area_height = self.SCREEN_HEIGHT - 80
        self.cell_size = min(
            (self.SCREEN_WIDTH - 100) // self.GRID_COLS, 
            self.grid_area_height // self.GRID_ROWS
        )
        self.grid_width = self.cell_size * self.GRID_COLS
        self.grid_height = self.cell_size * self.GRID_ROWS
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2 + 20
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.previous_space_held = False
        self.animations = []
        
        # Generate a board with at least one valid move
        while True:
            self._generate_board()
            if self._check_for_any_valid_move():
                break
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        if not self.game_over:
            # --- Handle Cursor Movement ---
            if movement == 1:  # Up
                self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_ROWS) % self.GRID_ROWS
            elif movement == 2:  # Down
                self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_ROWS
            elif movement == 3:  # Left
                self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_COLS) % self.GRID_COLS
            elif movement == 4:  # Right
                self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_COLS

            # --- Handle Click Action (on press, not hold) ---
            is_click = space_held and not self.previous_space_held
            if is_click:
                self.moves_left -= 1
                row, col = self.cursor_pos
                color_index = self.grid[row][col]

                if color_index > 0:
                    connected = self._find_connected_squares(row, col)
                    if len(connected) > 1:
                        # Valid move
                        num_cleared = len(connected)
                        reward = num_cleared  # +1 per square
                        self.score += num_cleared * num_cleared # Bonus for larger groups
                        
                        for r, c in connected:
                            self._add_particle_effect(r, c, self.SQUARE_COLORS[self.grid[r][c] - 1])
                            self.grid[r][c] = 0 # Mark as empty
                        
                        # sfx: block_clear.wav
                        self._apply_gravity_and_refill()
                    else:
                        # Invalid move (single block)
                        reward = -0.2
                        # sfx: invalid_move.wav
                else:
                    # Clicked an empty square
                    reward = -0.2
                    # sfx: invalid_move.wav
        
        self.previous_space_held = space_held
        self.steps += 1

        # --- Update Animations ---
        self._update_animations()

        # --- Check Termination Conditions ---
        board_cleared = self._is_board_clear()
        no_moves_left = self.moves_left <= 0
        no_valid_moves = not self._check_for_any_valid_move()

        if board_cleared:
            reward += 100
            terminated = True
            self.game_over = True
            self.win_message = "BOARD CLEARED!"
            # sfx: win_jingle.wav
        elif no_moves_left or no_valid_moves:
            terminated = True
            self.game_over = True
            self.win_message = "GAME OVER"
            # sfx: lose_sound.wav
        
        if self.steps >= 1000: # Max episode length
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_ROWS + 1):
            y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_width, y), 1)
        for i in range(self.GRID_COLS + 1):
            x = self.grid_offset_x + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_height), 1)

        # Draw squares
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_index = self.grid[r][c]
                if color_index > 0:
                    color = self.SQUARE_COLORS[color_index - 1]
                    rect = pygame.Rect(
                        self.grid_offset_x + c * self.cell_size + 2,
                        self.grid_offset_y + r * self.cell_size + 2,
                        self.cell_size - 4,
                        self.cell_size - 4
                    )
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Draw animations
        for particle in self.animations:
            particle['timer'] -= 1
            if particle['timer'] > 0:
                alpha = int(255 * (particle['timer'] / particle['lifespan']))
                size = int(particle['size'] * (particle['timer'] / particle['lifespan']))
                surf = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.rect(surf, particle['color'] + (alpha,), surf.get_rect(), border_radius=3)
                self.screen.blit(surf, (particle['x'] - size // 2, particle['y'] - size // 2))

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cursor_c * self.cell_size,
            self.grid_offset_y + cursor_r * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        # Pulsing effect for cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2 * 50 + 205
        pygame.draw.rect(self.screen, (pulse, pulse, pulse), cursor_rect, 4, border_radius=6)

    def _render_ui(self):
        # Render Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, self.font_medium, self.COLOR_TEXT, 22, 22, shadow=True)
        
        # Render Moves Left
        moves_text = f"MOVES: {self.moves_left}"
        text_width = self.font_medium.size(moves_text)[0]
        self._draw_text(moves_text, self.font_medium, self.COLOR_TEXT, self.SCREEN_WIDTH - text_width - 20, 22, shadow=True)

        # Render Game Over/Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            text_width, text_height = self.font_large.size(self.win_message)
            self._draw_text(self.win_message, self.font_large, self.COLOR_TEXT, 
                            (self.SCREEN_WIDTH - text_width) / 2, 
                            (self.SCREEN_HEIGHT - text_height) / 2, shadow=True)

    def _draw_text(self, text, font, color, x, y, shadow=False):
        if shadow:
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (x + 2, y + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, (x, y))

    def _generate_board(self):
        if self.np_random is None: self.np_random = np.random.default_rng()
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_ROWS, self.GRID_COLS)).tolist()

    def _find_connected_squares(self, start_row, start_col):
        target_color = self.grid[start_row][start_col]
        if target_color == 0:
            return []

        q = [(start_row, start_col)]
        visited = set(q)
        connected = []

        while q:
            r, c = q.pop(0)
            connected.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                    if (nr, nc) not in visited and self.grid[nr][nc] == target_color:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return connected

    def _apply_gravity_and_refill(self):
        if self.np_random is None: self.np_random = np.random.default_rng()
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] != 0:
                    self.grid[empty_row][c], self.grid[r][c] = self.grid[r][c], self.grid[empty_row][c]
                    empty_row -= 1
            
            # Refill from top
            for r in range(empty_row, -1, -1):
                self.grid[r][c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _check_for_any_valid_move(self):
        visited = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (r, c) not in visited and self.grid[r][c] != 0:
                    connected = self._find_connected_squares(r, c)
                    if len(connected) > 1:
                        return True
                    for cell in connected:
                        visited.add(cell)
        return False

    def _is_board_clear(self):
        return all(self.grid[r][c] == 0 for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS))
    
    def _add_particle_effect(self, r, c, color):
        lifespan = 20 # frames
        x = self.grid_offset_x + c * self.cell_size + self.cell_size // 2
        y = self.grid_offset_y + r * self.cell_size + self.cell_size // 2
        self.animations.append({
            'x': x, 'y': y, 'color': color, 'timer': lifespan, 
            'lifespan': lifespan, 'size': self.cell_size - 4
        })

    def _update_animations(self):
        self.animations = [p for p in self.animations if p['timer'] > 0]

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # To display the game, we need a separate pygame window
    pygame.display.set_caption("Click-a-Square")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    while running:
        # Map pygame keys to gymnasium actions
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Since auto_advance is False, we only step on an event
        # For human play, we step every frame a key is potentially pressed
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            terminated = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

    pygame.quit()