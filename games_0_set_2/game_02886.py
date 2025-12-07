
# Generated: 2025-08-27T21:43:39.013735
# Source Brief: brief_02886.md
# Brief Index: 2886

        
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
    user_guide = "Controls: Use arrow keys to move the cursor. Press space to reveal a square. Hold shift to flag/unflag a square."

    # Must be a short, user-facing description of the game:
    game_description = "A classic mine-clearing puzzle game. Reveal all safe squares without detonating any hidden mines."

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # Constants
    GRID_WIDTH = 6
    GRID_HEIGHT = 5
    NUM_MINES = 10
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (50, 50, 60)
    COLOR_HIDDEN = (70, 70, 85)
    COLOR_REVEALED = (180, 180, 190)
    COLOR_CURSOR = (0, 255, 255)
    COLOR_FLAG = (255, 220, 0)
    COLOR_MINE = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    
    NUM_COLORS = {
        1: (50, 150, 255),
        2: (50, 200, 50),
        3: (255, 50, 50),
        4: (50, 50, 200),
        5: (150, 50, 50),
        6: (50, 200, 200),
        7: (20, 20, 20),
        8: (100, 100, 100)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width = 640
        self.height = 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        
        # Game state variables (initialized in reset)
        self.grid_content = None
        self.grid_state = None
        self.cursor_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.num_safe_squares = (self.GRID_WIDTH * self.GRID_HEIGHT) - self.NUM_MINES
        self.revealed_safe_count = None
        
        self.cell_size = min((self.width - 100) // self.GRID_WIDTH, (self.height - 100) // self.GRID_HEIGHT)
        self.grid_origin_x = (self.width - self.GRID_WIDTH * self.cell_size) // 2
        self.grid_origin_y = (self.height - self.GRID_HEIGHT * self.cell_size) // 2
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.revealed_safe_count = 0
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self._place_mines_and_numbers()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _place_mines_and_numbers(self):
        # 0: hidden, 1: revealed, 2: flagged
        self.grid_state = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        # -1: mine, 0-8: number of adjacent mines
        self.grid_content = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)

        mine_positions = random.sample(range(self.GRID_WIDTH * self.GRID_HEIGHT), self.NUM_MINES)
        for pos in mine_positions:
            x, y = pos % self.GRID_WIDTH, pos // self.GRID_WIDTH
            self.grid_content[x, y] = -1

        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid_content[x, y] == -1:
                    continue
                count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid_content[nx, ny] == -1:
                            count += 1
                self.grid_content[x, y] = count

    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0
        terminated = self.game_over
        
        if not terminated:
            # 1. Handle cursor movement
            if movement == 1: # Up
                self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
            elif movement == 2: # Down
                self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
            elif movement == 3: # Left
                self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
            elif movement == 4: # Right
                self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH

            # 2. Handle actions (Reveal > Flag)
            x, y = self.cursor_pos
            current_state = self.grid_state[x, y]

            if space_held: # Reveal
                if current_state == 0: # Can only reveal hidden squares
                    # SFX: click_sound
                    content = self.grid_content[x, y]
                    if content == -1: # Hit a mine
                        # SFX: explosion_sound
                        self.game_over = True
                        reward = -100
                        self.grid_state[x, y] = 1 # Reveal the mine
                    else: # Hit a safe square
                        initial_revealed = self.revealed_safe_count
                        self._flood_fill_reveal(x, y)
                        newly_revealed = self.revealed_safe_count - initial_revealed
                        reward = float(newly_revealed)
                        
                        if self.revealed_safe_count == self.num_safe_squares:
                            # SFX: win_jingle
                            self.game_over = True
                            self.win = True
                            reward += 100
            
            elif shift_held: # Flag/Unflag
                if current_state == 0: # Flag a hidden square
                    # SFX: flag_place_sound
                    self.grid_state[x, y] = 2
                    if self.grid_content[x, y] != -1:
                        reward = -0.1
                elif current_state == 2: # Unflag
                    # SFX: flag_remove_sound
                    self.grid_state[x, y] = 0

        self.steps += 1
        self.score += reward
        
        if not self.game_over and self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        terminated = self.game_over

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _flood_fill_reveal(self, x, y):
        stack = [(x, y)]
        visited = set()

        while stack:
            cx, cy = stack.pop()
            
            if not (0 <= cx < self.GRID_WIDTH and 0 <= cy < self.GRID_HEIGHT):
                continue
            if (cx, cy) in visited or self.grid_state[cx, cy] != 0:
                continue

            visited.add((cx, cy))
            self.grid_state[cx, cy] = 1
            self.revealed_safe_count += 1

            if self.grid_content[cx, cy] == 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        stack.append((cx + dx, cy + dy))

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid squares
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(
                    self.grid_origin_x + x * self.cell_size,
                    self.grid_origin_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                state = self.grid_state[x, y]
                content = self.grid_content[x, y]

                if state == 0: # Hidden
                    pygame.draw.rect(self.screen, self.COLOR_HIDDEN, rect)
                elif state == 1: # Revealed
                    pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                    if content > 0:
                        num_text = self.font_large.render(str(content), True, self.NUM_COLORS[content])
                        text_rect = num_text.get_rect(center=rect.center)
                        self.screen.blit(num_text, text_rect)
                    elif content == -1 and self.game_over:
                        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.cell_size // 3, self.COLOR_MINE)
                        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, self.cell_size // 3, self.COLOR_MINE)
                elif state == 2: # Flagged
                    pygame.draw.rect(self.screen, self.COLOR_HIDDEN, rect)
                    pole_rect = pygame.Rect(rect.centerx - 2, rect.top + 10, 4, self.cell_size - 20)
                    pygame.draw.rect(self.screen, self.COLOR_FLAG, pole_rect)
                    flag_points = [
                        (rect.centerx + 2, rect.top + 10),
                        (rect.centerx + 2, rect.centery),
                        (rect.centerx - self.cell_size // 3, rect.centery - self.cell_size // 6)
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)
                    pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FLAG)

        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            start = (self.grid_origin_x + i * self.cell_size, self.grid_origin_y)
            end = (self.grid_origin_x + i * self.cell_size, self.grid_origin_y + self.GRID_HEIGHT * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 2)
        for i in range(self.GRID_HEIGHT + 1):
            start = (self.grid_origin_x, self.grid_origin_y + i * self.cell_size)
            end = (self.grid_origin_x + self.GRID_WIDTH * self.cell_size, self.grid_origin_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 2)
            
        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_origin_x + self.cursor_pos[0] * self.cell_size,
            self.grid_origin_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width)

    def _render_ui(self):
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        steps_text = self.font_medium.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.width - steps_text.get_width() - 20, 20))

        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else self.COLOR_MINE
            
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

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
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test game-specific assertions
        assert self.NUM_MINES == 10, "Number of mines should be 10"
        
        self.reset()
        mine_pos_indices = np.where(self.grid_content == -1)
        if len(mine_pos_indices[0]) > 0:
            mx, my = mine_pos_indices[0][0], mine_pos_indices[1][0]
            self.cursor_pos = [mx, my]
            _, reward, terminated, _, _ = self.step([0, 1, 0]) # Reveal action
            assert reward == -100, f"Revealing a mine should yield -100 reward, got {reward}"
            assert terminated, "Revealing a mine should terminate the episode"
        
        self.reset()
        self.revealed_safe_count = self.num_safe_squares - 1
        safe_pos_indices = np.where(self.grid_content != -1)
        if len(safe_pos_indices[0]) > 0:
            sx, sy = safe_pos_indices[0][0], safe_pos_indices[1][0]
            while self.grid_state[sx, sy] == 1:
                sx, sy = random.choice(list(zip(*safe_pos_indices)))
            self.cursor_pos = [sx, sy]
            _, reward, terminated, _, _ = self.step([0, 1, 0])
            if terminated:
                 assert reward >= 100, f"Winning should yield >= +100 reward, got {reward}"

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Save first frame
    img = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    pygame.image.save(img, "frame_000.png")
    
    # Run a few random steps to demonstrate functionality
    for i in range(1, 25):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
        
        img = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        pygame.image.save(img, f"frame_{i:03d}.png")
        
        if terminated:
            print("Episode finished.")
            obs, info = env.reset()
            print("Environment reset.")
            
    env.close()