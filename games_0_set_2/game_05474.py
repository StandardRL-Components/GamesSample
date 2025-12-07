
# Generated: 2025-08-28T05:08:31.444176
# Source Brief: brief_05474.md
# Brief Index: 5474

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press SHIFT to cycle through colors. Press SPACE to paint the selected tile and all connected same-colored tiles."
    )

    game_description = (
        "A strategic puzzle game. Your goal is to make the entire grid a single color within a limited number of moves. Choose your colors and tiles wisely to flood-fill the board."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_SIZE = 12
        self.NUM_COLORS = 4
        self.MAX_MOVES = 25
        self.MAX_STEPS = 1000
        self.WIDTH, self.HEIGHT = 640, 400

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 255)
        self.TILE_COLORS = [
            (231, 76, 60),    # Red
            (46, 204, 113),   # Green
            (52, 152, 219),  # Blue
            (241, 196, 15),   # Yellow
        ]

        # Exact spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18)

        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.moves_left = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.anim_tick = 0
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # Initialize game state
        self.grid = self.rng.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_color_idx = 0
        self.moves_left = self.MAX_MOVES
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Input state
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # 1. Cursor Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Cursor wrapping
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE

        # 2. Color Selection (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_color_idx = (self.selected_color_idx + 1) % self.NUM_COLORS
            # sfx: UI_CYCLE_COLOR

        # 3. Paint Action (on press)
        if space_held and not self.prev_space_held and self.moves_left > 0:
            self.moves_left -= 1
            reward -= 1 # Cost of making a move

            target_x, target_y = self.cursor_pos
            original_color = self.grid[target_y, target_x]
            new_color = self.selected_color_idx

            if original_color != new_color:
                # sfx: PAINT_SPLAT
                changed_tiles_count = self._flood_fill(target_x, target_y, new_color, original_color)
                
                # Reward based on expanding the anchor (top-left) color region
                anchor_color = self.grid[0, 0]
                if new_color == anchor_color:
                    reward += changed_tiles_count * 0.5  # Good move
                else:
                    reward -= changed_tiles_count * 0.1 # Potentially bad move
            else:
                # sfx: ACTION_FAIL
                reward -= 2 # Penalty for useless move

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Termination Check ---
        terminated = False
        win = self._check_win()
        if win:
            reward += 100
            terminated = True
            self.game_over = True
            # sfx: GAME_WIN
        elif self.moves_left <= 0:
            reward -= 50
            terminated = True
            self.game_over = True
            # sfx: GAME_LOSE
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _flood_fill(self, x, y, new_color, original_color):
        if new_color == original_color:
            return 0
        
        q = deque([(x, y)])
        count = 0
        
        while q:
            cx, cy = q.popleft()
            if not (0 <= cx < self.GRID_SIZE and 0 <= cy < self.GRID_SIZE):
                continue
            if self.grid[cy, cx] == original_color:
                self.grid[cy, cx] = new_color
                count += 1
                q.append((cx + 1, cy))
                q.append((cx - 1, cy))
                q.append((cx, cy + 1))
                q.append((cx, cy - 1))
        return count

    def _check_win(self):
        first_color = self.grid[0, 0]
        return np.all(self.grid == first_color)

    def _get_observation(self):
        self.anim_tick += 1
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        grid_pixel_size = 360
        self.tile_size = grid_pixel_size // self.GRID_SIZE
        self.grid_offset_x = (self.WIDTH - grid_pixel_size) // 2
        self.grid_offset_y = (self.HEIGHT - grid_pixel_size) // 2

        # Draw tiles
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = self.grid[y, x]
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.tile_size,
                    self.grid_offset_y + y * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                pygame.draw.rect(self.screen, self.TILE_COLORS[color_idx], rect)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.grid_offset_x + i * self.tile_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.tile_size, self.grid_offset_y + grid_pixel_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.tile_size)
            end_pos = (self.grid_offset_x + grid_pixel_size, self.grid_offset_y + i * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_cursor(self):
        cursor_x, cursor_y = self.cursor_pos
        rect = pygame.Rect(
            self.grid_offset_x + cursor_x * self.tile_size,
            self.grid_offset_y + cursor_y * self.tile_size,
            self.tile_size,
            self.tile_size
        )
        
        # Pulsing effect for cursor
        pulse = (math.sin(self.anim_tick * 0.2) + 1) / 2 # 0 to 1
        line_width = int(2 + pulse * 2)
        
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, line_width)

    def _render_ui(self):
        # Moves Left display
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 15, 10))
        
        # Score display
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Color Palette display
        palette_width = self.NUM_COLORS * 40 + (self.NUM_COLORS - 1) * 10
        start_x = (self.WIDTH - palette_width) // 2
        
        for i, color in enumerate(self.TILE_COLORS):
            rect = pygame.Rect(start_x + i * 50, self.HEIGHT - 45, 40, 30)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            
            if i == self.selected_color_idx:
                # Highlight selected color
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 3, border_radius=5)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self._check_win() else "OUT OF MOVES"
            end_text = self.font_main.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "win": self._check_win() if not self.game_over else False,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

# Example usage:
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Color Flood")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    running = True
    
    # Map keyboard keys to MultiDiscrete actions
    # action = [movement, space, shift]
    action = np.array([0, 0, 0])
    
    print(env.user_guide)

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                action.fill(0) # Reset action state

        # --- Key State Polling ---
        keys = pygame.key.get_pressed()
        
        # Movement (only one direction at a time)
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0

        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # --- Step the Environment ---
        # Since auto_advance is False, we need to send an action to see a change.
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()

        # --- Rendering ---
        # The observation is the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()