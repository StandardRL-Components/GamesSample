
# Generated: 2025-08-27T23:13:52.280736
# Source Brief: brief_03396.md
# Brief Index: 3396

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move selector. Space to select a tile. "
        "Use an arrow key again to swap with an adjacent tile. Shift to cancel selection."
    )

    game_description = (
        "Match 3 or more colored gems to clear them from the board. "
        "Plan your swaps to create combos and clear the level before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.TILE_SIZE = 40
        self.GRID_WIDTH = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.TILE_SIZE
        self.X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20
        self.MAX_STEPS = 2000
        self.MAX_LEVEL = 3

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.COLOR_SELECTED_TILE = (255, 255, 255)
        self.TILE_COLORS = [
            (220, 50, 50),   # Red
            (50, 220, 50),   # Green
            (50, 100, 220),  # Blue
            (220, 220, 50),  # Yellow
            (150, 50, 220),  # Purple
            (220, 120, 50),  # Orange
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.selector_pos = None
        self.selected_tile = None
        self.level = None
        self.score = None
        self.moves_remaining = None
        self.steps = None
        self.game_over = None
        self.just_cleared_coords = None
        self.last_swap_reward = None

        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def _get_initial_moves(self, level):
        return 30 - (level - 1) * 5

    def _get_num_colors(self, level):
        return min(3 + level - 1, len(self.TILE_COLORS))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.level = options.get("level", 1) if options else 1
        self.score = 0
        self.moves_remaining = self._get_initial_moves(self.level)
        self.steps = 0
        self.game_over = False

        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile = None
        self.just_cleared_coords = []
        self.last_swap_reward = 0

        self._generate_valid_grid()

        return self._get_observation(), self._get_info()

    def _generate_valid_grid(self):
        while True:
            self.grid = self.np_random.integers(
                0, self._get_num_colors(self.level), size=(self.GRID_SIZE, self.GRID_SIZE)
            )
            if not self._find_matches() and self._find_possible_moves():
                break

    def _find_possible_moves(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Check swap right
                if c < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r, c + 1] = self.grid[r, c + 1], self.grid[r, c]
                    if self._find_matches():
                        self.grid[r, c], self.grid[r, c + 1] = self.grid[r, c + 1], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r, c + 1] = self.grid[r, c + 1], self.grid[r, c]
                # Check swap down
                if r < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r + 1, c] = self.grid[r + 1, c], self.grid[r, c]
                    if self._find_matches():
                        self.grid[r, c], self.grid[r + 1, c] = self.grid[r + 1, c], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r + 1, c] = self.grid[r + 1, c], self.grid[r, c]
        return False
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.just_cleared_coords.clear()
        reward = 0

        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        if shift_press and self.selected_tile:
            self.selected_tile = None
        elif space_press:
            if self.selected_tile and self.selector_pos == self.selected_tile:
                self.selected_tile = None # Deselect
            else:
                self.selected_tile = list(self.selector_pos) # Select
        elif movement != 0:
            dr = [-1, 1, 0, 0]
            dc = [0, 0, -1, 1]
            move_idx = movement - 1

            if self.selected_tile: # Perform a swap
                target_pos = [self.selected_tile[0] + dr[move_idx], self.selected_tile[1] + dc[move_idx]]
                
                # Check if target is adjacent to selected tile
                dist = abs(target_pos[0] - self.selector_pos[0]) + abs(target_pos[1] - self.selector_pos[1])
                
                if self.selector_pos == target_pos:
                    reward, terminated = self._execute_swap(self.selected_tile, target_pos)
                    self.selected_tile = None
                else: # Invalid swap direction
                    self.selected_tile = None # Cancel selection on invalid move
            else: # Move selector
                self.selector_pos[0] = np.clip(self.selector_pos[0] + dr[move_idx], 0, self.GRID_SIZE - 1)
                self.selector_pos[1] = np.clip(self.selector_pos[1] + dc[move_idx], 0, self.GRID_SIZE - 1)
        
        # If no action resulted in a swap, check for termination from other causes
        if 'terminated' not in locals():
            terminated = self._check_termination()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _execute_swap(self, pos1, pos2):
        self.moves_remaining -= 1
        r1, c1 = pos1
        r2, c2 = pos2

        # Swap tiles in grid
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        # Check for matches
        matches = self._find_matches()
        if not matches:
            # No match, swap back and penalize
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            self.last_swap_reward = -0.1
            return self.last_swap_reward, self._check_termination()

        # Process matches
        combo_multiplier = 1
        total_reward = 0
        while matches:
            # Calculate reward
            num_cleared = len(matches)
            reward_this_turn = num_cleared * combo_multiplier
            if num_cleared > 3:
                reward_this_turn += 5 * combo_multiplier
            total_reward += reward_this_turn
            self.score += int(reward_this_turn * 10) # Scale score for display

            # Clear tiles
            for r, c in matches:
                self.grid[r, c] = -1
                self.just_cleared_coords.append((r, c)) # For visual effect
            
            # Sound: tile_clear.wav

            # Handle gravity and refill
            self._apply_gravity()
            self._refill_grid()
            
            # Check for new matches (combos)
            matches = self._find_matches()
            combo_multiplier += 1
        
        # After all combos, check for anti-softlock
        if not self._find_possible_moves():
            self._generate_valid_grid() # Reshuffle
            # Sound: reshuffle.wav

        self.last_swap_reward = total_reward
        return total_reward, self._check_termination()

    def _find_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r, c + 1] == self.grid[r, c + 2]:
                    matches.update([(r, c), (r, c + 1), (r, c + 2)])
        # Vertical
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r + 1, c] == self.grid[r + 2, c]:
                    matches.update([(r, c), (r + 1, c), (r + 2, c)])
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1

    def _refill_grid(self):
        num_colors = self._get_num_colors(self.level)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, num_colors)

    def _is_level_complete(self):
        return np.all(self.grid == self.grid[0, 0])

    def _check_termination(self):
        if self.game_over:
            return True
        if self.moves_remaining <= 0:
            self.game_over = True
            self.last_swap_reward = -50
            return True
        if self._is_level_complete():
            self.game_over = True
            self.last_swap_reward = 100
            if self.level < self.MAX_LEVEL:
                self.level += 1 # This won't take effect until reset, but is good for info
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
            "level": self.level,
            "moves_remaining": self.moves_remaining,
            "last_swap_reward": self.last_swap_reward,
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.X_OFFSET + i * self.TILE_SIZE, self.Y_OFFSET)
            end_pos = (self.X_OFFSET + i * self.TILE_SIZE, self.Y_OFFSET + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.X_OFFSET, self.Y_OFFSET + i * self.TILE_SIZE)
            end_pos = (self.WIDTH - self.X_OFFSET, self.Y_OFFSET + i * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_val = self.grid[r, c]
                if tile_val != -1:
                    color = self.TILE_COLORS[tile_val]
                    rect = pygame.Rect(
                        self.X_OFFSET + c * self.TILE_SIZE + 1,
                        self.Y_OFFSET + r * self.TILE_SIZE + 1,
                        self.TILE_SIZE - 2,
                        self.TILE_SIZE - 2
                    )
                    pygame.draw.rect(self.screen, color, rect, border_radius=5)
                    
                    # Add a subtle highlight for depth
                    highlight_color = tuple(min(255, x + 40) for x in color)
                    pygame.draw.rect(self.screen, highlight_color, (rect.x, rect.y, rect.width, 4), border_top_left_radius=5, border_top_right_radius=5)

        # Draw visual effects for cleared tiles
        for r, c in self.just_cleared_coords:
            center_x = self.X_OFFSET + c * self.TILE_SIZE + self.TILE_SIZE // 2
            center_y = self.Y_OFFSET + r * self.TILE_SIZE + self.TILE_SIZE // 2
            for i in range(8):
                angle = i * (math.pi / 4) + (self.steps * 0.2) # Animate rotation
                line_len = 15
                start_x = center_x + math.cos(angle) * 5
                start_y = center_y + math.sin(angle) * 5
                end_x = center_x + math.cos(angle) * line_len
                end_y = center_y + math.sin(angle) * line_len
                pygame.draw.line(self.screen, (255, 255, 255), (start_x, start_y), (end_x, end_y), 2)

        # Draw selected tile highlight
        if self.selected_tile:
            r, c = self.selected_tile
            rect = pygame.Rect(
                self.X_OFFSET + c * self.TILE_SIZE,
                self.Y_OFFSET + r * self.TILE_SIZE,
                self.TILE_SIZE,
                self.TILE_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTED_TILE, rect, 4, border_radius=7)

        # Draw selector
        r, c = self.selector_pos
        rect = pygame.Rect(
            self.X_OFFSET + c * self.TILE_SIZE,
            self.Y_OFFSET + r * self.TILE_SIZE,
            self.TILE_SIZE,
            self.TILE_SIZE
        )
        # Pulsing effect for selector
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, width, border_radius=7)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Moves Remaining
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)

        # Level
        level_text = self.font_medium.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
        self.screen.blit(level_text, level_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self._is_level_complete():
                msg = "Level Complete!"
                # Sound: level_complete.wav
            else:
                msg = "Game Over"
                # Sound: game_over.wav

            end_text = self.font_large.render(msg, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
if __name__ == '__main__':
    # Set this to "human" to play the game, or "rgb_array" for no display.
    render_mode = "human"

    if render_mode == "human":
        import os
        os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'windows' or 'x11' or 'dummy'
        pygame.display.set_caption("Gem Swap Environment")
        screen = pygame.display.set_mode((640, 400))

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    # --- Manual Play Loop ---
    selected = False
    while not done:
        action = [0, 0, 0] # Default no-op
        
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    elif event.key == pygame.K_SPACE:
                        action[1] = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        action[2] = 1
                    elif event.key == pygame.K_r: # Manual reset
                        obs, info = env.reset()
                        print("--- Game Reset ---")

        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")
            if done:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Total Steps: {info['steps']}")

        if render_mode == "human":
            # Blit the environment's rendering to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30) # Limit FPS for human play

    env.close()