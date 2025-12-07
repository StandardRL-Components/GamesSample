import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a tile. "
        "Space on an adjacent tile to swap. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced match-3 puzzle game. Swap tiles to create matches of three or more. "
        "Clear as many as you can before the time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_DIM = 8
        self.NUM_COLORS = 3
        self.MAX_STEPS = 1800 # 60 seconds at 30fps
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

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
        try:
            self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
            self.font_small = pygame.font.SysFont("Arial", 24)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 30)


        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_BG = (30, 40, 50)
        self.COLOR_GRID_LINES = (50, 60, 70)
        self.TILE_COLORS = [(255, 80, 80), (80, 255, 80), (80, 120, 255)] # Red, Green, Blue
        self.CURSOR_COLOR = (255, 255, 0, 150) # Yellow, semi-transparent
        self.SELECTED_COLOR = (255, 255, 255) # White
        self.UI_TEXT_COLOR = (240, 240, 240)
        self.FLASH_COLOR = (255, 255, 255)

        # Grid layout
        self.GRID_AREA_SIZE = 320
        self.TILE_SIZE = self.GRID_AREA_SIZE // self.GRID_DIM
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_AREA_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_AREA_SIZE) // 2
        
        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_tile_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.rng = None
        self.last_space_held = None
        self.animations = []

        # The original code called these, we keep them for consistency.
        # The fix in reset() will prevent the timeout during validation.
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = np.array([self.GRID_DIM // 2, self.GRID_DIM // 2])
        self.selected_tile_pos = None
        self.last_space_held = False
        self.animations = []

        # Generate a valid initial board (no pre-existing matches and at least one move)
        while True:
            self.grid = self.rng.integers(0, self.NUM_COLORS, size=(self.GRID_DIM, self.GRID_DIM))
            
            # Resolve any matches that were generated at the start
            while True:
                matches = self._check_matches()
                if not matches:
                    break
                for r, c in matches:
                    self.grid[r, c] = -1
                self._apply_gravity()
                self._refill_board()
            
            # Ensure at least one move is possible
            if self._find_possible_moves():
                break
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Input Handling & Action Logic ---

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos = np.clip(self.cursor_pos, 0, self.GRID_DIM - 1)

        # 2. Handle deselect with shift key
        if shift_held:
            self.selected_tile_pos = None

        # 3. Handle selection and swap attempts on space press (rising edge)
        space_press = space_held and not self.last_space_held
        if space_press:
            if self.selected_tile_pos is None:
                # No tile is selected -> select the one under the cursor
                self.selected_tile_pos = tuple(self.cursor_pos)
            else:
                # A tile is already selected -> this action is a swap/deselect/reselect attempt
                p1 = self.selected_tile_pos
                p2 = tuple(self.cursor_pos)

                if p1 == p2:
                    # Pressed space on the same selected tile -> deselect
                    self.selected_tile_pos = None
                elif abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1:
                    # Pressed space on an adjacent tile -> attempt swap
                    reward = self._attempt_swap(p1, p2)
                    self.selected_tile_pos = None  # Clear selection after any swap attempt
                else:
                    # Pressed space on a non-adjacent tile -> change selection
                    self.selected_tile_pos = p2
        
        self.last_space_held = space_held
        
        # --- Game State Update ---
        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True

        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _attempt_swap(self, p1, p2):
        # Perform swap
        self._swap_tiles(p1, p2)

        # Check if swap creates a match
        all_matches = self._check_matches()
        if not all_matches:
            # Invalid move, swap back
            self._swap_tiles(p1, p2)
            return 0 # No reward for invalid move

        # Valid move, start cascade
        total_reward = 0
        chain_level = 1
        while all_matches:
            # Calculate reward and score
            reward_for_cascade = self._calculate_reward(all_matches)
            total_reward += reward_for_cascade * chain_level
            self.score += len(all_matches) * chain_level

            # Create animations and clear tiles
            for r, c in all_matches:
                self.animations.append({'type': 'flash', 'pos': (r, c), 'timer': 10})
                self.grid[r, c] = -1 # Mark as empty
            
            # Apply gravity and refill
            self._apply_gravity()
            self._refill_board()

            # Check for new matches from cascade
            all_matches = self._check_matches()
            if all_matches:
                chain_level += 1
        
        return total_reward

    def _swap_tiles(self, p1, p2):
        self.grid[p1], self.grid[p2] = self.grid[p2], self.grid[p1]

    def _check_matches(self):
        matched_tiles = set()
        # Check horizontal
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    for i in range(3): matched_tiles.add((r, c+i))
                    # Check for longer matches
                    for i in range(3, self.GRID_DIM - c):
                        if self.grid[r, c] == self.grid[r, c+i]:
                            matched_tiles.add((r, c+i))
                        else:
                            break

        # Check vertical
        for c in range(self.GRID_DIM):
            for r in range(self.GRID_DIM - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    for i in range(3): matched_tiles.add((r+i, c))
                    # Check for longer matches
                    for i in range(3, self.GRID_DIM - r):
                        if self.grid[r, c] == self.grid[r+i, c]:
                            matched_tiles.add((r+i, c))
                        else:
                            break
        return matched_tiles

    def _calculate_reward(self, matches):
        num_matched = len(matches)
        reward = num_matched * 0.1
        if num_matched == 4:
            reward += 1
        elif num_matched >= 5:
            reward += 2
        return reward

    def _apply_gravity(self):
        for c in range(self.GRID_DIM):
            empty_row = self.GRID_DIM - 1
            for r in range(self.GRID_DIM - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self._swap_tiles((r, c), (empty_row, c))
                    empty_row -= 1

    def _refill_board(self):
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.rng.integers(0, self.NUM_COLORS)

    def _find_possible_moves(self):
        # Helper to ensure board is not stuck
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                # Try swapping right
                if c < self.GRID_DIM - 1:
                    self._swap_tiles((r, c), (r, c + 1))
                    if self._check_matches():
                        self._swap_tiles((r, c), (r, c + 1)) # Swap back
                        return True
                    self._swap_tiles((r, c), (r, c + 1)) # Swap back
                # Try swapping down
                if r < self.GRID_DIM - 1:
                    self._swap_tiles((r, c), (r + 1, c))
                    if self._check_matches():
                        self._swap_tiles((r, c), (r + 1, c)) # Swap back
                        return True
                    self._swap_tiles((r, c), (r + 1, c)) # Swap back
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_AREA_SIZE, self.GRID_AREA_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw tiles and animations
        self._update_and_draw_animations()
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                tile_color_idx = self.grid[r, c]
                if tile_color_idx != -1:
                    self._draw_tile(r, c, self.TILE_COLORS[tile_color_idx])
        
        # Draw grid lines
        for i in range(self.GRID_DIM + 1):
            # Vertical
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES,
                             (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y),
                             (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y + self.GRID_AREA_SIZE))
            # Horizontal
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES,
                             (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.TILE_SIZE),
                             (self.GRID_OFFSET_X + self.GRID_AREA_SIZE, self.GRID_OFFSET_Y + i * self.TILE_SIZE))

        # Draw cursor
        cursor_x = self.GRID_OFFSET_X + self.cursor_pos[0] * self.TILE_SIZE
        cursor_y = self.GRID_OFFSET_Y + self.cursor_pos[1] * self.TILE_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.TILE_SIZE, self.TILE_SIZE)
        
        # Pulsating effect for cursor
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # Varies between 0 and 1
        alpha = 100 + pulse * 100
        s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, (self.CURSOR_COLOR[0], self.CURSOR_COLOR[1], self.CURSOR_COLOR[2], alpha), s.get_rect(), border_radius=8)
        self.screen.blit(s, (cursor_x, cursor_y))
        pygame.draw.rect(self.screen, self.CURSOR_COLOR[:3], cursor_rect, 2, border_radius=8)


        # Draw selection
        if self.selected_tile_pos:
            sel_x = self.GRID_OFFSET_X + self.selected_tile_pos[0] * self.TILE_SIZE
            sel_y = self.GRID_OFFSET_Y + self.selected_tile_pos[1] * self.TILE_SIZE
            sel_rect = pygame.Rect(sel_x, sel_y, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.SELECTED_COLOR, sel_rect, 4, border_radius=8)

    def _draw_tile(self, r, c, color):
        tile_rect = pygame.Rect(
            self.GRID_OFFSET_X + c * self.TILE_SIZE + 3,
            self.GRID_OFFSET_Y + r * self.TILE_SIZE + 3,
            self.TILE_SIZE - 6,
            self.TILE_SIZE - 6
        )
        pygame.draw.rect(self.screen, color, tile_rect, border_radius=6)
        
        # Add a subtle highlight for 3D effect
        highlight_color = tuple(min(255, x + 40) for x in color)
        pygame.draw.line(self.screen, highlight_color, tile_rect.topleft, tile_rect.topright, 1)
        pygame.draw.line(self.screen, highlight_color, tile_rect.topleft, tile_rect.bottomleft, 1)

    def _update_and_draw_animations(self):
        active_animations = []
        for anim in self.animations:
            if anim['type'] == 'flash':
                r, c = anim['pos']
                alpha = int(255 * (anim['timer'] / 10))
                flash_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                flash_surf.fill((self.FLASH_COLOR[0], self.FLASH_COLOR[1], self.FLASH_COLOR[2], alpha))
                pos_x = self.GRID_OFFSET_X + c * self.TILE_SIZE
                pos_y = self.GRID_OFFSET_Y + r * self.TILE_SIZE
                self.screen.blit(flash_surf, (pos_x, pos_y))

            anim['timer'] -= 1
            if anim['timer'] > 0:
                active_animations.append(anim)
        self.animations = active_animations

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.UI_TEXT_COLOR)
        self.screen.blit(score_text, (20, 10))

        # Time
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_large.render(f"TIME: {time_left // 30}", True, self.UI_TEXT_COLOR)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "TIME'S UP!"
            win_surf = self.font_large.render(win_text, True, (255, 200, 0))
            win_rect = win_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(win_surf, win_rect)

            final_score_surf = self.font_small.render(f"Final Score: {self.score}", True, self.UI_TEXT_COLOR)
            final_score_rect = final_score_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_surf, final_score_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.MAX_STEPS - self.steps,
            "cursor_pos": self.cursor_pos.tolist(),
            "selected_tile": self.selected_tile_pos
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # To see the game window, comment out the os.environ line at the top of the file
    
    env = GameEnv(render_mode="rgb_array")
    
    # To play manually, we need a real display
    try:
        pygame.display.set_caption(env.game_description)
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        manual_play = True
    except pygame.error:
        print("No display available, cannot run manual play. Running a short automated test.")
        manual_play = False

    obs, info = env.reset(seed=42)
    terminated = False
    
    if manual_play:
        # Game loop for manual play
        while not terminated:
            movement = 0 # no-op
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = np.array([movement, space, shift])
            obs, reward, terminated, truncated, info = env.step(action)

            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Run at 30 FPS for smooth visuals
    else:
        # Automated test loop
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps. Final Score: {info['score']}")
                obs, info = env.reset(seed=i)


    print(f"Game Over! Final Score: {info['score']}")
    env.close()