
# Generated: 2025-08-27T22:28:03.732681
# Source Brief: brief_03130.md
# Brief Index: 3130

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric match-3 puzzle game where the player swaps adjacent tiles
    to create matches of three or more. The goal is to reach a target score
    within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a tile, "
        "then move to an adjacent tile and press space again to swap."
    )

    game_description = (
        "An isometric match-3 puzzle game. Swap tiles to create matches of three "
        "or more to reach the target score before you run out of moves."
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_COLORS = 5
    MAX_MOVES = 20
    TARGET_SCORE = 5000
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Visuals
    TILE_WIDTH = 60
    TILE_HEIGHT = 30
    TILE_WIDTH_HALF = TILE_WIDTH // 2
    TILE_HEIGHT_HALF = TILE_HEIGHT // 2
    
    # Colors
    COLOR_BG = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 0)
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    TILE_HIGHLIGHT_COLOR = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24)
        self.font_game_over = pygame.font.SysFont("Arial", 64, bold=True)
        
        # Centering the grid
        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.TILE_HEIGHT_HALF) // 2 + 20

        # These will be initialized in reset()
        self.grid = None
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.selected_tile = None
        self.steps = 0
        self.np_random = None

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            # Fallback if seed is None
            self.np_random = np.random.default_rng()

        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_tile = None
        self.steps = 0
        self._create_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_press, _ = action[0], action[1] == 1, action[2] == 1

        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[0] = min(self.GRID_HEIGHT - 1, self.cursor_pos[0] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[1] = min(self.GRID_WIDTH - 1, self.cursor_pos[1] + 1)

        # 2. Handle selection/swap logic
        if space_press:
            if not self.selected_tile:
                # sfx: select_tile.wav
                self.selected_tile = list(self.cursor_pos)
            else:
                r1, c1 = self.selected_tile
                r2, c2 = self.cursor_pos
                
                # Deselect if same tile is pressed again
                if r1 == r2 and c1 == c2:
                    self.selected_tile = None
                # Check for adjacency for a swap
                elif abs(r1 - r2) + abs(c1 - c2) == 1:
                    # sfx: swap.wav
                    reward = self._attempt_swap(r1, c1, r2, c2)
                    self.selected_tile = None
                # If not adjacent, deselect
                else:
                    # sfx: deselect.wav
                    self.selected_tile = None
                    reward = -0.1 # Small penalty for invalid selection

        terminated = self._check_termination()
        if terminated:
            if self.score >= self.TARGET_SCORE:
                reward += 100
            else: # Ran out of moves
                reward -= 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_swap(self, r1, c1, r2, c2):
        self.moves_left -= 1
        
        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        total_tiles_cleared, total_bonus_reward = self._process_cascades()

        if total_tiles_cleared > 0:
            # sfx: match_success.wav
            return total_tiles_cleared + total_bonus_reward
        else:
            # No match, swap back
            # sfx: invalid_swap.wav
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            return -1 # Penalty for a wasted move

    def _process_cascades(self):
        total_tiles_cleared = 0
        total_bonus_reward = 0
        combo_multiplier = 1

        while True:
            matched_coords, bonus_reward = self._find_and_score_matches()
            
            if not matched_coords:
                break

            num_cleared = len(matched_coords)
            total_tiles_cleared += num_cleared
            total_bonus_reward += bonus_reward
            
            # Update game score (different from agent reward)
            self.score += (num_cleared * 10 + bonus_reward * 5) * combo_multiplier
            # sfx: match_clear_wave.wav

            for r, c in matched_coords:
                self.grid[r, c] = -1 # Mark for clearing

            self._apply_gravity()
            self._refill_grid()
            # sfx: tiles_fall.wav
            
            combo_multiplier += 1
        
        return total_tiles_cleared, total_bonus_reward

    def _find_and_score_matches(self):
        matched_coords = set()
        bonus = 0

        # Horizontal matches
        for r in range(self.GRID_HEIGHT):
            c = 0
            while c < self.GRID_WIDTH - 2:
                color = self.grid[r, c]
                if color != -1 and color == self.grid[r, c+1] and color == self.grid[r, c+2]:
                    line_len = 2
                    while c + line_len < self.GRID_WIDTH and self.grid[r, c + line_len] == color:
                        line_len += 1
                    
                    for i in range(line_len):
                        matched_coords.add((r, c + i))
                    
                    if line_len == 4: bonus += 10
                    elif line_len >= 5: bonus += 20
                    
                    c += line_len
                else:
                    c += 1
        
        # Vertical matches
        for c in range(self.GRID_WIDTH):
            r = 0
            while r < self.GRID_HEIGHT - 2:
                color = self.grid[r, c]
                if color != -1 and color == self.grid[r+1, c] and color == self.grid[r+2, c]:
                    line_len = 2
                    while r + line_len < self.GRID_HEIGHT and self.grid[r + line_len, c] == color:
                        line_len += 1

                    for i in range(line_len):
                        matched_coords.add((r + i, c))

                    if line_len == 4: bonus += 10
                    elif line_len >= 5: bonus += 20
                    
                    r += line_len
                else:
                    r += 1
            
        return matched_coords, bonus

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            write_idx = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != write_idx:
                        self.grid[write_idx, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    write_idx -= 1

    def _refill_grid(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_COLORS)

    def _create_grid(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_and_score_matches()[0]:
                break

    def _check_termination(self):
        if self.score >= self.TARGET_SCORE or self.moves_left <= 0:
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
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
            "selected_tile": self.selected_tile,
            "steps": self.steps
        }

    def _iso_to_screen(self, r, c):
        x = self.grid_origin_x + (c - r) * self.TILE_WIDTH_HALF
        y = self.grid_origin_y + (c + r) * self.TILE_HEIGHT_HALF
        return int(x), int(y)

    def _get_tile_points(self, r, c, scale=1.0):
        center_x, center_y = self._iso_to_screen(r, c)
        w = self.TILE_WIDTH_HALF * scale
        h = self.TILE_HEIGHT_HALF * scale
        return [
            (center_x, center_y - h),
            (center_x + w, center_y),
            (center_x, center_y + h),
            (center_x - w, center_y),
        ]

    def _render_game(self):
        # Render tiles from back to front
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r, c]
                if color_idx == -1: continue

                color = self.TILE_COLORS[color_idx]
                points = self._get_tile_points(r, c)
                
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Render selected tile highlight
        if self.selected_tile:
            r, c = self.selected_tile
            pulse = (math.sin(self.steps * 0.4) + 1) / 2 # 0 to 1
            alpha = int(100 + pulse * 100)
            highlight_color = self.TILE_HIGHLIGHT_COLOR + (alpha,)
            points = self._get_tile_points(r, c, scale=1.1)
            pygame.gfxdraw.aapolygon(self.screen, points, highlight_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, highlight_color)
        
        # Render cursor
        r, c = self.cursor_pos
        cursor_points = self._get_tile_points(r, c, scale=1.15)
        pygame.gfxdraw.aapolygon(self.screen, cursor_points, self.COLOR_CURSOR)
        # Draw a second, smaller polygon for thickness
        cursor_points_inner = self._get_tile_points(r, c, scale=1.1)
        pygame.gfxdraw.aapolygon(self.screen, cursor_points_inner, self.COLOR_CURSOR)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Moves Left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        text_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(moves_text, text_rect)

        # Game Over Message
        if self.game_over:
            message = "YOU WIN!" if self.score >= self.TARGET_SCORE else "GAME OVER"
            color = (150, 255, 150) if self.score >= self.TARGET_SCORE else (255, 150, 150)
            
            # Text with a shadow
            game_over_text = self.font_game_over.render(message, True, (0,0,0))
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2 + 3, self.SCREEN_HEIGHT / 2 + 3))
            self.screen.blit(game_over_text, text_rect)

            game_over_text = self.font_game_over.render(message, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def close(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    action = [0, 0, 0] # no-op, no-space, no-shift

    while not done:
        # --- Human controls ---
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
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
            
        # --- Step the environment ---
        # The game only advances on an action
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
        
        # Reset action for next frame
        action = [0, 0, 0]

        # --- Render the game to the window ---
        # The observation is a numpy array, convert it to a Pygame surface
        # Pygame uses (width, height), numpy uses (height, width)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS

    print("Game Over!")
    # Keep the window open for a few seconds to see the final message
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        clock.tick(30)
    
    pygame.quit()