
# Generated: 2025-08-27T12:26:33.919582
# Source Brief: brief_00040.md
# Brief Index: 40

        
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
        "Controls: ↑↓←→ to move the cursor. Press space to select a gem and clear matching groups."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically match colored gems on a grid to collect them and achieve the target score within a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 10
        self.NUM_GEM_TYPES = 5
        self.MATCH_MIN_SIZE = 3
        self.MAX_MOVES = 20
        self.GEM_GOAL = 10

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (50, 60, 70)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        
        # --- Sizing and Layout ---
        self.UI_HEIGHT = 50
        self.GRID_AREA_HEIGHT = self.HEIGHT - self.UI_HEIGHT
        self.GEM_SIZE = min(self.WIDTH // self.GRID_COLS, self.GRID_AREA_HEIGHT // self.GRID_ROWS)
        self.GRID_WIDTH = self.GEM_SIZE * self.GRID_COLS
        self.GRID_HEIGHT = self.GEM_SIZE * self.GRID_ROWS
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = self.UI_HEIGHT + (self.GRID_AREA_HEIGHT - self.GRID_HEIGHT) // 2

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
        self.font_ui = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 48, bold=True)

        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.moves_left = 0
        self.gems_collected = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.last_space_press = False
        
        # --- Final Validation ---
        self.validate_implementation()
    
    def _grid_to_pixel(self, r, c):
        x = self.GRID_OFFSET_X + c * self.GEM_SIZE
        y = self.GRID_OFFSET_Y + r * self.GEM_SIZE
        return x, y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.gems_collected = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.last_space_press = False
        
        self._create_valid_grid()

        return self._get_observation(), self._get_info()

    def _create_valid_grid(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
            if self._has_possible_moves():
                break

    def _has_possible_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if len(self._find_connected_gems(r, c)) >= self.MATCH_MIN_SIZE:
                    return True
        return False

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action
        
        # --- Handle Cursor Movement (doesn't consume a move) ---
        if movement != 0:
            # Debounce cursor movement for turn-based play
            if not hasattr(self, '_last_move_step') or self.steps > self._last_move_step:
                self._last_move_step = self.steps
                if movement == 1: self.cursor_pos[0] -= 1  # Up
                elif movement == 2: self.cursor_pos[0] += 1  # Down
                elif movement == 3: self.cursor_pos[1] -= 1  # Left
                elif movement == 4: self.cursor_pos[1] += 1  # Right
                
                # Wrap cursor around
                self.cursor_pos[0] %= self.GRID_ROWS
                self.cursor_pos[1] %= self.GRID_COLS

        # --- Handle Selection (consumes a move) ---
        is_space_press = space_held and not self.last_space_press
        if is_space_press:
            self.moves_left -= 1
            
            r, c = self.cursor_pos
            connected_gems = self._find_connected_gems(r, c)

            if len(connected_gems) >= self.MATCH_MIN_SIZE:
                # --- Successful Match ---
                num_matched = len(connected_gems)
                self.gems_collected += num_matched
                match_reward = num_matched
                if num_matched >= 4:
                    match_reward += 5 # Bonus for larger cluster
                self.score += match_reward
                reward += match_reward

                for r_gem, c_gem in connected_gems:
                    self.grid[r_gem, c_gem] = -1 # Mark for removal
                    # sound: gem_pop.wav
                
                self._apply_gravity_and_refill()
                
                # Reshuffle if no more moves are possible
                if not self._has_possible_moves() and self.gems_collected < self.GEM_GOAL:
                    self._create_valid_grid()
                    # sound: board_shuffle.wav
            else:
                # --- Failed Match ---
                reward += -0.1 # Small penalty for a useless move
                # sound: no_match.wav
        
        self.last_space_press = bool(space_held)

        # --- Check Termination Conditions ---
        if self.gems_collected >= self.GEM_GOAL:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
        elif self.moves_left <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward += -50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _find_connected_gems(self, r, c):
        if not (0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS):
            return []
            
        target_color = self.grid[r, c]
        if target_color == -1: return []

        q = [(r, c)]
        visited = set([(r, c)])
        connected = []

        while q:
            curr_r, curr_c = q.pop(0)
            connected.append((curr_r, curr_c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_r, next_c = curr_r + dr, curr_c + dc
                if (0 <= next_r < self.GRID_ROWS and
                    0 <= next_c < self.GRID_COLS and
                    (next_r, next_c) not in visited and
                    self.grid[next_r, next_c] == target_color):
                    visited.add((next_r, next_c))
                    q.append((next_r, next_c))
        return connected

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
            
            # Refill from top
            for r in range(empty_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "moves_left": self.moves_left,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_gems()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_and_gems(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT), 1)

        # Draw gems
        gem_radius = int(self.GEM_SIZE * 0.4)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    px, py = self._grid_to_pixel(r, c)
                    center_x = px + self.GEM_SIZE // 2
                    center_y = py + self.GEM_SIZE // 2
                    color = self.GEM_COLORS[gem_type]
                    
                    # Draw a filled, anti-aliased circle for the gem
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, gem_radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, gem_radius, color)

    def _render_cursor(self):
        if self.game_over: return
        r, c = self.cursor_pos
        px, py = self._grid_to_pixel(r, c)
        
        # Pulsing effect for the cursor
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
        thickness = 2 + int(pulse * 2)
        
        rect = pygame.Rect(px, py, self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, thickness)

    def _render_ui(self):
        score_text = f"Score: {self.score}"
        gems_text = f"Gems: {self.gems_collected} / {self.GEM_GOAL}"
        moves_text = f"Moves: {self.moves_left}"

        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        gems_surf = self.font_ui.render(gems_text, True, self.COLOR_UI_TEXT)
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_UI_TEXT)

        self.screen.blit(score_surf, (20, 15))
        self.screen.blit(gems_surf, (self.WIDTH // 2 - gems_surf.get_width() // 2, 15))
        self.screen.blit(moves_surf, (self.WIDTH - moves_surf.get_width() - 20, 15))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "YOU WIN!" if self.win else "GAME OVER"
        color = (100, 255, 100) if self.win else (255, 100, 100)
        
        msg_surf = self.font_msg.render(message, True, color)
        msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(msg_surf, msg_rect)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Gem Matcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    game_done = False
    
    print(env.user_guide)
    
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if not game_done:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    elif event.key == pygame.K_SPACE: action[1] = 1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                game_done = False

        if not game_done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if (terminated or truncated) and not game_done:
            print(f"Episode finished. Total Reward: {total_reward}")
            print(info)
            print("Press 'R' to play again, or close the window to quit.")
            game_done = True
            
        clock.tick(15)

    pygame.quit()