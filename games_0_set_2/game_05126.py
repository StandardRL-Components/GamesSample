
# Generated: 2025-08-28T04:04:40.036952
# Source Brief: brief_05126.md
# Brief Index: 5126

        
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
        "Controls: Arrows to move cursor. Space to select/swap. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap gems to create matches of 3 or more. Collect 100 gems in 50 moves to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.GEM_SIZE = 40
        self.NUM_GEM_TYPES = 6
        self.WIN_SCORE = 100
        self.MAX_MOVES = 50

        self.GRID_WIDTH = self.GRID_COLS * self.GEM_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.GEM_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20 # Move down for UI

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_BG = (30, 35, 50)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_SELECT = (255, 255, 0, 150)
        self.COLOR_TEXT = (220, 220, 230)
        self.GEM_COLORS = [
            (255, 50, 50),   # Red
            (50, 255, 50),   # Green
            (80, 80, 255),   # Blue
            (255, 255, 50),  # Yellow
            (200, 50, 255),  # Purple
            (255, 150, 50),  # Orange
        ]
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('sans-serif', 36, bold=True)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.space_was_held = False
        self.last_match_coords = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._initialize_grid()
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_pos = None
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.space_was_held = False
        self.last_match_coords = []
        
        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
        while self._find_all_match_lines():
            # Re-create grid until no matches exist on spawn
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        is_space_click = space_held and not self.space_was_held
        self.space_was_held = space_held
        
        self.last_match_coords = []

        if movement != 0:
            dr = [-1, 1, 0, 0] # Up, Down
            dc = [0, 0, -1, 1] # Left, Right
            self.cursor_pos[0] = (self.cursor_pos[0] + dr[movement - 1]) % self.GRID_ROWS
            self.cursor_pos[1] = (self.cursor_pos[1] + dc[movement - 1]) % self.GRID_COLS
        
        if shift_held:
            self.selected_pos = None

        if is_space_click:
            if self.selected_pos is None:
                # sound: select.wav
                self.selected_pos = list(self.cursor_pos)
            else:
                if self._is_adjacent(self.selected_pos, self.cursor_pos):
                    # This is a swap attempt, which is the main turn-based action
                    return self._handle_swap(self.selected_pos, self.cursor_pos)
                else:
                    # Selected a non-adjacent gem, so just re-select
                    # sound: select.wav
                    self.selected_pos = list(self.cursor_pos)

        # If no swap was attempted, it's a "zero-cost" action
        return self._get_observation(), 0.0, False, False, self._get_info()

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _handle_swap(self, pos1, pos2):
        self.moves_left -= 1
        reward = 0.0
        
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        all_match_lines = self._find_all_match_lines()
        
        if not all_match_lines:
            # sound: invalid_swap.wav
            # Invalid swap, swap back immediately.
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            self.selected_pos = None
        else:
            # Valid swap, process initial matches
            reward += self._process_matches(all_match_lines)
            
            # Process chain reactions
            while True:
                self._drop_and_refill()
                new_match_lines = self._find_all_match_lines()
                if not new_match_lines:
                    break
                # sound: chain_reaction.wav
                reward += self._process_matches(new_match_lines)

            self.selected_pos = None
        
        terminated = self.score >= self.WIN_SCORE or self.moves_left <= 0
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Win bonus
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_all_match_lines(self):
        match_lines = []
        
        # Horizontal check
        for r in range(self.GRID_ROWS):
            c = 0
            while c < self.GRID_COLS - 2:
                gem_type = self.grid[r, c]
                if gem_type == -1:
                    c += 1
                    continue
                if self.grid[r, c+1] == gem_type and self.grid[r, c+2] == gem_type:
                    line = [(r, c), (r, c+1), (r, c+2)]
                    c_scan = c + 3
                    while c_scan < self.GRID_COLS and self.grid[r, c_scan] == gem_type:
                        line.append((r, c_scan))
                        c_scan += 1
                    match_lines.append(line)
                    c = c_scan
                else:
                    c += 1
        
        # Vertical check
        for c in range(self.GRID_COLS):
            r = 0
            while r < self.GRID_ROWS - 2:
                gem_type = self.grid[r, c]
                if gem_type == -1:
                    r += 1
                    continue
                if self.grid[r+1, c] == gem_type and self.grid[r+2, c] == gem_type:
                    line = [(r, c), (r+1, c), (r+2, c)]
                    r_scan = r + 3
                    while r_scan < self.GRID_ROWS and self.grid[r_scan, c] == gem_type:
                        line.append((r_scan, c))
                        r_scan += 1
                    match_lines.append(line)
                    r = r_scan
                else:
                    r += 1
        return match_lines

    def _process_matches(self, all_match_lines):
        reward = 0.0
        gems_to_remove = set()
        for line in all_match_lines:
            gems_to_remove.update(map(tuple, line))
            if len(line) == 3: reward += 5.0
            elif len(line) == 4: reward += 10.0
            else: reward += 20.0
        
        if not gems_to_remove:
            return 0.0
            
        reward += len(gems_to_remove) # +1 per gem
        self.score += len(gems_to_remove)
        self.last_match_coords.extend(list(gems_to_remove))
        
        for r, c in gems_to_remove:
            self.grid[r, c] = -1 # Mark for removal
        # sound: match.wav
        return reward

    def _drop_and_refill(self):
        # Drop existing gems down
        for c in range(self.GRID_COLS):
            write_idx = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != write_idx:
                        self.grid[write_idx, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    write_idx -= 1
        
        # Refill empty spots at the top
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw gems
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    self._draw_gem(gem_type, r, c)
        
        # Draw selection highlight
        if self.selected_pos:
            r, c = self.selected_pos
            rect = pygame.Rect(
                self.GRID_X_OFFSET + c * self.GEM_SIZE,
                self.GRID_Y_OFFSET + r * self.GEM_SIZE,
                self.GEM_SIZE, self.GEM_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect.inflate(6, 6), 3, border_radius=8)

        # Draw cursor
        cur_r, cur_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + cur_c * self.GEM_SIZE,
            self.GRID_Y_OFFSET + cur_r * self.GEM_SIZE,
            self.GEM_SIZE, self.GEM_SIZE
        )
        s = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), 5, border_radius=6)
        self.screen.blit(s, cursor_rect.topleft)

        # Draw match sparkles
        if self.last_match_coords:
            for r, c in self.last_match_coords:
                center_x = self.GRID_X_OFFSET + c * self.GEM_SIZE + self.GEM_SIZE // 2
                center_y = self.GRID_Y_OFFSET + r * self.GEM_SIZE + self.GEM_SIZE // 2
                for i in range(5):
                    angle = (i / 5) * 2 * math.pi + (pygame.time.get_ticks() / 200)
                    len1 = 10 + 5 * math.sin(pygame.time.get_ticks() / 100)
                    len2 = len1 * 0.5
                    start_pos = (center_x + len2 * math.cos(angle), center_y + len2 * math.sin(angle))
                    end_pos = (center_x + len1 * math.cos(angle), center_y + len1 * math.sin(angle))
                    pygame.draw.line(self.screen, (255, 255, 200), start_pos, end_pos, 2)

    def _draw_gem(self, gem_type, r, c):
        rect = pygame.Rect(
            self.GRID_X_OFFSET + c * self.GEM_SIZE,
            self.GRID_Y_OFFSET + r * self.GEM_SIZE,
            self.GEM_SIZE, self.GEM_SIZE
        )
        center = rect.center
        radius = self.GEM_SIZE // 2 - 5
        color = self.GEM_COLORS[gem_type]
        
        if gem_type == 0: # Circle
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color)
        elif gem_type == 1: # Square
            inner_rect = rect.inflate(-10, -10)
            pygame.draw.rect(self.screen, color, inner_rect, border_radius=3)
        elif gem_type == 2: # Triangle Up
            points = [(center[0], center[1] - radius), 
                      (center[0] - radius, center[1] + radius*0.7), 
                      (center[0] + radius, center[1] + radius*0.7)]
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)
        elif gem_type == 3: # Diamond
            points = [(center[0], center[1] - radius), (center[0] + radius, center[1]), 
                      (center[0], center[1] + radius), (center[0] - radius, center[1])]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 4: # Hexagon
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 5: # Triangle Down
            points = [(center[0], center[1] + radius), 
                      (center[0] - radius, center[1] - radius*0.7), 
                      (center[0] + radius, center[1] - radius*0.7)]
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)
        
        # Add a subtle shine
        shine_rect = pygame.Rect(0, 0, radius, radius)
        shine_rect.center = (center[0] - radius * 0.2, center[1] - radius * 0.2)
        s = pygame.Surface(shine_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(s, (255,255,255,60), s.get_rect())
        self.screen.blit(s, shine_rect)


    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                end_text = self.font_large.render("You Win!", True, (255, 255, 100))
            else:
                end_text = self.font_large.render("Game Over", True, (255, 100, 100))
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "steps": 0 # Not meaningful for this turn-based game
        }

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # To run the game interactively
    pygame.display.set_caption("Gem Swap Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # No-op, release space, release shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward > 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Redraw one last time to show final state
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(3000) # Pause for 3 seconds before reset
            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for interactive play

    env.close()