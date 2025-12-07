
# Generated: 2025-08-28T07:04:23.086124
# Source Brief: brief_03129.md
# Brief Index: 3129

        
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
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to select a gem group. "
        "Clear connected gems of the same color to score."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Select groups of matching gems to clear them from the board. "
        "Clear 100 gems before the time runs out to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 10
    GRID_ROWS = 8
    GEM_SIZE = 40
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_COLS * GEM_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_ROWS * GEM_SIZE) // 2 + 20

    MAX_STEPS = 1200
    WIN_CONDITION_GEMS = 100 # Changed from 10 to make the game more substantial

    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_TEXT = (220, 230, 240)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_HIGHLIGHT = (255, 255, 255)
    
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 150, 50),  # Orange
    ]
    NUM_GEM_TYPES = len(GEM_COLORS)

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        self.cursor_pos = [0, 0]
        self.steps = 0
        self.score = 0
        self.collected_gems = 0
        self.game_over = False
        self.win = False
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.collected_gems = 0
        self.game_over = False
        self.win = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.particles = []

        self._generate_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        # 1. Movement
        if movement != 0:
            last_pos = list(self.cursor_pos)
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            
            # Wrap around grid
            self.cursor_pos[0] %= self.GRID_COLS
            self.cursor_pos[1] %= self.GRID_ROWS
            
            if last_pos != self.cursor_pos:
                # Sound: Cursor move
                pass

        # 2. Gem Selection (Space)
        if space_held:
            match_group = self._find_connected_gems(self.cursor_pos[0], self.cursor_pos[1])
            if len(match_group) > 1:
                # Successful match
                # Sound: Match success
                num_cleared = len(match_group)
                self.collected_gems += num_cleared
                
                # Base reward: +1 per gem
                base_reward = num_cleared
                
                # Bonus for larger groups
                bonus_reward = 0
                if num_cleared >= 4:
                    bonus_reward = 5 + (num_cleared - 4) * 2 # +5 for 4, +7 for 5, etc.
                
                reward += base_reward + bonus_reward
                self.score += base_reward + bonus_reward

                for x, y in match_group:
                    self._create_particles(x, y, self.grid[x, y])
                    self.grid[x, y] = -1 # Mark for clearing
                
                self._apply_gravity_and_refill()

            else:
                # Invalid selection
                # Sound: Match fail
                reward -= 0.1

        self.steps += 1
        self._update_particles()
        
        # --- Termination Check ---
        if self.collected_gems >= self.WIN_CONDITION_GEMS:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100 # Win bonus
            self.score += 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 50 # Lose penalty
            self.score -= 50

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "collected_gems": self.collected_gems}

    # --- Game Logic Helpers ---
    def _generate_grid(self):
        while True:
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    self.grid[c, r] = self.np_random.integers(0, self.NUM_GEM_TYPES)
            if self._check_for_any_match():
                break

    def _check_for_any_match(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if len(self._find_connected_gems(c, r)) > 1:
                    return True
        return False

    def _find_connected_gems(self, start_x, start_y):
        if not (0 <= start_x < self.GRID_COLS and 0 <= start_y < self.GRID_ROWS):
            return []

        target_color = self.grid[start_x, start_y]
        if target_color == -1: return []

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        match_group = []

        while q:
            x, y = q.popleft()
            match_group.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and
                        (nx, ny) not in visited and self.grid[nx, ny] == target_color):
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return match_group

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[c, r] != -1:
                    self.grid[c, empty_row] = self.grid[c, r]
                    empty_row -= 1
            for r in range(empty_row, -1, -1):
                self.grid[c, r] = self.np_random.integers(0, self.NUM_GEM_TYPES)
        
        # Ensure the new board has matches
        if not self._check_for_any_match():
            self._generate_grid() # Regenerate if no matches are possible

    # --- Rendering ---
    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_COLS * self.GEM_SIZE, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_ROWS * self.GEM_SIZE), 1)

        # Highlight potential match
        potential_match = self._find_connected_gems(self.cursor_pos[0], self.cursor_pos[1])
        if len(potential_match) > 1:
            for x, y in potential_match:
                self._draw_gem(x, y, self.grid[x, y], highlight=True)

        # Draw gems
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (c, r) not in potential_match or len(potential_match) <= 1:
                    gem_type = self.grid[c, r]
                    if gem_type != -1:
                        self._draw_gem(c, r, gem_type)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (p['x'], p['y']), int(p['size']))

        # Draw cursor
        cursor_x = self.GRID_X_OFFSET + self.cursor_pos[0] * self.GEM_SIZE
        cursor_y = self.GRID_Y_OFFSET + self.cursor_pos[1] * self.GEM_SIZE
        pulse = (math.sin(self.steps * 0.3) + 1) / 2
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.GEM_SIZE, self.GEM_SIZE), line_width)

    def _draw_gem(self, grid_x, grid_y, gem_type, highlight=False):
        x = self.GRID_X_OFFSET + grid_x * self.GEM_SIZE
        y = self.GRID_Y_OFFSET + grid_y * self.GEM_SIZE
        rect = pygame.Rect(x, y, self.GEM_SIZE, self.GEM_SIZE)
        
        color = self.GEM_COLORS[gem_type]
        padding = 4
        
        # Main gem body
        gem_rect = rect.inflate(-padding*2, -padding*2)
        pygame.gfxdraw.box(self.screen, gem_rect, color)
        
        # Shine effect
        shine_rect = gem_rect.copy()
        shine_rect.height //= 2
        shine_color = (min(255, c + 50) for c in color)
        pygame.gfxdraw.box(self.screen, shine_rect, (*shine_color, 60))

        # Border
        border_color = tuple(min(255, c + 30) for c in color)
        pygame.draw.rect(self.screen, border_color, gem_rect, 1)

        if highlight:
            highlight_rect = rect.inflate(-2, -2)
            pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, highlight_rect, 2, border_radius=4)
    
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Gems Collected
        gems_text = self.font_small.render(f"GEMS: {self.collected_gems} / {self.WIN_CONDITION_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gems_text, (15, 30))

        # Time bar
        time_ratio = 1.0 - (self.steps / self.MAX_STEPS)
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 15
        bar_y = 15
        
        # Bar color changes based on time remaining
        if time_ratio > 0.5:
            bar_color = (100, 220, 100) # Green
        elif time_ratio > 0.2:
            bar_color = (220, 220, 100) # Yellow
        else:
            bar_color = (220, 100, 100) # Red
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, int(bar_width * time_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0, 0))

    # --- Visual Effects ---
    def _create_particles(self, grid_x, grid_y, gem_type):
        center_x = self.GRID_X_OFFSET + grid_x * self.GEM_SIZE + self.GEM_SIZE // 2
        center_y = self.GRID_Y_OFFSET + grid_y * self.GEM_SIZE + self.GEM_SIZE // 2
        color = self.GEM_COLORS[gem_type]

        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': center_x,
                'y': center_y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'size': random.uniform(2, 5),
                'life': random.randint(10, 20),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['size'] -= 0.1
            p['life'] -= 1
        
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]

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
        
        # Test game-specific logic
        self._generate_grid()
        assert self._check_for_any_match(), "Initial grid generation failed to create a valid match."
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Map pygame keys to gymnasium actions
    key_to_action = {
        pygame.K_UP:    1,
        pygame.K_DOWN:  2,
        pygame.K_LEFT:  3,
        pygame.K_RIGHT: 4,
    }

    print(GameEnv.user_guide)

    # We need to render the environment to a window to play
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while running:
        movement = 0
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    movement = key_to_action[event.key]
                if event.key == pygame.K_SPACE:
                    space = 1
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        action = [movement, space, 0] # Shift is not used
        
        # Only step if an action is taken
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
        
        # Render the observation to the display window
        # The observation is (H, W, C), but pygame surfaces are (W, H)
        # We need to transpose back from the gym format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()