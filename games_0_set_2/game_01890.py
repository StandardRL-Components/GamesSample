
# Generated: 2025-08-27T18:36:40.181836
# Source Brief: brief_01890.md
# Brief Index: 1890

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to clear a selected group of 3 or more matching fruits."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced fruit-matching puzzle game. Clear groups of fruit to score points and trigger cascades. Reach the target score before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 8
    CELL_SIZE = 40
    GRID_X_OFFSET = (640 - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y_OFFSET = (400 - GRID_HEIGHT * CELL_SIZE) + 10

    NUM_FRUIT_TYPES = 5
    MIN_MATCH_SIZE = 3
    WIN_SCORE = 500
    MAX_TIME_STEPS = 1800  # 60 seconds * 30 fps

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HIGHLIGHT = (255, 255, 255)
    COLOR_INVALID_HIGHLIGHT = (255, 80, 80)
    
    FRUIT_COLORS = [
        (255, 80, 80),   # Red (Cherry)
        (80, 255, 80),   # Green (Apple)
        (80, 150, 255),  # Blue (Blueberry)
        (255, 255, 80),  # Yellow (Lemon)
        (200, 80, 255),  # Purple (Grape)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = 640
        self.screen_height = 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 18)
        self.font_huge = pygame.font.SysFont("Arial", 48, bold=True)

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.cursor_pos = [0, 0]
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.win_state = False
        self.particles = []
        self.highlighted_group = []

        self.np_random = None # Will be initialized in reset
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback to a new random generator if seed is not provided
            if self.np_random is None:
                 self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_TIME_STEPS
        self.game_over = False
        self.win_state = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles = []
        
        self._fill_board()
        while self._find_and_remove_all_matches():
            self._apply_gravity()
            self._refill_grid()

        self._update_highlighted_group()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        reward = 0
        terminated = False
        
        self.time_remaining -= 1
        self.steps += 1
        
        # --- Handle Input ---
        moved = self._handle_movement(movement)
        if moved:
            self._update_highlighted_group()
        
        if space_held:
            # sfx: click_sound
            if len(self.highlighted_group) >= self.MIN_MATCH_SIZE:
                chain = 0
                while True:
                    if chain == 0: # First match
                        cleared_count = len(self.highlighted_group)
                        reward += cleared_count
                        self.score += 10 * (cleared_count - 2)
                        self._create_particles(self.highlighted_group)
                        self._remove_fruits(self.highlighted_group)
                        # sfx: match_success_sound
                    else: # Chain reaction
                        # sfx: chain_reaction_sound
                        reward += 5 # Chain bonus
                        self.score += 5 # Chain bonus
                    
                    self._apply_gravity()
                    self._refill_grid()
                    
                    new_matches = self._find_all_matches()
                    if not new_matches:
                        break
                    
                    chain += 1
                    cleared_count = len(new_matches)
                    reward += cleared_count
                    self.score += 10 * (cleared_count - 2)
                    self._create_particles(new_matches)
                    self._remove_fruits(new_matches)

                self._update_highlighted_group()
            else:
                # Invalid move
                # sfx: error_sound
                reward -= 0.1
                self.time_remaining -= 30 # 1 second penalty
        
        # --- Update Game State ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['vy'] += 0.1 # Gravity on particles

        # --- Check Termination ---
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win_state = True
            terminated = True
            reward += 100
            # sfx: win_jingle
        elif self.time_remaining <= 0:
            self.time_remaining = 0
            self.game_over = True
            self.win_state = False
            terminated = True
            reward -= 10
            # sfx: lose_fanfare
        elif self.steps >= 2000: # Max episode length fallback
            self.game_over = True
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Game Logic Helpers ---
    def _handle_movement(self, movement):
        moved = False
        prev_pos = list(self.cursor_pos)
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)
        
        if prev_pos != self.cursor_pos:
            moved = True
            # sfx: cursor_move_sound
        return moved
        
    def _fill_board(self):
        self.grid = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))

    def _find_and_remove_all_matches(self):
        all_matches = self._find_all_matches()
        if all_matches:
            self._remove_fruits(all_matches)
            return True
        return False

    def _find_all_matches(self):
        all_matches = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] != 0:
                    # Check horizontal
                    if x < self.GRID_WIDTH - 2 and self.grid[x, y] == self.grid[x+1, y] == self.grid[x+2, y]:
                        all_matches.update([(x, y), (x+1, y), (x+2, y)])
                    # Check vertical
                    if y < self.GRID_HEIGHT - 2 and self.grid[x, y] == self.grid[x, y+1] == self.grid[x, y+2]:
                        all_matches.update([(x, y), (x, y+1), (x, y+2)])
        return list(all_matches)

    def _update_highlighted_group(self):
        x, y = self.cursor_pos
        if self.grid[x, y] == 0:
            self.highlighted_group = []
            return

        q = [(x, y)]
        visited = set(q)
        group = list(q)
        
        while q:
            cx, cy = q.pop(0)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    if self.grid[nx, ny] == self.grid[x, y]:
                        visited.add((nx, ny))
                        q.append((nx, ny))
                        group.append((nx, ny))
        self.highlighted_group = group

    def _remove_fruits(self, group):
        for x, y in group:
            self.grid[x, y] = 0

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] != 0:
                    if y != empty_row:
                        self.grid[x, empty_row] = self.grid[x, y]
                        self.grid[x, y] = 0
                    empty_row -= 1
    
    def _refill_grid(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] == 0:
                    self.grid[x, y] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)

    def _create_particles(self, group):
        for x, y in group:
            fruit_type = self.grid[x, y]
            if fruit_type == 0: continue
            color = self.FRUIT_COLORS[fruit_type - 1]
            px = self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2
            py = self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
            for _ in range(5):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                self.particles.append({
                    'x': px, 'y': py,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': random.randint(15, 30),
                    'color': color
                })

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_fruits()
        if not self.game_over:
            self._render_highlight()
        self._render_particles_draw()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            end = (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _render_fruits(self):
        radius = self.CELL_SIZE // 2 - 4
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                fruit_type = self.grid[x, y]
                if fruit_type > 0:
                    color = self.FRUIT_COLORS[fruit_type - 1]
                    center_x = self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2
                    center_y = self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
                    
                    # Draw anti-aliased filled circle
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    
                    # Add a little shine
                    shine_x = center_x + radius // 3
                    shine_y = center_y - radius // 3
                    pygame.gfxdraw.filled_circle(self.screen, shine_x, shine_y, radius // 4, (255, 255, 255, 100))

    def _render_highlight(self):
        is_valid_match = len(self.highlighted_group) >= self.MIN_MATCH_SIZE
        highlight_color = self.COLOR_HIGHLIGHT if is_valid_match else self.COLOR_INVALID_HIGHLIGHT
        
        # Draw outline around all fruits in the potential group
        for x, y in self.highlighted_group:
            rect = pygame.Rect(
                self.GRID_X_OFFSET + x * self.CELL_SIZE + 1,
                self.GRID_Y_OFFSET + y * self.CELL_SIZE + 1,
                self.CELL_SIZE - 2,
                self.CELL_SIZE - 2
            )
            pygame.draw.rect(self.screen, highlight_color, rect, 2, border_radius=4)
        
        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + cx * self.CELL_SIZE,
            self.GRID_Y_OFFSET + cy * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, cursor_rect, 3, border_radius=6)
    
    def _render_particles_draw(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = p['color'] + (alpha,)
            size = int(max(1, 6 * (p['life'] / 30.0)))
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['x']) - size, int(p['y']) - size))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))
        
        # Timer
        time_sec = self.time_remaining / 30
        time_color = self.COLOR_UI_TEXT if time_sec > 10 else (255, 100, 100)
        time_text = self.font_large.render(f"Time: {time_sec:.1f}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.screen_width - 15, 15))
        self.screen.blit(time_text, time_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "YOU WIN!" if self.win_state else "TIME'S UP!"
        color = (100, 255, 100) if self.win_state else (255, 100, 100)
        
        title_text = self.font_huge.render(message, True, color)
        title_rect = title_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 - 30))
        
        score_text = self.font_large.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 30))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(title_text, title_rect)
        self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "cursor_pos": list(self.cursor_pos),
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
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Requires pygame to be installed with display drivers
    import os
    if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
         print("Cannot run interactive test in dummy mode. Exiting.")
         exit()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Fruit Cascade")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # In manual play, we only step when an action is taken
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            if terminated:
                print("Game Over!")
                pygame.time.wait(2000) # Pause for 2 seconds
                obs, info = env.reset()
                total_reward = 0

        # We need to get the latest observation even if no action is taken
        # to update the highlight from cursor movement
        else:
            obs = env._get_observation()
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    env.close()