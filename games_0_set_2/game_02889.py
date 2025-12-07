
# Generated: 2025-08-28T06:18:39.322172
# Source Brief: brief_02889.md
# Brief Index: 2889

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a number and clear matching clusters."
    )

    game_description = (
        "A fast-paced grid-based number matching puzzle. Race against time to clear the board by selecting adjacent identical numbers for points."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_TIME = 60.0  # seconds
    MAX_STEPS = int(MAX_TIME * FPS)

    GRID_COLS, GRID_ROWS = 12, 8
    GRID_LINE_WIDTH = 2
    CELL_WIDTH = 40
    CELL_HEIGHT = 40
    GRID_WIDTH = GRID_COLS * CELL_WIDTH
    GRID_HEIGHT = GRID_ROWS * CELL_HEIGHT
    GRID_X = (WIDTH - GRID_WIDTH) // 2
    GRID_Y = (HEIGHT - GRID_HEIGHT) // 2 + 20

    # --- Colors ---
    COLOR_BG = (25, 28, 32)
    COLOR_GRID = (60, 65, 70)
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TIMER_BG = (80, 80, 80)
    COLOR_TIMER_FG = (100, 200, 255)
    
    NUMBER_COLORS = {
        1: (255, 100, 100), 2: (100, 255, 100), 3: (100, 150, 255),
        4: (255, 255, 100), 5: (255, 100, 255), 6: (100, 255, 255),
        7: (255, 150, 50),  8: (150, 100, 255), 9: (200, 200, 200)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_number = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_ui = pygame.font.SysFont("tahoma", 20, bold=True)

        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.cursor_pos = [0, 0]
        self.timer = 0.0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.last_space_press = False
        
        self.reset()
        self.validate_implementation()
    
    def _generate_grid(self):
        grid = self.np_random.integers(1, 10, size=(self.GRID_ROWS, self.GRID_COLS))
        
        # Guarantee at least two solvable clusters of size 2
        for _ in range(2):
            attempts = 0
            while attempts < 100:
                r = self.np_random.integers(0, self.GRID_ROWS)
                c = self.np_random.integers(0, self.GRID_COLS)
                
                neighbors = []
                if r > 0: neighbors.append((r - 1, c))
                if r < self.GRID_ROWS - 1: neighbors.append((r + 1, c))
                if c > 0: neighbors.append((r, c - 1))
                if c < self.GRID_COLS - 1: neighbors.append((r, c + 1))
                
                if neighbors:
                    nr, nc = random.choice(neighbors)
                    if grid[nr, nc] != grid[r, c]:
                        grid[nr, nc] = grid[r, c]
                        break
                attempts += 1
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_grid()
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.timer = self.MAX_TIME
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.last_space_press = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        terminated = self.game_over

        if not terminated:
            self.steps += 1
            self.timer -= 1.0 / self.FPS
            
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            self._handle_movement(movement)
            
            is_space_pressed = space_held and not self.last_space_press
            if is_space_pressed:
                reward += self._handle_click()
            self.last_space_press = space_held
            
            self._update_particles()
            
            if self.timer <= 0:
                terminated = True
                reward += -50.0 # Loss penalty
                self.game_over = True
            elif np.all(self.grid == 0):
                terminated = True
                reward += 100.0 # Win bonus
                self.game_over = True
            elif self.steps >= self.MAX_STEPS:
                terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_COLS
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_ROWS
            
    def _handle_click(self):
        cx, cy = self.cursor_pos
        target_num = self.grid[cy, cx]
        
        if target_num == 0:
            # Clicked an empty cell
            return -0.01

        # Find cluster using Breadth-First Search (BFS)
        q = [(cy, cx)]
        visited = set([(cy, cx)])
        cluster = []

        while q:
            r, c = q.pop(0)
            cluster.append((r, c))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and \
                   (nr, nc) not in visited and self.grid[nr, nc] == target_num:
                    visited.add((nr, nc))
                    q.append((nr, nc))

        if len(cluster) > 1:
            # Successful match
            # SFX: match_clear.wav
            for r, c in cluster:
                self.grid[r, c] = 0
                self._create_particles(c, r, self.NUMBER_COLORS[target_num])
            
            self.score += len(cluster)
            
            reward = len(cluster) * 0.1
            if len(cluster) >= 4:
                reward += 1.0
            return reward
        else:
            # Clicked a lone number
            # SFX: invalid_click.wav
            return -0.01

    def _create_particles(self, c, r, color):
        px = self.GRID_X + c * self.CELL_WIDTH + self.CELL_WIDTH / 2
        py = self.GRID_Y + r * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 25) # frames
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life, 'color': color, 'radius': radius})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.98 # friction
            p['vel'][1] *= 0.98
            p['life'] -= 1
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + r * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y), self.GRID_LINE_WIDTH)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X + c * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT), self.GRID_LINE_WIDTH)

        # Draw numbers
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                num = self.grid[r, c]
                if num > 0:
                    color = self.NUMBER_COLORS[num]
                    text_surf = self.font_number.render(str(num), True, color)
                    text_rect = text_surf.get_rect(center=(
                        self.GRID_X + c * self.CELL_WIDTH + self.CELL_WIDTH / 2,
                        self.GRID_Y + r * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
                    ))
                    self.screen.blit(text_surf, text_rect)

        # Draw cursor
        cursor_x = self.GRID_X + self.cursor_pos[0] * self.CELL_WIDTH
        cursor_y = self.GRID_Y + self.cursor_pos[1] * self.CELL_HEIGHT
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.CELL_WIDTH, self.CELL_HEIGHT)
        
        s = pygame.Surface((self.CELL_WIDTH, self.CELL_HEIGHT), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, (cursor_x, cursor_y))
        pygame.draw.rect(self.screen, (255,255,255), cursor_rect, 2)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'] * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        # Draw score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Draw timer bar
        timer_bar_width = self.WIDTH - 30
        timer_bar_height = 15
        timer_bar_x = 15
        timer_bar_y = 40
        
        # Background
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BG, (timer_bar_x, timer_bar_y, timer_bar_width, timer_bar_height), border_radius=4)
        
        # Foreground
        current_width = max(0, timer_bar_width * (self.timer / self.MAX_TIME))
        if current_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_TIMER_FG, (timer_bar_x, timer_bar_y, current_width, timer_bar_height), border_radius=4)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cursor_pos": self.cursor_pos,
        }

    def close(self):
        pygame.font.quit()
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

# Example usage to run and visualize the game
if __name__ == '__main__':
    # Set SDL_VIDEODRIVER to a dummy value to run headless
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    # To run with visualization, comment out the line above and uncomment the block below
    # This part is for human play and visualization, not part of the core environment
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for visualization ---
    pygame.display.set_caption("GridNumber Puzzle")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    running = True

    action = env.action_space.sample()
    action.fill(0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = np.array([movement, 1 if space_held else 0, 1 if shift_held else 0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()