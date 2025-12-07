import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:57:01.169924
# Source Brief: brief_01295.md
# Brief Index: 1295
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A puzzle game where you place falling blocks to create 3x3 squares of solid color. "
        "Complete five color changes before time runs out to win."
    )
    user_guide = (
        "Controls: ←/→ to move, ↑ to rotate clockwise, ↓ to rotate counter-clockwise. "
        "Press space to hard drop the block."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 4
    CELL_SIZE = 80
    GRID_WIDTH = GRID_SIZE * CELL_SIZE
    GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    MAX_STEPS = 1800 # 60 seconds at 30 FPS
    
    # --- Colors ---
    COLOR_BG = (15, 23, 42)
    COLOR_GRID_BG = (23, 37, 64)
    COLOR_GRID_LINES = (51, 65, 85)
    COLOR_TEXT = (226, 232, 240)
    COLOR_TEXT_SHADOW = (15, 23, 42)
    COLOR_TIMER_BAR = (34, 197, 94)
    COLOR_TIMER_BAR_WARN = (239, 68, 68)
    COLOR_SCORE_GLOW = (250, 204, 21, 100)
    COLOR_WIN = (52, 211, 153)
    COLOR_LOSE = (248, 113, 113)

    BLOCK_COLORS = [
        (59, 130, 246),  # Blue (I)
        (234, 179, 8),   # Yellow (L)
        (139, 92, 246),  # Purple (T)
        (239, 68, 68),   # Red (S)
        (16, 185, 129),  # Green (Z)
    ]
    
    CHANGED_SQUARE_COLOR = (253, 224, 71)

    # --- Tetromino Shapes ---
    SHAPES = [
        [[1, 1, 1, 1]], # I
        [[1, 0, 0], [1, 1, 1]], # L
        [[0, 1, 0], [1, 1, 1]], # T
        [[0, 1, 1], [1, 1, 0]], # S
        [[1, 1, 0], [0, 1, 1]], # Z
    ]

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
        self.font_large = pygame.font.Font(None, 60)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = None
        self.changed_squares = None
        self.current_block = None
        self.particles = []
        
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is for debugging, not for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.changed_squares = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.particles = []
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        hard_drop = action[1] == 1
        # shift_held = action[2] == 1 # Unused in this design

        reward = 0
        self.steps += 1
        
        # --- Handle player input ---
        # Using a simple cooldown to prevent overly fast actions
        if self.steps % 3 == 0:
            if movement == 1: # Rotate Clockwise
                self._rotate_block(1)
            elif movement == 2: # Rotate Counter-Clockwise
                self._rotate_block(-1)
            elif movement == 3: # Move Left
                self._move_block(-1)
            elif movement == 4: # Move Right
                self._move_block(1)
        
        if hard_drop:
            # Move down until collision
            while not self._check_collision(self.current_block['shape'], (self.current_block['pos'][0], self.current_block['pos'][1] + 1)):
                self.current_block['pos'][1] += 1
            # Force lock in this step
            self.current_block['fall_timer'] = self.current_block['fall_speed']
            # Sound: Hard drop thud

        # --- Update game logic ---
        self.current_block['fall_timer'] += 1
        if self.current_block['fall_timer'] >= self.current_block['fall_speed']:
            self.current_block['fall_timer'] = 0
            
            new_pos = (self.current_block['pos'][0], self.current_block['pos'][1] + 1)
            if self._check_collision(self.current_block['shape'], new_pos):
                lock_reward = self._lock_block()
                reward += lock_reward
                
                change_reward = self._check_for_color_changes()
                reward += change_reward
                
                if not self.game_over:
                    self._spawn_new_block()
                    if self._check_collision(self.current_block['shape'], self.current_block['pos']):
                        self.game_over = True # Grid is full
                        reward = -100.0
            else:
                self.current_block['pos'][1] += 1
        
        # --- Update particles ---
        self._update_particles()
        
        # --- Check termination conditions ---
        terminated = self.game_over
        truncated = False
        if self.score >= 5:
            self.game_over = True
            terminated = True
            reward = 100.0
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True # Use truncated for time limit
            if self.score < 5:
                reward = -100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_new_block(self):
        shape_idx = self.np_random.integers(0, len(self.SHAPES))
        shape = self.SHAPES[shape_idx]
        color_idx = shape_idx + 1
        
        start_x = (self.GRID_SIZE - len(shape[0])) // 2
        start_y = 0

        self.current_block = {
            'shape': shape,
            'color_idx': color_idx,
            'pos': [start_x, start_y],
            'fall_speed': self.np_random.integers(15, 31), # Slower fall speed
            'fall_timer': 0
        }
        # Sound: New block appears

    def _check_collision(self, shape, pos):
        shape_h = len(shape)
        shape_w = len(shape[0])
        
        for r in range(shape_h):
            for c in range(shape_w):
                if shape[r][c] == 1:
                    grid_x = int(pos[0] + c)
                    grid_y = int(pos[1] + r)
                    
                    if not (0 <= grid_x < self.GRID_SIZE): return True # Out of bounds horizontally
                    if grid_y >= self.GRID_SIZE: return True # Out of bounds vertically
                    if grid_y < 0: continue # Above grid is fine
                    if self.grid[grid_y][grid_x] != 0: return True # Collides with existing block
        return False

    def _lock_block(self):
        # Sound: Block locks
        shape = self.current_block['shape']
        pos = self.current_block['pos']
        color = self.current_block['color_idx']
        
        cells_filled = 0
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c] == 1:
                    grid_x, grid_y = int(pos[0] + c), int(pos[1] + r)
                    if 0 <= grid_y < self.GRID_SIZE and 0 <= grid_x < self.GRID_SIZE:
                        self.grid[grid_y][grid_x] = color
                        cells_filled += 1
                        self._create_particles(grid_x, grid_y, self.BLOCK_COLORS[color - 1], 5)
        
        self.current_block = None
        return cells_filled * 0.1 # Reward for placing a block

    def _check_for_color_changes(self):
        reward = 0
        for r in range(self.GRID_SIZE - 2):
            for c in range(self.GRID_SIZE - 2):
                if not self.changed_squares[r][c]:
                    sub_grid = self.grid[r:r+3, c:c+3]
                    if np.all(sub_grid > 0):
                        first_color = sub_grid[0, 0]
                        if np.all(sub_grid == first_color):
                            self.changed_squares[r:r+3, c:c+3] = True
                            self.score += 1
                            reward += 10.0 # Bigger reward for a successful change
                            # Sound: Color change success
                            # Visual effect for color change
                            for sr in range(3):
                                for sc in range(3):
                                    self._create_particles(c+sc, r+sr, self.CHANGED_SQUARE_COLOR, 20, is_large=True)
        return reward

    def _rotate_block(self, direction):
        if self.current_block is None: return
        
        shape = self.current_block['shape']
        if direction == 1: # Clockwise
            new_shape = [list(row) for row in zip(*shape[::-1])]
        else: # Counter-clockwise
            new_shape = [list(row) for row in zip(*shape)][::-1]

        # Wall kick logic
        original_pos = self.current_block['pos']
        for offset in [0, 1, -1, 2, -2]:
            new_pos = [original_pos[0] + offset, original_pos[1]]
            if not self._check_collision(new_shape, new_pos):
                self.current_block['shape'] = new_shape
                self.current_block['pos'] = new_pos
                # Sound: Rotate
                return
    
    def _move_block(self, dx):
        if self.current_block is None: return
        
        new_pos = [self.current_block['pos'][0] + dx, self.current_block['pos'][1]]
        if not self._check_collision(self.current_block['shape'], new_pos):
            self.current_block['pos'] = new_pos
            # Sound: Move

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT))
        
        # Draw landed blocks
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.grid[r][c]
                if color_idx > 0:
                    color = self.BLOCK_COLORS[color_idx - 1]
                    if self.changed_squares[r][c]:
                        color = self.CHANGED_SQUARE_COLOR
                    self._draw_cell(c, r, color)

        # Draw ghost piece
        if self.current_block and not self.game_over:
            ghost_pos = list(self.current_block['pos'])
            while not self._check_collision(self.current_block['shape'], (ghost_pos[0], ghost_pos[1] + 1)):
                ghost_pos[1] += 1
            
            shape = self.current_block['shape']
            color = self.BLOCK_COLORS[self.current_block['color_idx'] - 1]
            for r in range(len(shape)):
                for c in range(len(shape[0])):
                    if shape[r][c] == 1:
                        self._draw_cell(ghost_pos[0] + c, ghost_pos[1] + r, color, is_ghost=True)

        # Draw current falling block
        if self.current_block and not self.game_over:
            shape = self.current_block['shape']
            pos = self.current_block['pos']
            color = self.BLOCK_COLORS[self.current_block['color_idx'] - 1]
            for r in range(len(shape)):
                for c in range(len(shape[0])):
                    if shape[r][c] == 1:
                        self._draw_cell(pos[0] + c, pos[1] + r, color)

        # Draw grid lines on top
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, 
                             (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE), 
                             (self.GRID_X + self.GRID_WIDTH, self.GRID_Y + i * self.CELL_SIZE), 2)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, 
                             (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y), 
                             (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT), 2)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

    def _draw_cell(self, x, y, color, is_ghost=False):
        px, py = self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE
        
        if is_ghost:
            rect = (px + 2, py + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.draw.rect(self.screen, color, rect, 3, border_radius=5)
        else:
            inset = 4
            border = 8
            rect = (px + inset, py + inset, self.CELL_SIZE - inset*2, self.CELL_SIZE - inset*2)
            pygame.draw.rect(self.screen, tuple(min(255, c + 40) for c in color), rect, 0, border_radius=8)
            pygame.draw.rect(self.screen, color, (px + border, py + border, self.CELL_SIZE - border*2, self.CELL_SIZE - border*2), 0, border_radius=5)

    def _render_ui(self):
        # Score
        score_text = f"CHANGES: {self.score} / 5"
        self._draw_text(score_text, self.font_medium, (self.SCREEN_WIDTH // 2, 30))
        
        # Timer bar
        time_ratio = 1.0 - (self.steps / self.MAX_STEPS)
        bar_width = self.GRID_WIDTH * time_ratio
        bar_color = self.COLOR_TIMER_BAR if time_ratio > 0.25 else self.COLOR_TIMER_BAR_WARN
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X, self.SCREEN_HEIGHT - 40, self.GRID_WIDTH, 20))
        pygame.draw.rect(self.screen, bar_color, (self.GRID_X, self.SCREEN_HEIGHT - 40, bar_width, 20), border_radius=5)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((15, 23, 42, 200))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= 5:
                self._draw_text("VICTORY!", self.font_large, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 30), self.COLOR_WIN)
            else:
                self._draw_text("GAME OVER", self.font_large, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 30), self.COLOR_LOSE)

    def _draw_text(self, text, font, center_pos, color=None, shadow=True):
        if color is None: color = self.COLOR_TEXT
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=center_pos)
        
        if shadow:
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surf.get_rect(center=(center_pos[0] + 2, center_pos[1] + 2))
            self.screen.blit(shadow_surf, shadow_rect)
            
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, grid_x, grid_y, color, count, is_large=False):
        px, py = self.GRID_X + grid_x * self.CELL_SIZE + self.CELL_SIZE/2, self.GRID_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE/2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5) if is_large else self.np_random.uniform(1, 3)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'size': self.np_random.uniform(3, 8) if is_large else self.np_random.uniform(2, 4),
                'life': self.np_random.integers(20, 40),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]

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

if __name__ == '__main__':
    # --- Example Usage & Manual Play ---
    # Set the video driver to a real one for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_over = False
    
    # Create a window for human play
    pygame.display.set_caption("Color Grid Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0]) # No-op, no space, no shift

    while running:
        # --- Human Input ---
        action[0] = 0 # Reset movement action each frame
        action[1] = 0 # Reset hard drop action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if game_over:
                    obs, info = env.reset()
                    game_over = False
                    continue
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_LEFT:
                    action[0] = 3 # Move Left
                if event.key == pygame.K_RIGHT:
                    action[0] = 4 # Move Right
                if event.key == pygame.K_UP:
                    action[0] = 1 # Rotate CW
                if event.key == pygame.K_DOWN:
                    action[0] = 2 # Rotate CCW
                if event.key == pygame.K_SPACE:
                    action[1] = 1 # Hard Drop
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
        
        if not game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            game_over = terminated or truncated

            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for human play

    env.close()