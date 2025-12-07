
# Generated: 2025-08-28T01:46:41.182043
# Source Brief: brief_04222.md
# Brief Index: 4222

        
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


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = np_random.integers(20, 40)
        self.radius = np_random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.radius -= 0.1
        return self.life > 0 and self.radius > 0

    def draw(self, surface):
        if self.life > 0 and self.radius > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.radius))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to push rows/columns. Use Shift/Space to change which row/column is selected."
    )

    game_description = (
        "Push blocks to create lines of five identical colors to clear them. Clear all blocks before you run out of moves!"
    )

    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 5
    CELL_SIZE = 60
    GRID_LINE_WIDTH = 2
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_SIZE * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_SIZE * CELL_SIZE) // 2 + 20

    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (50, 60, 70)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_SELECTOR = (255, 255, 100, 80)
    BLOCK_COLORS = [
        (0, 0, 0),  # 0 is empty
        (52, 152, 219),  # Blue
        (231, 76, 60),   # Red
        (46, 204, 113),  # Green
        (241, 196, 15),  # Yellow
    ]
    INITIAL_MOVES = 50
    ANIMATION_SPEED = 0.2 # Slower for better visual feedback

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.visual_grid_state = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.animations = []
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = self.INITIAL_MOVES
        
        self.selected_row = self.np_random.integers(0, self.GRID_SIZE)
        self.selected_col = self.np_random.integers(0, self.GRID_SIZE)
        
        self.animations.clear()
        self.particles.clear()
        
        self._populate_grid()
        self.visual_grid_state = self.grid.copy()
        
        return self._get_observation(), self._get_info()

    def _populate_grid(self):
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        num_blocks = self.np_random.integers(18, 23)
        possible_indices = list(range(self.GRID_SIZE * self.GRID_SIZE))
        self.np_random.shuffle(possible_indices)
        
        for i in range(num_blocks):
            idx = possible_indices[i]
            row, col = divmod(idx, self.GRID_SIZE)
            color_idx = self.np_random.integers(1, len(self.BLOCK_COLORS))
            self.grid[row, col] = color_idx

    def step(self, action):
        reward = 0
        terminated = self.game_over

        if self._update_animations():
            return self._get_observation(), 0, terminated, False, self._get_info()

        if self.game_over:
            return self._get_observation(), 0, terminated, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        is_noop = movement == 0 and not space_held and not shift_held
        if is_noop:
            return self._get_observation(), 0, terminated, False, self._get_info()

        if space_held: self.selected_col = (self.selected_col + 1) % self.GRID_SIZE
        if shift_held: self.selected_row = (self.selected_row + 1) % self.GRID_SIZE
        
        if movement != 0:
            self.steps += 1
            self.moves_left -= 1
            
            reward += -0.2
            self.score += -0.2

            start_grid = self.grid.copy()
            pushed_grid = self._perform_push(start_grid, movement)
            cleared_info = self._check_and_process_clears(pushed_grid)
            
            self.grid = cleared_info['final_grid']
            clear_reward = cleared_info['reward']
            reward += clear_reward
            self.score += clear_reward
            
            self._setup_animations(start_grid, pushed_grid, cleared_info, movement)

            if np.count_nonzero(self.grid) == 0:
                self.game_over, self.win, terminal_reward = True, True, 50
                reward += terminal_reward
                self.score += terminal_reward
            elif self.moves_left <= 0:
                self.game_over, self.win, terminal_reward = True, False, -50
                reward += terminal_reward
                self.score += terminal_reward
            
            terminated = self.game_over

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _perform_push(self, grid, movement):
        new_grid = grid.copy()
        if movement == 3: # left
            row = new_grid[self.selected_row, :]; row = row[row != 0]
            new_grid[self.selected_row, :] = 0; new_grid[self.selected_row, :len(row)] = row
        elif movement == 4: # right
            row = new_grid[self.selected_row, :]; row = row[row != 0]
            new_grid[self.selected_row, :] = 0; new_grid[self.selected_row, -len(row):] = row
        elif movement == 1: # up
            col = new_grid[:, self.selected_col]; col = col[col != 0]
            new_grid[:, self.selected_col] = 0; new_grid[:len(col), self.selected_col] = col
        elif movement == 2: # down
            col = new_grid[:, self.selected_col]; col = col[col != 0]
            new_grid[:, self.selected_col] = 0; new_grid[-len(col):, self.selected_col] = col
        return new_grid

    def _check_and_process_clears(self, grid):
        to_clear = np.zeros_like(grid, dtype=bool)
        reward, cleared_lines = 0, 0
        for i in range(self.GRID_SIZE):
            if grid[i, 0] != 0 and np.all(grid[i, :] == grid[i, 0]):
                to_clear[i, :] = True; cleared_lines += 1
            if grid[0, i] != 0 and np.all(grid[:, i] == grid[0, i]):
                to_clear[:, i] = True; cleared_lines += 1
        
        cleared_block_count = np.sum(to_clear)
        if cleared_block_count > 0:
            reward += cleared_block_count * 1.0 + cleared_lines * 5.0

        final_grid = grid.copy(); final_grid[to_clear] = 0
        return {'reward': reward, 'final_grid': final_grid, 'cleared_coords': np.argwhere(to_clear)}
    
    def _setup_animations(self, start_grid, pushed_grid, cleared_info, movement):
        self.animations.clear()
        push_map = {}
        if movement in [3, 4]: # Horizontal
            r = self.selected_row
            start_indices = [c for c, v in enumerate(start_grid[r, :]) if v != 0]
            end_indices = [c for c, v in enumerate(pushed_grid[r, :]) if v != 0]
            for i in range(len(start_indices)): push_map[(r, start_indices[i])] = (r, end_indices[i])
        elif movement in [1, 2]: # Vertical
            c = self.selected_col
            start_indices = [r for r, v in enumerate(start_grid[:, c]) if v != 0]
            end_indices = [r for r, v in enumerate(pushed_grid[:, c]) if v != 0]
            for i in range(len(start_indices)): push_map[(start_indices[i], c)] = (end_indices[i], c)
        
        self.animations.append({'type': 'push', 'progress': 0.0, 'map': push_map})
        if len(cleared_info['cleared_coords']) > 0:
            self.animations.append({'type': 'flash', 'progress': 0.0, 'coords': cleared_info['cleared_coords'], 'grid_at_flash': pushed_grid})
            self.animations.append({'type': 'particles', 'coords': cleared_info['cleared_coords'], 'grid_at_flash': pushed_grid})

    def _update_animations(self):
        if not self.animations: return False
        anim = self.animations[0]
        anim['progress'] = min(1.0, anim['progress'] + self.ANIMATION_SPEED)
        
        if anim['progress'] >= 1.0:
            if anim['type'] == 'push': self.visual_grid_state = self.animations[1]['grid_at_flash'] if len(self.animations) > 1 else self.grid.copy()
            if anim['type'] == 'particles':
                for r, c in anim['coords']:
                    color_idx = anim['grid_at_flash'][r, c]
                    px = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE / 2
                    py = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE / 2
                    for _ in range(10): self.particles.append(Particle(px, py, self.BLOCK_COLORS[color_idx], self.np_random))
            self.animations.pop(0)
            if not self.animations: self.visual_grid_state = self.grid.copy()
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_selector()
        self._render_blocks()
        self._render_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid_and_selector(self):
        s_col = pygame.Surface((self.CELL_SIZE, self.GRID_SIZE * self.CELL_SIZE), pygame.SRCALPHA)
        s_col.fill(self.COLOR_SELECTOR)
        self.screen.blit(s_col, (self.GRID_OFFSET_X + self.selected_col * self.CELL_SIZE, self.GRID_OFFSET_Y))
        s_row = pygame.Surface((self.GRID_SIZE * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s_row.fill(self.COLOR_SELECTOR)
        self.screen.blit(s_row, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + self.selected_row * self.CELL_SIZE))
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y), (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE), self.GRID_LINE_WIDTH)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE), (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE), self.GRID_LINE_WIDTH)

    def _draw_block(self, x, y, color_idx, override_color=None):
        color = override_color if override_color else self.BLOCK_COLORS[color_idx]
        margin = 4
        block_rect = pygame.Rect(int(x) + margin, int(y) + margin, self.CELL_SIZE - margin * 2, self.CELL_SIZE - margin * 2)
        pygame.draw.rect(self.screen, color, block_rect, border_radius=5)

    def _render_blocks(self):
        anim = self.animations[0] if self.animations else None
        animated_coords = set()
        
        if anim and anim['type'] == 'push':
            for (r, c), (dr, dc) in anim['map'].items():
                color_idx = self.visual_grid_state[r, c]
                if color_idx == 0: continue
                
                start_x, start_y = self.GRID_OFFSET_X + c * self.CELL_SIZE, self.GRID_OFFSET_Y + r * self.CELL_SIZE
                target_x, target_y = self.GRID_OFFSET_X + dc * self.CELL_SIZE, self.GRID_OFFSET_Y + dr * self.CELL_SIZE
                x, y = start_x + (target_x - start_x) * anim['progress'], start_y + (target_y - start_y) * anim['progress']
                
                self._draw_block(x, y, color_idx)
                animated_coords.add((r, c))

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r, c) in animated_coords: continue
                
                color_idx = self.visual_grid_state[r, c]
                if color_idx == 0: continue
                
                x, y = self.GRID_OFFSET_X + c * self.CELL_SIZE, self.GRID_OFFSET_Y + r * self.CELL_SIZE
                color = self.BLOCK_COLORS[color_idx]

                if anim and anim['type'] == 'flash' and any(np.array_equal([r, c], coord) for coord in anim['coords']):
                    p = 1 - abs(anim['progress'] - 0.5) * 2
                    color = tuple(int(col + (255 - col) * p) for col in color)
                elif anim and anim['type'] == 'particles' and any(np.array_equal([r, c], coord) for coord in anim['coords']):
                    continue
                
                self._draw_block(x, y, color_idx, override_color=color)

    def _render_particles(self):
        self.particles = [p for p in self.particles if p.update()]
        for p in self.particles: p.draw(self.screen)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        moves_text = self.font_main.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (10, 10))
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg, color = ("YOU WIN!", (100, 255, 100)) if self.win else ("GAME OVER", (255, 100, 100))
            end_text = self.font_title.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left, "game_over": self.game_over}

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        obs = self._get_observation()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(info, dict)
        action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(reward, (int, float)) and isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Pusher")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0]
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: action[2] = 1
                elif event.key == pygame.K_r: obs, info = env.reset()

        # Step on an explicit action, or if animations are running
        if action_taken or env.animations:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0: print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_left']}")
            if terminated: print(f"Game Over! Final Score: {info['score']}")
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if info.get('game_over') and not env.animations:
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(60)
    pygame.quit()