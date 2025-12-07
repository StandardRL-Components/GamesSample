
# Generated: 2025-08-28T06:26:21.433963
# Source Brief: brief_05904.md
# Brief Index: 5904

        
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
        "Controls: ←→ to move, ↑ to rotate, ↓ for hard drop. Blocks fall automatically."
    )

    game_description = (
        "A fast-paced, grid-based block-clearing puzzle. Clear 10 lines to win before the "
        "time runs out or the stack reaches the top."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.BLOCK_SIZE = 18
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2
        self.FPS = 30
        self.MAX_TIME = 60 * self.FPS # 60 seconds
        self.MAX_STEPS = 1000
        self.WIN_LINES = 10

        # Colors (Neon on Dark)
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_TEXT_ACCENT = (100, 255, 255)
        self.COLOR_GAMEOVER = (255, 50, 50)
        self.COLOR_WIN = (50, 255, 50)
        
        # Block shapes (I, L, J, T, S, Z, O) and their colors
        self.BLOCK_SHAPES = [
            [[1, 1, 1, 1]], # I
            [[1, 0, 0], [1, 1, 1]], # L
            [[0, 0, 1], [1, 1, 1]], # J
            [[0, 1, 0], [1, 1, 1]], # T
            [[0, 1, 1], [1, 1, 0]], # S
            [[1, 1, 0], [0, 1, 1]], # Z
            [[1, 1], [1, 1]]  # O
        ]
        self.BLOCK_COLORS = [
            (0, 255, 255),    # Cyan (I)
            (255, 165, 0),    # Orange (L)
            (0, 0, 255),      # Blue (J)
            (128, 0, 128),    # Purple (T)
            (0, 255, 0),      # Green (S)
            (255, 0, 0),      # Red (Z)
            (255, 255, 0)     # Yellow (O)
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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables are initialized in reset()
        self.reset()

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.win = False
        self.time_left = self.MAX_TIME
        
        self.fall_timer = 0
        self.fall_speed = self.FPS // 2  # Block falls every 0.5 seconds initially

        self.input_cooldowns = {'move': 0, 'rotate': 0}
        self.cooldown_frames = {'move': 5, 'rotate': 8}

        self.particles = []
        self.reward_this_step = 0
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_this_step = -0.01  # Small penalty per step
        self.steps += 1
        self.time_left -= 1
        
        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            self._update_particles()
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.win:
                self.reward_this_step += 100
            elif self.time_left <= 0:
                self.reward_this_step -= 5
            else: # Top out
                self.reward_this_step -= 10
            self.game_over = True
            
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]  # 0-4: none/up/down/left/right

        # Update cooldowns
        for key in self.input_cooldowns:
            if self.input_cooldowns[key] > 0:
                self.input_cooldowns[key] -= 1

        if movement == 1 and self.input_cooldowns['rotate'] == 0:  # Up -> Rotate
            self._rotate_block()
            self.input_cooldowns['rotate'] = self.cooldown_frames['rotate']
        elif movement == 2:  # Down -> Hard Drop
            self._hard_drop()
        elif movement == 3 and self.input_cooldowns['move'] == 0:  # Left
            self._move_block(-1)
            self.input_cooldowns['move'] = self.cooldown_frames['move']
        elif movement == 4 and self.input_cooldowns['move'] == 0:  # Right
            self._move_block(1)
            self.input_cooldowns['move'] = self.cooldown_frames['move']

    def _update_game_state(self):
        self.fall_timer += 1
        if self.fall_timer >= self.fall_speed:
            self.fall_timer = 0
            self.current_block['y'] += 1
            if self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'])):
                self.current_block['y'] -= 1
                self._lock_block()
                self._spawn_new_block()
    
    def _spawn_new_block(self):
        shape_index = self.np_random.integers(0, len(self.BLOCK_SHAPES))
        self.current_block = {
            'shape': self.BLOCK_SHAPES[shape_index],
            'color_index': shape_index + 1,
            'x': self.GRID_WIDTH // 2 - len(self.BLOCK_SHAPES[shape_index][0]) // 2,
            'y': 0
        }
        if self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'])):
            self.game_over = True
            self.win = False

    def _check_collision(self, shape, offset):
        off_x, off_y = offset
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = off_x + x, off_y + y
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT and self.grid[grid_x, grid_y] == 0):
                        return True
        return False

    def _rotate_block(self):
        # Sound: Block rotate
        original_shape = self.current_block['shape']
        rotated_shape = list(zip(*original_shape[::-1]))
        if not self._check_collision(rotated_shape, (self.current_block['x'], self.current_block['y'])):
            self.current_block['shape'] = rotated_shape
        else: # Wall kick
            for dx in [-1, 1, -2, 2]:
                if not self._check_collision(rotated_shape, (self.current_block['x'] + dx, self.current_block['y'])):
                    self.current_block['x'] += dx
                    self.current_block['shape'] = rotated_shape
                    return

    def _move_block(self, dx):
        # Sound: Block move
        if not self._check_collision(self.current_block['shape'], (self.current_block['x'] + dx, self.current_block['y'])):
            self.current_block['x'] += dx

    def _hard_drop(self):
        # Sound: Hard drop
        while not self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'] + 1)):
            self.current_block['y'] += 1
        self._lock_block()
        self._spawn_new_block()

    def _lock_block(self):
        # Sound: Block lock
        shape = self.current_block['shape']
        off_x, off_y = self.current_block['x'], self.current_block['y']
        
        # Risky placement reward calculation
        risky_neighbors = 0
        placed_cells = 0
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    placed_cells += 1
                    grid_x, grid_y = off_x + x, off_y + y
                    # Check neighbors (down, left, right)
                    for dx, dy in [(0, 1), (-1, 0), (1, 0)]:
                        nx, ny = grid_x + dx, grid_y + dy
                        if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT) or self.grid[nx, ny] != 0:
                            risky_neighbors += 1

        if placed_cells > 0 and risky_neighbors / placed_cells >= 1.5: # Heuristic for "risky"
             self.reward_this_step += 2.0
        else:
             self.reward_this_step -= 0.2

        # Place block on grid
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[off_x + x, off_y + y] = self.current_block['color_index']
        
        self._clear_lines()
        
    def _clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[:, y] != 0):
                lines_to_clear.append(y)

        if lines_to_clear:
            # Sound: Line clear
            num_cleared = len(lines_to_clear)
            self.lines_cleared += num_cleared
            self.score += (num_cleared ** 2) * 10 # Bonus for multi-line clears
            self.reward_this_step += num_cleared * 1.0

            # Create particles
            for y in lines_to_clear:
                color = self.BLOCK_COLORS[self.np_random.integers(0, len(self.BLOCK_COLORS))]
                for i in range(30):
                    px = self.GRID_X + self.np_random.uniform(0, self.GRID_WIDTH * self.BLOCK_SIZE)
                    py = self.GRID_Y + y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
                    self.particles.append({
                        'pos': [px, py],
                        'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 1)],
                        'life': self.np_random.integers(15, 30),
                        'color': color,
                        'size': self.np_random.uniform(2, 5)
                    })

            # Remove lines and shift down
            for y in sorted(lines_to_clear, reverse=True):
                self.grid[:, y:] = np.roll(self.grid[:, y:], -1, axis=1)
                self.grid[:, -1] = 0

    def _check_termination(self):
        if self.lines_cleared >= self.WIN_LINES:
            self.win = True
            return True
        if self.game_over: # Set by top-out
            return True
        if self.time_left <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, (0,0,0), (self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE))
        for x in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X + x * self.BLOCK_SIZE, self.GRID_Y), (self.GRID_X + x * self.BLOCK_SIZE, self.GRID_Y + self.GRID_HEIGHT * self.BLOCK_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y + y * self.BLOCK_SIZE), (self.GRID_X + self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_Y + y * self.BLOCK_SIZE))

        # Draw locked blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] != 0:
                    color_index = int(self.grid[x, y]) - 1
                    self._draw_cell(x, y, self.BLOCK_COLORS[color_index])

        # Draw current block and ghost
        if not self.game_over:
            # Ghost piece
            ghost_y = self.current_block['y']
            while not self._check_collision(self.current_block['shape'], (self.current_block['x'], ghost_y + 1)):
                ghost_y += 1
            color = self.BLOCK_COLORS[self.current_block['color_index'] - 1]
            self._draw_block(self.current_block['shape'], (self.current_block['x'], ghost_y), color, alpha=50)

            # Current block
            self._draw_block(self.current_block['shape'], (self.current_block['x'], self.current_block['y']), color, glow=True)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_block(self, shape, offset, color, alpha=255, glow=False):
        off_x, off_y = offset
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self._draw_cell(off_x + x, off_y + y, color, alpha, glow)

    def _draw_cell(self, grid_x, grid_y, color, alpha=255, glow=False):
        px, py = self.GRID_X + grid_x * self.BLOCK_SIZE, self.GRID_Y + grid_y * self.BLOCK_SIZE
        
        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        if glow:
            glow_size = self.BLOCK_SIZE * 1.8
            s = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            glow_alpha = 90 if alpha == 255 else alpha // 3
            pygame.draw.circle(s, (*color, glow_alpha), (glow_size/2, glow_size/2), glow_size/2)
            self.screen.blit(s, (px - (glow_size - self.BLOCK_SIZE)/2, py - (glow_size - self.BLOCK_SIZE)/2), special_flags=pygame.BLEND_RGBA_ADD)

        # Main block color
        block_surf = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
        block_surf.fill((*color, alpha))
        
        # Inner bevel
        inset_color = tuple(min(255, c + 60) for c in color)
        pygame.draw.rect(block_surf, (*inset_color, alpha), (2, 2, self.BLOCK_SIZE - 4, self.BLOCK_SIZE - 4), 0, border_radius=2)
        
        # Center highlight
        highlight_color = tuple(min(255, c + 90) for c in color)
        pygame.draw.circle(block_surf, (*highlight_color, alpha), (self.BLOCK_SIZE // 2, self.BLOCK_SIZE // 2), self.BLOCK_SIZE // 4)

        self.screen.blit(block_surf, rect.topleft)

    def _render_ui(self):
        # Score and Lines
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        lines_text = self.font_main.render(f"LINES: {self.lines_cleared}/{self.WIN_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(lines_text, (20, 50))
        
        # Timer
        time_str = f"TIME: {self.time_left / self.FPS:.1f}"
        time_color = self.COLOR_TEXT_ACCENT if self.time_left > 10 * self.FPS else self.COLOR_GAMEOVER
        time_text = self.font_main.render(time_str, True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 20))

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            # Draw a semi-transparent black background for the text
            bg_rect = text_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "time_left_seconds": self.time_left / self.FPS,
        }
        
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'windows' or 'x11' or 'dummy'
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Fall")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)

    env.close()