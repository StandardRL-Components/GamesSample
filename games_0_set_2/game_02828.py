
# Generated: 2025-08-27T21:33:56.221885
# Source Brief: brief_02828.md
# Brief Index: 2828

        
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
        "Controls: ←→ to move, ↑ to rotate, ↓ to soft drop. Hold Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a 60-second onslaught of falling blocks in this fast-paced arcade puzzler. Clear lines to score points and perform risky edge placements for bonuses."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.PLAYFIELD_WIDTH, self.PLAYFIELD_HEIGHT = 10, 15
        self.CELL_SIZE = 24
        self.PLAYFIELD_OFFSET_X = (self.SCREEN_WIDTH - self.PLAYFIELD_WIDTH * self.CELL_SIZE) / 2
        self.PLAYFIELD_OFFSET_Y = (self.SCREEN_HEIGHT - self.PLAYFIELD_HEIGHT * self.CELL_SIZE) - 20

        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_GHOST = (255, 255, 255, 60)
        self.BLOCK_COLORS = [
            (0, 0, 0),          # 0: Empty
            (230, 60, 60),      # 1: J-Block (Red)
            (60, 230, 60),      # 2: L-Block (Green)
            (60, 150, 230),     # 3: I-Block (Blue)
            (230, 230, 60),     # 4: O-Block (Yellow)
        ]

        # Tetromino shapes
        self.TETROMINOES = {
            'J': [[[1, 0, 0], [1, 1, 1], [0, 0, 0]], [[0, 1, 1], [0, 1, 0], [0, 1, 0]], [[0, 0, 0], [1, 1, 1], [0, 0, 1]], [[0, 1, 0], [0, 1, 0], [1, 1, 0]]],
            'L': [[[0, 0, 1], [1, 1, 1], [0, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 1]], [[0, 0, 0], [1, 1, 1], [1, 0, 0]], [[1, 1, 0], [0, 1, 0], [0, 1, 0]]],
            'I': [[[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]], [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]], [[0,0,0,0], [0,0,0,0], [1,1,1,1], [0,0,0,0]], [[0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0]]],
            'O': [[[1, 1], [1, 1]]],
        }
        self.TETROMINO_MAP = ['J', 'L', 'I', 'O']
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # State variables are initialized in reset()
        self.playfield = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.time_remaining = 0
        self.current_block = None
        self.next_block_type = None
        self.base_fall_speed = 0
        self.move_cooldown = 0
        self.rotate_cooldown = 0
        self.hard_drop_used_this_piece = False
        self.particles = []
        self.line_clear_flash = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.playfield = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.base_fall_speed = 0.5  # cells per second
        
        self.move_cooldown = 0
        self.rotate_cooldown = 0
        self.particles.clear()
        self.line_clear_flash.clear()

        self.next_block_type = self.np_random.choice(len(self.TETROMINO_MAP)) + 1
        self._spawn_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.001  # Small penalty per frame to encourage speed
        self.steps += 1
        self.time_remaining -= 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update difficulty every 10 seconds
        if self.steps > 0 and self.steps % (self.FPS * 10) == 0:
            self.base_fall_speed += 0.25

        # Cooldowns
        if self.move_cooldown > 0: self.move_cooldown -= 1
        if self.rotate_cooldown > 0: self.rotate_cooldown -= 1
        
        # Unpack factorized action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Action Handling
        if space_held and not self.hard_drop_used_this_piece:
            # Sound: Hard drop thud
            reward += self._hard_drop()
            reward += self._lock_and_clear()
            self._spawn_block()
        else:
            # Handle horizontal movement
            if movement in [3, 4] and self.move_cooldown == 0:
                self._move_block(movement - 3.5) # 3 -> -0.5, 4 -> 0.5
                self.move_cooldown = 4 # frames
            
            # Handle rotation
            if movement == 1 and self.rotate_cooldown == 0:
                self._rotate_block()
                self.rotate_cooldown = 6 # frames

            # Physics Update (fall)
            soft_drop_multiplier = 10.0 if movement == 2 else 1.0
            fall_increment = (self.base_fall_speed / self.FPS) * soft_drop_multiplier
            self.current_block['y'] += fall_increment

            # Check for landing
            shape = self.TETROMINOES[self.TETROMINO_MAP[self.current_block['type']-1]][self.current_block['rotation']]
            if self._check_collision(shape, (self.current_block['x'], math.ceil(self.current_block['y']))):
                self.current_block['y'] = math.ceil(self.current_block['y']) - 1
                reward += self._lock_and_clear()
                self._spawn_block()

        terminated = self.game_over or self.time_remaining <= 0
        if self.time_remaining <= 0 and not self.game_over:
            reward += 100 # Survival bonus

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _spawn_block(self):
        self.current_block = {
            'type': self.next_block_type,
            'rotation': 0,
            'x': self.PLAYFIELD_WIDTH // 2 - 1,
            'y': 0.0
        }
        self.next_block_type = self.np_random.integers(1, len(self.TETROMINO_MAP) + 1)
        self.hard_drop_used_this_piece = False

        shape = self.TETROMINOES[self.TETROMINO_MAP[self.current_block['type']-1]][self.current_block['rotation']]
        if self._check_collision(shape, (self.current_block['x'], self.current_block['y'])):
            self.game_over = True
            # Sound: Game over
    
    def _check_collision(self, shape, pos):
        px, py = pos
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = int(px + c_idx), int(py + r_idx)
                    if not (0 <= grid_x < self.PLAYFIELD_WIDTH and 0 <= grid_y < self.PLAYFIELD_HEIGHT):
                        return True  # Out of bounds
                    if self.playfield[grid_y, grid_x] != 0:
                        return True  # Collides with another block
        return False

    def _move_block(self, dx_sign):
        dx = int(dx_sign * 2)
        shape = self.TETROMINOES[self.TETROMINO_MAP[self.current_block['type']-1]][self.current_block['rotation']]
        if not self._check_collision(shape, (self.current_block['x'] + dx, self.current_block['y'])):
            self.current_block['x'] += dx
            # Sound: Move tick

    def _rotate_block(self):
        rotations = self.TETROMINOES[self.TETROMINO_MAP[self.current_block['type']-1]]
        new_rotation = (self.current_block['rotation'] + 1) % len(rotations)
        new_shape = rotations[new_rotation]
        
        # Wall kick logic
        for offset in [0, 1, -1, 2, -2]:
            if not self._check_collision(new_shape, (self.current_block['x'] + offset, self.current_block['y'])):
                self.current_block['rotation'] = new_rotation
                self.current_block['x'] += offset
                # Sound: Rotate swoosh
                return

    def _hard_drop(self):
        shape = self.TETROMINOES[self.TETROMINO_MAP[self.current_block['type']-1]][self.current_block['rotation']]
        drop_y = self.current_block['y']
        while not self._check_collision(shape, (self.current_block['x'], drop_y + 1)):
            drop_y += 1
        self.current_block['y'] = drop_y
        self.hard_drop_used_this_piece = True
        return 0

    def _lock_and_clear(self):
        # Sound: Block lock
        shape = self.TETROMINOES[self.TETROMINO_MAP[self.current_block['type']-1]][self.current_block['rotation']]
        block_x, block_y = self.current_block['x'], int(self.current_block['y'])
        
        is_at_edge = False
        block_width = len(shape[0])

        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = block_x + c_idx, block_y + r_idx
                    if 0 <= grid_x < self.PLAYFIELD_WIDTH and 0 <= grid_y < self.PLAYFIELD_HEIGHT:
                        self.playfield[grid_y, grid_x] = self.current_block['type']
                        if grid_x == 0 or grid_x == self.PLAYFIELD_WIDTH - 1:
                            is_at_edge = True

        placement_reward = 2.0 if is_at_edge else -0.2
        
        # Line clearing
        lines_cleared = 0
        full_rows = []
        for r in range(self.PLAYFIELD_HEIGHT):
            if np.all(self.playfield[r, :] != 0):
                lines_cleared += 1
                full_rows.append(r)
        
        if lines_cleared > 0:
            # Sound: Line clear
            self.score += lines_cleared * 100
            for r in full_rows:
                self._create_particles(r)
                self.line_clear_flash.append({'y': r, 'timer': 10})
            
            # Remove full rows and shift down
            self.playfield = np.delete(self.playfield, full_rows, axis=0)
            new_rows = np.zeros((lines_cleared, self.PLAYFIELD_WIDTH), dtype=int)
            self.playfield = np.vstack((new_rows, self.playfield))

        return placement_reward + (lines_cleared * 1.0)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw playfield grid
        pf_rect = pygame.Rect(self.PLAYFIELD_OFFSET_X, self.PLAYFIELD_OFFSET_Y, self.PLAYFIELD_WIDTH * self.CELL_SIZE, self.PLAYFIELD_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, (10, 10, 20), pf_rect)
        for i in range(self.PLAYFIELD_WIDTH + 1):
            x = self.PLAYFIELD_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.PLAYFIELD_OFFSET_Y), (x, self.PLAYFIELD_OFFSET_Y + self.PLAYFIELD_HEIGHT * self.CELL_SIZE))
        for i in range(self.PLAYFIELD_HEIGHT + 1):
            y = self.PLAYFIELD_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAYFIELD_OFFSET_X, y), (self.PLAYFIELD_OFFSET_X + self.PLAYFIELD_WIDTH * self.CELL_SIZE, y))

        # Draw locked blocks
        for r in range(self.PLAYFIELD_HEIGHT):
            for c in range(self.PLAYFIELD_WIDTH):
                if self.playfield[r, c] != 0:
                    self._render_cell(c, r, self.BLOCK_COLORS[self.playfield[r, c]])

        # Draw ghost piece
        if not self.game_over and self.current_block:
            shape = self.TETROMINOES[self.TETROMINO_MAP[self.current_block['type']-1]][self.current_block['rotation']]
            ghost_y = self.current_block['y']
            while not self._check_collision(shape, (self.current_block['x'], ghost_y + 1)):
                ghost_y += 1
            self._render_block(shape, (self.current_block['x'], int(ghost_y)), self.COLOR_GHOST, is_ghost=True)

        # Draw current block
        if not self.game_over and self.current_block:
            shape = self.TETROMINOES[self.TETROMINO_MAP[self.current_block['type']-1]][self.current_block['rotation']]
            color = self.BLOCK_COLORS[self.current_block['type']]
            self._render_block(shape, (self.current_block['x'], self.current_block['y']), color)
            
        # Update and render particles
        self._update_and_render_particles()

        # Render line clear flash
        for flash in self.line_clear_flash[:]:
            y = self.PLAYFIELD_OFFSET_Y + flash['y'] * self.CELL_SIZE
            alpha = int(255 * (flash['timer'] / 10))
            flash_surface = pygame.Surface((self.PLAYFIELD_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (self.PLAYFIELD_OFFSET_X, y))
            flash['timer'] -= 1
            if flash['timer'] <= 0:
                self.line_clear_flash.remove(flash)

    def _render_block(self, shape, pos, color, is_ghost=False):
        px, py = pos
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    self._render_cell(px + c_idx, py + r_idx, color, is_ghost)
    
    def _render_cell(self, c, r, color, is_ghost=False):
        x = self.PLAYFIELD_OFFSET_X + c * self.CELL_SIZE
        y = self.PLAYFIELD_OFFSET_Y + r * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2)
        else:
            # Main cell color
            pygame.draw.rect(self.screen, color, rect)
            # 3D-ish effect
            highlight = tuple(min(255, c_val + 40) for c_val in color)
            shadow = tuple(max(0, c_val - 40) for c_val in color)
            pygame.draw.line(self.screen, highlight, (x, y), (x + self.CELL_SIZE - 1, y), 2)
            pygame.draw.line(self.screen, highlight, (x, y), (x, y + self.CELL_SIZE - 1), 2)
            pygame.draw.line(self.screen, shadow, (x + self.CELL_SIZE - 1, y), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1), 2)
            pygame.draw.line(self.screen, shadow, (x, y + self.CELL_SIZE - 1), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1), 2)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Time remaining
        time_sec = self.time_remaining // self.FPS
        time_text = self.font_main.render(f"TIME: {time_sec}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 20, 20))
        
        # Next block preview
        preview_x, preview_y = self.SCREEN_WIDTH - 140, 70
        pygame.draw.rect(self.screen, self.COLOR_GRID, (preview_x, preview_y, 120, 120), 0, 5)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (preview_x, preview_y, 120, 120), 2, 5)
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (preview_x + (120 - next_text.get_width()) / 2, preview_y + 5))

        if self.next_block_type:
            shape = self.TETROMINOES[self.TETROMINO_MAP[self.next_block_type-1]][0]
            color = self.BLOCK_COLORS[self.next_block_type]
            shape_w = len(shape[0]) * self.CELL_SIZE
            shape_h = len(shape) * self.CELL_SIZE
            block_x = preview_x + (120 - shape_w) / 2
            block_y = preview_y + (120 - shape_h) / 2 + 10
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        rect = pygame.Rect(block_x + c_idx * self.CELL_SIZE, block_y + r_idx * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                        pygame.draw.rect(self.screen, color, rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            end_text = "GAME OVER" if self.time_remaining > 0 else "YOU SURVIVED!"
            text_surf = self.font_main.render(end_text, True, (255, 50, 50) if self.game_over and self.time_remaining > 0 else (50, 255, 50))
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _create_particles(self, row_y):
        # Sound: Particle burst
        world_y = self.PLAYFIELD_OFFSET_Y + row_y * self.CELL_SIZE + self.CELL_SIZE / 2
        for i in range(self.PLAYFIELD_WIDTH * 3): # 3 particles per cell
            world_x = self.PLAYFIELD_OFFSET_X + (i/3) * self.CELL_SIZE + self.CELL_SIZE / 2
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(30, 60)
            color = random.choice([(255,255,255), (200,200,255), (255,255,200)])
            self.particles.append({'pos': [world_x, world_y], 'vel': vel, 'life': life, 'color': color})

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # gravity
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            size = max(0, int(p['life'] / 10))
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], pos_int, size)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")