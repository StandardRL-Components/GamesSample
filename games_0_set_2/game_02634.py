
# Generated: 2025-08-27T20:58:15.656480
# Source Brief: brief_02634.md
# Brief Index: 2634

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a block stacking game.
    The goal is to stack falling blocks as high as possible without any part
    of a block being placed outside the designated grid area.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the block. Press space to drop it instantly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade puzzler. Stack falling blocks as high as you can. "
        "Reach a height of 20 to win. Don't let any blocks fall off the side!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 25
        self.BLOCK_SIZE = self.HEIGHT // self.GRID_HEIGHT
        self.GRID_PIXEL_WIDTH = self.GRID_WIDTH * self.BLOCK_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_PIXEL_WIDTH) // 2

        self.TARGET_HEIGHT = 20
        self.MAX_STEPS = 1000
        self.FALL_SPEED = 0.1  # Grid units per frame
        self.SPAWN_DELAY_FRAMES = 10

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_STACKED_FILL = (80, 90, 100)
        self.COLOR_STACKED_BORDER = (120, 130, 140)
        self.COLOR_TARGET_LINE = (0, 255, 128)
        self.BLOCK_COLORS = [
            (255, 87, 34),   # Deep Orange
            (255, 193, 7),   # Amber
            (76, 175, 80),   # Green
            (33, 150, 243),  # Blue
            (156, 39, 176),  # Purple
        ]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GAMEOVER = (255, 82, 82)
        self.COLOR_WIN = (100, 255, 218)

        # --- Block Shapes (horizontal bars of different lengths) ---
        self.BLOCK_SHAPES = [
            [(0, 0), (1, 0)],
            [(0, 0), (1, 0), (2, 0)],
            [(-1, 0), (0, 0), (1, 0)],
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(-1, 0), (0, 0), (1, 0), (2, 0)],
        ]

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
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Internal State (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = None
        self.placed_blocks = None
        self.stack_height = 0
        self.falling_block = None
        self.particles = None
        self.spawn_timer = 0
        
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.placed_blocks = []
        self.stack_height = 0
        self.particles = []
        self.spawn_timer = 0

        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1

        if self.falling_block is None:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_new_block()
        else:
            movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1

            # Horizontal Movement
            if movement == 3:  # Left
                self.falling_block['pos'][0] -= 1
            elif movement == 4:  # Right
                self.falling_block['pos'][0] += 1
            
            # Clamp to grid boundaries to prevent moving off-screen
            min_x = min(c[0] for c in self.falling_block['shape'])
            max_x = max(c[0] for c in self.falling_block['shape'])
            self.falling_block['pos'][0] = max(-min_x, self.falling_block['pos'][0])
            self.falling_block['pos'][0] = min(self.GRID_WIDTH - 1 - max_x, self.falling_block['pos'][0])

            # Vertical Movement & Placement
            if space_pressed:
                # Sound: Whoosh
                landing_y = self._get_landing_y()
                self.falling_block['pos'][1] = landing_y
                reward += self._place_block()
            else:
                self.falling_block['pos'][1] += self.FALL_SPEED
                landing_y = self._get_landing_y()
                if self.falling_block['pos'][1] >= landing_y:
                    self.falling_block['pos'][1] = landing_y
                    reward += self._place_block()

        self._update_particles()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
             self.game_over = True # Game ends due to time limit

        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_landing_y(self):
        block_x = int(self.falling_block['pos'][0])
        max_stack_y = 0
        for dx, _ in self.falling_block['shape']:
            col = block_x + dx
            if 0 <= col < self.GRID_WIDTH:
                col_height = 0
                # Find the highest block in this column by scanning from the top
                for r in range(self.GRID_HEIGHT):
                    if self.grid[r][col] != 0:
                        col_height = self.GRID_HEIGHT - r
                        break
                max_stack_y = max(max_stack_y, col_height)
        # The landing y is just above the highest point of the stack below
        return self.GRID_HEIGHT - 1 - max_stack_y

    def _place_block(self):
        block_pos = [int(self.falling_block['pos'][0]), int(round(self.falling_block['pos'][1]))]

        # 1. Check for loss condition (part of block is off the horizontal grid)
        for dx, _ in self.falling_block['shape']:
            if not (0 <= block_pos[0] + dx < self.GRID_WIDTH):
                self.game_over = True
                # Sound: Failure
                return -100.0

        # 2. Place block on grid and calculate rewards
        is_centered = self._is_perfectly_centered()
        current_max_height = self.stack_height

        for dx, dy in self.falling_block['shape']:
            gx, gy = block_pos[0] + dx, block_pos[1] + dy
            if 0 <= gx < self.GRID_WIDTH and 0 <= gy < self.GRID_HEIGHT:
                self.grid[gy][gx] = 1  # Mark as occupied
                
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + gx * self.BLOCK_SIZE,
                    gy * self.BLOCK_SIZE,
                    self.BLOCK_SIZE, self.BLOCK_SIZE
                )
                self.placed_blocks.append({'rect': rect, 'color': self.falling_block['color']})
                self._create_particles(rect.center, self.falling_block['color'])

        self.stack_height = self._calculate_stack_height()
        self.falling_block = None
        self.spawn_timer = self.SPAWN_DELAY_FRAMES
        
        # Sound: Block place
        reward = 0.1  # Base reward for any successful placement
        if is_centered:
            reward += 1.0  # Bonus for centering

        # 3. Check for win condition
        if self.stack_height >= self.TARGET_HEIGHT:
            self.game_over = True
            # Sound: Victory
            return 100.0

        return reward

    def _calculate_stack_height(self):
        for r in range(self.GRID_HEIGHT):
            if np.any(self.grid[r, :]):
                return self.GRID_HEIGHT - r
        return 0

    def _is_perfectly_centered(self):
        block_x_int = int(self.falling_block['pos'][0])
        landing_y_int = int(round(self.falling_block['pos'][1]))
        
        com_x_falling = block_x_int + np.mean([dx for dx, dy in self.falling_block['shape']])

        supporting_blocks_x = []
        for dx, _ in self.falling_block['shape']:
            col = block_x_int + dx
            if 0 <= col < self.GRID_WIDTH and landing_y_int + 1 < self.GRID_HEIGHT:
                if self.grid[landing_y_int + 1][col] != 0:
                    supporting_blocks_x.append(col)
        
        if not supporting_blocks_x:
            com_x_support = (self.GRID_WIDTH - 1) / 2.0
        else:
            com_x_support = np.mean(supporting_blocks_x)
            
        return abs(com_x_falling - com_x_support) < 0.5

    def _spawn_new_block(self):
        shape_idx = self.np_random.integers(0, len(self.BLOCK_SHAPES))
        color_idx = self.np_random.integers(0, len(self.BLOCK_COLORS))
        
        self.falling_block = {
            'shape': self.BLOCK_SHAPES[shape_idx],
            'pos': [self.GRID_WIDTH // 2, 0.0],  # [grid_x, float_grid_y]
            'color': self.BLOCK_COLORS[color_idx],
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stack_height": self.stack_height}

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_OFFSET_X + x * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_PIXEL_WIDTH, py))
            
        # Draw target line
        target_y = (self.GRID_HEIGHT - self.TARGET_HEIGHT) * self.BLOCK_SIZE
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (self.GRID_OFFSET_X, target_y), (self.GRID_OFFSET_X + self.GRID_PIXEL_WIDTH, target_y), 2)

        # Draw placed blocks
        for block in self.placed_blocks:
            pygame.draw.rect(self.screen, self.COLOR_STACKED_FILL, block['rect'])
            pygame.draw.rect(self.screen, self.COLOR_STACKED_BORDER, block['rect'], 1)

        # Draw falling block with a "ghost" showing where it will land
        if self.falling_block:
            color = self.falling_block['color']
            # Draw ghost
            landing_y = self._get_landing_y()
            ghost_color = (*color, 60) # Semi-transparent
            for dx, _ in self.falling_block['shape']:
                gx = int(self.falling_block['pos'][0]) + dx
                gy = landing_y
                
                rect_surf = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(rect_surf, ghost_color, rect_surf.get_rect())
                self.screen.blit(rect_surf, (self.GRID_OFFSET_X + gx * self.BLOCK_SIZE, gy * self.BLOCK_SIZE))

            # Draw actual block
            for dx, _ in self.falling_block['shape']:
                gx = int(self.falling_block['pos'][0]) + dx
                gy_float = self.falling_block['pos'][1]
                
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + gx * self.BLOCK_SIZE, gy_float * self.BLOCK_SIZE,
                    self.BLOCK_SIZE, self.BLOCK_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, tuple(min(255, c + 50) for c in color), rect.inflate(-4, -4))

        self._render_particles()

    def _render_ui(self):
        height_text = self.font_medium.render(f"Height: {self.stack_height}/{self.TARGET_HEIGHT}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 10))

        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))

        if self.game_over:
            msg, color = ("YOU WIN!", self.COLOR_WIN) if self.stack_height >= self.TARGET_HEIGHT else ("GAME OVER", self.COLOR_GAMEOVER)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            pygame.draw.rect(self.screen, (*self.COLOR_BG, 200), text_rect.inflate(20,20), border_radius=10)
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            size = max(0, int(p['life'] / 4))
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p['color'])

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2], "Action space mismatch"
        print("✓ Action space OK")
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3) and test_obs.dtype == np.uint8, "Observation space mismatch"
        print("✓ Observation space OK")
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(info, dict), "reset() return format error"
        print("✓ reset() OK")
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3), "Step obs shape error"
        assert isinstance(reward, float), "Step reward type error"
        assert isinstance(term, bool), "Step terminated type error"
        assert not trunc, "Step truncated should be False"
        assert isinstance(info, dict), "Step info type error"
        print("✓ step() OK")
        
        print("✓ Implementation validated successfully")