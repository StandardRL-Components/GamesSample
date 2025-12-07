
# Generated: 2025-08-27T19:54:54.368618
# Source Brief: brief_02290.md
# Brief Index: 2290

        
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
    user_guide = "Controls: Use Left/Right arrow keys to move the paddle and intercept falling blocks."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced, grid-based block breaker. Intercept falling blocks to score points, but let one reach the bottom and it's game over!"

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 20
        self.CELL_WIDTH = 32
        self.CELL_HEIGHT = 18
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_WIDTH
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_HEIGHT
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2

        self.PADDLE_WIDTH = self.CELL_WIDTH * 2
        self.PADDLE_HEIGHT = 10
        self.PADDLE_Y = self.GRID_Y_OFFSET + self.GRID_HEIGHT
        self.PADDLE_SPEED = 12

        self.INITIAL_FALL_SPEED = 1.0
        self.SPEED_INCREASE = 0.05
        self.BLOCK_SPAWN_RATE = 0.025

        self.MAX_STEPS = 10000
        self.WIN_CONDITION = 50

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PADDLE = (240, 240, 240)
        self.BLOCK_COLORS = {
            1: (50, 205, 50),   # Green
            2: (65, 105, 225),  # Blue
            3: (220, 20, 60),   # Red
        }
        self.COLOR_TEXT = (255, 255, 255)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # State Variables
        self.np_random = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.blocks_cleared = 0
        self.fall_speed = 0.0
        self.paddle_x = 0.0
        self.blocks = []
        self.particles = []
        
        # Initialize state
        self.reset()
        
        # Final validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.blocks_cleared = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        
        self.paddle_x = self.WIDTH / 2
        self.blocks = []
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        reward = -0.01  # Time penalty to encourage speed
        
        # Unpack factorized action
        movement = action[0]
        
        # 1. Update paddle position based on action
        if movement == 3:  # Left
            self.paddle_x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_x += self.PADDLE_SPEED
        
        # Clamp paddle position to stay within the grid
        min_paddle_x = self.GRID_X_OFFSET + self.PADDLE_WIDTH / 2
        max_paddle_x = self.GRID_X_OFFSET + self.GRID_WIDTH - self.PADDLE_WIDTH / 2
        self.paddle_x = np.clip(self.paddle_x, min_paddle_x, max_paddle_x)
        
        # 2. Update game state
        self._update_blocks()
        self._update_particles()
        
        # 3. Spawn new blocks
        if self.np_random.random() < self.BLOCK_SPAWN_RATE:
            self._spawn_block()

        # 4. Handle collisions
        reward += self._handle_collisions()

        # 5. Check for termination conditions
        terminated = self._check_termination()
        
        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_collisions(self):
        reward = 0.0
        collided_blocks_indices = []
        paddle_rect = pygame.Rect(
            self.paddle_x - self.PADDLE_WIDTH / 2, 
            self.PADDLE_Y, 
            self.PADDLE_WIDTH, 
            self.PADDLE_HEIGHT
        )

        for i, block in enumerate(self.blocks):
            block_rect = pygame.Rect(block['x'], block['y'], self.CELL_WIDTH, self.CELL_HEIGHT)
            if paddle_rect.colliderect(block_rect):
                collided_blocks_indices.append(i)
                # Sound: block_break.wav
                self._create_particles(block['x'] + self.CELL_WIDTH / 2, block['y'], self.BLOCK_COLORS[block['points']])
                
                # Base reward for destroying block
                reward += block['points']

                # Risky play reward calculation
                paddle_center_norm = (self.paddle_x - self.GRID_X_OFFSET) / self.GRID_WIDTH
                if paddle_center_norm < 0.15 or paddle_center_norm > 0.85:
                    reward += 2.0  # Risky play bonus
                elif 0.4 < paddle_center_norm < 0.6:
                    reward -= 0.2  # Safe play penalty

                self.blocks_cleared += 1
                
                # Increase difficulty every 5 blocks
                if self.blocks_cleared > 0 and self.blocks_cleared % 5 == 0:
                    self.fall_speed += self.SPEED_INCREASE
                    # Sound: level_up.wav

        # Remove collided blocks safely
        for i in sorted(collided_blocks_indices, reverse=True):
            del self.blocks[i]
        
        self.score += reward
        return reward

    def _check_termination(self):
        terminated = False
        # Loss condition: a block reaches the bottom row
        for block in self.blocks:
            if block['y'] + self.CELL_HEIGHT >= self.HEIGHT:
                terminated = True
                self.score -= 100
                # Sound: game_over.wav
                break
        
        # Win condition: 50 blocks cleared
        if not terminated and self.blocks_cleared >= self.WIN_CONDITION:
            terminated = True
            self.score += 100
            # Sound: victory.wav

        # Timeout condition
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        return terminated

    def _spawn_block(self):
        col = self.np_random.integers(0, self.GRID_COLS)
        points = self.np_random.choice(list(self.BLOCK_COLORS.keys()))
        self.blocks.append({
            'x': self.GRID_X_OFFSET + col * self.CELL_WIDTH,
            'y': self.GRID_Y_OFFSET,
            'points': points
        })

    def _update_blocks(self):
        for block in self.blocks:
            block['y'] += self.fall_speed

    def _create_particles(self, x, y, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [x, y], 'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'color': color, 'life': self.np_random.integers(20, 40)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity effect
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self.clock.tick(30)
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + i * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT))
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + i * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH, y))

        # Draw blocks
        for block in self.blocks:
            rect = pygame.Rect(int(block['x']), int(block['y']), self.CELL_WIDTH, self.CELL_HEIGHT)
            color = self.BLOCK_COLORS[block['points']]
            pygame.draw.rect(self.screen, color, rect)
            border_color = tuple(min(255, c * 0.7) for c in color)
            pygame.draw.rect(self.screen, border_color, rect, 2)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, p['color'])

        # Draw paddle
        paddle_rect = pygame.Rect(
            int(self.paddle_x - self.PADDLE_WIDTH / 2), int(self.PADDLE_Y),
            int(self.PADDLE_WIDTH), int(self.PADDLE_HEIGHT)
        )
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)
        
        # Add a glow effect to the paddle for visibility
        glow_rect = paddle_rect.inflate(6, 6)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PADDLE, 50), glow_surface.get_rect(), border_radius=6)
        self.screen.blit(glow_surface, glow_rect.topleft)

    def _render_ui(self):
        # Score display
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Blocks cleared display
        blocks_text = self.font_small.render(f"CLEARED: {self.blocks_cleared} / {self.WIN_CONDITION}", True, self.COLOR_TEXT)
        blocks_rect = blocks_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(blocks_text, blocks_rect)

        # Game Over / Win message
        if self.game_over:
            msg, color = ("YOU WIN!", (100, 255, 100)) if self.blocks_cleared >= self.WIN_CONDITION else ("GAME OVER", (255, 100, 100))
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 50))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_cleared": self.blocks_cleared,
            "fall_speed": self.fall_speed,
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