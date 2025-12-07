
# Generated: 2025-08-27T15:06:59.013138
# Source Brief: brief_00888.md
# Brief Index: 888

        
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
        "Controls: Use ←→ to move the block, and press Space to drop it. "
        "Stack blocks to reach the target line without letting any fall off."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game where you stack colored blocks to reach a target height. "
        "Careful placement is key to building a stable tower and achieving a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.BLOCK_HEIGHT_UNITS = 1
        self.BLOCK_PIXEL_HEIGHT = self.GRID_SIZE * self.BLOCK_HEIGHT_UNITS
        self.TARGET_HEIGHT_UNITS = 15
        self.MAX_STEPS = 1000
        self.MOVE_SPEED = self.GRID_SIZE // 2 

        # Colors
        self.COLOR_BG = (44, 62, 80)
        self.COLOR_GRID = (52, 73, 94)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_TARGET_LINE = (231, 76, 60, 200)
        self.BLOCK_COLORS = [
            (26, 188, 156), (46, 204, 113), (52, 152, 219),
            (155, 89, 182), (241, 196, 15), (230, 126, 34),
            (231, 76, 60)
        ]
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24)
        
        # State variables will be initialized in reset()
        self.stacked_blocks = []
        self.falling_block = None
        self.falling_block_color = None
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.particles.clear()
        
        # Create the base platform
        self.stacked_blocks = []
        base_width = self.WIDTH * 0.8
        base_rect = pygame.Rect(
            (self.WIDTH - base_width) / 2,
            self.HEIGHT - self.BLOCK_PIXEL_HEIGHT,
            base_width,
            self.BLOCK_PIXEL_HEIGHT
        )
        self.stacked_blocks.append({'rect': base_rect, 'color': self.COLOR_GRID})
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()

    def _spawn_new_block(self):
        width_units = random.choice([4, 5, 5, 6, 7])
        block_pixel_width = self.GRID_SIZE * width_units

        self.falling_block = pygame.Rect(
            (self.WIDTH - block_pixel_width) / 2,
            self.GRID_SIZE * 2,
            block_pixel_width,
            self.BLOCK_PIXEL_HEIGHT
        )
        self.falling_block_color = random.choice(self.BLOCK_COLORS)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0.0
        
        # --- Handle Movement (if block is not being dropped) ---
        if not space_held:
            if movement == 3:  # Left
                self.falling_block.x -= self.MOVE_SPEED
            elif movement == 4:  # Right
                self.falling_block.x += self.MOVE_SPEED
            
            self.falling_block.x = max(0, self.falling_block.x)
            self.falling_block.x = min(self.WIDTH - self.falling_block.width, self.falling_block.x)

        # --- Handle Dropping the Block ---
        else: # space_held is True, this is the main turn action
            # Find the highest support block directly underneath the falling block
            potential_supports = [b['rect'] for b in self.stacked_blocks if self.falling_block.colliderect(b['rect'].x, 0, b['rect'].width, self.HEIGHT)]
            
            support_block = None
            if potential_supports:
                support_block = min(potential_supports, key=lambda r: r.top)

            is_stable = False
            if support_block:
                overlap_left = max(self.falling_block.left, support_block.left)
                overlap_right = min(self.falling_block.right, support_block.right)
                overlap_width = overlap_right - overlap_left
                
                if overlap_width > 0:
                    is_stable = True
                    self.falling_block.top = support_block.top - self.BLOCK_PIXEL_HEIGHT
                    
                    overlap_ratio = overlap_width / self.falling_block.width
                    if overlap_ratio < 0.5:
                        reward += 2.0
                    elif overlap_ratio > 0.8:
                        reward -= 0.2

            if is_stable:
                # sound_effect: 'block_land.wav'
                reward += 1.0
                
                new_block_data = {'rect': self.falling_block.copy(), 'color': self.falling_block_color}
                self.stacked_blocks.append(new_block_data)
                
                stack_height_units = (self.HEIGHT - self.falling_block.top) / self.BLOCK_PIXEL_HEIGHT
                reward += stack_height_units * 0.1
                
                self._create_particles(self.falling_block.midbottom)

                if stack_height_units >= self.TARGET_HEIGHT_UNITS:
                    self.game_over = True
                    self.win = True
                    reward += 100.0
                else:
                    self._spawn_new_block()
            
            else:
                # sound_effect: 'block_fall.wav'
                self.game_over = True
                reward = -100.0
        
        self.score += reward
        self.steps += 1
        self._update_particles()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
             self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos):
        # sound_effect: 'particle_burst.wav'
        for _ in range(15):
            particle = {
                'pos': list(pos),
                'vel': [random.uniform(-1.5, 1.5), random.uniform(-2.5, -0.5)],
                'size': random.uniform(2, 5),
                'life': random.randint(20, 40),
                'color': (255, 255, 255)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['size'] -= 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        target_y = self.HEIGHT - self.TARGET_HEIGHT_UNITS * self.BLOCK_PIXEL_HEIGHT
        pygame.gfxdraw.hline(self.screen, 0, self.WIDTH, int(target_y), self.COLOR_TARGET_LINE)
        
        for block in self.stacked_blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            pygame.draw.rect(self.screen, tuple(min(255, c * 0.8) for c in block['color']), block['rect'], 2, border_radius=3)

        if self.falling_block and not self.game_over:
            start_pos = self.falling_block.midbottom
            end_y = self.HEIGHT
            potential_supports = [b['rect'] for b in self.stacked_blocks if self.falling_block.colliderect(b['rect'].x, 0, b['rect'].width, self.HEIGHT)]
            if potential_supports:
                end_y = min(r.top for r in potential_supports)
            pygame.draw.line(self.screen, (255, 255, 255, 50), start_pos, (start_pos[0], end_y), 1)

            glow_surf = pygame.Surface((self.falling_block.width + 20, self.falling_block.height + 20), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (255, 255, 255, 40), glow_surf.get_rect(), border_radius=8)
            self.screen.blit(glow_surf, (self.falling_block.x - 10, self.falling_block.y - 10), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.draw.rect(self.screen, self.falling_block_color, self.falling_block, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, self.falling_block, 2, border_radius=3)

        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'])
            if size > 0:
                alpha = max(0, min(255, int(p['life'] * 255 / 20)))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p['color'] + (alpha,))

    def _render_ui(self):
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        height_so_far = 0
        if len(self.stacked_blocks) > 1:
            top_block_y = min(b['rect'].top for b in self.stacked_blocks[1:])
            height_so_far = (self.HEIGHT - top_block_y) / self.BLOCK_PIXEL_HEIGHT

        height_text = self.font_small.render(f"Height: {height_so_far:.0f}/{self.TARGET_HEIGHT_UNITS}", True, self.COLOR_TEXT)
        height_rect = height_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(height_text, height_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)