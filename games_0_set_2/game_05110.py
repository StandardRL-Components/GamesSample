
# Generated: 2025-08-28T03:59:50.659181
# Source Brief: brief_05110.md
# Brief Index: 5110

        
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


# Helper class for particle effects
class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, lifetime=20, size=5, p_type='explode'):
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
        
        if p_type == 'explode':
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed
        elif p_type == 'implode':
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2)
            self.vx = math.cos(angle) * speed * -1 # Move inward
            self.vy = math.sin(angle) * speed * -1
        elif p_type == 'trail':
            self.vx = random.uniform(-0.5, 0.5)
            self.vy = random.uniform(-0.5, 0.5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            progress = self.lifetime / self.max_lifetime
            alpha = int(255 * progress * 0.8) # Fade out
            radius = int(self.size * progress)
            if radius > 0:
                color_with_alpha = (*self.color, alpha)
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), radius, color_with_alpha)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ↑/↓ to move paddle. Press Space to cycle color."

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match your paddle's color to the incoming ball. Score 10 points to win, or miss 5 balls to lose."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    GRID_CELL_SIZE = 36
    PLAY_AREA_SIZE = GRID_SIZE * GRID_CELL_SIZE
    PLAY_AREA_X = (SCREEN_WIDTH - PLAY_AREA_SIZE) // 2
    PLAY_AREA_Y = (SCREEN_HEIGHT - PLAY_AREA_SIZE) // 2
    PADDLE_COLUMN = GRID_SIZE - 1

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_WHITE = (255, 255, 255)
    PALETTE = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.game_over_message = ""
        self.paddle_y = 0
        self.paddle_color_index = 0
        self.ball_x = 0.0
        self.ball_y = 0
        self.ball_color_index = 0
        self.ball_speed = 0.0
        self.particles = []
        self.space_was_held = False
        self.last_paddle_dist_to_ball = 0.0
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.game_over_message = ""
        
        self.paddle_y = self.GRID_SIZE // 2
        self.paddle_color_index = 0
        self.ball_speed = 0.2  # Grid units per step
        self.particles = []
        self.space_was_held = False
        
        self._spawn_ball()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = 0
        
        if not self.game_over:
            # 1. Unpack and handle actions
            movement = action[0]
            space_held = action[1] == 1
            
            if movement == 1:  # Up
                self.paddle_y = max(0, self.paddle_y - 1)
            elif movement == 2:  # Down
                self.paddle_y = min(self.GRID_SIZE - 1, self.paddle_y + 1)
            
            if space_held and not self.space_was_held:
                self.paddle_color_index = (self.paddle_color_index + 1) % len(self.PALETTE)
                # SFX: color_change.wav
            self.space_was_held = space_held
            
            # 2. Update game logic
            self.steps += 1
            
            # Continuous reward for paddle positioning
            current_dist = abs(self.ball_y - self.paddle_y)
            reward += (self.last_paddle_dist_to_ball - current_dist) * 0.1
            self.last_paddle_dist_to_ball = current_dist

            # Ball movement and trail
            prev_ball_x = self.ball_x
            self.ball_x += self.ball_speed
            if self.steps % 3 == 0:
                self._create_particles(self.ball_x, self.ball_y, self.PALETTE[self.ball_color_index], 1, 'trail')

            # 3. Collision detection
            if prev_ball_x < self.PADDLE_COLUMN and self.ball_x >= self.PADDLE_COLUMN:
                if self.paddle_y == self.ball_y:
                    if self.paddle_color_index == self.ball_color_index:
                        self.score += 1
                        reward += 1.0
                        self.ball_speed += 0.02
                        self._create_particles(self.PADDLE_COLUMN, self.ball_y, self.PALETTE[self.ball_color_index], 30, 'explode')
                        # SFX: success.wav
                    else:
                        self.misses += 1
                        reward -= 1.0
                        self._create_particles(self.PADDLE_COLUMN, self.ball_y, (128,128,128), 20, 'implode')
                        # SFX: mismatch.wav
                    self._spawn_ball()

            if self.ball_x >= self.GRID_SIZE:
                self.misses += 1
                reward -= 1.0
                # SFX: miss.wav
                self._spawn_ball()

        self._update_particles()
        
        # 4. Check for termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= 10:
                reward += 10.0
                self.game_over_message = "YOU WIN!"
            elif self.misses >= 5:
                reward -= 10.0
                self.game_over_message = "GAME OVER"
            elif self.steps >= 1000:
                self.game_over_message = "TIME'S UP"
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _spawn_ball(self):
        self.ball_x = 0.0
        self.ball_y = random.randint(0, self.GRID_SIZE - 1)
        self.ball_color_index = random.randint(0, len(self.PALETTE) - 1)
        self.last_paddle_dist_to_ball = abs(self.ball_y - self.paddle_y)

    def _check_termination(self):
        return self.score >= 10 or self.misses >= 5 or self.steps >= 1000

    def _create_particles(self, grid_x, grid_y, color, count, p_type='explode'):
        px, py = self._grid_to_pixel(grid_x, grid_y, center=True)
        for _ in range(count):
            self.particles.append(Particle(px, py, color, p_type=p_type))
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _get_observation(self):
        # Clear screen with a background color
        self.screen.fill(self.COLOR_BG)
        
        # Create a separate surface for elements that need blending
        blend_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        self._render_grid()
        self._render_ball(blend_surface)
        self._render_paddle(blend_surface)
        self._render_particles(blend_surface)
        
        # Blit the blended elements onto the main screen
        self.screen.blit(blend_surface, (0,0), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Render non-blended elements like UI on top
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
        }

    def _grid_to_pixel(self, grid_x, grid_y, center=False):
        x = self.PLAY_AREA_X + grid_x * self.GRID_CELL_SIZE
        y = self.PLAY_AREA_Y + grid_y * self.GRID_CELL_SIZE
        if center:
            x += self.GRID_CELL_SIZE // 2
            y += self.GRID_CELL_SIZE // 2
        return int(x), int(y)

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            start_pos_v = (self.PLAY_AREA_X + i * self.GRID_CELL_SIZE, self.PLAY_AREA_Y)
            end_pos_v = (self.PLAY_AREA_X + i * self.GRID_CELL_SIZE, self.PLAY_AREA_Y + self.PLAY_AREA_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos_v, end_pos_v)
            
            start_pos_h = (self.PLAY_AREA_X, self.PLAY_AREA_Y + i * self.GRID_CELL_SIZE)
            end_pos_h = (self.PLAY_AREA_X + self.PLAY_AREA_SIZE, self.PLAY_AREA_Y + i * self.GRID_CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos_h, end_pos_h)

    def _render_particles(self, surface):
        for p in self.particles:
            p.draw(surface)

    def _render_ball(self, surface):
        color = self.PALETTE[self.ball_color_index]
        px, py = self._grid_to_pixel(self.ball_x, self.ball_y, center=True)
        radius = self.GRID_CELL_SIZE // 2 - 4
        
        # Main ball on main screen for solid color
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_WHITE)

        # Glow effect on blend surface
        glow_radius = radius + 6
        pygame.gfxdraw.filled_circle(surface, px, py, glow_radius, (*color, 60))

    def _render_paddle(self, surface):
        color = self.PALETTE[self.paddle_color_index]
        px, py = self._grid_to_pixel(self.PADDLE_COLUMN, self.paddle_y)
        paddle_rect = pygame.Rect(px, py, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
        
        # Main paddle on main screen
        pygame.draw.rect(self.screen, color, paddle_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, paddle_rect, width=2, border_radius=4)

        # Glow effect on blend surface
        glow_rect = pygame.Rect(paddle_rect.x - 6, paddle_rect.y - 6, self.GRID_CELL_SIZE + 12, self.GRID_CELL_SIZE + 12)
        pygame.draw.rect(surface, (*color, 120), glow_rect, border_radius=8)
        
    def _render_ui(self):
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        miss_text = self.font_medium.render(f"MISSES: {self.misses} / 5", True, self.COLOR_UI_TEXT)
        miss_rect = miss_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(miss_text, miss_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg_text = self.font_large.render(self.game_over_message, True, self.COLOR_WHITE)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def validate_implementation(self):
        print("Running implementation validation...")
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
        
        print("✓ Implementation validated successfully")