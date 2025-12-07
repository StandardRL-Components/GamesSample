
# Generated: 2025-08-28T06:29:42.996128
# Source Brief: brief_05914.md
# Brief Index: 5914

        
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
    user_guide = "Controls: Use Left/Right arrows to move the paddle."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced, top-down block breaker where risky plays are rewarded and safe plays are penalized."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 5.0
    MAX_STEPS = 10000

    BLOCK_ROWS = 5
    BLOCK_COLS = 14
    BLOCK_WIDTH = 40
    BLOCK_HEIGHT = 20
    BLOCK_SPACING = 5
    BLOCK_AREA_TOP = 50

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (200, 200, 0, 50)
    BLOCK_COLORS = [
        (255, 50, 50), (255, 150, 50), (255, 255, 50),
        (50, 255, 50), (50, 255, 255), (50, 150, 255), (150, 50, 255)
    ]
    COLOR_TEXT = (220, 220, 220)
    COLOR_GAMEOVER = (255, 0, 0)
    COLOR_WIN = (0, 255, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_main = pygame.font.SysFont("Impact", 60)

        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.lives = 3
        self.game_over = False
        self.win = False

        self.paddle_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.last_paddle_center_x = self.paddle_rect.centerx

        self.ball_pos = np.array([self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS - 1], dtype=np.float64)
        self.ball_speed = self.INITIAL_BALL_SPEED
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)

        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - total_block_width) / 2
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                block_x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                block_y = self.BLOCK_AREA_TOP + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                rect = pygame.Rect(block_x, block_y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({
                    'rect': rect,
                    'color': self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)],
                    'alive': True
                })
        self.blocks_destroyed_count = 0

        self.last_bounce_type = 'static'
        self.particles = []

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0.0
        terminated = self.game_over or self.win

        if terminated:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self._update_particles()
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are not used in this game
        
        # --- Paddle Movement ---
        self.last_paddle_center_x = self.paddle_rect.centerx
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        self.paddle_rect.x = np.clip(self.paddle_rect.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)
        paddle_movement = self.paddle_rect.centerx - self.last_paddle_center_x

        # --- Ball Movement and Collisions ---
        self.ball_pos += self.ball_vel * self.ball_speed
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collision
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
        self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
        
        # Paddle collision
        if self.ball_vel[1] > 0 and ball_rect.colliderect(self.paddle_rect):
            # sfx_paddle_hit
            reward += 0.1
            self.ball_vel[1] *= -1
            ball_hit_offset = self.ball_pos[0] - self.paddle_rect.centerx
            if paddle_movement == 0:
                self.last_bounce_type = 'static'
            elif math.copysign(1, paddle_movement) != math.copysign(1, ball_hit_offset):
                self.last_bounce_type = 'towards'
            else:
                self.last_bounce_type = 'away'
            
            influence = ball_hit_offset / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = (self.ball_vel[0] + influence) / 2.0
            norm = np.linalg.norm(self.ball_vel)
            if norm > 0: self.ball_vel /= norm
            if abs(self.ball_vel[1]) < 0.1: self.ball_vel[1] = -0.1
            
            self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS

        # Block collision
        block_broken_this_step = False
        for block in self.blocks:
            if block['alive'] and ball_rect.colliderect(block['rect']):
                # sfx_block_break
                block['alive'] = False
                block_broken_this_step = True
                self.blocks_destroyed_count += 1
                self._create_particles(block['rect'].center, block['color'])
                reward += 1.0
                if self.last_bounce_type == 'towards': reward += 5.0
                elif self.last_bounce_type == 'away': reward -= 2.0
                
                self._reflect_ball_from_block(ball_rect, block['rect'])
                
                if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 20 == 0:
                    self.ball_speed += 0.2
                break

        if not block_broken_this_step:
            reward -= 0.02
            
        # --- Game State Update ---
        if self.ball_pos[1] >= self.SCREEN_HEIGHT - self.BALL_RADIUS:
            # sfx_lose_life
            self.lives -= 1
            if self.lives > 0:
                self.ball_pos = np.array([self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS - 5], dtype=np.float64)
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            else:
                self.game_over = True

        if self.game_over:
            terminated = True
            reward -= 100.0
        elif self.blocks_destroyed_count == len(self.blocks):
            self.win = True
            terminated = True
            reward += 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.score += reward
        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _reflect_ball_from_block(self, ball_rect, block_rect):
        overlap = ball_rect.clip(block_rect)
        if overlap.width < overlap.height:
            self.ball_vel[0] *= -1
        else:
            self.ball_vel[1] *= -1

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            size = self.np_random.uniform(2, 5)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color, 'size': size})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _render_game(self):
        # Blocks
        for block in self.blocks:
            if block['alive']:
                pygame.draw.rect(self.screen, block['color'], block['rect'])
                brighter_color = tuple(min(255, c + 30) for c in block['color'])
                darker_color = tuple(max(0, c - 30) for c in block['color'])
                pygame.draw.line(self.screen, brighter_color, block['rect'].topleft, block['rect'].topright, 2)
                pygame.draw.line(self.screen, brighter_color, block['rect'].topleft, block['rect'].bottomleft, 2)
                pygame.draw.line(self.screen, darker_color, block['rect'].bottomright, block['rect'].topright, 2)
                pygame.draw.line(self.screen, darker_color, block['rect'].bottomright, block['rect'].bottomleft, 2)
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30.0))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (int(p['size']), int(p['size'])), int(p['size']))
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        # Ball
        ball_center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (self.BALL_RADIUS * 2, self.BALL_RADIUS * 2), self.BALL_RADIUS * 1.5)
        self.screen.blit(glow_surf, (ball_center[0] - self.BALL_RADIUS * 2, ball_center[1] - self.BALL_RADIUS * 2))
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        lives_text = "LIVES: " + "♥ " * self.lives
        lives_surf = self.font_ui.render(lives_text, True, self.COLOR_TEXT)
        self.screen.blit(lives_surf, (self.SCREEN_WIDTH - lives_surf.get_width() - 10, 10))

        if self.game_over or self.win:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_WIN if self.win else self.COLOR_GAMEOVER
            msg_surf = self.font_main.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        # Render all game elements
        self._render_game()
        # Render UI overlay
        self._render_ui()
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")