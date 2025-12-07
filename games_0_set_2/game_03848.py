import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle for effects."""
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = 30  # 1 second at 30 FPS
        self.max_lifespan = 30
        self.color = color
        self.radius = np_random.integers(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / self.max_lifespan))))
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.x - self.radius), int(self.y - self.radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: ←→ to move the paddle. Press space to launch the ball."
    game_description = "Bounce a pixelated ball off a paddle to destroy all the blocks within a 60-second time limit."
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_TIME_SECONDS = 60
    MAX_STEPS = GAME_TIME_SECONDS * FPS

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_PADDLE = (220, 220, 220)
    COLOR_BALL = (255, 255, 0)
    COLOR_WALL = (80, 80, 100)
    COLOR_TEXT = (240, 240, 240)
    BLOCK_COLORS = [(217, 87, 99), (217, 143, 87), (217, 217, 87), (143, 217, 87), (87, 217, 217)]
    
    # Game element properties
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 7
    BALL_INITIAL_SPEED_Y = -6
    BALL_MAX_SPEED_Y = 7
    BALL_MAX_SPEED_X = 8
    
    BLOCK_ROWS = 5
    BLOCK_COLS = 10
    BLOCK_WIDTH = 58
    BLOCK_HEIGHT = 20
    BLOCK_SPACING = 4
    BLOCK_AREA_TOP = 60
    WALL_THICKNESS = 10

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.game_over = False
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.game_state = 'waiting'
        self.score = 0
        self.steps = 0
        self.timer = self.GAME_TIME_SECONDS
        
        # Call reset to initialize game state before validation
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_over = False
        self.steps = 0
        self.score = 0
        self.timer = self.GAME_TIME_SECONDS
        self.game_state = 'waiting'
        self.particles = []

        # Paddle setup
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT * 2
        self.paddle = pygame.Rect((self.SCREEN_WIDTH - self.PADDLE_WIDTH) // 2, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball setup (attached to paddle)
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top
        self.ball_vel = [0, 0]

        # Block setup
        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - total_block_width) // 2
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_AREA_TOP + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                self.blocks.append((block_rect, color))
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.clock.tick(self.FPS)
        self.steps += 1
        self.timer -= 1 / self.FPS
        reward = 0.0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.left = max(self.WALL_THICKNESS, self.paddle.left)
        self.paddle.right = min(self.SCREEN_WIDTH - self.WALL_THICKNESS, self.paddle.right)

        # --- Game Logic ---
        if self.game_state == 'waiting':
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
            if space_held:
                # sfx: launch_ball.wav
                self.game_state = 'playing'
                initial_vx = self.np_random.uniform(-2, 2)
                self.ball_vel = [initial_vx, self.BALL_INITIAL_SPEED_Y]
        
        elif self.game_state == 'playing':
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]
            
            # --- Collisions ---
            # Walls
            if self.ball.left <= self.WALL_THICKNESS or self.ball.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
                self.ball_vel[0] *= -1
                self.ball.left = max(self.ball.left, self.WALL_THICKNESS + 1)
                self.ball.right = min(self.ball.right, self.SCREEN_WIDTH - self.WALL_THICKNESS - 1)
                # sfx: wall_bounce.wav
            if self.ball.top <= self.WALL_THICKNESS:
                self.ball_vel[1] *= -1
                self.ball.top = max(self.ball.top, self.WALL_THICKNESS + 1)
                # sfx: wall_bounce.wav
            # The ball bounces off the bottom wall in this timed mode
            if self.ball.bottom >= self.SCREEN_HEIGHT - self.WALL_THICKNESS:
                self.ball_vel[1] *= -1
                self.ball.bottom = min(self.ball.bottom, self.SCREEN_HEIGHT - self.WALL_THICKNESS - 1)
                # sfx: wall_bounce.wav
            
            # Paddle
            if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
                reward += 0.1
                # sfx: paddle_hit.wav
                offset = self.ball.centerx - self.paddle.centerx
                normalized_offset = offset / (self.PADDLE_WIDTH / 2)
                normalized_offset = max(-1.0, min(1.0, normalized_offset))
                self.ball_vel[0] = normalized_offset * self.BALL_MAX_SPEED_X
                self.ball_vel[1] *= -1
                self.ball_vel[1] = -min(abs(self.ball_vel[1]), self.BALL_MAX_SPEED_Y)
                self.ball.bottom = self.paddle.top
            
            # Blocks
            hit_block = self.ball.collidelist([b[0] for b in self.blocks])
            if hit_block != -1:
                block_rect, block_color = self.blocks[hit_block]
                # sfx: block_destroy.wav
                reward += 1.0
                self.score += 10
                
                # Create particles
                for _ in range(15):
                    self.particles.append(Particle(self.ball.centerx, self.ball.centery, block_color, self.np_random))
                
                # Bounce logic
                # Determine if collision was vertical or horizontal
                # A simple way: check overlap
                overlap_rect = self.ball.clip(block_rect)
                if overlap_rect.width > overlap_rect.height:
                    self.ball_vel[1] *= -1 # Vertical collision
                else:
                    self.ball_vel[0] *= -1 # Horizontal collision

                self.blocks.pop(hit_block)
        
        # --- Update Particles ---
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

        # --- Termination Conditions ---
        terminated = False
        win = not self.blocks
        lose = self.timer <= 0

        if win:
            reward += 50.0
            self.score += 1000 # Win bonus
            terminated = True
        elif lose:
            reward -= 50.0
            terminated = True
        
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # Background
        self.screen.fill(self.COLOR_BG)

        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS)) # Top
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.SCREEN_HEIGHT - self.WALL_THICKNESS, self.SCREEN_WIDTH, self.WALL_THICKNESS)) # Bottom
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT)) # Left
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT)) # Right

        # Blocks
        for block_rect, color in self.blocks:
            pygame.draw.rect(self.screen, color, block_rect)
            # Add a subtle 3D effect
            pygame.draw.rect(self.screen, tuple(min(255, c + 30) for c in color), block_rect, 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        pygame.draw.circle(self.screen, self.COLOR_BALL, self.ball.center, self.BALL_RADIUS)
        # Add a glow effect to the ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS + 2, (255, 255, 150, 100))

        # Particles
        for p in self.particles:
            p.draw(self.screen)
            
        # UI
        self._render_ui()

        # Final message on game over
        if self.game_over:
            message = "YOU WIN!" if not self.blocks else "TIME'S UP!"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WALL_THICKNESS + 10, self.WALL_THICKNESS + 5))

        # Timer
        time_left = max(0, self.timer)
        timer_text = f"TIME: {int(time_left):02d}"
        
        # Color transition for timer: Green -> Yellow -> Red
        time_ratio = time_left / self.GAME_TIME_SECONDS
        if time_ratio > 0.5:
            # Green to Yellow
            r = int(255 * (1 - (time_ratio - 0.5) * 2))
            g = 255
        else:
            # Yellow to Red
            r = 255
            g = int(255 * (time_ratio * 2))
        timer_color = (r, g, 0)

        timer_surf = self.font_small.render(timer_text, True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - self.WALL_THICKNESS - 10, self.WALL_THICKNESS + 5))
        self.screen.blit(timer_surf, timer_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.timer,
            "blocks_remaining": len(self.blocks),
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset, which also implicitly tests the observation space
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8, f"Obs dtype is {obs.dtype}"
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