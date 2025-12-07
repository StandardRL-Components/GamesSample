
# Generated: 2025-08-28T02:19:23.413404
# Source Brief: brief_04410.md
# Brief Index: 4410

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Use ← and → to move the paddle. Your goal is to break all the blocks."
    )

    # Short, user-facing description of the game
    game_description = (
        "A minimalist, neon-drenched block-breaking game. Control the paddle to bounce the ball, "
        "destroy all the blocks, and achieve a high score."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors (Neon on Dark)
    COLOR_BG = (15, 15, 25)
    COLOR_BG_GRID = (30, 30, 45)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (0, 255, 255) # Cyan
    COLOR_TEXT = (220, 220, 220)
    COLOR_LIVES = (255, 100, 100)

    # Game Parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 12
    PADDLE_SPEED = 10
    PADDLE_Y = SCREEN_HEIGHT - 30

    BALL_RADIUS = 7
    BALL_INITIAL_SPEED = 6
    BALL_MAX_SPEED = 9
    BALL_SPEED_INCREASE = 0.05

    BLOCK_ROWS = 10
    BLOCK_COLS = 10
    BLOCK_WIDTH = 58
    BLOCK_HEIGHT = 15
    BLOCK_SPACING = 6
    BLOCK_AREA_TOP = 50

    MAX_STEPS = 2000
    INITIAL_LIVES = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_lives = pygame.font.SysFont("Consolas", 20, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 32)
            self.font_lives = pygame.font.SysFont(None, 28)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.blocks = []
        self.particles = []
        self.last_block_break_step = -100
        self.total_blocks = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES

        # Paddle
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, self.PADDLE_Y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball
        self._reset_ball()

        # Blocks
        self.blocks = []
        self.total_blocks = self.BLOCK_ROWS * self.BLOCK_COLS
        block_total_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - block_total_width) / 2
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                hue = int((i / self.BLOCK_ROWS) * 360)
                color = pygame.Color(0)
                color.hsla = (hue, 100, 50, 100)
                
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_AREA_TOP + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({"rect": block_rect, "color": color})

        # Particles & Effects
        self.particles = []
        self.last_block_break_step = -100

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_speed = self.BALL_INITIAL_SPEED
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards
        self.ball_vel = [math.cos(angle) * self.ball_speed, math.sin(angle) * self.ball_speed]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Reward for surviving

        # --- 1. Handle Action ---
        movement = action[0]
        # action[1] (space) and action[2] (shift) are unused.
        
        paddle_moved = False
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            paddle_moved = True
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            paddle_moved = True

        if paddle_moved:
            reward -= 0.02 # Small penalty for movement

        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        # --- 2. Update Game Logic ---
        # Ball movement
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # --- 3. Handle Collisions ---
        # Wall collisions
        if ball_rect.left <= 0:
            ball_rect.left = 0
            self.ball_vel[0] *= -1
            # sfx: wall_bounce
        if ball_rect.right >= self.SCREEN_WIDTH:
            ball_rect.right = self.SCREEN_WIDTH
            self.ball_vel[0] *= -1
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            ball_rect.top = 0
            self.ball_vel[1] *= -1
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_bounce
            self.ball_vel[1] *= -1
            ball_rect.bottom = self.paddle.top
            
            # Change horizontal velocity based on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = offset * self.ball_speed * 1.2

            # Normalize velocity to maintain constant speed
            current_speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if current_speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / current_speed) * self.ball_speed
                self.ball_vel[1] = (self.ball_vel[1] / current_speed) * self.ball_speed

        # Block collisions
        block_hit_index = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if block_hit_index != -1:
            # sfx: block_break
            hit_block = self.blocks.pop(block_hit_index)
            self._create_particles(hit_block['rect'].center, hit_block['color'])
            
            # Reward for breaking a block
            reward += 1.0
            self.score += 10
            
            # Chain reaction bonus
            if self.steps - self.last_block_break_step <= 3:
                reward += 2.0
                self.score += 20 # Bonus score
            self.last_block_break_step = self.steps

            # Slightly increase ball speed
            self.ball_speed = min(self.BALL_MAX_SPEED, self.ball_speed + self.BALL_SPEED_INCREASE)

            # Collision response
            # A simple but effective method: reverse velocity based on which axis has less overlap
            overlap = ball_rect.clip(hit_block['rect'])
            if overlap.width < overlap.height:
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[1] *= -1

        # Ball out of bounds (bottom)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            if self.lives > 0:
                self._reset_ball()
                # sfx: lose_life
            else:
                self.game_over = True
                reward -= 100  # Terminal penalty
                # sfx: game_over
        
        self.ball_pos = [ball_rect.centerx, ball_rect.centery]

        # --- 4. Update Particles ---
        self._update_particles()

        # --- 5. Check Termination ---
        terminated = self.game_over
        if len(self.blocks) == 0:
            terminated = True
            reward += 100 # Win bonus
            self.score += 1000 # Win score bonus
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, 20):
            for y in range(0, self.SCREEN_HEIGHT, 20):
                pygame.gfxdraw.pixel(self.screen, x, y, self.COLOR_BG_GRID)

        # --- Game Elements ---
        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
        
        # Particles
        for p in self.particles:
            alpha = int(p['life'] / p['max_life'] * 255)
            color = p['color']
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (color[0], color[1], color[2], alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Paddle with glow
        glow_rect = self.paddle.inflate(6, 6)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PADDLE, 50), glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Ball with glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 3, (*self.COLOR_BALL, 80))
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        
        # --- UI Overlay ---
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Lives
        for i in range(self.lives):
            life_rect = pygame.Rect(self.SCREEN_WIDTH - 25 - (i * 25), 15, 20, 5)
            pygame.draw.rect(self.screen, self.COLOR_LIVES, life_rect, border_radius=2)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(1, 4)
            })
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

if __name__ == '__main__':
    # --- Example of how to run the environment ---
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Main game loop
    running = True
    while running:
        # Human controls
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # Default: no-op
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        # Event handling (for closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        # Control the frame rate
        env.clock.tick(30)
        
    env.close()