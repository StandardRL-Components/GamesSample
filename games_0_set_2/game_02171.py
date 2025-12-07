
# Generated: 2025-08-28T03:57:34.790478
# Source Brief: brief_02171.md
# Brief Index: 2171

        
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

    user_guide = (
        "Controls: ←→ to move the paddle. Destroy all the blocks to win."
    )

    game_description = (
        "A retro arcade game. Control a paddle to deflect a ball and destroy all the blocks in a cosmic arena."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        
        # Colors
        self.COLOR_BG = (25, 15, 40)
        self.COLOR_PADDLE = (0, 200, 255)
        self.COLOR_PADDLE_GLOW = (100, 220, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 50, 50), (255, 150, 50), (255, 255, 50),
            (50, 255, 50), (50, 200, 255), (150, 50, 255)
        ]

        # Fonts
        try:
            self.FONT_UI = pygame.font.SysFont("Consolas", 24)
            self.FONT_MSG = pygame.font.SysFont("Consolas", 48)
        except pygame.error:
            self.FONT_UI = pygame.font.SysFont(None, 28)
            self.FONT_MSG = pygame.font.SysFont(None, 52)

        # Game constants
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_MIN_SPEED = 6
        self.BALL_MAX_SPEED = 12
        self.MAX_STEPS = 2000
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.lives = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_message = None
        self.bg_stars = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.win_message = ""
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * self.BALL_MIN_SPEED
        
        self.blocks = self._create_blocks()
        self.particles = []

        if self.bg_stars is None:
            self.bg_stars = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
                for _ in range(100)
            ]
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        blocks = []
        block_rows = 6
        block_cols = 12
        block_width = self.WIDTH // block_cols
        block_height = 20
        top_offset = 50
        
        for i in range(block_rows):
            for j in range(block_cols):
                block_rect = pygame.Rect(
                    j * block_width,
                    top_offset + i * block_height,
                    block_width - 2,
                    block_height - 2
                )
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                blocks.append({'rect': block_rect, 'color': color})
        return blocks
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # 1. Handle Input
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)
        
        # 2. Update Game Logic
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        
        self.steps += 1
        
        # 3. Check Termination
        terminated = False
        if not self.blocks:
            self.game_over = True
            terminated = True
            reward += 100
            self.win_message = "YOU WIN!"
        elif self.lives <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
            self.win_message = "GAME OVER"
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            self.win_message = "TIME UP"
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_ball(self):
        self.ball_pos += self.ball_vel

    def _handle_collisions(self):
        reward = 0
        
        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            # sfx: wall_bounce

        # Bottom wall (lose life)
        if self.ball_pos.y + self.BALL_RADIUS >= self.HEIGHT:
            self.lives -= 1
            reward -= 10
            # sfx: lose_life
            if self.lives > 0:
                self.paddle.centerx = self.WIDTH / 2
                self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
                angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
                self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * self.BALL_MIN_SPEED
            return reward

        # Paddle collision
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS # Prevent sticking

            # Add spin based on hit location
            hit_offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += hit_offset * 3
            self.ball_vel.scale_to_length(np.clip(self.ball_vel.length() * 1.02, self.BALL_MIN_SPEED, self.BALL_MAX_SPEED))
            # sfx: paddle_hit

        # Block collisions
        for i in range(len(self.blocks) - 1, -1, -1):
            block_item = self.blocks[i]
            if ball_rect.colliderect(block_item['rect']):
                # sfx: block_break
                self.score += 10
                reward += 1
                self._create_particles(block_item['rect'].center, block_item['color'])
                
                # Simple bounce logic
                self.ball_vel.y *= -1 
                
                del self.blocks[i]
                break # Only break one block per frame
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            if p['life'] <= 0:
                del self.particles[i]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw stars
        for x, y, size in self.bg_stars:
            pygame.draw.rect(self.screen, (200, 200, 255, 50), (x, y, size, size))

        # Draw blocks
        for block_item in self.blocks:
            rect = block_item['rect']
            color = block_item['color']
            darker_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, darker_color, rect, width=2, border_radius=3)

        # Draw paddle with glow
        glow_rect = self.paddle.inflate(6, 6)
        glow_rect.center = self.paddle.center
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_GLOW, glow_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Draw ball with glow
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, (*self.COLOR_BALL_GLOW, 100))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'] * (p['life'] / 40))
            if radius > 0:
                # Using a surface for alpha blending
                particle_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (radius, radius), radius)
                self.screen.blit(particle_surf, (pos_int[0] - radius, pos_int[1] - radius))
    
    def _render_ui(self):
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.FONT_UI.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 180, 10))
        for i in range(self.lives):
            life_rect = pygame.Rect(self.WIDTH - 90 + i * 30, 12, 25, 8)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_rect, border_radius=3)

        # Game Over Message
        if self.game_over:
            msg_surf = self.FONT_MSG.render(self.win_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
        }
        
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cosmic Breakout")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Get keyboard input
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        action = [movement, 0, 0] # Space and Shift are not used

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Optionally, auto-reset after a delay
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()