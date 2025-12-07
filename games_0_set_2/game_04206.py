
# Generated: 2025-08-28T01:43:39.045871
# Source Brief: brief_04206.md
# Brief Index: 4206

        
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
    user_guide = "Controls: Use ← and → to move the paddle."

    # Must be a short, user-facing description of the game:
    game_description = (
        "A vibrant top-down Breakout game with a retro-neon aesthetic. Destroy all blocks to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 5.0
        self.LIVES = 3
        
        # --- Colors ---
        self.COLOR_BG = (10, 10, 30)
        self.COLOR_GRID = (20, 20, 40)
        self.COLOR_PADDLE = (255, 255, 0)
        self.COLOR_BALL = (255, 0, 255)
        self.COLOR_UI = (200, 200, 255)
        self.BLOCK_COLORS = [
            (0, 255, 255), (0, 255, 128), (128, 255, 0),
            (255, 128, 0), (255, 0, 128)
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.blocks = []
        self.particles = []
        self.ball_trail = []
        self.blocks_destroyed_count = 0
        
        # Initialize state variables
        self.reset()

        # --- Self-Validation ---
        # self.validate_implementation() # Uncomment for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.LIVES
        self.blocks_destroyed_count = 0
        
        # Paddle
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle_rect = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        # Ball
        self.ball_speed = self.INITIAL_BALL_SPEED
        self._reset_ball()
        
        # Blocks (procedural generation)
        self.blocks = []
        block_width, block_height = 60, 20
        gap = 4
        rows = 5
        cols = self.WIDTH // (block_width + gap)
        start_x = (self.WIDTH - cols * (block_width + gap) + gap) / 2
        start_y = 50
        for r in range(rows):
            for c in range(cols):
                if self.np_random.random() > 0.1: # 90% chance of block spawning
                    x = start_x + c * (block_width + gap)
                    y = start_y + r * (block_height + gap)
                    color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                    block = {
                        "rect": pygame.Rect(x, y, block_width, block_height),
                        "color": color
                    }
                    self.blocks.append(block)

        # Effects
        self.particles = []
        self.ball_trail = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = np.array([self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS - 1], dtype=np.float64)
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upward direction
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * self.ball_speed
        self.ball_trail = []

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # --- Action Handling ---
        movement = action[0]
        
        if movement == 3: # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
            reward -= 0.02
        elif movement == 4: # Right
            self.paddle_rect.x += self.PADDLE_SPEED
            reward -= 0.02
            
        # Clamp paddle to screen
        self.paddle_rect.x = max(0, min(self.paddle_rect.x, self.WIDTH - self.PADDLE_WIDTH))

        # --- Ball Physics ---
        self.ball_pos += self.ball_vel
        self.ball_trail.append(self.ball_pos.copy())
        if len(self.ball_trail) > 10:
            self.ball_trail.pop(0)

        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        
        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS
            
            # Change ball angle based on hit location
            offset = (self.ball_pos[0] - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.0
            
            # Reward for paddle hit style
            if abs(offset) < 0.5: # Safe hit
                reward += 0.4
            else: # Risky hit
                reward -= 0.2

            # Normalize velocity to maintain constant speed
            self.ball_vel = self.ball_vel / np.linalg.norm(self.ball_vel) * self.ball_speed

        # Block collisions
        hit_block = None
        for block in self.blocks:
            if ball_rect.colliderect(block["rect"]):
                hit_block = block
                break
        
        if hit_block:
            reward += 1.0
            self.score += 10
            self.blocks.remove(hit_block)
            self._create_particles(hit_block["rect"].center, hit_block["color"])
            
            # Determine bounce direction
            # A simple but effective method: check which side is closest
            dx = self.ball_pos[0] - hit_block["rect"].centerx
            dy = self.ball_pos[1] - hit_block["rect"].centery
            w, h = hit_block["rect"].width / 2, hit_block["rect"].height / 2
            
            if abs(dx / w) > abs(dy / h):
                self.ball_vel[0] *= -1 # Horizontal collision
            else:
                self.ball_vel[1] *= -1 # Vertical collision
            
            self.blocks_destroyed_count += 1
            if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 10 == 0:
                self.ball_speed += 0.5 # Difficulty scaling
                self.ball_vel = self.ball_vel / np.linalg.norm(self.ball_vel) * self.ball_speed
                
        # --- Update Particles ---
        self._update_particles()
        
        # --- Termination Conditions ---
        # Lose a life
        if self.ball_pos[1] > self.HEIGHT:
            self.lives -= 1
            reward -= 10
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True
        
        # Win condition
        if not self.blocks:
            reward += 100
            self.score += 1000
            self.game_over = True
            
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": np.array(pos, dtype=np.float64),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Drag
            p["life"] -= 1

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
        }

    def _render_all(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            # Inner glow effect
            inner_rect = block["rect"].inflate(-6, -6)
            s = pygame.Surface(inner_rect.size, pygame.SRCALPHA)
            s.fill((255, 255, 255, 60))
            self.screen.blit(s, inner_rect.topleft)

        # Draw paddle with glow
        paddle_color = self.COLOR_PADDLE
        pygame.draw.rect(self.screen, paddle_color, self.paddle_rect, border_radius=5)
        glow_rect = self.paddle_rect.inflate(4, 4)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*paddle_color, 60), s.get_rect(), border_radius=7)
        self.screen.blit(s, glow_rect.topleft)

        # Draw ball trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(255 * (i / len(self.ball_trail)))
            color = (*self.COLOR_BALL, alpha)
            radius = int(self.BALL_RADIUS * (i / len(self.ball_trail)))
            if radius > 0:
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (pos[0] - radius, pos[1] - radius))

        # Draw ball with glow
        ball_color = self.COLOR_BALL
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        for i in range(5, 0, -1):
            alpha = 100 - i * 20
            pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + i, (*ball_color, alpha))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, ball_color)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, ball_color)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30.0))
            color = (*p["color"], alpha)
            size = max(1, int(3 * (p["life"] / 30.0)))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (size, size), size)
            self.screen.blit(s, (int(p["pos"][0] - size), int(p["pos"][1] - size)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_large.render(f"LIVES: {self.lives}", True, self.COLOR_UI)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            msg = "LEVEL CLEARED!" if not self.blocks else "GAME OVER"
            color = (0, 255, 0) if not self.blocks else (255, 0, 0)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            game_over_text = self.font_large.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(game_over_text, text_rect)

            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_UI)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, score_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    env.reset()
    
    # Override screen to be a display surface
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Breakout Neon")

    running = True
    terminated = False
    
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_r]: # Reset game
            terminated = False
            env.reset()
            
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            env._render_all() # Re-render to the display surface
            pygame.display.flip()
        
        env.clock.tick(60) # Run at 60 FPS for smooth human play
        
    env.close()