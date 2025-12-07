
# Generated: 2025-08-28T02:10:17.243660
# Source Brief: brief_01621.md
# Brief Index: 1621

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker. Clear all blocks by bouncing the ball off your paddle. Lose all your balls and it's game over."
    )

    # Frames auto-advance at 30fps.
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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_PADDLE_OUTLINE = (100, 200, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_WALL = (50, 50, 70)
        self.COLOR_TEXT = (200, 200, 220)
        self.BLOCK_COLORS = [
            (255, 0, 128), (255, 128, 0), (224, 224, 0),
            (0, 255, 0), (0, 128, 255)
        ]
        
        # Game constants
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BALL_SPEED_INITIAL = 6
        self.BALL_MAX_SPEED_X = 8
        self.MAX_STEPS = 30 * 90 # 90 seconds at 30fps
        
        # Initialize state variables
        self.np_random = None
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.balls_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.wall_bounces_since_last_hit = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize RNG
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Paddle state
        paddle_y = self.HEIGHT - 40
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        # Ball state (initially attached to paddle)
        self.ball_launched = False
        self._reset_ball()

        # Blocks state
        self._create_blocks()
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        
        # Particles and effects
        self.particles = []
        self.wall_bounces_since_last_hit = 0
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )
        self.ball_vel = [0, 0]
        self.ball_launched = False
        self.wall_bounces_since_last_hit = 0
    
    def _create_blocks(self):
        self.blocks = []
        block_width, block_height = 60, 20
        gap = 4
        num_cols = 10
        num_rows = 5
        total_block_width = num_cols * (block_width + gap) - gap
        start_x = (self.WIDTH - total_block_width) / 2
        start_y = 50

        for i in range(num_rows):
            for j in range(num_cols):
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                x = start_x + j * (block_width + gap)
                y = start_y + i * (block_height + gap)
                self.blocks.append({"rect": pygame.Rect(x, y, block_width, block_height), "color": color})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = 0.0

        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        if space_held and not self.ball_launched:
            # Sound effect placeholder: sfx_launch
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
            self.ball_vel = [self.BALL_SPEED_INITIAL * math.cos(angle), self.BALL_SPEED_INITIAL * math.sin(angle)]

        # 2. Update game logic
        if not self.ball_launched:
            self.ball.centerx = self.paddle.centerx
        else:
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]

        # 3. Collision detection
        # Ball vs Walls
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.x = max(0, min(self.WIDTH - self.ball.width, self.ball.x))
            self.wall_bounces_since_last_hit += 1
            # Sound effect placeholder: sfx_wall_bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.y = max(0, self.ball.y)
            self.wall_bounces_since_last_hit += 1
            # Sound effect placeholder: sfx_wall_bounce

        # Ball vs Paddle
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Sound effect placeholder: sfx_paddle_hit
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            
            offset = (self.ball.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_vel[0] = self.BALL_MAX_SPEED_X * offset
            
            # Speed up ball slightly on each paddle hit to prevent stalemates
            speed_magnitude = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel[0] *= (speed_magnitude + 0.1) / speed_magnitude
            self.ball_vel[1] *= (speed_magnitude + 0.1) / speed_magnitude

            # Reward for paddle hits
            if abs(offset) > 0.7:
                reward += 0.1  # Risky hit
            else:
                reward -= 0.02 # Safe hit
            
            self.wall_bounces_since_last_hit = 0

        # Ball vs Blocks
        hit_block_idx = self.ball.collidelist([b["rect"] for b in self.blocks])
        if hit_block_idx != -1:
            # Sound effect placeholder: sfx_block_break
            block_hit = self.blocks[hit_block_idx]
            
            # Determine collision side to correctly reflect
            # This is a simplified but effective approach
            dx = self.ball.centerx - block_hit["rect"].centerx
            dy = self.ball.centery - block_hit["rect"].centery
            if abs(dx / block_hit["rect"].width) > abs(dy / block_hit["rect"].height):
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[1] *= -1

            self._create_particles(block_hit["rect"].center, block_hit["color"])
            self.blocks.pop(hit_block_idx)
            
            self.score += 1
            reward += 1.0
            self.wall_bounces_since_last_hit = 0

        # Ball out of bounds
        if self.ball.top > self.HEIGHT:
            # Sound effect placeholder: sfx_lose_ball
            self.balls_left -= 1
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
                reward -= 100.0

        # 4. Anti-softlock mechanism
        if self.wall_bounces_since_last_hit > 15:
            self.ball_vel[0] += self.np_random.uniform(-0.5, 0.5)
            self.ball_vel[1] -= 0.5 # Bias downwards
            self.wall_bounces_since_last_hit = 0

        # 5. Update particles
        self._update_particles()

        # 6. Check for termination conditions
        terminated = False
        if not self.blocks: # Win condition
            self.game_over = True
            terminated = True
            reward += 100.0
        elif self.balls_left <= 0: # Lose condition
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS: # Max steps
            terminated = True
        
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['vel'][1] += 0.1 # Gravity

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            r, g, b = p['color']
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (r, g, b, alpha)
            )

        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            brighter_color = tuple(min(255, c + 50) for c in block["color"])
            pygame.draw.rect(self.screen, brighter_color, block["rect"], 2)

        # Render paddle with neon glow
        glow_rect = self.paddle.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE, 50), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_OUTLINE, self.paddle, 2, border_radius=3)

        # Render ball with neon glow
        pygame.gfxdraw.filled_circle(self.screen, self.ball.centerx, self.ball.centery, self.BALL_RADIUS + 3, (*self.COLOR_BALL, 50))
        pygame.gfxdraw.filled_circle(self.screen, self.ball.centerx, self.ball.centery, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, self.ball.centerx, self.ball.centery, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        for i in range(self.balls_left):
            x = self.WIDTH - 20 - i * 20
            pygame.gfxdraw.filled_circle(self.screen, x, 20, 6, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, x, 20, 6, self.COLOR_PADDLE_OUTLINE)

        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            color = (0, 255, 128) if not self.blocks else (255, 0, 100)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Get human input
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and (terminated or truncated):
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'R' to restart
            pass
        
        clock.tick(30) # Match the auto_advance rate
        
    env.close()