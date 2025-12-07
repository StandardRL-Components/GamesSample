
# Generated: 2025-08-28T00:17:15.054962
# Source Brief: brief_03746.md
# Brief Index: 3746

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro block-breaker. Clear all blocks to advance through levels. Don't let the ball fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.MAX_LEVELS = 3
        self.INITIAL_LIVES = 3

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_LIVES = (220, 40, 40)
        self.BLOCK_COLORS = [
            (50, 50, 200), (50, 200, 50), (200, 50, 50),
            (200, 200, 50), (50, 200, 200), (200, 50, 200)
        ]

        # Paddle
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 12
        self.PADDLE_Y = self.HEIGHT - 40
        self.PADDLE_SPEED = 8

        # Ball
        self.BALL_RADIUS = 6
        self.INITIAL_BALL_SPEED = 3.0

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.level = 0
        self.paddle_pos = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = True
        self.blocks = []
        self.particles = []
        self.current_reward = 0
        self.game_over = False
        self.game_won = False
        self.np_random = None

        # Call validation at the end of __init__
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.level = 1
        self.game_over = False
        self.game_won = False
        self.particles = []
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.current_reward = -0.01 # Small penalty for time passing
        
        # --- Handle Actions ---
        movement = action[0]
        space_pressed = action[1] == 1
        
        self._update_paddle(movement)
        self._handle_ball_launch(space_pressed)

        # --- Update Game Logic ---
        if not self.ball_attached:
            self._update_ball()
            self._handle_collisions()
        
        self._update_particles()
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = self.game_over or self.game_won or (self.steps >= self.MAX_STEPS)
        if terminated and not (self.game_over or self.game_won): # Max steps reached
            self.game_over = True
        
        return (
            self._get_observation(),
            self.current_reward,
            terminated,
            False,
            self._get_info()
        )

    def _setup_level(self):
        self.paddle_pos = pygame.Vector2(self.WIDTH / 2, self.PADDLE_Y)
        self.ball_attached = True
        self._attach_ball_to_paddle()
        
        self.blocks = []
        rows = 4 + self.level
        cols = 10
        block_width = self.WIDTH / cols
        block_height = 20
        
        for r in range(rows):
            for c in range(cols):
                # Create varied patterns based on level
                if (self.level == 2 and (c+r) % 2 == 0) or \
                   (self.level == 3 and (c*r) % 4 == 0):
                    continue
                
                point_value = min(5, self.level + r // 2)
                color = self.BLOCK_COLORS[point_value % len(self.BLOCK_COLORS)]
                
                block_rect = pygame.Rect(
                    c * block_width,
                    50 + r * block_height,
                    block_width - 2,
                    block_height - 2
                )
                self.blocks.append({"rect": block_rect, "color": color, "points": point_value})

    def _update_paddle(self, movement):
        if movement == 3:  # Left
            self.paddle_pos.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_pos.x += self.PADDLE_SPEED
        
        self.paddle_pos.x = np.clip(
            self.paddle_pos.x, self.PADDLE_WIDTH / 2, self.WIDTH - self.PADDLE_WIDTH / 2
        )
        
        if self.ball_attached:
            self._attach_ball_to_paddle()

    def _handle_ball_launch(self, space_pressed):
        if self.ball_attached and space_pressed:
            self.ball_attached = False
            ball_speed = self.INITIAL_BALL_SPEED + (self.level - 1) * 0.5
            launch_angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = pygame.Vector2(
                math.cos(launch_angle) * ball_speed,
                math.sin(launch_angle) * ball_speed
            )
            # sfx: launch_ball.wav

    def _attach_ball_to_paddle(self):
        self.ball_pos = pygame.Vector2(self.paddle_pos.x, self.paddle_pos.y - self.PADDLE_HEIGHT / 2 - self.BALL_RADIUS)

    def _update_ball(self):
        self.ball_pos += self.ball_vel

    def _handle_collisions(self):
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        paddle_rect = pygame.Rect(self.paddle_pos.x - self.PADDLE_WIDTH / 2, self.PADDLE_Y - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Wall collisions
        if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS + 1, self.WIDTH - self.BALL_RADIUS - 1)
            # sfx: bounce_wall.wav
            # Anti-stuck mechanism
            if abs(self.ball_vel.y) < 0.1:
                self.ball_vel.y += self.np_random.uniform(-0.2, 0.2)

        if self.ball_pos.y <= self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS + 1
            # sfx: bounce_wall.wav

        # Paddle collision
        if self.ball_vel.y > 0 and ball_rect.colliderect(paddle_rect):
            # sfx: bounce_paddle.wav
            hit_pos = (self.ball_pos.x - self.paddle_pos.x) / (self.PADDLE_WIDTH / 2)
            hit_pos = np.clip(hit_pos, -1, 1)

            # Reward for risky edge hits
            if abs(hit_pos) > 0.8:
                self.current_reward += 5.0
            else:
                self.current_reward += 0.1
            
            # Change angle based on hit position
            angle = math.pi * 0.5 * hit_pos
            speed = self.ball_vel.length()
            self.ball_vel.x = speed * math.sin(angle)
            self.ball_vel.y = -speed * math.cos(angle)
            
            # Ensure ball is above paddle to prevent multiple collisions
            self.ball_pos.y = self.PADDLE_Y - self.PADDLE_HEIGHT / 2 - self.BALL_RADIUS

        # Ball out of bounds (lose life)
        if self.ball_pos.y > self.HEIGHT:
            self.lives -= 1
            # sfx: lose_life.wav
            if self.lives <= 0:
                self.game_over = True
                self.current_reward -= 100
                # sfx: game_over.wav
            else:
                self.ball_attached = True
                self._attach_ball_to_paddle()

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                # sfx: break_block.wav
                self.current_reward += block["points"]
                self.score += block["points"]
                self._create_particles(block["rect"].center, block["color"])
                
                # Determine which side was hit for bounce logic
                # A simple but effective method
                dx = self.ball_pos.x - block["rect"].centerx
                dy = self.ball_pos.y - block["rect"].centery
                w, h = block["rect"].width / 2, block["rect"].height / 2
                
                if abs(dx / w) > abs(dy / h): # Horizontal collision
                    self.ball_vel.x *= -1
                else: # Vertical collision
                    self.ball_vel.y *= -1
                
                self.blocks.remove(block)
                break # Only break one block per frame

        # Level complete
        if not self.blocks:
            self.level += 1
            if self.level > self.MAX_LEVELS:
                self.game_won = True
                self.current_reward += 300
                # sfx: game_win.wav
            else:
                self.current_reward += 100
                self._setup_level()
                # sfx: level_up.wav

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9 # Damping
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "level": self.level,
        }

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
    
    def _render_game(self):
        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            
        # Render paddle
        paddle_rect = pygame.Rect(0, 0, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        paddle_rect.center = (int(self.paddle_pos.x), int(self.paddle_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=4)
        
        # Render ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            size = max(1, int(self.BALL_RADIUS / 3 * (p["lifespan"] / 30)))
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p["pos"].x - size), int(p["pos"].y - size)))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Level
        level_text = self.font_ui.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))
        
        # Lives
        heart_size = 12
        for i in range(self.lives):
            x_pos = self.WIDTH / 2 - (self.lives * (heart_size*2) / 2) + i * (heart_size*2)
            self._draw_heart(int(x_pos), 10 + heart_size, heart_size)
            
        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
        elif self.game_won:
            msg = "YOU WIN!"
        else:
            return

        text = self.font_game_over.render(msg, True, self.COLOR_LIVES if self.game_over else self.COLOR_BALL)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _draw_heart(self, x, y, size):
        # A simple heart shape using two circles and a triangle
        pygame.gfxdraw.filled_circle(self.screen, x - size // 2, y, size // 2, self.COLOR_LIVES)
        pygame.gfxdraw.filled_circle(self.screen, x + size // 2, y, size // 2, self.COLOR_LIVES)
        pygame.gfxdraw.filled_trigon(self.screen, x - size, y, x + size, y, x, y + size, self.COLOR_LIVES)

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

# Example of how to run the environment
if __name__ == '__main__':
    import time
    
    # Set this to run headlessly
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # For human play
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    running = True
    
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            time.sleep(3) # Pause to show final message
            obs, info = env.reset()
            total_reward = 0

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        env.clock.tick(60) # Limit to 60 FPS for human play

    env.close()