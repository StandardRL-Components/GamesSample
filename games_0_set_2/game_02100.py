
# Generated: 2025-08-27T19:15:36.455331
# Source Brief: brief_02100.md
# Brief Index: 2100

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to move the paddle. Press space to launch the ball."

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced block-breaking game. Break all the blocks to win, but lose a life if the ball falls. "
        "Risky hits near the paddle's edge grant bonus points and speed."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.BALL_RADIUS = 8
        self.PADDLE_SPEED = 8
        self.MAX_STEPS = 10000

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
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)
        
        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150)
        self.COLOR_WALL = (80, 80, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = {
            1: (0, 200, 100),  # Green
            2: (100, 150, 255), # Blue
            3: (255, 100, 100), # Red
        }

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        self.ball_launched = False
        self.blocks_broken_since_speedup = 0
        self.initial_ball_speed = 4.0
        
        self.reset()
        
        # Run self-validation
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.win = False
        self.ball_launched = False
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

        self.blocks = []
        num_cols, num_rows = 10, 5
        block_width = (self.WIDTH - 20) // num_cols
        block_height = 20
        for r in range(num_rows):
            for c in range(num_cols):
                points = 3 - r // 2
                block = pygame.Rect(
                    10 + c * block_width + 2,
                    50 + r * block_height + 2,
                    block_width - 4,
                    block_height - 4,
                )
                self.blocks.append({"rect": block, "points": points, "color": self.BLOCK_COLORS[points]})
        
        self.particles = []
        self.blocks_broken_since_speedup = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed

        if not self.game_over:
            self._handle_input(action)
            reward += self._update_game_state()

        self.steps += 1
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if self.game_over:
            if self.win:
                reward += 100
            else:
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))
        
        if space_held and not self.ball_launched:
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.initial_ball_speed
            # sfx: launch_ball

    def _update_game_state(self):
        step_reward = 0
        
        if not self.ball_launched:
            self.ball_pos.x = self.paddle.centerx
            return step_reward
            
        self.ball_pos += self.ball_vel

        # Ball collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # sfx: wall_bounce
        
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = max(self.BALL_RADIUS, self.ball_pos.y)
            # sfx: wall_bounce

        # Paddle collision
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
            
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x = offset * self.ball_vel.magnitude() * 0.8
            self.ball_vel.normalize_ip()
            self.ball_vel *= self._get_current_ball_speed()
            
            # Risky hit reward
            if abs(offset) > 0.8: # Outer 20% (10% on each side)
                step_reward += 0.1
                # sfx: risky_hit
                self._create_particles(self.ball_pos, self.COLOR_PADDLE, 15, 3)
            else:
                # sfx: paddle_hit
                self._create_particles(self.ball_pos, self.COLOR_PADDLE, 5, 2)

        # Block collisions
        for block_data in self.blocks[:]:
            if block_data["rect"].colliderect(ball_rect):
                step_reward += block_data["points"]
                self.score += block_data["points"]
                
                # Collision response
                self._handle_block_collision(block_data["rect"])
                
                self._create_particles(block_data["rect"].center, block_data["color"], 20)
                self.blocks.remove(block_data)
                # sfx: block_break
                
                self.blocks_broken_since_speedup += 1
                if self.blocks_broken_since_speedup >= 10:
                    self.blocks_broken_since_speedup = 0
                    current_speed = self.ball_vel.magnitude()
                    self.ball_vel.normalize_ip()
                    self.ball_vel *= (current_speed + 0.5)
                
                if not self.blocks:
                    self.win = True
                    self.game_over = True
                break

        # Ball out of bounds
        if self.ball_pos.y + self.BALL_RADIUS > self.HEIGHT:
            self.lives -= 1
            # sfx: lose_life
            if self.lives <= 0:
                self.game_over = True
            else:
                self.ball_launched = False
                self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
                self.ball_vel = pygame.Vector2(0, 0)
        
        # Anti-softlock
        if self.ball_launched and abs(self.ball_vel.y) < 0.2:
            self.ball_vel.y = 0.2 * (-1 if self.ball_vel.y < 0 else 1)
        
        self._update_particles()
        
        return step_reward

    def _handle_block_collision(self, block_rect):
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        
        # Determine collision side to correctly reflect the ball
        # This is a simplified but effective method
        dx = self.ball_pos.x - block_rect.centerx
        dy = self.ball_pos.y - block_rect.centery
        w = (ball_rect.width + block_rect.width) / 2
        h = (ball_rect.height + block_rect.height) / 2
        
        if abs(dx) / w > abs(dy) / h:
            # Horizontal collision
            self.ball_vel.x *= -1
        else:
            # Vertical collision
            self.ball_vel.y *= -1

    def _get_current_ball_speed(self):
        speed_increases = (50 - len(self.blocks) - self.blocks_broken_since_speedup) // 10
        return self.initial_ball_speed + speed_increases * 0.5

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw gradient background
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.rect(s, (*p['color'], alpha), s.get_rect())
                self.screen.blit(s, (int(p['pos'].x) - size, int(p['pos'].y) - size))

        # Render blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data["color"], block_data["rect"], border_radius=3)
            # Add a subtle inner highlight
            highlight_color = tuple(min(255, c + 40) for c in block_data["color"])
            pygame.draw.rect(self.screen, highlight_color, block_data["rect"].inflate(-6, -6), border_radius=2)
            
        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        
        # Render ball with glow
        glow_radius = int(self.BALL_RADIUS * 1.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_BALL_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (int(self.ball_pos.x) - glow_radius, int(self.ball_pos.y) - glow_radius))
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        heart_char = "♥"
        lives_text = " ".join([heart_char] * self.lives)
        lives_render = self.font_small.render(lives_text, True, (255, 80, 80))
        self.screen.blit(lives_render, (self.WIDTH - lives_render.get_width() - 10, 10))

        # Game state messages
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            msg_render = self.font_big.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(msg_render, msg_render.get_rect(center=self.screen.get_rect().center))
        elif not self.ball_launched:
            msg_render = self.font_small.render("PRESS SPACE TO LAUNCH", True, self.COLOR_TEXT)
            self.screen.blit(msg_render, msg_render.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 80)))

    def _create_particles(self, pos, color, count, speed=5):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(1, speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'max_life': 40,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks)
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be run by the evaluation system, which imports the class.
    
    # Re-enable video driver for direct play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Override the screen for display
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")

    terminated = False
    
    # Main game loop
    while not terminated:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        env.clock.tick(60) # Run at 60 FPS for smooth human gameplay
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()