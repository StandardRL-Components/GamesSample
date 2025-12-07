
# Generated: 2025-08-27T23:30:35.038991
# Source Brief: brief_03492.md
# Brief Index: 3492

        
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
        "A retro block-breaking game. Clear all the blocks to win, but you only have 3 lives!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
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
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (35, 35, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            10: (50, 200, 50),   # Green
            20: (50, 100, 255),  # Blue
            30: (220, 50, 50),   # Red
        }

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Game parameters
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BALL_SPEED_INITIAL = 5
        self.MAX_BALL_ANGLE_FACTOR = 0.8
        self.MAX_STEPS = 2000

        # Initialize state variables to be defined in reset()
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_held = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        
        # Initialize state
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.particles = []

        # Paddle
        paddle_y = self.HEIGHT - 40
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Blocks
        self._create_blocks()

        # Ball
        self._reset_ball()
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks = []
        block_width = 58
        block_height = 20
        gap = 5
        rows = 5
        cols = 10
        start_x = (self.WIDTH - (cols * (block_width + gap) - gap)) / 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                value = (1 + (r // 2)) * 10 # Rows 0,1=10; 2,3=20; 4=30
                color = self.BLOCK_COLORS[value]
                x = start_x + c * (block_width + gap)
                y = start_y + r * (block_height + gap)
                self.blocks.append({
                    "rect": pygame.Rect(x, y, block_width, block_height),
                    "color": color,
                    "value": value
                })
    
    def _reset_ball(self):
        self.ball_held = True
        ball_x = self.paddle.centerx
        ball_y = self.paddle.top - self.BALL_RADIUS
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball.center = (ball_x, ball_y)
        self.ball_vel = [0, 0]

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1
        
        reward = -0.01  # Small penalty for each step to encourage speed
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Input
        if movement == 3:  # Move left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Move right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.clamp_ip(self.screen.get_rect())

        if self.ball_held and space_pressed:
            # Launch ball
            self.ball_held = False
            launch_angle = (self.np_random.random() - 0.5) * 0.5 # Small random angle
            self.ball_vel = [
                self.BALL_SPEED_INITIAL * math.sin(launch_angle),
                -self.BALL_SPEED_INITIAL * math.cos(launch_angle)
            ]
            # sfx: launch_sound

        # 2. Update Game Logic
        if self.ball_held:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
        else:
            reward += self._update_ball()

        self._update_particles()
        self.steps += 1

        # 3. Check Termination
        terminated = False
        if self.lives <= 0:
            terminated = True
            self.game_over = True
        
        if not self.blocks:
            terminated = True
            self.game_over = True
            reward += 100  # Victory bonus

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        reward = 0
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.clamp_ip(self.screen.get_rect())
            # sfx: wall_bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.clamp_ip(self.screen.get_rect())
            # sfx: wall_bounce

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on where it hit the paddle
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * self.BALL_SPEED_INITIAL * self.MAX_BALL_ANGLE_FACTOR
            
            # Clamp ball speed to prevent it from getting too fast/slow
            speed = math.hypot(*self.ball_vel)
            if speed > self.BALL_SPEED_INITIAL * 1.5:
                self.ball_vel = [v / speed * self.BALL_SPEED_INITIAL * 1.5 for v in self.ball_vel]
            
            self.ball.bottom = self.paddle.top
            # sfx: paddle_bounce

        # Block collisions
        hit_block_idx = self.ball.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            block = self.blocks.pop(hit_block_idx)
            self.score += block['value']
            reward += block['value']  # Use block value as reward
            # sfx: block_break
            
            # Create particle explosion
            self._create_particles(block['rect'].center, block['color'])

            # Determine bounce direction
            # A simple approach: reverse vertical velocity
            self.ball_vel[1] *= -1

        # Missed ball
        if self.ball.top >= self.HEIGHT:
            self.lives -= 1
            reward -= 50
            if self.lives > 0:
                self._reset_ball()
            # sfx: life_lost

        # Anti-stuck mechanism
        if abs(self.ball_vel[1]) < 0.2:
            self.ball_vel[1] = 0.3 * np.sign(self.ball_vel[1] or 1)
            
        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            size = self.np_random.integers(2, 5)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'color': color, 'size': size})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background Grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block['color']), block['rect'], 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifetime'] / 30))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, p['size'], p['size']))
            self.screen.blit(temp_surf, (int(p['pos'][0]), int(p['pos'][1])))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Lives
        life_text = self.font_small.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(life_text, (self.WIDTH - 150, 15))
        for i in range(self.lives):
            life_rect = pygame.Rect(self.WIDTH - 80 + i * 25, 15, 20, 10)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_rect, border_radius=2)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if not self.blocks else "GAME OVER"
            end_text = self.font_large.render(message, True, self.COLOR_PADDLE)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
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

# Example usage:
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # For human play
    import pygame
    
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Main loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            # Get keyboard input for human play
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_pressed = 1 if keys[pygame.K_SPACE] else 0
            
            action = [movement, space_pressed, 0] # Shift is unused
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation from the environment
            # Pygame uses (width, height), numpy uses (height, width)
            # Transpose the observation back to pygame's format
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        else:
            # If game is over, wait for a key press to reset
            keys = pygame.key.get_pressed()
            if any(keys):
                obs, info = env.reset()
                terminated = False
                total_reward = 0

        # Control the frame rate
        env.clock.tick(30) # 30 FPS
        
    env.close()