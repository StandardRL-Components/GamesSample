
# Generated: 2025-08-28T02:56:38.573987
# Source Brief: brief_04616.md
# Brief Index: 4616

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to move the paddle. Press Space to launch the ball."
    )

    # Short, user-facing description of the game
    game_description = (
        "A minimalist, neon-themed block breaker. Use the paddle to bounce the ball and destroy all the blocks."
    )

    # Frames auto-advance at 30fps
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_PADDLE = (255, 0, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_BLOCK = (0, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PARTICLE = (0, 180, 180)

    # Game dimensions
    WIDTH, HEIGHT = 640, 400
    
    # Paddle properties
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12

    # Ball properties
    BALL_RADIUS = 8
    BALL_SPEED = 8
    MIN_BALL_SPEED_Y = 4

    # Block properties
    BLOCK_ROWS = 5
    BLOCK_COLS = 12
    BLOCK_WIDTH = 50
    BLOCK_HEIGHT = 15
    BLOCK_SPACING = 4
    BLOCK_AREA_TOP = 50

    # Game mechanics
    MAX_LIVES = 3
    MAX_STEPS = 30 * 60 # 60 seconds at 30fps
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        
        self.reset()
        
        # Validate implementation after setup
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.ball_launched = False
        
        # Paddle
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT * 2
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        # Ball
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )
        self.ball_vel = [0, 0]
        
        # Blocks
        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - total_block_width) / 2
        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_AREA_TOP + r * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                self.blocks.append(pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT))

        # Particles
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = 0.01  # Small reward for surviving
        
        # 1. Handle player input
        predicted_ball_x = self.ball.centerx
        if self.ball_launched and self.ball_vel[1] > 0:
             # Simple projection to paddle y-level
            time_to_paddle = (self.paddle.top - self.ball.centery) / self.ball_vel[1]
            predicted_ball_x = self.ball.centerx + self.ball_vel[0] * time_to_paddle

        if movement == 3:  # Move Left
            if self.paddle.centerx > predicted_ball_x:
                reward += 0.02 # Reward for moving towards ball
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Move Right
            if self.paddle.centerx < predicted_ball_x:
                reward += 0.02 # Reward for moving towards ball
            self.paddle.x += self.PADDLE_SPEED
            
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)
        
        # Launch ball
        if space_pressed and not self.ball_launched:
            # Sound: launch.wav
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            self.ball_vel = [self.BALL_SPEED * math.sin(angle), -self.BALL_SPEED * math.cos(angle)]

        # 2. Update game state
        if self.ball_launched:
            self._move_ball()
        else:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
        
        # 3. Handle collisions and events
        reward += self._handle_collisions()
        
        # 4. Update particles
        self._update_particles()
        
        # 5. Check for termination
        self.steps += 1
        terminated = False
        if self.lives <= 0:
            terminated = True
            self.game_over = True
            reward -= 100 # Loss penalty
        elif not self.blocks:
            terminated = True
            self.game_over = True
            reward += 100 # Win bonus
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_ball(self):
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]
        
        # Anti-softlock: ensure vertical speed is maintained
        if abs(self.ball_vel[1]) < self.MIN_BALL_SPEED_Y:
            self.ball_vel[1] = np.sign(self.ball_vel[1]) * self.MIN_BALL_SPEED_Y if self.ball_vel[1] != 0 else -self.MIN_BALL_SPEED_Y

    def _handle_collisions(self):
        reward = 0
        
        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.x = np.clip(self.ball.x, 0, self.WIDTH - self.ball.width)
            # Sound: wall_bounce.wav
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.y = np.clip(self.ball.y, 0, self.HEIGHT - self.ball.height)
            # Sound: wall_bounce.wav

        # Bottom wall (lose life)
        if self.ball.top >= self.HEIGHT:
            self.lives -= 1
            reward -= 5
            self.ball_launched = False
            self.ball_vel = [0, 0]
            # Sound: lose_life.wav
            if self.lives > 0:
                self.ball.centerx = self.paddle.centerx
                self.ball.bottom = self.paddle.top

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Sound: paddle_hit.wav
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on hit location
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 4
            # Cap horizontal speed
            self.ball_vel[0] = np.clip(self.ball_vel[0], -self.BALL_SPEED, self.BALL_SPEED)

        # Block collisions
        collided_block_index = self.ball.collidelist(self.blocks)
        if collided_block_index != -1:
            # Sound: block_break.wav
            block = self.blocks.pop(collided_block_index)
            self.score += 10
            reward += 1
            
            # Create particles
            self._spawn_particles(block.center)
            
            # Simple but effective collision response
            prev_ball_rect = self.ball.copy()
            prev_ball_rect.x -= self.ball_vel[0]
            prev_ball_rect.y -= self.ball_vel[1]
            
            if prev_ball_rect.centery < block.top or prev_ball_rect.centery > block.bottom:
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1
        
        return reward

    def _spawn_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(10, 20)
            self.particles.append([list(pos), vel, lifetime])

    def _update_particles(self):
        for p in self.particles[:]:
            p[0][0] += p[1][0]  # pos.x
            p[0][1] += p[1][1]  # pos.y
            p[1][1] += 0.1     # gravity
            p[2] -= 1          # lifetime
            if p[2] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw blocks with a slight 3D effect
        for block in self.blocks:
            brighter_color = tuple(min(255, c + 30) for c in self.COLOR_BLOCK)
            darker_color = tuple(max(0, c - 30) for c in self.COLOR_BLOCK)
            pygame.draw.rect(self.screen, darker_color, block)
            inner_rect = block.inflate(-4, -4)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK, inner_rect)

        # Draw paddle with glow
        self._draw_glow_rect(self.paddle, self.COLOR_PADDLE, 15)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with glow
        self._draw_glow_circle(self.ball.center, self.BALL_RADIUS, self.COLOR_BALL, 20)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            size = max(0, p[2] / 5)
            alpha = max(0, p[2] * 15)
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_PARTICLE, alpha), (size, size), size)
            self.screen.blit(s, (p[0][0] - size, p[0][1] - size))

    def _draw_glow_rect(self, rect, color, blur_radius):
        glow_surf = pygame.Surface((rect.width + blur_radius * 2, rect.height + blur_radius * 2), pygame.SRCALPHA)
        for i in range(blur_radius, 0, -2):
            alpha = 100 * (1 - i / blur_radius)
            pygame.draw.rect(glow_surf, (*color, alpha), glow_surf.get_rect(), border_radius=i+3)
        self.screen.blit(glow_surf, (rect.x - blur_radius, rect.y - blur_radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_glow_circle(self, center, radius, color, blur_radius):
        glow_surf = pygame.Surface((radius * 2 + blur_radius * 2, radius * 2 + blur_radius * 2), pygame.SRCALPHA)
        center_glow = (glow_surf.get_width() // 2, glow_surf.get_height() // 2)
        for i in range(blur_radius, 0, -2):
            alpha = 80 * (1 - i / blur_radius)**2
            pygame.gfxdraw.aacircle(glow_surf, center_glow[0], center_glow[1], radius + i, (*color, alpha))
        self.screen.blit(glow_surf, (center[0] - center_glow[0], center[1] - center_glow[1]), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            if not self.blocks:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_BALL)
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to use a visible display for testing
    import os
    os.environ.pop('SDL_VIDEODRIVER', None)

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Block Breaker")
    # terminated = False
    # while not terminated:
    #     movement, space, shift = 0, 0, 0
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]: movement = 3
    #     if keys[pygame.K_RIGHT]: movement = 4
    #     if keys[pygame.K_SPACE]: space = 1
        
    #     action = [movement, space, shift]
    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     # Display the observation
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             terminated = True
                
    #     env.clock.tick(30)
    
    # env.close()

    # --- Agent Random Play ---
    obs, info = env.reset()
    total_reward = 0
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"Episode finished. Final Info: {info}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
    env.close()