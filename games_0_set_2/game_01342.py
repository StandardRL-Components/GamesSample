
# Generated: 2025-08-27T16:49:36.023413
# Source Brief: brief_01342.md
# Brief Index: 1342

        
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
        "Controls: ←→ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based block breaker. Clear all blocks to win, but lose a life if the ball falls past your paddle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 12
        self.PADDLE_SPEED = 8.0
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 4.0
        self.POWERUP_SIZE = 15
        self.POWERUP_SPEED = 1.5

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = [
            (255, 60, 60), (60, 255, 60), (60, 60, 255),
            (255, 255, 60), (255, 60, 255), (60, 255, 255)
        ]
        self.POWERUP_COLORS = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255)
        ]

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
        self.font_ui = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_end = pygame.font.SysFont("consolas", 50, bold=True)
        
        # Initialize state variables (to be properly set in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.powerups = []
        self.base_ball_speed = self.INITIAL_BALL_SPEED
        self.blocks_destroyed_count = 0
        self.wide_paddle_timer = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.blocks_destroyed_count = 0
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        
        # Start ball in a random upward direction
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = [math.cos(angle), math.sin(angle)]

        self.base_ball_speed = self.INITIAL_BALL_SPEED
        
        self.blocks = self._create_blocks()
        self.total_blocks = len(self.blocks)
        self.particles = []
        self.powerups = []
        self.wide_paddle_timer = 0
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        blocks = []
        block_width = 58
        block_height = 20
        rows = 5
        cols = 10
        
        # Center the block grid
        grid_width = cols * (block_width + 2)
        start_x = (self.WIDTH - grid_width) / 2
        
        for i in range(rows):
            for j in range(cols):
                powerup_type = None
                # 15% chance for a powerup
                if self.np_random.random() < 0.15:
                    powerup_type = self.np_random.choice(['WIDE_PADDLE', 'EXTRA_LIFE'])
                
                color = self.np_random.choice(self.BLOCK_COLORS)
                
                block_rect = pygame.Rect(
                    start_x + j * (block_width + 2),
                    50 + i * (block_height + 2),
                    block_width,
                    block_height
                )
                blocks.append({'rect': block_rect, 'color': color, 'powerup': powerup_type})
        return blocks

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # Handle input
        reward += self._handle_input(action)
        
        # Update game logic
        self._update_powerups()
        reward += self._update_ball()
        self._update_particles()
        
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.lives <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            # sfx: game_over_lose
        elif not self.blocks:
            reward += 100
            terminated = True
            self.game_over = True
            # sfx: game_over_win
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

    def _handle_input(self, action):
        movement = action[0]
        reward = 0
        
        moved = False
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            moved = True
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            moved = True
            
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.paddle.width))

        # Small penalty for moving when ball is moving away (up)
        if moved and self.ball_vel[1] < 0:
            reward -= 0.02
            
        return reward

    def _update_ball(self):
        reward = 0
        
        # Move ball
        self.ball_pos[0] += self.ball_vel[0] * self.base_ball_speed
        self.ball_pos[1] += self.ball_vel[1] * self.base_ball_speed
        
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS, min(self.ball_pos[0], self.WIDTH - self.BALL_RADIUS))
            # sfx: wall_bounce
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce
            
        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_vel[0] = np.clip(offset, -1.5, 1.5)
            # Normalize velocity
            norm = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel[0] /= norm
            self.ball_vel[1] /= norm
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS - 1
            # sfx: paddle_bounce

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                reward += 0.1
                self.score += 10
                
                self._create_particles(block['rect'].center, block['color'])
                
                if block['powerup']:
                    self._spawn_powerup(block['rect'].center, block['powerup'])
                
                # AABB collision response
                overlap = ball_rect.clip(block['rect'])
                if overlap.width < overlap.height:
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[1] *= -1

                self.blocks.remove(block)
                self.blocks_destroyed_count += 1
                
                # Increase difficulty
                if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 10 == 0:
                    self.base_ball_speed += 0.2
                
                # sfx: block_hit
                break # Handle one collision per frame

        # Lost life
        if self.ball_pos[1] > self.HEIGHT + self.BALL_RADIUS:
            self.lives -= 1
            reward -= 1
            if self.lives > 0:
                self.paddle.x = (self.WIDTH - self.paddle.width) / 2
                self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = [math.cos(angle), math.sin(angle)]
                # sfx: life_lost
        
        return reward

    def _update_powerups(self):
        reward = 0
        # Wide paddle timer
        if self.wide_paddle_timer > 0:
            self.wide_paddle_timer -= 1
            if self.wide_paddle_timer == 0:
                # Reset paddle size
                center_x = self.paddle.centerx
                self.paddle.width = self.PADDLE_WIDTH
                self.paddle.centerx = center_x
                self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.paddle.width))

        # Move and check collection
        for pu in self.powerups[:]:
            pu['rect'].y += self.POWERUP_SPEED
            if pu['rect'].top > self.HEIGHT:
                self.powerups.remove(pu)
            elif pu['rect'].colliderect(self.paddle):
                reward += 1
                self.score += 50
                if pu['type'] == 'EXTRA_LIFE':
                    self.lives = min(5, self.lives + 1)
                    # sfx: powerup_life
                elif pu['type'] == 'WIDE_PADDLE':
                    self.wide_paddle_timer = 600 # 20 seconds at 30fps
                    center_x = self.paddle.centerx
                    self.paddle.width = self.PADDLE_WIDTH * 1.5
                    self.paddle.centerx = center_x
                    self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.paddle.width))
                    # sfx: powerup_wide
                self.powerups.remove(pu)
        return reward

    def _spawn_powerup(self, pos, type):
        rect = pygame.Rect(pos[0] - self.POWERUP_SIZE/2, pos[1] - self.POWERUP_SIZE/2, self.POWERUP_SIZE, self.POWERUP_SIZE)
        self.powerups.append({'rect': rect, 'type': type, 'spawn_time': self.steps})

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Render blocks
        for block in self.blocks:
            r = block['rect']
            c = block['color']
            # Bevel effect
            pygame.draw.rect(self.screen, c, r, border_radius=3)
            brighter_c = tuple(min(255, val + 40) for val in c)
            pygame.draw.rect(self.screen, brighter_c, (r.x, r.y, r.width, 3), border_top_left_radius=3, border_top_right_radius=3)

        # Render powerups
        for pu in self.powerups:
            # Pulsating rainbow effect
            pulse = abs(math.sin((self.steps - pu['spawn_time']) * 0.2))
            size = int(self.POWERUP_SIZE * (1 + pulse * 0.2))
            color_index = (self.steps // 5) % len(self.POWERUP_COLORS)
            color = self.POWERUP_COLORS[color_index]
            
            # Draw a 'P' for WIDE_PADDLE, 'L' for EXTRA_LIFE
            text = 'W' if pu['type'] == 'WIDE_PADDLE' else 'L'
            text_surf = self.font_ui.render(text, True, (255,255,255))
            
            icon_rect = pygame.Rect(0,0,size,size)
            icon_rect.center = pu['rect'].center
            
            pygame.gfxdraw.box(self.screen, icon_rect, (*color, 150))
            self.screen.blit(text_surf, text_surf.get_rect(center=icon_rect.center))
            
        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Render ball with glow
        ball_center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = (*p['color'], alpha)
            size = max(1, int(4 * (p['lifespan'] / 30)))
            temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            temp_surf.fill(color)
            self.screen.blit(temp_surf, (int(p['pos'][0]), int(p['pos'][1])))

        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if not self.blocks:
                end_text = self.font_end.render("YOU WIN!", True, self.COLOR_BALL)
            else:
                end_text = self.font_end.render("GAME OVER", True, self.BLOCK_COLORS[0])
            
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
        }
    
    def close(self):
        pygame.font.quit()
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Grid Breaker")
    
    done = False
    total_reward = 0
    
    # Use a dictionary to track held keys for smoother control
    keys_held = {
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False
    }

    while not done:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # Map keyboard state to action
        movement = 0 # no-op
        if keys_held[pygame.K_LEFT]:
            movement = 3
        elif keys_held[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # space and shift are not used

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation from the environment
        # The observation is (H, W, C), but pygame wants (W, H) for display
        # and surfarray.make_surface expects (W, H, C)
        obs_transposed = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(obs_transposed)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(60)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Keep window open for a few seconds to see the end screen
    pygame.time.wait(3000)
    
    env.close()