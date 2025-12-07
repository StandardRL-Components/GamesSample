
# Generated: 2025-08-28T06:03:33.646303
# Source Brief: brief_05778.md
# Brief Index: 5778

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle for explosion effects."""
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        self.np_random = np_random
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        
        self.radius = self.np_random.uniform(3, 7)
        self.lifespan = self.np_random.integers(20, 40)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius -= 0.1
        self.radius = max(0, self.radius)

    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 30))))
            r, g, b = self.color
            
            # Use gfxdraw for anti-aliased circle
            pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), int(self.radius), (r, g, b, alpha))
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), (r, g, b, alpha))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker. Break all the blocks to win, but don't let the ball fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500

    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 40)
    COLOR_PADDLE = (50, 150, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_WALL = (100, 100, 110)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 87, 34),   # Deep Orange
        (3, 169, 244),   # Light Blue
        (139, 195, 74),  # Light Green
        (255, 235, 59),  # Yellow
        (156, 39, 176),  # Purple
        (233, 30, 99),   # Pink
    ]
    
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 16
    PADDLE_SPEED = 12
    
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 5.0
    
    BLOCK_ROWS = 5
    BLOCK_COLS = 20
    BLOCK_WIDTH = 30
    BLOCK_HEIGHT = 15
    BLOCK_GAP = 2
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Initialize state variables (will be properly set in reset)
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.ball_launched = False
        self.blocks_broken_count = 0
        self.ball_speed = 0.0

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.ball_launched = False
        self.blocks_broken_count = 0
        self.ball_speed = self.INITIAL_BALL_SPEED
        
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) // 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        
        self.particles = []
        
        self._create_blocks()
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks = []
        grid_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_GAP) - self.BLOCK_GAP
        start_x = (self.SCREEN_WIDTH - grid_width) // 2
        start_y = 50
        
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_GAP)
                y = start_y + i * (self.BLOCK_HEIGHT + self.BLOCK_GAP)
                color = self.np_random.choice(len(self.BLOCK_COLORS))
                block = {
                    "rect": pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
                    "color": self.BLOCK_COLORS[color]
                }
                self.blocks.append(block)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        reward = -0.01  # Small penalty for each step to encourage speed
        
        self._handle_input(movement, space_held)
        
        event_reward = self._update_game_state()
        reward += event_reward
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            if not self.blocks: # Win
                reward += 100
            elif self.lives <= 0: # Loss
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Move paddle
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # Launch ball
        if not self.ball_launched and space_held:
            self.ball_launched = True
            # // Play launch sound
            initial_angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = pygame.Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.ball_speed

    def _update_game_state(self):
        reward = 0
        if not self.ball_launched:
            self.ball_pos.x = self.paddle.centerx
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball_pos += self.ball_vel
            
            # Wall collisions
            if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.SCREEN_WIDTH:
                self.ball_vel.x *= -1
                self.ball_pos.x = max(self.BALL_RADIUS, min(self.SCREEN_WIDTH - self.BALL_RADIUS, self.ball_pos.x))
                # // Play wall bounce sound
            if self.ball_pos.y - self.BALL_RADIUS <= 0:
                self.ball_vel.y *= -1
                self.ball_pos.y = max(self.BALL_RADIUS, self.ball_pos.y)
                # // Play wall bounce sound
            
            # Bottom wall (lose life)
            if self.ball_pos.y + self.BALL_RADIUS >= self.SCREEN_HEIGHT:
                self.lives -= 1
                self.ball_launched = False
                # // Play life lost sound
                if self.lives > 0:
                    self.reset_ball()

            # Paddle collision
            ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
            if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
                self.ball_vel.y *= -1
                self.ball_pos.y = self.paddle.top - self.BALL_RADIUS # Prevent sticking

                dist_from_center = self.ball_pos.x - self.paddle.centerx
                normalized_dist = dist_from_center / (self.PADDLE_WIDTH / 2)
                
                # Reward for risky vs safe play
                if abs(normalized_dist) > 0.7: # Hit near edge
                    reward += 0.1
                else: # Hit near center
                    reward -= 0.2
                
                self.ball_vel.x += normalized_dist * 2.0
                # // Play paddle bounce sound

            # Block collisions
            for block in self.blocks[:]:
                if block["rect"].colliderect(ball_rect):
                    reward += 1.0
                    self.score += 10
                    self.blocks.remove(block)
                    self._create_explosion(block["rect"].center, block["color"])
                    self.blocks_broken_count += 1
                    
                    # Increase ball speed every 20 blocks
                    if self.blocks_broken_count > 0 and self.blocks_broken_count % 20 == 0:
                        self.ball_speed += 0.5

                    # Determine bounce direction
                    # A simple but effective method: check which side is closer
                    dx = abs(self.ball_pos.x - block["rect"].centerx)
                    dy = abs(self.ball_pos.y - block["rect"].centery)
                    
                    if dx / block["rect"].width > dy / block["rect"].height:
                        self.ball_vel.x *= -1
                    else:
                        self.ball_vel.y *= -1
                    # // Play block break sound
                    break # Only break one block per frame
            
            # Normalize velocity to maintain constant speed
            if self.ball_vel.length() > 0:
                self.ball_vel.normalize_ip()
                self.ball_vel *= self.ball_speed

        self._update_particles()
        return reward
    
    def reset_ball(self):
        self.ball_launched = False
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

    def _create_explosion(self, position, color):
        for _ in range(self.np_random.integers(15, 25)):
            self.particles.append(Particle(position[0], position[1], color, self.np_random))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
            
    def _check_termination(self):
        return self.lives <= 0 or not self.blocks or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=8)

        # Draw ball
        pos = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives (hearts)
        heart_color = (255, 50, 50)
        for i in range(self.lives):
            x = self.SCREEN_WIDTH - 30 - i * 35
            self._draw_heart(self.screen, x, 25, heart_color)

    def _draw_heart(self, surface, x, y, color):
        points = [
            (x, y - 5), (x + 5, y - 10), (x + 10, y - 5),
            (x, y + 5),
            (x - 10, y - 5), (x - 5, y - 10)
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks),
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
    # This block allows you to play the game manually
    # You might need to install pygame: pip install pygame
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a persistent key state dictionary
    key_state = {
        pygame.K_LEFT: 0,
        pygame.K_RIGHT: 0,
        pygame.K_SPACE: 0,
    }

    # Pygame setup for human play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    total_reward = 0
    total_steps = 0

    while not done:
        # --- Action mapping for human play ---
        movement = 0 # no-op
        space = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in key_state:
                    key_state[event.key] = 1
            if event.type == pygame.KEYUP:
                if event.key in key_state:
                    key_state[event.key] = 0
        
        if key_state[pygame.K_LEFT]:
            movement = 3
        elif key_state[pygame.K_RIGHT]:
            movement = 4
            
        if key_state[pygame.K_SPACE]:
            space = 1
            
        action = [movement, space, 0] # shift is unused

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        total_steps += 1
        
        # --- Render the game screen ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {total_steps}")

    env.close()