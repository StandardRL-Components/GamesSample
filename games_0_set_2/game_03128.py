
# Generated: 2025-08-27T22:26:51.232736
# Source Brief: brief_03128.md
# Brief Index: 3128

        
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
        "A retro block-breaking game. Destroy all blocks with the ball to win, but don't let it fall past your paddle!"
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
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_WALL = (200, 200, 220)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 87, 34),   # Deep Orange
            (255, 193, 7),   # Amber
            (76, 175, 80),   # Green
            (33, 150, 243),  # Blue
            (156, 39, 176),  # Purple
        ]
        
        # Game parameters
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_MAX_SPEED_Y = 8
        self.MAX_STEPS = 2000
        self.INITIAL_LIVES = 3
        
        # State variables are initialized in reset()
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.block_colors = None
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = None
        self.recent_breaks = None
        self.stuck_ball_tracker = None

        # Initialize state
        self.reset()

        # Self-validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        
        # Paddle
        paddle_y = self.HEIGHT - 40
        self.paddle_rect = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Ball
        self.ball_pos = [self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS]
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards
        self.ball_vel = [
            self.BALL_MAX_SPEED_Y * math.cos(angle),
            self.BALL_MAX_SPEED_Y * math.sin(angle)
        ]

        # Blocks
        self.blocks = []
        self.block_colors = []
        n_rows = 5
        n_cols = 10
        block_width = 60
        block_height = 20
        gap = 4
        total_block_width = n_cols * (block_width + gap) - gap
        start_x = (self.WIDTH - total_block_width) // 2
        start_y = 50
        for i in range(n_rows):
            for j in range(n_cols):
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                rect = pygame.Rect(
                    start_x + j * (block_width + gap),
                    start_y + i * (block_height + gap),
                    block_width,
                    block_height
                )
                self.blocks.append(rect)
                self.block_colors.append(color)

        # Effects and trackers
        self.particles = []
        self.recent_breaks = []
        self.stuck_ball_tracker = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01  # Time penalty to encourage efficiency
        
        # 1. Update Paddle Position
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        self.paddle_rect.left = max(0, self.paddle_rect.left)
        self.paddle_rect.right = min(self.WIDTH, self.paddle_rect.right)

        # 2. Update Ball Position
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS, 
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )

        # 3. Handle Collisions
        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(1, ball_rect.left)
            ball_rect.right = min(self.WIDTH - 1, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
            self.stuck_ball_tracker.append(False) # Not a block hit
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(1, ball_rect.top)
            self.ball_pos[1] = ball_rect.centery
            self.stuck_ball_tracker.append(False) # Not a block hit

        # Paddle collision
        if self.paddle_rect.colliderect(ball_rect) and self.ball_vel[1] > 0:
            # sound: paddle_hit.wav
            offset = (ball_rect.centerx - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = 5 * offset
            self.ball_vel[1] *= -1
            
            # Normalize Y speed to prevent it from getting too slow/fast
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            self.ball_vel[1] = -math.sqrt(max(1, self.BALL_MAX_SPEED_Y**2 - self.ball_vel[0]**2))
            
            # Ensure ball is above paddle to prevent sticking
            self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS

            # Reward for paddle hit style
            if abs(offset) > 0.75: # Edge hit
                reward += 0.1
            else: # Center hit
                reward -= 0.2
            self.stuck_ball_tracker.append(False) # Not a block hit

        # Block collisions
        hit_block_idx = ball_rect.collidelist(self.blocks)
        if hit_block_idx != -1:
            # sound: block_break.wav
            block = self.blocks[hit_block_idx]
            
            # Determine collision side
            prev_ball_rect = pygame.Rect(
                (self.ball_pos[0] - self.ball_vel[0]) - self.BALL_RADIUS,
                (self.ball_pos[1] - self.ball_vel[1]) - self.BALL_RADIUS,
                self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
            )
            if prev_ball_rect.bottom <= block.top or prev_ball_rect.top >= block.bottom:
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1

            # Reward for breaking a block
            reward += 1.0
            self.score += 10

            # Chain reaction reward
            if self.recent_breaks and self.steps - self.recent_breaks[-1] <= 3:
                reward += 5.0
                self.score += 50 # Bonus points
            self.recent_breaks.append(self.steps)
            if len(self.recent_breaks) > 5:
                self.recent_breaks.pop(0)

            # Create particles
            for _ in range(15):
                self.particles.append(self._create_particle(block.center, self.block_colors[hit_block_idx]))

            # Remove block
            self.blocks.pop(hit_block_idx)
            self.block_colors.pop(hit_block_idx)
            self.stuck_ball_tracker = [] # Reset stuck tracker on block hit

        # 4. Anti-softlock mechanism
        if len(self.stuck_ball_tracker) > 20: # 20 consecutive non-block hits
            if not any(self.stuck_ball_tracker):
                # Ball is likely in a horizontal loop
                self.ball_vel[1] += self.np_random.uniform(-0.5, 0.5)
                self.stuck_ball_tracker = []
            else:
                self.stuck_ball_tracker.pop(0)

        # 5. Update Particles
        self.particles = [p for p in self.particles if self._update_particle(p)]

        # 6. Check for termination conditions
        terminated = False
        # Miss
        if ball_rect.top >= self.HEIGHT:
            # sound: miss.wav
            self.lives -= 1
            if self.lives <= 0:
                # sound: game_over.wav
                reward -= 100
                terminated = True
                self.game_over = True
            else:
                # Reset ball on paddle
                self.ball_pos = [self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS]
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = [self.BALL_MAX_SPEED_Y * 0.7 * math.cos(angle), self.BALL_MAX_SPEED_Y * 0.7 * math.sin(angle)]
                self.stuck_ball_tracker = []

        # Win
        if not self.blocks:
            # sound: win.wav
            reward += 100
            terminated = True
            self.game_over = True

        # Max steps
        self.steps += 1
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

    def _create_particle(self, pos, color):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        vel = [speed * math.cos(angle), speed * math.sin(angle)]
        lifetime = self.np_random.integers(20, 40)
        return {'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'max_life': lifetime, 'color': color}

    def _update_particle(self, p):
        p['pos'][0] += p['vel'][0]
        p['pos'][1] += p['vel'][1]
        p['vel'][0] *= 0.95 # Damping
        p['vel'][1] *= 0.95
        p['lifetime'] -= 1
        return p['lifetime'] > 0

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render walls
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, 0), (0, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (self.WIDTH - 1, 0), (self.WIDTH - 1, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, 0), (self.WIDTH, 0), 2)
        
        # Render blocks
        for i, block in enumerate(self.blocks):
            pygame.draw.rect(self.screen, self.block_colors[i], block)
            
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_life']))
            color = p['color']
            # Create a temporary surface for transparency
            particle_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, (*color, alpha), (2, 2), 2)
            self.screen.blit(particle_surf, (int(p['pos'][0]) - 2, int(p['pos'][1]) - 2))

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        
        # Render ball (with antialiasing)
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.small_font.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 15))
        for i in range(self.lives):
            x = self.WIDTH - 80 + (i * (self.BALL_RADIUS * 2 + 5))
            pygame.gfxdraw.filled_circle(self.screen, x, 23, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, x, 23, self.BALL_RADIUS, self.COLOR_BALL)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = not self.blocks
            msg = "YOU WIN!" if win_condition else "GAME OVER"
            msg_color = (100, 255, 100) if win_condition else (255, 100, 100)
            
            end_text = self.font.render(msg, True, msg_color)
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
    env = GameEnv(render_mode="rgb_array")
    
    # To play manually, you would need to map keyboard inputs to actions
    # This example just runs random actions
    
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Simple agent: move randomly
        action = env.action_space.sample()
        
        # Basic manual control override
        keys = pygame.key.get_pressed()
        move_action = 0 # no-op
        if keys[pygame.K_LEFT]:
            move_action = 3
        elif keys[pygame.K_RIGHT]:
            move_action = 4
        action = [move_action, 0, 0]

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(60)

    env.close()