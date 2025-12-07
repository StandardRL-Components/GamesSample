import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to move the paddle."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced block breaker where risky paddle hits are rewarded. Clear all blocks to win, but lose all your balls and it's game over."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 10
    BALL_RADIUS = 7
    BALL_SPEED = 7
    WALL_THICKNESS = 10
    MAX_STEPS = 2500
    INITIAL_LIVES = 3

    # Colors
    COLOR_BG = (10, 10, 30)
    COLOR_GRID = (20, 20, 45)
    COLOR_WALL = (140, 140, 160)
    COLOR_PADDLE = (240, 240, 240)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    BLOCK_COLORS = [
        (255, 87, 34), (255, 193, 7), (139, 195, 74),
        (0, 188, 212), (33, 150, 243), (156, 39, 176)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Ensure Pygame runs headlessly
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables are initialized in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.total_blocks = 0
        self.ball_trail = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        
        # Initialize state variables
        # self.reset() is called by the validation method
        
        # Run validation
        # self.validate_implementation() # Commented out for submission, but useful for dev
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False
        self.ball_trail.clear()
        self.particles.clear()

        # Paddle
        paddle_y = self.SCREEN_HEIGHT - 40
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball
        self._reset_ball()

        # Blocks
        self.blocks.clear()
        block_rows = 5
        block_cols = 10
        block_width = 58
        block_height = 20
        gap = 4
        total_block_width = block_cols * (block_width + gap) - gap
        start_x = (self.SCREEN_WIDTH - total_block_width) / 2
        start_y = 50
        for i in range(block_rows):
            for j in range(block_cols):
                x = start_x + j * (block_width + gap)
                y = start_y + i * (block_height + gap)
                self.blocks.append(pygame.Rect(x, y, block_width, block_height))
        self.total_blocks = len(self.blocks)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _reset_ball(self):
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards angles
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED
        self.ball_trail.clear()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02 # Continuous penalty for time
        
        # --- Action Handling ---
        movement = action[0]
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, self.WALL_THICKNESS, self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS)

        # --- Ball Movement ---
        self.ball_pos += self.ball_vel
        # FIX: pygame.Vector2 does not have a .copy() method.
        # Create a new vector to copy it.
        self.ball_trail.append(pygame.Vector2(self.ball_pos))
        if len(self.ball_trail) > 10:
            self.ball_trail.pop(0)
        
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # --- Collision Detection ---
        # Walls
        if ball_rect.left <= self.WALL_THICKNESS:
            ball_rect.left = self.WALL_THICKNESS + 1
            self.ball_vel.x *= -1
            self.ball_pos.x = ball_rect.centerx
        if ball_rect.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
            ball_rect.right = self.SCREEN_WIDTH - self.WALL_THICKNESS - 1
            self.ball_vel.x *= -1
            self.ball_pos.x = ball_rect.centerx
        if ball_rect.top <= self.WALL_THICKNESS:
            ball_rect.top = self.WALL_THICKNESS + 1
            self.ball_vel.y *= -1
            self.ball_pos.y = ball_rect.centery
        
        # Bottom (lose life)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
                reward -= 100 # Terminal penalty for losing all lives
            else:
                self._reset_ball()
            # Early exit for this step to avoid other collisions after reset
            return self._get_observation(), reward, self.game_over, False, self._get_info()


        # Paddle
        if self.ball_vel.y > 0 and ball_rect.colliderect(self.paddle):
            hit_pos_norm = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            hit_pos_norm = np.clip(hit_pos_norm, -1, 1)

            # Reward for paddle hit quality
            if abs(hit_pos_norm) < 0.2: # Center 20%
                reward += 0.1
            elif abs(hit_pos_norm) > 0.9: # Outer 10% on each side
                reward -= 0.1

            angle = (1 - (hit_pos_norm + 1) / 2) * math.pi * 0.8 + 0.1 * math.pi
            self.ball_vel.x = math.cos(angle) * self.BALL_SPEED
            self.ball_vel.y = -math.sin(angle) * self.BALL_SPEED
            
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS - 1
            ball_rect.bottom = self.paddle.top - 1

        # Blocks
        collided_idx = ball_rect.collidelist(self.blocks)
        if collided_idx != -1:
            block = self.blocks.pop(collided_idx)
            self._create_particles(block.center)
            reward += 1
            self.score += 1

            prev_ball_pos = self.ball_pos - self.ball_vel
            if prev_ball_pos.y + self.BALL_RADIUS <= block.top or prev_ball_pos.y - self.BALL_RADIUS >= block.bottom:
                 self.ball_vel.y *= -1
            else:
                 self.ball_vel.x *= -1

        self.ball_pos.x, self.ball_pos.y = ball_rect.centerx, ball_rect.centery
        
        # --- Particle Update ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'].y += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- Termination Check ---
        self.steps += 1
        terminated = self.game_over
        if not self.blocks:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100 # Win bonus
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _create_particles(self, pos):
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': self.BLOCK_COLORS[self.np_random.integers(len(self.BLOCK_COLORS))]
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # Blocks
        for i, block in enumerate(self.blocks):
            color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
            pygame.draw.rect(self.screen, color, block)
            pygame.draw.rect(self.screen, tuple(int(c*0.7) for c in color), block, 2)

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 30))
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            s.fill((*p['color'], alpha))
            self.screen.blit(s, (int(p['pos'].x), int(p['pos'].y)))

        # Ball trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(255 * (i + 1) / (len(self.ball_trail) + 1)) * 0.5
            radius = int(self.BALL_RADIUS * (i + 1) / (len(self.ball_trail) + 1))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, (*self.COLOR_BALL, int(alpha)))

        # Ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS + 2, (*self.COLOR_BALL, 60))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        glow_rect = self.paddle.inflate(6, 6)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PADDLE, 40), glow_surface.get_rect(), border_radius=6)
        self.screen.blit(glow_surface, glow_rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.WALL_THICKNESS + 2))
        
        # Lives
        for i in range(self.lives):
            x = self.SCREEN_WIDTH - self.WALL_THICKNESS - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            y = self.WALL_THICKNESS + 10 + self.BALL_RADIUS
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            text = self.font_game_over.render(message, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        self.reset()
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage for human play
if __name__ == '__main__':
    # To play, you must remove the os.environ line from __init__
    # or set it to a valid display driver.
    # For example:
    # del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    # env.validate_implementation()
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(60)
        
    pygame.quit()