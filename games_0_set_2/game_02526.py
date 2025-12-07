
# Generated: 2025-08-27T20:37:27.097295
# Source Brief: brief_02526.md
# Brief Index: 2526

        
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
        "Controls: Use ← and → to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Use the paddle to bounce the ball and destroy all the blocks to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Game world
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    FPS = 30
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_PARTICLE = (200, 200, 200)
    COLOR_TEXT = (220, 220, 220)
    COLOR_BLOCK_OUTLINE = (50, 50, 70)
    BLOCK_COLORS = {
        10: (200, 70, 70),   # Red
        20: (70, 200, 70),   # Green
        30: (70, 70, 200)    # Blue
    }

    # Paddle
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12

    # Ball
    BALL_RADIUS = 8
    BALL_SPEED = 8

    # Blocks
    BLOCK_ROWS = 5
    BLOCK_COLS = 10
    BLOCK_WIDTH = 58
    BLOCK_HEIGHT = 20
    BLOCK_SPACING = 6
    BLOCK_AREA_TOP = 50
    
    # Rewards
    REWARD_PER_STEP = -0.01
    REWARD_PADDLE_HIT = 0.1
    REWARD_LIFE_LOST = -5.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.active_blocks = 0
        
        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Optional: for development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.particles = []

        # Paddle
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT * 2
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball
        self._reset_ball()
        
        # Blocks
        self._generate_blocks()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = self.REWARD_PER_STEP
        
        # --- Action Handling ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # --- Game Logic ---
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        
        self.steps += 1
        
        # --- Termination Check ---
        terminated = False
        if self.lives <= 0:
            self.game_over = True
            terminated = True
            reward += self.REWARD_LOSE
        elif self.active_blocks <= 0:
            self.game_over = True
            terminated = True
            reward += self.REWARD_WIN
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _reset_ball(self):
        """Resets the ball's position and velocity."""
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        self.ball_vel = [self.BALL_SPEED * math.cos(angle), self.BALL_SPEED * math.sin(angle)]

    def _generate_blocks(self):
        """Creates a grid of blocks with procedurally assigned point values."""
        self.blocks = []
        total_block_area_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - total_block_area_width) / 2
        
        point_values = list(self.BLOCK_COLORS.keys())
        num_blocks = self.BLOCK_ROWS * self.BLOCK_COLS
        
        # Create a shuffled list of point assignments for procedural feel
        assignments = []
        for val in point_values:
            assignments.extend([val] * (num_blocks // len(point_values)))
        while len(assignments) < num_blocks: # Ensure correct number of blocks
            assignments.append(point_values[0])
        self.np_random.shuffle(assignments)

        for i in range(num_blocks):
            row = i // self.BLOCK_COLS
            col = i % self.BLOCK_COLS
            x = start_x + col * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
            y = self.BLOCK_AREA_TOP + row * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
            points = assignments[i]
            
            block = {
                "rect": pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
                "points": points,
                "color": self.BLOCK_COLORS[points],
                "active": True
            }
            self.blocks.append(block)
        self.active_blocks = len(self.blocks)

    def _update_ball(self):
        """Moves the ball according to its velocity."""
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

    def _handle_collisions(self):
        """Checks and resolves all ball collisions and returns rewards."""
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH, ball_rect.right)
            # sfx: wall_bounce.wav
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            # sfx: wall_bounce.wav

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            reward += self.REWARD_PADDLE_HIT
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on where it hit the paddle
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.BALL_SPEED * offset
            
            # Normalize velocity to maintain constant speed
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED
            
            # Ensure ball is above paddle to prevent sticking
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            # sfx: paddle_hit.wav

        # Block collisions
        for block in self.blocks:
            if block["active"] and ball_rect.colliderect(block["rect"]):
                block["active"] = False
                self.active_blocks -= 1
                self.score += block["points"]
                reward += block["points"]
                
                self._create_particles(block["rect"].center, block["color"])
                
                # Determine bounce direction
                prev_ball_rect = pygame.Rect(ball_rect.x - self.ball_vel[0], ball_rect.y - self.ball_vel[1], ball_rect.width, ball_rect.height)
                
                if prev_ball_rect.bottom <= block["rect"].top or prev_ball_rect.top >= block["rect"].bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1
                
                # sfx: block_destroy.wav
                break # Only handle one block collision per frame

        # Ball out of bounds (lose life)
        if ball_rect.top > self.SCREEN_HEIGHT:
            self.lives -= 1
            reward += self.REWARD_LIFE_LOST
            if self.lives > 0:
                self._reset_ball()
            # sfx: life_lost.wav
            
        return reward
    
    def _create_particles(self, pos, color):
        """Generates particles for a destruction effect."""
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        """Updates position and lifespan of all particles."""
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _get_observation(self):
        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        
        self._render_particles()
        self._render_blocks()
        self._render_paddle()
        self._render_ball()
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 20.0))
            color = (p["color"][0], p["color"][1], p["color"][2], alpha)
            size = int(max(1, 3 * (p["lifespan"] / 20.0)))
            rect = pygame.Rect(int(p["pos"][0] - size/2), int(p["pos"][1] - size/2), size, size)
            
            # Create a temporary surface for transparency
            temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, temp_surf.get_rect())
            self.screen.blit(temp_surf, rect.topleft)

    def _render_blocks(self):
        for block in self.blocks:
            if block["active"]:
                pygame.draw.rect(self.screen, block["color"], block["rect"])
                pygame.draw.rect(self.screen, self.COLOR_BLOCK_OUTLINE, block["rect"], 2)

    def _render_paddle(self):
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

    def _render_ball(self):
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.active_blocks <= 0 else "GAME OVER"
            color = (100, 255, 100) if self.active_blocks <= 0 else (255, 100, 100)
            
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "active_blocks": self.active_blocks
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    while running:
        # Action defaults
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # space and shift are not used
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
        if terminated:
            # Wait a bit before resetting on game over
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    env.close()