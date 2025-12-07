
# Generated: 2025-08-28T05:37:06.362148
# Source Brief: brief_05634.md
# Brief Index: 5634

        
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

    # User-facing control string, corrected for the game genre.
    user_guide = (
        "Controls: ←→ to move the paddle. Break all the blocks to win."
    )

    # User-facing description of the game, corrected for the game genre.
    game_description = (
        "A retro arcade game. Use the paddle to keep the ball in play and break all the blocks."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 1000
        self.INITIAL_LIVES = 3

        # Colors
        self.COLOR_BG = (15, 15, 40)
        self.COLOR_GRID = (30, 30, 60)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 50, 50), (255, 150, 50), (255, 255, 50),
            (50, 255, 50), (50, 150, 255), (150, 50, 255)
        ]

        # Paddle
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12

        # Ball
        self.BALL_RADIUS = 7
        self.BALL_MAX_SPEED = 7
        self.BALL_MIN_SPEED = 5

        # Blocks
        self.BLOCK_COLS = 10
        self.BLOCK_ROWS = 5
        self.BLOCK_WIDTH = 58
        self.BLOCK_HEIGHT = 20
        self.BLOCK_SPACING = 6

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        
        # Initialize state variables
        self.reset()
        
        # --- Validation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        
        # Paddle
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT * 2
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        # Ball
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_MIN_SPEED

        # Blocks
        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - total_block_width) / 2
        start_y = 50
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = start_y + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT), "color": color})

        # Particles
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small time penalty to encourage efficiency
        
        # --- Handle Action ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle.x))
        
        # --- Update Game State ---
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.lives <= 0:
            terminated = True
            self.game_over = True
        elif not self.blocks:
            reward += 100  # Victory bonus
            terminated = True
            self.game_over = True
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

    def _update_ball(self):
        self.ball_pos += self.ball_vel

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.SCREEN_WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # SFX placeholder: # Wall bounce sound

        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # SFX placeholder: # Wall bounce sound

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS

            # Influence horizontal velocity based on hit location
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x = self.BALL_MAX_SPEED * offset
            
            # Normalize speed
            speed = self.ball_vel.length()
            if speed > self.BALL_MAX_SPEED:
                self.ball_vel = self.ball_vel.normalize() * self.BALL_MAX_SPEED
            elif speed < self.BALL_MIN_SPEED:
                self.ball_vel = self.ball_vel.normalize() * self.BALL_MIN_SPEED
            # SFX placeholder: # Paddle hit sound

        # Block collisions
        hit_block_idx = ball_rect.collidelist([b["rect"] for b in self.blocks])
        if hit_block_idx != -1:
            block_data = self.blocks.pop(hit_block_idx)
            block_rect = block_data["rect"]
            
            self._create_particles(block_rect.center, block_data["color"])
            reward += 1
            self.score += 10
            
            # Simple but effective collision response
            self.ball_vel.y *= -1
            # SFX placeholder: # Block break sound
        
        # Bottom wall (lose life)
        if self.ball_pos.y + self.BALL_RADIUS >= self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 10
            # SFX placeholder: # Life lost sound
            if self.lives > 0:
                self._reset_ball()
        
        return reward

    def _reset_ball(self):
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_MIN_SPEED

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([pygame.Vector2(pos), vel, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1]  # Update position
            p[2] -= 1     # Decrease lifetime
            p[1] *= 0.95  # Apply friction
        self.particles = [p for p in self.particles if p[2] > 0]

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
            
        # Blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data["color"], block_data["rect"], border_radius=3)
            
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Particles
        for p in self.particles:
            pos, _, lifetime, color = p
            radius = max(0, (lifetime / 30) * 4)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)

        # Ball
        ball_x, ball_y = int(self.ball_pos.x), int(self.ball_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_icon_width = self.PADDLE_WIDTH / 3
        life_icon_height = self.PADDLE_HEIGHT / 2
        for i in range(self.lives):
            x = self.SCREEN_WIDTH - (i + 1) * (life_icon_width + 5) - 5
            y = 10
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, (x, y, life_icon_width, life_icon_height), border_radius=3)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WON!" if not self.blocks else "GAME OVER"
            text_surf = self.font_game_over.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
    
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
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    # --- Example Usage ---
    # To run this, you will need to install pygame: pip install pygame
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Map keys to actions
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4
    }

    # Set up a window for human play
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Breakout")
    clock = pygame.time.Clock()

    while running:
        action = np.array([0, 0, 0])  # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            print("--- New Game Started ---")

        clock.tick(30) # Run at 30 FPS

    env.close()