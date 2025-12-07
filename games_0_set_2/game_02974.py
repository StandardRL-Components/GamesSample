
# Generated: 2025-08-27T21:59:29.778543
# Source Brief: brief_02974.md
# Brief Index: 2974

        
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
    """
    A Gymnasium environment for a retro arcade brick-breaking game.

    The player controls a paddle at the bottom of the screen to bounce a ball
    upwards, destroying a grid of bricks. The goal is to achieve the highest
    score by clearing bricks and chaining consecutive hits for a score multiplier.
    The game ends when the player loses all three lives or reaches the victory score.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use ← and → to move the paddle. Press space to launch the ball."
    )
    game_description = (
        "A retro arcade brick-breaker. Bounce the ball to destroy bricks, "
        "chain hits for a score multiplier, and aim for a high score before losing all your lives."
    )

    # Frame advance setting
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.VICTORY_SCORE = 500
        self.MAX_STEPS = 10000
        self.INITIAL_LIVES = 3
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 80, 10
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 6
        self.INITIAL_BALL_SPEED = 4.0
        self.BRICK_ROWS, self.BRICK_COLS = 10, 10
        self.BRICK_WIDTH, self.BRICK_HEIGHT = 58, 15
        self.BRICK_GAP = 6
        self.BRICK_AREA_TOP = 50

        # --- Colors ---
        self.COLOR_BG = (16, 16, 24)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_UI = (220, 220, 220)
        self.BRICK_COLORS = {
            10: (64, 192, 64),   # Green
            20: (64, 128, 192),  # Blue
            30: (192, 64, 64)    # Red
        }

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_multiplier = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game State (initialized in reset) ---
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = False
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        self.bricks = []
        self.particles = []
        self.ball_trail = []
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.consecutive_hits = 0
        self.multiplier = 1
        self.game_over = False
        self.score_checkpoint = 100
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Optional: Call for self-check during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_launched = False
        self._reset_ball()
        self._create_bricks()

        self.particles = []
        self.ball_trail = []
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.steps = 0
        self.consecutive_hits = 0
        self.multiplier = 1
        self.game_over = False
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        self.score_checkpoint = 100

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = -0.02  # Small penalty for time passing

        # --- Action Handling ---
        movement = action[0]
        space_pressed = action[1] == 1

        self._update_paddle(movement)
        reward += self._update_ball(space_pressed)
        
        self._update_particles()

        # --- Game State Updates ---
        self._update_difficulty()
        if not self.bricks:
            self._create_bricks()

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            if self.score >= self.VICTORY_SCORE:
                reward += 100 # Victory bonus
            elif self.lives <= 0:
                reward -= 100 # Game over penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_paddle(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.clamp_ip(self.screen.get_rect())

    def _update_ball(self, space_pressed):
        step_reward = 0
        
        if not self.ball_launched:
            if space_pressed:
                self.ball_launched = True
                # Sound: Ball Launch
                angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.current_ball_speed
            else:
                self._reset_ball()
            return step_reward

        # --- Update Ball Position and Trail ---
        self.ball_trail.append(self.ball_pos.copy())
        if len(self.ball_trail) > 10:
            self.ball_trail.pop(0)
        self.ball_pos += self.ball_vel

        # --- Wall Collisions ---
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # Sound: Wall Bounce
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            # Sound: Wall Bounce

        # --- Bottom Wall (Life Lost) ---
        if self.ball_pos.y + self.BALL_RADIUS >= self.HEIGHT:
            self.lives -= 1
            self.ball_launched = False
            self._reset_ball()
            self._reset_multiplier()
            # Sound: Life Lost
            return step_reward

        # --- Paddle Collision ---
        if self.paddle.collidepoint(self.ball_pos.x, self.ball_pos.y + self.BALL_RADIUS) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x = offset * 1.5
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.current_ball_speed
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
            self._reset_multiplier()
            # Sound: Paddle Bounce
            return step_reward

        # --- Brick Collisions ---
        hit_a_brick = False
        for brick_data in self.bricks[:]:
            brick_rect, color, points = brick_data
            if brick_rect.collidepoint(self.ball_pos):
                hit_a_brick = True
                
                # Reward logic
                step_reward += 0.1
                self.consecutive_hits += 1
                if self.consecutive_hits == 2:
                    step_reward += 1.0 # Start multiplier chain
                elif self.consecutive_hits > 2:
                    step_reward += 2.0 # Continue chain
                self.multiplier = 1 + (self.consecutive_hits - 1) // 2

                # Game logic
                self.score += points * self.multiplier
                self._create_particles(brick_rect.center, color)
                self.bricks.remove(brick_data)
                
                # Determine bounce direction
                # A simple vertical bounce feels best for this kind of game
                self.ball_vel.y *= -1
                
                # Sound: Brick Break
                break # Only one brick per frame

        return step_reward

    def _update_difficulty(self):
        if self.score >= self.score_checkpoint:
            self.current_ball_speed += 0.1
            self.score_checkpoint += 100
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.current_ball_speed

    def _reset_multiplier(self):
        self.consecutive_hits = 0
        self.multiplier = 1

    def _reset_ball(self):
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_trail.clear()

    def _create_bricks(self):
        self.bricks.clear()
        total_brick_width = self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_GAP) - self.BRICK_GAP
        start_x = (self.WIDTH - total_brick_width) // 2
        
        for row in range(self.BRICK_ROWS):
            for col in range(self.BRICK_COLS):
                if row > 5:
                    points, color = 30, self.BRICK_COLORS[30]
                elif row > 2:
                    points, color = 20, self.BRICK_COLORS[20]
                else:
                    points, color = 10, self.BRICK_COLORS[10]
                
                x = start_x + col * (self.BRICK_WIDTH + self.BRICK_GAP)
                y = self.BRICK_AREA_TOP + row * (self.BRICK_HEIGHT + self.BRICK_GAP)
                
                brick_rect = pygame.Rect(x, y, self.BRICK_WIDTH, self.BRICK_HEIGHT)
                self.bricks.append((brick_rect, color, points))

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            particle = {
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.VICTORY_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives}

    def _render_game(self):
        # Bricks
        for brick_rect, color, _ in self.bricks:
            pygame.draw.rect(self.screen, color, brick_rect, border_radius=3)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p["color"], alpha), (1, 1), 1)
            self.screen.blit(s, (int(p["pos"].x), int(p["pos"].y)))
            
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Ball trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(255 * (i / len(self.ball_trail)) * 0.5)
            pygame.gfxdraw.filled_circle(
                self.screen, int(pos.x), int(pos.y), self.BALL_RADIUS, (*self.COLOR_BALL, alpha)
            )

        # Ball
        if self.lives > 0:
            pygame.gfxdraw.filled_circle(
                self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL
            )


    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_UI)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Multiplier
        if self.multiplier > 1:
            mult_text = self.font_multiplier.render(f"x{self.multiplier}", True, self.COLOR_UI)
            text_rect = mult_text.get_rect(center=(self.WIDTH // 2, 25))
            self.screen.blit(mult_text, text_rect)
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a window to display the game
    pygame.display.set_caption("Brick Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Map keyboard keys to MultiDiscrete actions
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated:
        # --- Human Controls ---
        movement_action = 0 # No-op
        space_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
            
        if keys[pygame.K_SPACE]:
            space_action = 1

        action = [movement_action, space_action, 0] # Shift is not used

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        env.clock.tick(30) # Match the intended FPS

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            total_reward = 0

    env.close()