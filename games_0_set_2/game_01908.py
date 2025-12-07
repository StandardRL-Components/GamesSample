
# Generated: 2025-08-27T18:40:07.767663
# Source Brief: brief_01908.md
# Brief Index: 1908

        
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
    A fast-paced, grid-based Brick Breaker with risk/reward scoring where players must strategically aim
    to maximize points before running out of lives.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use ← and → to move the paddle."
    )

    # User-facing game description
    game_description = (
        "A retro arcade brick breaker. Break colored bricks for points, but don't miss the ball!"
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (26, 28, 44)  # Dark Blue
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    BRICK_COLORS = {
        "green": (50, 205, 50),
        "blue": (65, 105, 225),
        "red": (220, 20, 60),
        "gold": (255, 215, 0),
    }
    BRICK_VALUES = {
        "green": 1,
        "blue": 3,
        "red": 5,
        "gold": 10,
    }

    # Paddle properties
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12

    # Ball properties
    BALL_RADIUS = 7
    BALL_INITIAL_SPEED = 6
    BALL_MAX_SPIN_EFFECT = 1.5

    # Brick properties
    BRICK_ROWS = 5
    BRICK_COLS = 12
    BRICK_WIDTH = 50
    BRICK_HEIGHT = 20
    BRICK_GAP = 4
    BRICK_OFFSET_TOP = 50
    BRICK_OFFSET_LEFT = (SCREEN_WIDTH - (BRICK_COLS * (BRICK_WIDTH + BRICK_GAP) - BRICK_GAP)) // 2

    # Game rules
    INITIAL_LIVES = 3
    WIN_SCORE = 100
    MAX_STEPS = 1000 * (30 // FPS) # Scale max steps to FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = []
        self.particles = []

        self.reset()
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
        self.paddle_rect = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball
        self.ball_pos = np.array([self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS - 1], dtype=float)
        angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=float) * self.BALL_INITIAL_SPEED

        # Bricks
        self.bricks = []
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                brick_x = self.BRICK_OFFSET_LEFT + c * (self.BRICK_WIDTH + self.BRICK_GAP)
                brick_y = self.BRICK_OFFSET_TOP + r * (self.BRICK_HEIGHT + self.BRICK_GAP)
                
                if r == 0: color_key = "gold"
                elif r <= 1: color_key = "red"
                elif r <= 3: color_key = "blue"
                else: color_key = "green"

                self.bricks.append({
                    "rect": pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT),
                    "color_key": color_key,
                    "value": self.BRICK_VALUES[color_key],
                    "color": self.BRICK_COLORS[color_key],
                })
        
        # Particles
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0

        # --- Handle Input ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
            reward -= 0.02 # Cost for moving
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
            reward -= 0.02 # Cost for moving
        
        # Clamp paddle to screen
        self.paddle_rect.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle_rect.x))

        # --- Update Game Logic ---
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if self.lives <= 0:
            reward -= 50  # Lose penalty
            terminated = True
        elif self.score >= self.WIN_SCORE:
            reward += 100  # Win bonus
            terminated = True
        elif not self.bricks: # Cleared all bricks
            reward += 100 # Win bonus
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_ball(self):
        self.ball_pos += self.ball_vel

        # Anti-softlock: ensure ball doesn't get stuck horizontally
        if abs(self.ball_vel[1]) < 0.5:
            self.ball_vel[1] = math.copysign(0.5, self.ball_vel[1] if self.ball_vel[1] != 0 else 1.0)

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(1, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH - 1, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(1, ball_rect.top)
            self.ball_pos[1] = ball_rect.centery
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            # Add spin based on where the ball hits the paddle
            offset = (ball_rect.centerx - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * self.BALL_MAX_SPIN_EFFECT
            # Normalize speed
            speed = np.linalg.norm(self.ball_vel)
            self.ball_vel = self.ball_vel / speed * self.BALL_INITIAL_SPEED
            
            ball_rect.bottom = self.paddle_rect.top - 1
            self.ball_pos[1] = ball_rect.centery
            # sfx: paddle_hit

        # Brick collisions
        for brick in self.bricks[:]:
            if ball_rect.colliderect(brick["rect"]):
                # Determine collision side to correctly reflect the ball
                # A simple approximation: check if ball center is mostly horizontal or vertical
                # relative to the brick center
                brick_center = np.array(brick["rect"].center)
                collision_vector = self.ball_pos - brick_center
                
                if abs(collision_vector[0] / brick["rect"].width) > abs(collision_vector[1] / brick["rect"].height):
                    self.ball_vel[0] *= -1 # Horizontal collision
                else:
                    self.ball_vel[1] *= -1 # Vertical collision

                # Rewards and score
                reward += 0.1 # Continuous reward for hitting any brick
                reward += brick["value"]
                self.score += brick["value"]
                
                # Visual effects and game state change
                self._create_particles(brick["rect"].center, brick["color"])
                self.bricks.remove(brick)
                # sfx: brick_break
                break

        # Miss (bottom wall)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 1 # Miss penalty
            # Reset ball position
            self.ball_pos = np.array([self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS - 1], dtype=float)
            angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
            self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=float) * self.BALL_INITIAL_SPEED
            # sfx: life_lost

        return reward

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "lifespan": self.np_random.uniform(10, 20),
                "color": color
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render bricks with a subtle 3D effect
        for brick in self.bricks:
            r = brick["rect"]
            color_dark = tuple(c * 0.6 for c in brick["color"])
            color_light = tuple(min(255, c * 1.2) for c in brick["color"])
            pygame.draw.rect(self.screen, brick["color"], r)
            pygame.draw.line(self.screen, color_dark, r.bottomleft, r.bottomright, 2)
            pygame.draw.line(self.screen, color_dark, r.topright, r.bottomright, 2)
            pygame.draw.line(self.screen, color_light, r.topleft, r.bottomleft, 2)
            pygame.draw.line(self.screen, color_light, r.topleft, r.topright, 2)

        # Render paddle with a subtle 3D effect
        p = self.paddle_rect
        color_dark = tuple(c * 0.6 for c in self.COLOR_PADDLE)
        color_light = tuple(min(255, c * 1.2) for c in self.COLOR_PADDLE)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, p)
        pygame.draw.rect(self.screen, color_dark, p, 2)


        # Render ball with a glow
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Render particles
        for p in self.particles:
            alpha = p["lifespan"] / 20
            color = (
                int(p["color"][0] * alpha),
                int(p["color"][1] * alpha),
                int(p["color"][2] * alpha)
            )
            size = max(1, int(3 * alpha))
            pygame.draw.rect(self.screen, color, (int(p["pos"][0]), int(p["pos"][1]), size, size))

    def _render_ui(self):
        # Render score
        score_text = self.font_big.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render lives as icons
        life_icon_rect = pygame.Rect(0, 0, self.PADDLE_WIDTH // 4, self.PADDLE_HEIGHT // 2)
        for i in range(self.lives):
            life_icon_rect.topright = (self.SCREEN_WIDTH - 10 - i * (life_icon_rect.width + 5), 15)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_icon_rect, border_radius=2)
            
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            
            message = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            text_surf = self.font_big.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
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


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a window.
    # The environment itself is headless, but we can create a window to display its output.
    
    # Set up a display for human play
    pygame.display.set_caption("Brick Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print("MANUAL PLAY INSTRUCTIONS")
    print(GameEnv.user_guide)
    print("="*30 + "\n")

    while not done:
        # Construct action from keyboard input
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # [movement, space, shift]

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'r' key

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        # The observation is (H, W, C), but pygame surfaces expect (W, H)
        # and surfarray.make_surface expects transposed (W, H, C)
        # So we transpose the observation from (H, W, C) to (W, H, C)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control the frame rate
        env.clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()