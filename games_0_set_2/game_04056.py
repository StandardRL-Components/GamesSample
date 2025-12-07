import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set SDL_VIDEODRIVER to dummy for headless execution, which is required for the environment
# This should be done before pygame.init()
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←/→ to tilt paddle. Space to set paddle flat. Keep the ball in the air."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist, skill-based arcade game. Control the paddle's angle to keep the ball bouncing and score points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    WIN_SCORE = 15
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_WALL = (60, 65, 80)
    COLOR_PADDLE = (137, 221, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_PARTICLE = (255, 204, 102)
    COLOR_GAMEOVER = (255, 80, 80)
    COLOR_WIN = (80, 255, 80)

    # Physics
    PADDLE_WIDTH, PADDLE_HEIGHT = 120, 16
    PADDLE_Y = HEIGHT - 40
    BALL_RADIUS = 10
    BALL_HORIZ_SPEED = 7.0
    GRAVITY = 0.35
    PADDLE_BOUNCE_EFFECT = 1.05
    PADDLE_ANGLE_INFLUENCE = 4.0
    WALL_THICKNESS = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
        self.font_small = pygame.font.Font(pygame.font.get_default_font(), 24)

        # Game state variables are initialized in reset()
        self.ball_pos = None
        self.ball_vel = None
        self.paddle_angle = None
        self.paddle_rect = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        
        # This check is for development and ensures compliance
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Paddle state
        self.paddle_angle = 0.0
        self.paddle_rect = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.PADDLE_Y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Ball state - Start on the paddle for a stable beginning
        # This prevents the ball from immediately falling and causing termination,
        # which fixes the stability test failure.
        self.ball_pos = pygame.Vector2(self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS)
        
        # Give it a slight random horizontal velocity and a strong upward velocity
        initial_vx = self.np_random.uniform(-self.BALL_HORIZ_SPEED / 4, self.BALL_HORIZ_SPEED / 4)
        initial_vy = -self.np_random.uniform(9, 12)
        self.ball_vel = pygame.Vector2(initial_vx, initial_vy)

        # Effects
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and return the last state
            reward = 0
            terminated = True
            return (
                self._get_observation(),
                reward,
                terminated,
                False, # Truncated is false if terminated
                self._get_info()
            )

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        target_angle = self.paddle_angle
        if space_held:
            target_angle = 0.0
        elif movement == 3:  # Left
            target_angle = 15.0 # Positive angle tilts left side up
        elif movement == 4:  # Right
            target_angle = -15.0 # Negative angle tilts right side up
        
        # Smoothly interpolate paddle angle for better visual feel
        self.paddle_angle += (target_angle - self.paddle_angle) * 0.4

        # --- Game Logic ---
        reward = 0.1  # Base reward for surviving a frame

        # 1. Update Ball Position and Velocity
        self.ball_vel.y += self.GRAVITY
        self.ball_pos += self.ball_vel

        # 2. Collision Detection
        # Walls
        if self.ball_pos.x - self.BALL_RADIUS < self.WALL_THICKNESS or self.ball_pos.x + self.BALL_RADIUS > self.WIDTH - self.WALL_THICKNESS:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.ball_pos.x, self.BALL_RADIUS + self.WALL_THICKNESS)
            self.ball_pos.x = min(self.ball_pos.x, self.WIDTH - self.BALL_RADIUS - self.WALL_THICKNESS)
            self._create_particles(pygame.Vector2(self.ball_pos.x, self.ball_pos.y), 5, self.COLOR_WALL)

        # Top wall
        if self.ball_pos.y - self.BALL_RADIUS < 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS

        # Paddle
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.paddle_rect.colliderect(ball_rect) and self.ball_vel.y > 0:
            self.ball_pos.y = self.paddle_rect.top - self.BALL_RADIUS
            
            reward += 1.0
            self.score += 1
            
            self.ball_vel.y *= -self.PADDLE_BOUNCE_EFFECT
            self.ball_vel.y = min(self.ball_vel.y, -2.0)
            
            angle_rad = math.radians(self.paddle_angle)
            self.ball_vel.x -= math.sin(angle_rad) * self.PADDLE_ANGLE_INFLUENCE

            self.ball_vel.x = max(-15, min(15, self.ball_vel.x))
            
            bounce_angle_deg = math.degrees(math.atan2(-self.ball_vel.y, abs(self.ball_vel.x)))
            if bounce_angle_deg > 70:
                reward -= 0.2
            elif bounce_angle_deg < 35:
                reward += 0.5
            
            self._create_particles(self.ball_pos, 20, self.COLOR_PARTICLE)

        # 3. Update Particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = (p['life'] / p['max_life']) * p['max_radius']

        # 4. Check Termination and Truncation Conditions
        terminated = False
        truncated = False

        if self.ball_pos.y - self.BALL_RADIUS > self.HEIGHT:
            self.game_over = True
            terminated = True
            reward = -10.0
        
        if not terminated and self.score >= self.WIN_SCORE:
            self.game_over = True
            terminated = True
            reward += 10.0

        # MAX_STEPS condition should set truncated to True, not terminated.
        # Check is for >= MAX_STEPS - 1 because steps is incremented after this.
        if not terminated and self.steps >= self.MAX_STEPS - 1:
            truncated = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))

        for p in self.particles:
            if p['radius'] > 0:
                pos = (int(p['pos'].x), int(p['pos'].y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        glow_color = (self.COLOR_BALL[0], self.COLOR_BALL[1], self.COLOR_BALL[2], 50)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 4, glow_color)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

        paddle_surf = pygame.Surface((self.PADDLE_WIDTH, self.PADDLE_HEIGHT), pygame.SRCALPHA)
        paddle_surf.fill(self.COLOR_PADDLE)
        rotated_paddle = pygame.transform.rotate(paddle_surf, self.paddle_angle)
        rotated_rect = rotated_paddle.get_rect(center=self.paddle_rect.center)
        self.screen.blit(rotated_paddle, rotated_rect.topleft)
        
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 15, 15))

        if self.game_over:
            if self.score >= self.WIN_SCORE:
                end_text = "YOU WIN!"
                end_color = self.COLOR_WIN
            else:
                end_text = "GAME OVER"
                end_color = self.COLOR_GAMEOVER
            
            text_surf = self.font_large.render(end_text, True, end_color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ball_pos": (self.ball_pos.x, self.ball_pos.y),
            "ball_vel": (self.ball_vel.x, self.ball_vel.y),
            "paddle_angle": self.paddle_angle,
        }

    def _create_particles(self, position, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pygame.Vector2(position),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': life,
                'max_life': life,
                'radius': self.np_random.uniform(2, 5),
                'max_radius': 5,
                'color': color
            })

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    # To play, you need a window. Un-comment the SDL_VIDEODRIVER line.
    # The main environment is headless by default.
    # import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.

    env = GameEnv(render_mode="rgb_array")
    
    try:
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Paddle Bounce")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        print("\n" + "="*30)
        print(f"GAME: {env.game_description}")
        print(f"CONTROLS: {env.user_guide}")
        print("="*30 + "\n")

        while not done:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        print("Resetting game.")
                        obs, info = env.reset()
                        done = False
                    if event.key == pygame.K_q:
                        done = True

            clock.tick(60)
            
    finally:
        env.close()
        pygame.quit()
        print("Game window closed.")