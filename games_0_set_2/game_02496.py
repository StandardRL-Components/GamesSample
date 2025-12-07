
# Generated: 2025-08-27T20:32:28.345143
# Source Brief: brief_02496.md
# Brief Index: 2496

        
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


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro arcade grid-based tennis game.

    The player controls a paddle at the bottom of the screen and must hit a
    bouncing ball to score points. Hitting the ball on the edges of the
    paddle (the "risky zone") yields a high reward, while hitting it in the
    center (the "safe zone") results in a small penalty. The game ends
    if the player scores 7 points (win) or misses 3 consecutive balls (loss).
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use ← and → to move the paddle. There are no other active controls."
    )

    # User-facing game description
    game_description = (
        "A fast-paced, grid-based arcade tennis game. Hit the bouncing ball with your paddle to score points. "
        "Aim for the edges of your paddle for bonus points, but be careful not to miss!"
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 16)

        # Game constants
        self.PADDLE_WIDTH = 120
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 20
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 6
        self.MAX_BALL_SPEED = 12
        self.MAX_STEPS = 1500
        self.WIN_SCORE = 7
        self.MAX_MISSES = 3

        # Color palette
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 50)
        self.COLOR_RISKY_ZONE = (255, 80, 80, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TRAJECTORY = (0, 255, 150)

        # Game state variables (initialized in reset)
        self.paddle_pos_x = 0
        self.ball_pos = np.zeros(2)
        self.ball_vel = np.zeros(2)
        self.score = 0
        self.consecutive_misses = 0
        self.steps = 0
        self.game_over = False
        self.particles = []

        self.reset()
        # self.validate_implementation() # Optional: Call to test during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle_pos_x = self.SCREEN_WIDTH / 2
        
        # Reset ball to center with a random downward velocity
        self.ball_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=float) * self.INITIAL_BALL_SPEED

        self.score = 0
        self.consecutive_misses = 0
        self.steps = 0
        self.game_over = False
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # --- Action Handling ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle_pos_x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_pos_x += self.PADDLE_SPEED

        # Clamp paddle position to be within screen bounds
        self.paddle_pos_x = np.clip(
            self.paddle_pos_x, self.PADDLE_WIDTH / 2, self.SCREEN_WIDTH - self.PADDLE_WIDTH / 2
        )

        # --- Game Logic ---
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        
        # --- Termination Check ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 10
            terminated = True
            self.game_over = True
        elif self.consecutive_misses >= self.MAX_MISSES:
            reward += -10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_ball(self):
        self.ball_pos += self.ball_vel

    def _handle_collisions(self):
        reward = 0
        
        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
            self._create_particles(self.ball_pos, 10, self.COLOR_GRID)
            # sfx: wall_bounce.wav
            
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            self._create_particles(self.ball_pos, 10, self.COLOR_GRID)
            # sfx: wall_bounce.wav

        # Paddle collision
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT
        if self.ball_vel[1] > 0 and self.ball_pos[1] >= paddle_y - self.BALL_RADIUS:
            paddle_left = self.paddle_pos_x - self.PADDLE_WIDTH / 2
            paddle_right = self.paddle_pos_x + self.PADDLE_WIDTH / 2
            
            if paddle_left <= self.ball_pos[0] <= paddle_right:
                self.ball_pos[1] = paddle_y - self.BALL_RADIUS
                self.consecutive_misses = 0
                # sfx: paddle_hit.wav

                # Calculate hit position and apply rewards/physics
                hit_offset = self.ball_pos[0] - self.paddle_pos_x
                normalized_offset = hit_offset / (self.PADDLE_WIDTH / 2)
                
                # Base reward for any hit
                reward += 0.1

                # Risky zone (outer 20%)
                if abs(normalized_offset) > 0.8:
                    reward += 1.0
                # Safe zone (inner 60%)
                elif abs(normalized_offset) < 0.6:
                    reward -= 0.2
                
                # Update ball velocity based on hit location
                self.ball_vel[1] *= -1
                self.ball_vel[0] += normalized_offset * 2.0
                
                # Clamp ball speed
                speed = np.linalg.norm(self.ball_vel)
                if speed > self.MAX_BALL_SPEED:
                    self.ball_vel = self.ball_vel / speed * self.MAX_BALL_SPEED

                self.score += 1
                self._create_particles(self.ball_pos, 20, self.COLOR_PADDLE)
            
        # Ball miss
        if self.ball_pos[1] > self.SCREEN_HEIGHT + self.BALL_RADIUS:
            self.consecutive_misses += 1
            reward -= 0.1
            # sfx: miss.wav
            
            # Reset ball
            self.ball_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
            angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
            self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=float) * self.INITIAL_BALL_SPEED
        
        return reward

    def _create_particles(self, position, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(position), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        # Iterate backwards to allow safe removal
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Damping
            p['vel'][1] *= 0.95 # Damping
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "consecutive_misses": self.consecutive_misses
        }

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_particles()
        self._render_trajectory()
        self._render_paddle()
        self._render_ball()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        grid_spacing = 40
        for x in range(0, self.SCREEN_WIDTH, grid_spacing):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, grid_spacing):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_paddle(self):
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT
        paddle_rect = pygame.Rect(
            self.paddle_pos_x - self.PADDLE_WIDTH / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)
        
        # Draw risky zones
        risky_zone_width = self.PADDLE_WIDTH * 0.2
        risky_surface = pygame.Surface((risky_zone_width, self.PADDLE_HEIGHT), pygame.SRCALPHA)
        risky_surface.fill(self.COLOR_RISKY_ZONE)
        
        self.screen.blit(risky_surface, (paddle_rect.left, paddle_rect.top))
        self.screen.blit(risky_surface, (paddle_rect.right - risky_zone_width, paddle_rect.top))

    def _render_ball(self):
        # Glow effect
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]),
            self.BALL_RADIUS + 5, self.COLOR_BALL_GLOW
        )
        # Main ball
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]),
            self.BALL_RADIUS, self.COLOR_BALL
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]),
            self.BALL_RADIUS, self.COLOR_BALL
        )

    def _render_trajectory(self):
        # Predict a short path for the ball
        temp_pos = self.ball_pos.copy()
        temp_vel = self.ball_vel.copy()
        points = []
        for _ in range(15):
            temp_pos += temp_vel
            if temp_pos[0] <= self.BALL_RADIUS or temp_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
                temp_vel[0] *= -1
            if temp_pos[1] <= self.BALL_RADIUS:
                temp_vel[1] *= -1
            if temp_pos[1] > self.SCREEN_HEIGHT - self.PADDLE_HEIGHT:
                break
            points.append(tuple(temp_pos))

        if len(points) > 2:
            pygame.draw.aalines(self.screen, self.COLOR_TRAJECTORY, False, points)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 30.0))
            color_with_alpha = (*p['color'], alpha)
            size = max(1, int(self.BALL_RADIUS / 4 * (p['life'] / 30.0)))
            pygame.draw.circle(self.screen, color_with_alpha, (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Misses
        miss_text = self.small_font.render(f"MISSES: {self.consecutive_misses}/{self.MAX_MISSES}", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (self.SCREEN_WIDTH - miss_text.get_width() - 10, 10))

        if self.game_over:
            outcome_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text_surf = self.font.render(outcome_text, True, self.COLOR_PADDLE)
            text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify the implementation."""
        print("Running implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # Set up a window to display the rendered frames
    pygame.display.set_caption("Grid Tennis")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Use a dictionary to track held keys for more responsive controls
    keys_held = {
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
    }
    
    running = True
    while running:
        # --- Human Input Handling ---
        movement_action = 0 # No-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        if keys_held[pygame.K_LEFT]:
            movement_action = 3
        elif keys_held[pygame.K_RIGHT]:
            movement_action = 4
        
        # Construct the action for the MultiDiscrete space
        action = [movement_action, 0, 0] # space and shift are not used

        # --- Environment Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Run at 30 FPS

    env.close()