
# Generated: 2025-08-28T06:13:18.100785
# Source Brief: brief_05828.md
# Brief Index: 5828

        
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
    """A simple particle class for visual effects."""
    def __init__(self, x, y, color, rng):
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 4)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.lifetime = rng.integers(15, 30)
        self.size = rng.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravity effect
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / 30))
            color = self.color + (alpha,)
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (self.size, self.size), self.size)
            surface.blit(temp_surf, (self.x - self.size, self.y - self.size))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = "Controls: Use ↑ and ↓ to move the paddle."

    # User-facing game description
    game_description = (
        "Hit the ball with your paddle to score points. "
        "Missing the ball costs you a life. Score 8 points to win!"
    )

    # Frames auto-advance for smooth real-time gameplay
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (20, 25, 40)
    COLOR_PADDLE = (255, 200, 0)
    COLOR_BALL = (0, 255, 255)
    COLOR_BALL_GLOW = (0, 150, 150)
    COLOR_WALL = (180, 180, 180)
    COLOR_UI = (255, 255, 255)
    COLOR_MISS = (255, 50, 50)
    COLOR_GRID = (40, 50, 70)

    PADDLE_WIDTH, PADDLE_HEIGHT = 10, 80
    PADDLE_SPEED = 12
    PADDLE_X = 60
    BALL_RADIUS = 10
    
    WIN_SCORE = 8
    LOSE_MISSES = 4
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_miss = pygame.font.SysFont("Consolas", 36, bold=True)

        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.paddle_y = 0
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_base_speed = 0
        self.last_ball_paddle_dist_y = 0
        self.particles = []

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.paddle_y = self.HEIGHT / 2
        self.ball_base_speed = 5.0
        self.particles = []
        
        self._reset_ball()
        
        self.last_ball_paddle_dist_y = abs(self.ball_pos[1] - self.paddle_y)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Calculate Continuous Reward ---
        reward = self._calculate_continuous_reward()

        # --- Handle Actions ---
        movement = action[0]
        self._update_paddle(movement)
        
        # --- Update Game Logic ---
        self._update_ball()
        self._update_particles()
        
        # --- Handle Collisions and Events ---
        hit_reward, miss_penalty = self._handle_collisions()
        reward += hit_reward + miss_penalty

        # --- Update Game State ---
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.misses >= self.LOSE_MISSES:
                reward -= 100 # Lose penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _calculate_continuous_reward(self):
        # Reward for moving the paddle closer to the ball's y-position
        current_dist_y = abs(self.ball_pos[1] - self.paddle_y)
        reward = 0.0
        if current_dist_y < self.last_ball_paddle_dist_y:
            reward += 0.1
        else:
            reward -= 0.1
        self.last_ball_paddle_dist_y = current_dist_y
        return reward

    def _update_paddle(self, movement):
        if movement == 1:  # Up
            self.paddle_y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle_y += self.PADDLE_SPEED
        
        # Clamp paddle to screen boundaries
        self.paddle_y = np.clip(
            self.paddle_y, self.PADDLE_HEIGHT / 2, self.HEIGHT - self.PADDLE_HEIGHT / 2
        )

    def _update_ball(self):
        self.ball_pos += self.ball_vel

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _handle_collisions(self):
        hit_reward = 0
        miss_penalty = 0

        # Ball hits top/bottom walls
        if (self.ball_pos[1] <= self.BALL_RADIUS and self.ball_vel[1] < 0) or \
           (self.ball_pos[1] >= self.HEIGHT - self.BALL_RADIUS and self.ball_vel[1] > 0):
            self.ball_vel[1] *= -1
            # sfx: wall_bounce

        # Ball hits right wall (after a paddle hit)
        if self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS and self.ball_vel[0] > 0:
            self.ball_vel[0] *= -1

        # Ball hits paddle
        paddle_rect = pygame.Rect(
            self.PADDLE_X, self.paddle_y - self.PADDLE_HEIGHT / 2, 
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
        )

        if self.ball_vel[0] < 0 and paddle_rect.colliderect(ball_rect):
            self.score += 1
            hit_reward += 1.0
            # sfx: paddle_hit
            self._create_hit_particles(self.ball_pos[0], self.ball_pos[1])
            
            # Increase difficulty every 2 points
            if self.score > 0 and self.score % 2 == 0:
                self.ball_base_speed += 0.5
            
            self._reset_ball()

        # Ball misses paddle (goes off left screen)
        elif self.ball_pos[0] < -self.BALL_RADIUS:
            self.misses += 1
            # sfx: miss
            self._reset_ball()

        return hit_reward, miss_penalty
        
    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True
        if self.misses >= self.LOSE_MISSES:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _reset_ball(self):
        y_start = self.np_random.uniform(self.HEIGHT * 0.2, self.HEIGHT * 0.8)
        self.ball_pos = np.array([float(self.WIDTH + self.BALL_RADIUS), y_start])
        
        # Angle towards a vertical band around the paddle's current position
        target_y = self.np_random.uniform(self.paddle_y - self.PADDLE_HEIGHT, self.paddle_y + self.PADDLE_HEIGHT)
        target_y = np.clip(target_y, 0, self.HEIGHT)
        
        direction_vector = np.array([self.PADDLE_X - self.ball_pos[0], target_y - self.ball_pos[1]])
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            self.ball_vel = (direction_vector / norm) * self.ball_base_speed
        else: # Failsafe
            angle = self.np_random.uniform(math.radians(160), math.radians(200))
            self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * self.ball_base_speed

    def _create_hit_particles(self, x, y):
        for _ in range(self.np_random.integers(15, 25)):
            self.particles.append(Particle(x, y, self.COLOR_PADDLE, self.np_random))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw perspective grid for "slope" effect
        for i in range(21):
            p = i / 20.0
            y = self.HEIGHT * p
            
            # Perspective effect: lines closer at the top
            persp_y = y * (1.0 + p * 0.5) - (self.HEIGHT * p * 0.5)
            
            # Fade out lines at the top and bottom
            alpha = int(150 * (1 - abs(p - 0.5) * 2))
            line_color = self.COLOR_GRID[:3] + (alpha,)
            
            temp_surf = pygame.Surface((self.WIDTH, 1), pygame.SRCALPHA)
            temp_surf.fill(line_color)
            self.screen.blit(temp_surf, (0, persp_y))

    def _render_game_elements(self):
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw paddle
        paddle_rect = pygame.Rect(
            self.PADDLE_X, int(self.paddle_y - self.PADDLE_HEIGHT / 2),
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)
        
        # Draw ball with glow
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        glow_radius = int(self.BALL_RADIUS * 1.8)
        
        # Glow effect
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW + (80,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (ball_pos_int[0] - glow_radius, ball_pos_int[1] - glow_radius))

        # Ball core
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Draw Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (20, 10))

        # Draw Misses
        for i in range(self.misses):
            miss_text = self.font_miss.render("X", True, self.COLOR_MISS)
            self.screen.blit(miss_text, (self.WIDTH - 30 - i * 25, 5))
            
        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            
            end_text = pygame.font.SysFont("Consolas", 60, bold=True).render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "misses": self.misses}

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

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption("Slope Ball")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
            
        action = [movement, 0, 0] # space and shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for a moment then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Match the intended FPS

    env.close()