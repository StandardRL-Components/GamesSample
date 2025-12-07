
# Generated: 2025-08-27T19:36:28.796145
# Source Brief: brief_02198.md
# Brief Index: 2198

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to rotate the paddle. Deflect the ball to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game where you rotate a paddle to deflect a bouncing ball. "
        "Score 15 points to win, but miss 3 balls and you lose. The ball gets faster as you score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Configuration Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_FG = (255, 255, 255)
        self.COLOR_SAFE = (0, 160, 255)
        self.COLOR_RISKY = (255, 65, 54)
        
        self.PADDLE_CENTER_Y = self.HEIGHT - 50
        self.PADDLE_CENTER_X = self.WIDTH // 2
        self.PADDLE_LENGTH = 100
        self.PADDLE_THICKNESS = 8
        self.PADDLE_ROTATION_SPEED = math.radians(4)

        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 2.5
        self.BALL_SPEED_INCREMENT = 0.5

        self.MAX_STEPS = 10000
        self.WIN_SCORE = 15
        self.MAX_LIVES = 3

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle_angle = 0.0
        self.ball_base_speed = 0.0
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        
        self.paddle_angle = 0.0
        self.ball_base_speed = self.INITIAL_BALL_SPEED
        
        self.particles = []
        
        self._spawn_ball()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        
        self.steps += 1
        reward = -0.01  # Small penalty for time passing

        # Update game logic
        self._update_paddle(movement)
        self._update_ball()
        collision_reward = self._handle_collisions()
        reward += collision_reward
        self._update_particles()
        
        terminated = (self.lives <= 0 or 
                      self.score >= self.WIN_SCORE or 
                      self.steps >= self.MAX_STEPS)

        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward = 50.0  # Win reward
            else:
                reward = -50.0 # Loss reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_ball(self):
        x = self.np_random.uniform(self.BALL_RADIUS * 2, self.WIDTH - self.BALL_RADIUS * 2)
        y = self.BALL_RADIUS + 10.0
        self.ball_pos = pygame.math.Vector2(x, y)
        
        target_x = self.np_random.uniform(
            self.PADDLE_CENTER_X - self.PADDLE_LENGTH, 
            self.PADDLE_CENTER_X + self.PADDLE_LENGTH
        )
        angle_to_target = math.atan2(self.PADDLE_CENTER_Y - y, target_x - x)
        
        self.ball_vel = pygame.math.Vector2(
            math.cos(angle_to_target),
            math.sin(angle_to_target)
        ) * self.ball_base_speed

    def _update_paddle(self, movement):
        if movement == 3:  # Left
            self.paddle_angle -= self.PADDLE_ROTATION_SPEED
        elif movement == 4: # Right
            self.paddle_angle += self.PADDLE_ROTATION_SPEED
        
        self.paddle_angle = max(-math.pi / 2.1, min(math.pi / 2.1, self.paddle_angle))

    def _update_ball(self):
        self.ball_pos += self.ball_vel

        if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # // sfx: wall_bounce

        if self.ball_pos.y <= self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # // sfx: wall_bounce

    def _handle_collisions(self):
        reward = 0.0

        if self.ball_pos.y > self.HEIGHT:
            self.lives -= 1
            reward -= 5.0
            # // sfx: miss
            if self.lives > 0:
                self._spawn_ball()
            return reward

        paddle_center = pygame.math.Vector2(self.PADDLE_CENTER_X, self.PADDLE_CENTER_Y)
        half_len_vec = pygame.math.Vector2(self.PADDLE_LENGTH / 2, 0).rotate_rad(self.paddle_angle)
        p1 = paddle_center - half_len_vec
        p2 = paddle_center + half_len_vec

        if self.ball_vel.y > 0 and abs(self.ball_pos.y - self.PADDLE_CENTER_Y) < 20:
            line_vec = p2 - p1
            pt_vec = self.ball_pos - p1
            
            if line_vec.length_squared() == 0: return 0.0
            
            t = pt_vec.dot(line_vec) / line_vec.length_squared()
            t = max(0, min(1, t))
            
            closest_point = p1 + t * line_vec
            distance_vec = self.ball_pos - closest_point
            
            if distance_vec.length() < self.BALL_RADIUS:
                self.score += 1
                reward += 1.1  # +1 for score, +0.1 for hit
                
                self._create_particles(self.ball_pos, 20)
                # // sfx: paddle_hit

                if self.score > 0 and self.score % 5 == 0:
                    self.ball_base_speed += self.BALL_SPEED_INCREMENT
                
                paddle_normal = pygame.math.Vector2(0, -1).rotate_rad(self.paddle_angle)
                self.ball_vel.reflect_ip(paddle_normal)
                self.ball_vel = self.ball_vel.normalize() * self.ball_base_speed
                
                self.ball_pos += paddle_normal * self.BALL_RADIUS

        return reward

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(10, 25)
            self.particles.append([pos.copy(), vel, lifespan])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_FG, (0, 0, self.WIDTH, self.HEIGHT), 2)
        
        for pos, vel, life in self.particles:
            alpha = max(0, min(255, int(255 * (life / 25.0))))
            color = (*self.COLOR_FG, alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(pos.x - 2), int(pos.y - 2)))
            
        paddle_center = pygame.math.Vector2(self.PADDLE_CENTER_X, self.PADDLE_CENTER_Y)
        half_len_vec = pygame.math.Vector2(self.PADDLE_LENGTH / 2, 0).rotate_rad(self.paddle_angle)
        p1, p2 = paddle_center - half_len_vec, paddle_center + half_len_vec

        paddle_normal = pygame.math.Vector2(0, -1).rotate_rad(self.paddle_angle)
        incoming_vel_norm = self.ball_vel.copy()
        if incoming_vel_norm.length() > 0:
            incoming_vel_norm.normalize_ip()
        
        alignment = max(0, min(1, paddle_normal.dot(-incoming_vel_norm)))
        
        paddle_color = tuple(
            int(r + (s - r) * alignment) for r, s in zip(self.COLOR_RISKY, self.COLOR_SAFE)
        )
        
        pygame.draw.line(self.screen, paddle_color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), self.PADDLE_THICKNESS)
        
        if self.lives > 0:
            pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_FG)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_FG)
            
    def _render_ui(self):
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_FG)
        score_rect = score_text.get_rect(center=(self.WIDTH // 2, 30))
        self.screen.blit(score_text, score_rect)

        for i in range(self.lives):
            pos_x = 30 + i * (self.BALL_RADIUS * 2 + 10)
            pos_y = 30
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, self.BALL_RADIUS, self.COLOR_FG)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.BALL_RADIUS, self.COLOR_FG)
            
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_FG)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            self.screen.blit(end_text, end_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_FG)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()