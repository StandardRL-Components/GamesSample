
# Generated: 2025-08-28T02:06:44.649701
# Source Brief: brief_04336.md
# Brief Index: 4336

        
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
        "Controls: ↑ to accelerate, ←→ to turn, and ↓ to reverse. Hold Shift to brake and execute sharp drifts."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Snail Racer! Guide your snail through three challenging stages, avoiding obstacles and racing against the clock. Master the drift to get the best time!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    TOTAL_STAGES = 3

    # Colors
    COLOR_BG_STAGES = [(40, 50, 40), (50, 45, 40), (40, 40, 55)]
    COLOR_SNAIL_BODY = (50, 180, 255)
    COLOR_SNAIL_SHELL = (255, 160, 80)
    COLOR_SNAIL_GLOW = (150, 220, 255)
    COLOR_OBSTACLE_ROCK = (110, 100, 90)
    COLOR_OBSTACLE_PLANT = (30, 130, 60)
    COLOR_START = (100, 255, 100, 100)
    COLOR_FINISH = (255, 100, 100, 100)
    COLOR_TRAIL = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    
    # Snail Physics
    MAX_SPEED = 6.0
    ACCELERATION = 0.15
    TURN_RATE = 0.07
    DRIFT_TURN_MULTIPLIER = 2.5
    FRICTION = 0.985
    BRAKE_FRICTION = 0.92
    REVERSE_ACCEL = 0.05
    COLLISION_DAMPING = 0.5

    # Game Rules
    STAGE_TIME_LIMIT = 60  # seconds
    BASE_OBSTACLES = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.game_over = True # Will be set to False in reset
        self.total_stages = self.TOTAL_STAGES

        # These will be initialized in reset
        self.steps = 0
        self.score = 0
        self.current_stage = 0
        self.time_remaining = 0
        self.player_pos = None
        self.player_vel = None
        self.player_angle = 0
        self.obstacles = []
        self.start_rect = None
        self.finish_rect = None
        self.particles = []
        self.collision_particles = []

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_over = False
        self.steps = 0
        self.score = 0
        self.current_stage = 0
        self._reset_for_stage()
        
        return self._get_observation(), self._get_info()

    def _reset_for_stage(self):
        """Resets the environment for the start of a new stage."""
        self.time_remaining = self.STAGE_TIME_LIMIT * self.FPS
        self.particles = []
        self.collision_particles = []

        if self.current_stage == 0:
            self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
            self.player_angle = -math.pi / 2
            self.start_rect = pygame.Rect(0, self.HEIGHT - 70, self.WIDTH, 20)
            self.finish_rect = pygame.Rect(0, 0, self.WIDTH, 50)
        elif self.current_stage == 1:
            self.player_pos = pygame.Vector2(50, self.HEIGHT / 2)
            self.player_angle = 0
            self.start_rect = pygame.Rect(30, 0, 20, self.HEIGHT)
            self.finish_rect = pygame.Rect(self.WIDTH - 50, 0, 50, self.HEIGHT)
        elif self.current_stage == 2:
            self.player_pos = pygame.Vector2(self.WIDTH / 2, 50)
            self.player_angle = math.pi / 2
            self.start_rect = pygame.Rect(0, 30, self.WIDTH, 20)
            self.finish_rect = pygame.Rect(0, self.HEIGHT - 50, self.WIDTH, 50)
        
        self.player_vel = pygame.Vector2(0, 0)
        self._generate_obstacles()

    def _generate_obstacles(self):
        self.obstacles = []
        num_obstacles = int(self.BASE_OBSTACLES * (1 + 0.1 * self.current_stage))
        
        for _ in range(num_obstacles):
            while True:
                size = self.np_random.integers(15, 35)
                pos = pygame.Vector2(
                    self.np_random.integers(size, self.WIDTH - size),
                    self.np_random.integers(size, self.HEIGHT - size)
                )
                new_obstacle_rect = pygame.Rect(pos.x - size/2, pos.y - size/2, size, size)
                
                if (not new_obstacle_rect.colliderect(self.finish_rect) and
                    not new_obstacle_rect.colliderect(self.start_rect) and
                    self.player_pos.distance_to(pos) > 100):
                    
                    color = self.COLOR_OBSTACLE_ROCK if self.np_random.random() > 0.5 else self.COLOR_OBSTACLE_PLANT
                    self.obstacles.append((new_obstacle_rect, color))
                    break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._update_player(movement, shift_held)
        self._update_game_state()
        collision_count = self._handle_collisions()
        
        self.steps += 1
        self.time_remaining -= 1
        
        reward = 0
        terminated = False
        
        # Continuous rewards
        speed = self.player_vel.length()
        if speed > 0.1:
            reward += 0.02 * (speed / self.MAX_SPEED)
        if collision_count > 0:
            reward -= 1.0 * collision_count
        
        # Event-based rewards & termination checks
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 10, 20, 20)
        
        if player_rect.colliderect(self.finish_rect):
            # --- Stage Clear ---
            # sfx: stage_clear.wav
            time_bonus = 50 * (self.time_remaining / (self.STAGE_TIME_LIMIT * self.FPS))
            reward += 10 + time_bonus
            self.current_stage += 1
            
            if self.current_stage >= self.total_stages:
                # --- Game Win ---
                # sfx: game_win.wav
                reward += 50
                terminated = True
                self.game_over = True
            else:
                # --- Advance to next stage ---
                self._reset_for_stage()

        elif self.time_remaining <= 0:
            # --- Game Over by Timeout ---
            # sfx: game_over.wav
            reward = -100
            terminated = True
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement, shift_held):
        if shift_held:
            self.player_vel *= self.BRAKE_FRICTION
        else:
            self.player_vel *= self.FRICTION

        turn_speed = self.TURN_RATE * (self.DRIFT_TURN_MULTIPLIER if shift_held else 1.0)
        if movement == 3: # Left
            self.player_angle -= turn_speed
        if movement == 4: # Right
            self.player_angle += turn_speed
        
        if movement == 1: # Up
            # sfx: accelerate.wav (loop)
            accel_vec = pygame.Vector2(math.cos(self.player_angle), math.sin(self.player_angle)) * self.ACCELERATION
            self.player_vel += accel_vec
        elif movement == 2: # Down
            accel_vec = pygame.Vector2(math.cos(self.player_angle), math.sin(self.player_angle)) * self.REVERSE_ACCEL
            self.player_vel -= accel_vec

        speed = self.player_vel.length()
        if speed > self.MAX_SPEED:
            self.player_vel.scale_to_length(self.MAX_SPEED)
        if speed < -self.MAX_SPEED / 2:
             self.player_vel.scale_to_length(-self.MAX_SPEED / 2)
    
    def _update_game_state(self):
        self.player_pos += self.player_vel

        if self.player_vel.length() > 2.0:
            particle_pos = self.player_pos - self.player_vel.normalize() * 10
            self.particles.append([particle_pos, self.player_vel.copy() * -0.1, self.np_random.integers(10, 20)])
        
        self.particles = [[p[0] + p[1], p[1] * 0.9, p[2] - 1] for p in self.particles if p[2] > 0]
        self.collision_particles = [[p[0] + p[1], p[1] * 0.95, p[2] - 1] for p in self.collision_particles if p[2] > 0]


    def _handle_collisions(self):
        collision_count = 0
        player_radius = 12

        if self.player_pos.x < player_radius:
            self.player_pos.x = player_radius
            self.player_vel.x *= -self.COLLISION_DAMPING
            collision_count += 1
        if self.player_pos.x > self.WIDTH - player_radius:
            self.player_pos.x = self.WIDTH - player_radius
            self.player_vel.x *= -self.COLLISION_DAMPING
            collision_count += 1
        if self.player_pos.y < player_radius:
            self.player_pos.y = player_radius
            self.player_vel.y *= -self.COLLISION_DAMPING
            collision_count += 1
        if self.player_pos.y > self.HEIGHT - player_radius:
            self.player_pos.y = self.HEIGHT - player_radius
            self.player_vel.y *= -self.COLLISION_DAMPING
            collision_count += 1

        player_rect = pygame.Rect(self.player_pos.x - player_radius, self.player_pos.y - player_radius, player_radius*2, player_radius*2)
        for obs_rect, _ in self.obstacles:
            if player_rect.colliderect(obs_rect):
                # sfx: thump.wav
                collision_count += 1
                self.player_vel *= self.COLLISION_DAMPING
                
                dx = self.player_pos.x - obs_rect.centerx
                dy = self.player_pos.y - obs_rect.centery
                if abs(dx) > abs(dy):
                    self.player_pos.x += 5 * np.sign(dx)
                else:
                    self.player_pos.y += 5 * np.sign(dy)

                for _ in range(10):
                    angle = self.np_random.random() * 2 * math.pi
                    speed = self.np_random.random() * 2 + 1
                    vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                    self.collision_particles.append([self.player_pos.copy(), vel, 20])
        
        return collision_count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG_STAGES[self.current_stage % len(self.COLOR_BG_STAGES)])
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        s = pygame.Surface((self.start_rect.width, self.start_rect.height), pygame.SRCALPHA)
        s.fill(self.COLOR_START)
        self.screen.blit(s, self.start_rect.topleft)

        s = pygame.Surface((self.finish_rect.width, self.finish_rect.height), pygame.SRCALPHA)
        s.fill(self.COLOR_FINISH)
        self.screen.blit(s, self.finish_rect.topleft)

        for rect, color in self.obstacles:
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, width=3, border_radius=5)

        for p in self.particles:
            size = max(0, int(p[2] / 4))
            pygame.draw.circle(self.screen, self.COLOR_TRAIL, p[0], size)
        
        for p in self.collision_particles:
            size = max(0, int(p[2] / 5))
            pygame.draw.circle(self.screen, self.COLOR_OBSTACLE_ROCK, p[0], size)

        if not self.game_over:
            self._render_player()

    def _render_player(self):
        pos = self.player_pos
        angle = self.player_angle
        
        glow_radius = 20 + math.sin(self.steps * 0.2) * 2
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(glow_radius), (*self.COLOR_SNAIL_GLOW, 50))
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(glow_radius), (*self.COLOR_SNAIL_GLOW, 70))

        body_radius = 12
        pygame.draw.circle(self.screen, self.COLOR_SNAIL_BODY, (int(pos.x), int(pos.y)), body_radius)
        pygame.draw.circle(self.screen, (255,255,255), (int(pos.x), int(pos.y)), body_radius, 1)

        shell_offset = -self.player_vel.normalize() * 5 if self.player_vel.length() > 0.1 else pygame.Vector2(-5,0)
        shell_bob = math.sin(self.steps * 0.3) * 2
        shell_pos = pos + shell_offset + pygame.Vector2(0, shell_bob)
        pygame.draw.circle(self.screen, self.COLOR_SNAIL_SHELL, (int(shell_pos.x), int(shell_pos.y)), 10)
        pygame.draw.circle(self.screen, (0,0,0, 50), (int(shell_pos.x), int(shell_pos.y)), 10, 2)

        eye_dir = pygame.Vector2(math.cos(angle), math.sin(angle))
        eye_base = pos + eye_dir * 8
        eye_perp = pygame.Vector2(-eye_dir.y, eye_dir.x)
        
        eye1_pos = eye_base + eye_perp * 4
        eye2_pos = eye_base - eye_perp * 4
        
        pygame.draw.circle(self.screen, (255, 255, 255), (int(eye1_pos.x), int(eye1_pos.y)), 4)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(eye2_pos.x), int(eye2_pos.y)), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (int(eye1_pos.x + eye_dir.x), int(eye1_pos.y + eye_dir.y)), 2)
        pygame.draw.circle(self.screen, (0, 0, 0), (int(eye2_pos.x + eye_dir.x), int(eye2_pos.y + eye_dir.y)), 2)

    def _render_ui(self):
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_str = f"{int(self.time_remaining / self.FPS // 60):02}:{int(self.time_remaining / self.FPS % 60):02}"
        time_color = (255, 100, 100) if self.time_remaining < 10 * self.FPS else self.COLOR_TEXT
        time_text = self.font_large.render(time_str, True, time_color)
        self.screen.blit(time_text, time_text.get_rect(topright=(self.WIDTH - 10, 10)))

        stage_text = self.font_small.render(f"Stage: {self.current_stage + 1}/{self.total_stages}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, stage_text.get_rect(topright=(self.WIDTH - 10, 50)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage + 1,
            "time_left_seconds": self.time_remaining / self.FPS,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        self.reset()
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

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Snail Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(env.user_guide)

    while not terminated:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        clock.tick(env.FPS)

    env.close()
    print(f"Game Over! Final Score: {info['score']}")