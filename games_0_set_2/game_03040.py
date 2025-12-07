
# Generated: 2025-08-27T22:12:16.305874
# Source Brief: brief_03040.md
# Brief Index: 3040

        
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
    """
    A retro-futuristic rhythm racing game where the player controls a car,
    avoids obstacles, and scores points by drifting and moving to the beat.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to steer. Hold space to drift for a score multiplier. "
        "Steer on the beat (bottom bar) for bonus points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-futuristic rhythm racer. Drift through a neon highway, "
        "avoiding obstacles and syncing your moves to the beat to maximize your score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    BPM = 120
    BEAT_INTERVAL = int(FPS / (BPM / 60))

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (30, 20, 50)
    COLOR_TRACK = (0, 100, 200)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 100, 100, 100)
    COLOR_CAR = (255, 255, 0)
    COLOR_CAR_GLOW = (255, 255, 150, 100)
    COLOR_BEAT_BAR = (200, 0, 150)
    COLOR_BEAT_FLASH = (255, 150, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (255, 200, 255)

    # Car physics
    CAR_ACCEL = 0.5
    CAR_FRICTION = 0.95
    CAR_MAX_SPEED = 8.0
    DRIFT_ACCEL_MULT = 1.2
    DRIFT_FRICTION_MULT = 1.02
    CAR_WIDTH = 30
    CAR_HEIGHT = 30

    # Obstacle properties
    OBSTACLE_WIDTH = 50
    OBSTACLE_HEIGHT = 20
    INITIAL_OBSTACLE_SPEED = 3.0
    OBSTACLE_SPEED_INCREASE_PER_LAP = 0.5

    # Game rules
    TOTAL_LAPS = 3
    TIME_PER_LAP_S = 60
    MAX_TIME_FRAMES = TOTAL_LAPS * TIME_PER_LAP_S * FPS
    BEAT_WINDOW = 3 # Frames before/after beat for bonus

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.car_poly_base = np.array([
            [0, -self.CAR_HEIGHT / 2],
            [-self.CAR_WIDTH / 2, self.CAR_HEIGHT / 4],
            [self.CAR_WIDTH / 2, self.CAR_HEIGHT / 4]
        ])
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.crashed = False
        
        self.car_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.8], dtype=float)
        self.car_vel_x = 0.0
        
        self.lap = 1
        self.time_remaining_frames = self.MAX_TIME_FRAMES
        
        self.obstacles = []
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.obstacle_spawn_timer = 0
        self.obstacle_spawn_interval = self.FPS * 2

        self.is_drifting = False
        self.drift_multiplier = 1.0
        self.drift_combo_timer = 0

        self.beat_timer = 0
        self.on_beat = False
        
        self.particles = []
        self.bg_scroll_y = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        
        self._handle_input(movement, space_held)
        self._update_car_physics()
        self._update_obstacles()
        self._update_particles()
        self._update_game_state()

        lap_reward, beat_reward, drift_reward = self._calculate_rewards(movement)
        reward += lap_reward + beat_reward + drift_reward
        self.score += lap_reward + beat_reward + drift_reward
        
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += terminal_reward
        self.game_over = terminated

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        self.is_drifting = space_held
        accel = self.CAR_ACCEL * (self.DRIFT_ACCEL_MULT if self.is_drifting else 1.0)
        if movement == 3: self.car_vel_x -= accel
        elif movement == 4: self.car_vel_x += accel

    def _update_car_physics(self):
        friction = self.CAR_FRICTION * (self.DRIFT_FRICTION_MULT if self.is_drifting else 1.0)
        self.car_vel_x *= friction
        self.car_vel_x = np.clip(self.car_vel_x, -self.CAR_MAX_SPEED, self.CAR_MAX_SPEED)
        self.car_pos[0] += self.car_vel_x
        if self.car_pos[0] < 0: self.car_pos[0] += self.SCREEN_WIDTH
        elif self.car_pos[0] > self.SCREEN_WIDTH: self.car_pos[0] -= self.SCREEN_WIDTH

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs[1] += self.obstacle_speed
        self.obstacles = [obs for obs in self.obstacles if obs[1] < self.SCREEN_HEIGHT + self.OBSTACLE_HEIGHT]
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            new_x = self.np_random.integers(0, self.SCREEN_WIDTH - self.OBSTACLE_WIDTH)
            self.obstacles.append([new_x, -self.OBSTACLE_HEIGHT])
            self.obstacle_spawn_timer = max(15, self.obstacle_spawn_interval - (self.lap * 10))

    def _update_particles(self):
        if self.is_drifting and abs(self.car_vel_x) > 1.0:
            # sfx: drift_sound
            for _ in range(2):
                p_pos = self.car_pos + [0, self.CAR_HEIGHT / 4]
                p_vel = [self.np_random.uniform(-1, 1) - self.car_vel_x * 0.1, self.np_random.uniform(0, 2)]
                self.particles.append({'pos': p_pos, 'vel': p_vel, 'life': self.np_random.integers(15, 30), 'radius': self.np_random.uniform(2, 5)})
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_game_state(self):
        self.time_remaining_frames -= 1
        current_lap = self.TOTAL_LAPS - (self.time_remaining_frames // (self.TIME_PER_LAP_S * self.FPS)) + 1
        if current_lap > self.lap and self.lap <= self.TOTAL_LAPS:
            self.lap = current_lap
            self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE_PER_LAP
            # sfx: lap_complete_sound
        self.beat_timer = (self.beat_timer + 1) % self.BEAT_INTERVAL
        self.on_beat = self.beat_timer < self.BEAT_WINDOW or self.beat_timer > self.BEAT_INTERVAL - self.BEAT_WINDOW
        if self.is_drifting:
            self.drift_combo_timer += 1
            self.drift_multiplier = 1.0 + (self.drift_combo_timer / (self.FPS * 2.0))
        else:
            self.drift_combo_timer = 0
            self.drift_multiplier = 1.0

    def _calculate_rewards(self, movement):
        lap_reward = 0.0
        if self.time_remaining_frames % (self.TIME_PER_LAP_S * self.FPS) == 0 and self.time_remaining_frames < self.MAX_TIME_FRAMES:
             if self.lap <= self.TOTAL_LAPS:
                 lap_reward = 1.0
        beat_reward = 0.1 if self.on_beat and (movement in [3, 4]) else 0.0
        drift_reward = 0.005 * self.drift_multiplier if self.is_drifting else 0.0
        return lap_reward, beat_reward, drift_reward

    def _check_termination(self):
        car_rect = pygame.Rect(self.car_pos[0] - self.CAR_WIDTH / 2, self.car_pos[1] - self.CAR_HEIGHT / 2, self.CAR_WIDTH, self.CAR_HEIGHT)
        for obs in self.obstacles:
            if car_rect.colliderect(pygame.Rect(obs[0], obs[1], self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT)):
                # sfx: crash_sound
                self.crashed = True
                return True, -100.0
        if self.time_remaining_frames <= 0:
            if not self.crashed:
                # sfx: win_sound
                return True, 50.0 + (self.score / 10.0)
            return True, 0.0
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_track()
        self._render_obstacles()
        self._render_particles()
        self._render_car()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lap": self.lap, "time_remaining": self.time_remaining_frames / self.FPS, "drift_multiplier": self.drift_multiplier}

    def _render_background(self):
        self.bg_scroll_y = (self.bg_scroll_y + 0.5) % 40
        for i in range(self.SCREEN_HEIGHT // 40 + 2):
            y = int(i * 40 + self.bg_scroll_y - 40)
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)
        for i in range(self.SCREEN_WIDTH // 40 + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (int(i * 40), 0), (int(i * 40), self.SCREEN_HEIGHT), 1)

    def _render_track(self):
        track_width = self.SCREEN_WIDTH * 0.9
        left_x, right_x = (self.SCREEN_WIDTH - track_width) / 2, self.SCREEN_WIDTH - (self.SCREEN_WIDTH - track_width) / 2
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.gfxdraw.line(self.screen, int(left_x), i, int(left_x), i + 10, self.COLOR_TRACK)
            pygame.gfxdraw.line(self.screen, int(right_x), i, int(right_x), i + 10, self.COLOR_TRACK)

    def _render_obstacles(self):
        for x, y in self.obstacles:
            rect = pygame.Rect(int(x), int(y), self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT)
            glow_surf = pygame.Surface(rect.inflate(10, 10).size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_OBSTACLE_GLOW, glow_surf.get_rect(), border_radius=8)
            self.screen.blit(glow_surf, rect.topleft - np.array([5, 5]))
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=5)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 30.0))
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*self.COLOR_PARTICLE, int(alpha)))

    def _render_car(self):
        pos = (int(self.car_pos[0]), int(self.car_pos[1]))
        glow_radius = int(self.CAR_WIDTH * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_CAR_GLOW)
        angle_rad = self.car_vel_x * 0.05
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_poly = (self.car_poly_base @ rot_matrix) + self.car_pos
        pygame.gfxdraw.aapolygon(self.screen, rotated_poly, self.COLOR_CAR)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_poly, self.COLOR_CAR)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        lap_text = self.font_small.render(f"LAP: {min(self.lap, self.TOTAL_LAPS)}/{self.TOTAL_LAPS}", True, self.COLOR_TEXT)
        time_s = max(0, self.time_remaining_frames // self.FPS)
        time_text = self.font_small.render(f"TIME: {time_s}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lap_text, (10, 35))
        self.screen.blit(time_text, (10, 60))
        if self.drift_multiplier > 1.1:
            mult_text = self.font_large.render(f"{self.drift_multiplier:.1f}x", True, self.COLOR_PARTICLE)
            self.screen.blit(mult_text, mult_text.get_rect(center=(self.SCREEN_WIDTH / 2, 50)))
        bar_y, bar_width = self.SCREEN_HEIGHT - 20, self.SCREEN_WIDTH * 0.5
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, 10), border_radius=5)
        fill_width = bar_width * (self.beat_timer / self.BEAT_INTERVAL)
        fill_color = self.COLOR_BEAT_FLASH if self.on_beat else self.COLOR_BEAT_BAR
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_width, 10), border_radius=5)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    running = True
    
    while running:
        movement, space_held, shift_held = 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
            
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            obs, info = env.reset()
            pygame.time.wait(1000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()