
# Generated: 2025-08-27T20:49:45.821953
# Source Brief: brief_02586.md
# Brief Index: 2586

        
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

    # Corrected user-facing strings to match the implemented game
    user_guide = "Controls: ← to move up, → to move down. Dodge the obstacles to reach the finish line."
    game_description = "High-speed side-scroller. Navigate a neon tunnel, dodging obstacles to survive and reach the finish line."

    # Frames auto-advance for smooth real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TRACK_LENGTH = 5000
    MAX_STEPS = 5000
    TARGET_TIME_STEPS = 1800  # 60 seconds at 30fps

    # Colors
    COLOR_BG = (10, 0, 20)
    COLOR_PLAYER = (0, 191, 255)
    COLOR_PLAYER_GLOW = (0, 75, 128)
    COLOR_OBSTACLE_SQUARE = (255, 100, 0)
    COLOR_OBSTACLE_CIRCLE = (255, 0, 100)
    COLOR_OBSTACLE_TRIANGLE = (255, 200, 0)
    COLOR_OBSTACLE_GLOW_FACTOR = 0.5
    COLOR_TRACK = (80, 80, 100)
    COLOR_FINISH = (0, 255, 0)
    COLOR_TEXT = (220, 220, 220)

    # Player
    PLAYER_X_POS = 100
    PLAYER_SIZE = 12
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = 0.92
    PLAYER_MAX_SPEED = 8

    # World
    FORWARD_SPEED = 5  # Pixels per frame the world scrolls
    TRACK_Y_TOP = 50
    TRACK_Y_BOTTOM = 350

    # Obstacles
    OBSTACLE_BASE_SPEED = 2.0
    OBSTACLE_SPEED_INCREASE_INTERVAL = 500
    OBSTACLE_SPEED_INCREASE_AMOUNT = 0.05
    OBSTACLE_BASE_SPAWN_INTERVAL = 40  # frames

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Initialize state variables
        self.np_random = None
        self.player_y = 0
        self.player_vy = 0
        self.distance_traveled = 0
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacle_current_speed = 0
        self.obstacle_spawn_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_y = self.SCREEN_HEIGHT / 2
        self.player_vy = 0
        self.distance_traveled = 0
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacle_current_speed = self.OBSTACLE_BASE_SPEED
        self.obstacle_spawn_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]

        self._update_player(movement)

        self.steps += 1
        self.distance_traveled += self.FORWARD_SPEED
        self._update_difficulty()
        self._update_obstacles()
        self._update_particles()

        collision = self._check_collisions()
        finished = self.distance_traveled >= self.TRACK_LENGTH
        max_steps_reached = self.steps >= self.MAX_STEPS

        terminated = collision or finished or max_steps_reached
        reward = self._calculate_reward(collision, finished)
        self.score += reward

        if terminated:
            self.game_over = True
            if collision:
                # Sound: Player Explosion
                self._create_particles(
                    (self.PLAYER_X_POS, self.player_y), 50, self.COLOR_PLAYER, 5, 30
                )

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        self.player_vy *= self.PLAYER_FRICTION

        if movement == 3:  # Left -> Up
            self.player_vy -= self.PLAYER_ACCEL
        elif movement == 4:  # Right -> Down
            self.player_vy += self.PLAYER_ACCEL

        self.player_vy = np.clip(self.player_vy, -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)
        self.player_y += self.player_vy
        self.player_y = np.clip(self.player_y, self.TRACK_Y_TOP + self.PLAYER_SIZE, self.TRACK_Y_BOTTOM - self.PLAYER_SIZE)

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.OBSTACLE_SPEED_INCREASE_INTERVAL == 0:
            self.obstacle_current_speed += self.OBSTACLE_SPEED_INCREASE_AMOUNT

    def _update_obstacles(self):
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            self._spawn_obstacle()
            spawn_interval_reduction = self.distance_traveled / self.TRACK_LENGTH * 20
            self.obstacle_spawn_timer = self.OBSTACLE_BASE_SPAWN_INTERVAL - spawn_interval_reduction

        new_obstacles = []
        for obs in self.obstacles:
            obs['pos'][0] -= self.FORWARD_SPEED
            obs['pos'][0] += obs['vel'][0]
            obs['pos'][1] += obs['vel'][1]

            if obs['type'] != 'triangle':
                if obs['pos'][1] <= self.TRACK_Y_TOP + obs['size'] or obs['pos'][1] >= self.TRACK_Y_BOTTOM - obs['size']:
                    obs['vel'][1] *= -1
                    obs['pos'][1] = np.clip(obs['pos'][1], self.TRACK_Y_TOP + obs['size'], self.TRACK_Y_BOTTOM - obs['size'])

            if obs['pos'][0] > -obs['size']:
                new_obstacles.append(obs)
        self.obstacles = new_obstacles

    def _spawn_obstacle(self):
        if self.distance_traveled < 300 or self.distance_traveled > self.TRACK_LENGTH - 500:
            return

        obstacle_type = self.np_random.choice(['square', 'circle', 'triangle'])
        size = self.np_random.integers(10, 25)
        pos_x = self.distance_traveled + self.SCREEN_WIDTH
        pos_y = self.np_random.uniform(self.TRACK_Y_TOP + size, self.TRACK_Y_BOTTOM - size)

        vel = [0, 0]
        if obstacle_type == 'square':
            vel = [-self.obstacle_current_speed * 0.5, self.np_random.uniform(-1, 1)]
        elif obstacle_type == 'circle':
            vel = [-self.obstacle_current_speed, self.np_random.uniform(-2, 2)]

        self.obstacles.append({'type': obstacle_type, 'pos': [pos_x, pos_y], 'size': size, 'vel': vel})

    def _check_collisions(self):
        player_pos = np.array([self.PLAYER_X_POS, self.player_y])
        for obs in self.obstacles:
            obs_screen_x = obs['pos'][0] - self.distance_traveled
            obs_pos = np.array([obs_screen_x, obs['pos'][1]])

            dist = np.linalg.norm(player_pos - obs_pos)
            if dist < obs['size'] + self.PLAYER_SIZE * 0.8:
                return True
        return False

    def _calculate_reward(self, collision, finished):
        if collision:
            return -10.0

        if finished:
            base_reward = 50.0
            time_bonus = 0
            if self.steps <= self.TARGET_TIME_STEPS:
                time_bonus = 100.0
            else:
                over_time = self.steps - self.TARGET_TIME_STEPS
                penalty_factor = min(1.0, over_time / self.TARGET_TIME_STEPS)
                time_bonus = 100.0 * (1.0 - penalty_factor)
            return base_reward + time_bonus

        return 0.1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_track()
        self._render_finish_line()
        self._render_obstacles()
        self._render_particles()
        self._render_player()

    def _render_track(self):
        self._draw_glowing_line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_TOP), (self.SCREEN_WIDTH, self.TRACK_Y_TOP), 3)
        self._draw_glowing_line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_BOTTOM), (self.SCREEN_WIDTH, self.TRACK_Y_BOTTOM), 3)

    def _render_finish_line(self):
        finish_x = self.TRACK_LENGTH - self.distance_traveled
        if 0 < finish_x < self.SCREEN_WIDTH:
            self._draw_glowing_line(self.screen, self.COLOR_FINISH, (finish_x, self.TRACK_Y_TOP), (finish_x, self.TRACK_Y_BOTTOM), 5)

    def _render_obstacles(self):
        for obs in self.obstacles:
            screen_x = obs['pos'][0] - self.distance_traveled
            screen_y = obs['pos'][1]
            size = obs['size']

            if screen_x > self.SCREEN_WIDTH + size or screen_x < -size:
                continue

            if obs['type'] == 'square':
                color = self.COLOR_OBSTACLE_SQUARE
                glow_color = tuple(int(c * self.COLOR_OBSTACLE_GLOW_FACTOR) for c in color)
                rect = pygame.Rect(int(screen_x - size / 2), int(screen_y - size / 2), int(size), int(size))
                self._draw_glowing_rect(self.screen, color, glow_color, rect, 4)
            elif obs['type'] == 'circle':
                color = self.COLOR_OBSTACLE_CIRCLE
                glow_color = tuple(int(c * self.COLOR_OBSTACLE_GLOW_FACTOR) for c in color)
                self._draw_glowing_circle(self.screen, color, glow_color, (int(screen_x), int(screen_y)), int(size), 4)
            elif obs['type'] == 'triangle':
                color = self.COLOR_OBSTACLE_TRIANGLE
                glow_color = tuple(int(c * self.COLOR_OBSTACLE_GLOW_FACTOR) for c in color)
                points = [
                    (screen_x - size / 2, screen_y + size / 2),
                    (screen_x - size / 2, screen_y - size / 2),
                    (screen_x + size / 2, screen_y)
                ]
                self._draw_glowing_polygon(self.screen, color, glow_color, points, int(size * 1.5))

    def _render_player(self):
        p_size = self.PLAYER_SIZE
        p_y = int(self.player_y)
        p_x = self.PLAYER_X_POS

        points = [
            (p_x + p_size, p_y),
            (p_x - p_size / 2, p_y - p_size / 2),
            (p_x - p_size / 2, p_y + p_size / 2)
        ]
        self._draw_glowing_polygon(self.screen, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, points, int(p_size * 2))

    def _render_ui(self):
        dist_text = self.font_large.render(f"DIST: {int(self.distance_traveled)}/{self.TRACK_LENGTH}", True, self.COLOR_TEXT)
        self.screen.blit(dist_text, (10, 10))

        time_sec = self.steps / 30.0
        time_text = self.font_large.render(f"TIME: {time_sec:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, self.SCREEN_HEIGHT - score_text.get_height() - 10))

    def _create_particles(self, pos, count, color, speed_max, life_max):
        # Sound: Particle burst
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, life_max)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _render_particles(self):
        for p in self.particles:
            screen_x = p['pos'][0] - self.distance_traveled
            alpha = 255 * (p['life'] / p['max_life'])
            color = (*p['color'], alpha)
            size = max(1, int(3 * (p['life'] / p['max_life'])))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(screen_x - size), int(p['pos'][1] - size)), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_glowing_polygon(self, surface, color, glow_color, points, glow_radius):
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        self._draw_glowing_circle(surface, color, glow_color, (center_x, center_y), glow_radius, 4)
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(surface, int_points, color)
        pygame.gfxdraw.filled_polygon(surface, int_points, color)

    def _draw_glowing_circle(self, surface, color, glow_color, center, radius, glow_radius):
        x, y = int(center[0]), int(center[1])
        for i in range(glow_radius, 0, -1):
            alpha = int(120 * (1 - i / glow_radius)**2)
            pygame.gfxdraw.filled_circle(surface, x, y, radius + i, (*glow_color, alpha))
        pygame.gfxdraw.aacircle(surface, x, y, radius, color)
        pygame.gfxdraw.filled_circle(surface, x, y, radius, color)

    def _draw_glowing_rect(self, surface, color, glow_color, rect, glow_radius):
        for i in range(glow_radius, 0, -1):
            alpha = int(100 * (1 - i / glow_radius)**2)
            glow_rect = rect.inflate(i * 2, i * 2)
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*glow_color, alpha), (0, 0, *glow_rect.size), border_radius=i+3)
            surface.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(surface, color, rect, border_radius=3)

    def _draw_glowing_line(self, surface, color, start, end, width):
        glow_color = tuple(min(255, c + 50) for c in color)
        pygame.draw.line(surface, (*glow_color, 50), start, end, width + 6)
        pygame.draw.line(surface, (*glow_color, 70), start, end, width + 3)
        pygame.draw.aaline(surface, color, start, end, True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_traveled": int(self.distance_traveled),
        }

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()