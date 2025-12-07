import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to set draw direction. No-op extends the line. "
        "Space for a long line, Shift for a short line."
    )

    game_description = (
        "Draw a track for a sledder to ride from the green start to the red finish. "
        "Balance speed and safety to get a high score. Crashing into the terrain ends the game."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Colors ---
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (40, 20, 60)
        self.COLOR_TERRAIN = (30, 40, 70)
        self.COLOR_TRACK = (255, 255, 255)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_RIDER_GLOW = (200, 200, 255)
        self.COLOR_START = (0, 255, 0)
        self.COLOR_FINISH = (255, 0, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SPARK = (255, 220, 180)

        # --- Fonts ---
        try:
            self.FONT_UI = pygame.font.SysFont("Consolas", 20)
            self.FONT_SPEED = pygame.font.SysFont("Consolas", 24)
        except pygame.error:
            self.FONT_UI = pygame.font.SysFont(None, 24)
            self.FONT_SPEED = pygame.font.SysFont(None, 28)

        # --- Game Constants ---
        self.START_X = 50
        self.FINISH_X = self.WIDTH - 50
        self.RIDER_RADIUS = 6
        self.GRAVITY = pygame.Vector2(0, 0.15)
        self.FRICTION = 0.995
        self.MAX_STEPS = 1000
        self.PHYSICS_SUBSTEPS = 8
        self.BASE_DRAW_LENGTH = 20

        # --- State variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.rider_pos = pygame.Vector2(0, 0)
        self.rider_vel = pygame.Vector2(0, 0)
        self.rider_on_track = False
        self.track_points = []
        self.terrain_points = []
        self.draw_direction = pygame.Vector2(1, 0)
        self.particles = []
        self.rider_trail = deque(maxlen=20)
        self.last_checkpoint_x = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_terrain()
        start_y = self._get_terrain_y(self.START_X) - 20

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.rider_pos = pygame.Vector2(self.START_X, start_y)
        self.rider_vel = pygame.Vector2(0, 0)
        self.rider_on_track = False
        
        self.track_points = [pygame.Vector2(self.START_X, start_y)]
        self.draw_direction = pygame.Vector2(1, 0)
        
        self.particles = []
        self.rider_trail.clear()
        
        self.last_checkpoint_x = self.START_X

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        old_rider_pos = self.rider_pos.copy()
        
        self._handle_action(action)
        crashed = self._simulate_physics()
        
        new_rider_pos = self.rider_pos.copy()

        reward = self._calculate_reward(crashed, old_rider_pos, new_rider_pos)
        self.score += reward
        
        terminated = self._check_termination(crashed)
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement != 0:
            directions = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            self.draw_direction = pygame.Vector2(directions[movement]).normalize()

        length = self.BASE_DRAW_LENGTH
        if space_held:
            length *= 2.0
        if shift_held:
            length *= 0.5
        
        last_point = self.track_points[-1]
        new_point = last_point + self.draw_direction * length
        
        # Clamp to screen bounds
        new_point.x = np.clip(new_point.x, 0, self.WIDTH)
        new_point.y = np.clip(new_point.y, 0, self.HEIGHT)
        
        self.track_points.append(new_point)

    def _simulate_physics(self):
        crashed = False
        for _ in range(self.PHYSICS_SUBSTEPS):
            if crashed: break

            self.rider_vel += self.GRAVITY
            self.rider_vel *= self.FRICTION
            self.rider_pos += self.rider_vel

            # Rider trail update
            if _ % 2 == 0:
                self.rider_trail.append(self.rider_pos.copy())

            # Terrain collision
            terrain_y = self._get_terrain_y(self.rider_pos.x)
            if self.rider_pos.y + self.RIDER_RADIUS > terrain_y:
                crashed = True
                self._create_sparks(self.rider_pos, 30)
                # Sound: Crash
                break

            # Track collision
            self.rider_on_track = False
            for i in range(len(self.track_points) - 1):
                p1 = self.track_points[i]
                p2 = self.track_points[i+1]
                
                closest_point, dist_sq = self._closest_point_on_segment(self.rider_pos, p1, p2)
                
                if dist_sq < self.RIDER_RADIUS ** 2:
                    self.rider_on_track = True
                    displacement = self.rider_pos - closest_point

                    # Only perform collision response if there is a non-zero displacement vector
                    # to avoid normalizing a zero vector.
                    if displacement.length_squared() > 1e-12:
                        dist = displacement.length()
                        overlap = self.RIDER_RADIUS - dist
                        
                        normal = displacement.normalize()
                        self.rider_pos += normal * overlap
                        
                        # Reflect velocity
                        vel_dot_normal = self.rider_vel.dot(normal)
                        if vel_dot_normal < 0:
                            self.rider_vel -= 2 * vel_dot_normal * normal
                        
                        # Create sparks on hard contact
                        if abs(vel_dot_normal) > 2:
                            self._create_sparks(closest_point, int(abs(vel_dot_normal)))

        return crashed

    def _calculate_reward(self, crashed, old_pos, new_pos):
        reward = 0.0
        
        # Penalty for existing
        reward -= 0.01

        # Reward for forward progress
        progress = new_pos.x - old_pos.x
        if progress > 0:
            reward += 0.1 * progress

        # Checkpoint reward
        current_checkpoint_idx = int(new_pos.x / 200)
        last_checkpoint_idx = int(self.last_checkpoint_x / 200)
        if current_checkpoint_idx > last_checkpoint_idx:
            reward += 10
            self.last_checkpoint_x = new_pos.x
        
        # Terminal rewards
        if crashed:
            reward = -5.0
        elif new_pos.x >= self.FINISH_X:
            reward = 100.0
        elif self.rider_pos.y > self.HEIGHT or self.rider_pos.y < 0:
            reward = -5.0

        return reward

    def _check_termination(self, crashed):
        if crashed:
            return True
        if self.rider_pos.x >= self.FINISH_X:
            return True
        if not (0 <= self.rider_pos.y <= self.HEIGHT):
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Terrain
        pygame.draw.polygon(self.screen, self.COLOR_TERRAIN, self.terrain_points + [(self.WIDTH, self.HEIGHT), (0, self.HEIGHT)])

        # Track
        if len(self.track_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_TRACK, False, self.track_points, 3)
        
        # Start and Finish lines
        start_y = self._get_terrain_y(self.START_X)
        finish_y = self._get_terrain_y(self.FINISH_X)
        pygame.draw.line(self.screen, self.COLOR_START, (self.START_X, start_y), (self.START_X, 0), 5)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_X, finish_y), (self.FINISH_X, 0), 5)

        # Particles
        self._update_and_draw_particles()

        # Rider Trail
        for i, pos in enumerate(self.rider_trail):
            alpha = int(255 * (i / len(self.rider_trail)))
            if alpha > 0:
                radius = int(self.RIDER_RADIUS * 0.5 * (i / len(self.rider_trail)))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, (*self.COLOR_RIDER_GLOW, alpha))

        # Rider
        rider_x, rider_y = int(self.rider_pos.x), int(self.rider_pos.y)
        glow_radius = int(self.RIDER_RADIUS * (1.5 + self.rider_vel.length() * 0.1))
        glow_alpha = 100 if self.rider_on_track else 50
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, glow_radius, (*self.COLOR_RIDER_GLOW, glow_alpha))
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)

    def _render_ui(self):
        score_text = self.FONT_UI.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.FONT_UI.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        speed = self.rider_vel.length() * 10
        speed_text = self.FONT_SPEED.render(f"{speed:.0f} km/h", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.WIDTH // 2 - speed_text.get_width() // 2, self.HEIGHT - 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    # --- Helper Methods ---

    def _generate_terrain(self):
        self.terrain_points = []
        octaves = 4
        freq = 0.005
        amp = 80
        y_offset = self.HEIGHT * 0.75
        
        for x in range(self.WIDTH + 1):
            noise_val = 0
            temp_amp = amp
            temp_freq = freq
            
            # Difficulty scales with horizontal distance
            max_slope_factor = 1.0 + (x / 200.0) * 0.2  # Corresponds to ~5 deg per 200px
            
            for _ in range(octaves):
                noise_val += self.np_random.uniform(-1, 1) * temp_amp * max_slope_factor
                temp_amp *= 0.5
                temp_freq *= 2.0
            
            y = y_offset + noise_val
            self.terrain_points.append((x, y))

    def _get_terrain_y(self, x):
        if not self.terrain_points: return self.HEIGHT
        x_clamped = int(np.clip(x, 0, self.WIDTH))
        return self.terrain_points[x_clamped][1]

    @staticmethod
    def _closest_point_on_segment(p, a, b):
        ap = p - a
        ab = b - a
        ab_len_sq = ab.length_squared()
        if ab_len_sq == 0:
            return a, (p - a).length_squared()
        
        t = ap.dot(ab) / ab_len_sq
        t = np.clip(t, 0, 1)
        
        closest = a + t * ab
        return closest, (p - closest).length_squared()

    def _create_sparks(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(10, 30)
            self.particles.append([pos.copy(), vel, lifespan])

    def _update_and_draw_particles(self):
        for p in self.particles:
            p[0] += p[1]  # Update position
            p[1] += self.GRAVITY * 0.2 # Particles affected by gravity
            p[2] -= 1     # Reduce lifespan
            
            alpha = int(255 * (p[2] / 30))
            if alpha > 0:
                pygame.draw.circle(self.screen, (*self.COLOR_SPARK, alpha), p[0], 1)
        
        self.particles = [p for p in self.particles if p[2] > 0]

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset(seed=1)
    terminated = False
    truncated = False
    
    # Pygame setup for interactive play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Game")
    
    print(env.user_guide)

    while not terminated and not truncated:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Buttons
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)
        
    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    env.close()