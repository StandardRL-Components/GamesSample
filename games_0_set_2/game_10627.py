import gymnasium as gym
import os
import pygame
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


# Ensure Pygame runs headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Helper class for vector math
Vec2 = pygame.math.Vector2

# --- Helper Classes ---

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, vel, color, lifetime):
        self.pos = Vec2(pos)
        self.vel = Vec2(vel)
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95  # Damping
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            size = int(5 * (self.lifetime / self.max_lifetime))
            if size > 0:
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, (*self.color, alpha), (size, size), size)
                surface.blit(particle_surf, (int(self.pos.x - size), int(self.pos.y - size)))

class Droplet:
    """Represents a water droplet."""
    def __init__(self, pos, mass=1.0):
        self.pos = Vec2(pos)
        self.vel = Vec2(0, 0)
        self.mass = mass
        self.radius = self._calculate_radius()
        self.on_tube_since = 0

    def _calculate_radius(self):
        return max(3, int(math.sqrt(self.mass) * 4))

    def apply_force(self, force):
        if self.mass > 0:
            self.vel += force / self.mass

    def update(self):
        self.pos += self.vel
        self.vel *= 0.99 # Air resistance/damping

    def merge(self, other):
        """Merges this droplet with another."""
        total_mass = self.mass + other.mass
        self.vel = (self.vel * self.mass + other.vel * other.mass) / total_mass
        self.pos = (self.pos * self.mass + other.pos * other.mass) / total_mass
        self.mass = total_mass
        self.radius = self._calculate_radius()

    def draw(self, surface):
        pos_int = (int(self.pos.x), int(self.pos.y))
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], self.radius, (100, 150, 255))
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], self.radius, (100, 150, 255))
        highlight_radius = max(1, int(self.radius * 0.5))
        highlight_offset = self.radius * 0.3
        highlight_pos = (int(self.pos.x - highlight_offset), int(self.pos.y - highlight_offset))
        pygame.gfxdraw.filled_circle(surface, highlight_pos[0], highlight_pos[1], highlight_radius, (200, 220, 255))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Guide water droplets through a network of tubes using wind. Merge droplets to grow larger and avoid obstacles to reach the goal."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to create wind and guide the droplets through the tubes."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (15, 20, 30)
    COLOR_TUBE = (80, 80, 90)
    COLOR_TUBE_OUTLINE = (60, 60, 70)
    COLOR_TARGET = (50, 180, 50)
    COLOR_OBSTACLE = (220, 50, 50)
    COLOR_TEXT = (230, 230, 230)

    GRAVITY = Vec2(0, 0.05)
    WIND_STRENGTH = 0.15
    TUBE_RADIUS = 20
    TUBE_RESTORING_FORCE = 0.1
    MAX_STEPS = 2000
    WIN_MASS_TARGET = 10
    INITIAL_DROPLETS = 20
    OBSTACLE_ADD_INTERVAL = 200
    MAX_OBSTACLES = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.render_mode = render_mode

    def _initialize_state(self):
        """Initializes all game state variables. Called by reset."""
        self.steps = 0
        self.score = 0.0

        self._define_tube_network()
        self.target_zone_pos = Vec2(self.WIDTH / 2, self.HEIGHT - 50)
        self.target_zone_radius = 40

        self.droplets = []
        self._spawn_initial_droplets()

        self.obstacles = []
        self._spawn_initial_obstacles()

        self.particles = []
        self.wind_vector = Vec2(0, 0)
        self.current_step_reward = 0.0

    def _define_tube_network(self):
        """Defines the geometry of the tube network."""
        self.tubes = [
            (Vec2(100, 0), Vec2(100, 200)),
            (Vec2(100, 200), Vec2(320, 250)),
            (Vec2(320, 250), Vec2(540, 200)),
            (Vec2(540, 200), Vec2(540, 0)),
            (Vec2(320, 250), Vec2(320, 400)),
        ]
        self.obstacle_spawn_points = [
            Vec2(100, 100), Vec2(210, 225), Vec2(430, 225),
            Vec2(540, 100), Vec2(320, 320), Vec2(100, 150),
            Vec2(265, 237), Vec2(375, 237), Vec2(540, 150),
            Vec2(320, 360)
        ]
        self.np_random.shuffle(self.obstacle_spawn_points)

    def _spawn_initial_droplets(self):
        self.droplets = []
        for _ in range(self.INITIAL_DROPLETS):
            spawn_x = self.np_random.uniform(80, 120)
            spawn_y = self.np_random.uniform(20, 60)
            self.droplets.append(Droplet(pos=(spawn_x, spawn_y), mass=1.0))

    def _spawn_initial_obstacles(self):
        self.obstacles = []
        self._add_obstacle()
        self._add_obstacle()

    def _add_obstacle(self):
        if self.obstacle_spawn_points and len(self.obstacles) < self.MAX_OBSTACLES:
            pos = self.obstacle_spawn_points.pop(0)
            self.obstacles.append({'pos': pos, 'radius': 10})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pygame.init() # Ensure pygame is initialized for each env instance
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, _, _ = action
        self.current_step_reward = 0.0

        self._handle_input(movement)
        self._update_game_state()

        self.steps += 1
        self.score += self.current_step_reward

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        # An episode is terminated if it's a win, but not if it's truncated
        if truncated:
            terminated = False

        return (
            self._get_observation(),
            self.current_step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1: self.wind_vector = Vec2(0, -1)
        elif movement == 2: self.wind_vector = Vec2(0, 1)
        elif movement == 3: self.wind_vector = Vec2(-1, 0)
        elif movement == 4: self.wind_vector = Vec2(1, 0)
        else: self.wind_vector = Vec2(0, 0)

    def _update_game_state(self):
        self._update_droplets()
        self._handle_collisions()
        self._update_particles()
        self._update_difficulty()

    def _update_droplets(self):
        for d in self.droplets:
            d.apply_force(self.GRAVITY * d.mass)
            d.apply_force(self.wind_vector * self.WIND_STRENGTH)

            closest_point, dist_sq = self._find_closest_point_on_network(d.pos)
            if dist_sq > self.TUBE_RADIUS**2:
                restoring_dir_vec = closest_point - d.pos
                if restoring_dir_vec.length_squared() > 0:
                    restoring_dir = restoring_dir_vec.normalize()
                    restoring_force = restoring_dir * self.TUBE_RESTORING_FORCE * (math.sqrt(dist_sq) - self.TUBE_RADIUS)
                    d.apply_force(restoring_force)
                d.on_tube_since = 0
            else:
                d.on_tube_since += 1

            d.update()

            if d.vel.y > 0.1:
                self.current_step_reward -= 0.01

    def _handle_collisions(self):
        merged_indices = set()
        for i in range(len(self.droplets)):
            for j in range(i + 1, len(self.droplets)):
                if i in merged_indices or j in merged_indices:
                    continue
                d1, d2 = self.droplets[i], self.droplets[j]
                if d1.pos.distance_squared_to(d2.pos) < (d1.radius + d2.radius)**2:
                    self._create_particles(d1.pos, 10, (150, 200, 255))
                    d1.merge(d2)
                    merged_indices.add(j)
                    self.current_step_reward += 1.1

        self.droplets = [d for i, d in enumerate(self.droplets) if i not in merged_indices]

        remaining_droplets = []
        for d in self.droplets:
            hit_obstacle = False
            for obs in self.obstacles:
                if d.pos.distance_squared_to(obs['pos']) < (d.radius + obs['radius'])**2:
                    hit_obstacle = True
                    self.current_step_reward -= 1.0
                    self._create_particles(d.pos, int(d.mass * 5), self.COLOR_OBSTACLE)
                    break
            if hit_obstacle:
                continue

            if 0 < d.pos.x < self.WIDTH and 0 < d.pos.y < self.HEIGHT:
                remaining_droplets.append(d)
            else:
                self.current_step_reward -= 1.0
        self.droplets = remaining_droplets

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.OBSTACLE_ADD_INTERVAL == 0:
            self._add_obstacle()

    def _check_termination(self):
        # Win condition
        for d in self.droplets:
            if d.mass >= self.WIN_MASS_TARGET and d.pos.distance_squared_to(self.target_zone_pos) < self.target_zone_radius**2:
                self.current_step_reward += 100.0
                return True
        # STABILITY FIX: Removed early termination on droplet loss.
        # The game now only terminates on a win or time limit (truncation).
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_target()
        self._render_tubes()
        self._render_wind_indicators()
        self._render_obstacles()
        self._render_particles()
        self._render_droplets()

    def _render_tubes(self):
        for start, end in self.tubes:
            pygame.draw.line(self.screen, self.COLOR_TUBE_OUTLINE, start, end, self.TUBE_RADIUS * 2 + 4)
            pygame.draw.line(self.screen, self.COLOR_TUBE, start, end, self.TUBE_RADIUS * 2)

    def _render_target(self):
        import pygame.gfxdraw
        pos_int = (int(self.target_zone_pos.x), int(self.target_zone_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.target_zone_radius, (*self.COLOR_TARGET, 50))
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.target_zone_radius, self.COLOR_TARGET)

    def _render_obstacles(self):
        import pygame.gfxdraw
        for obs in self.obstacles:
            pos_int = (int(obs['pos'].x), int(obs['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], obs['radius'], self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], obs['radius'], (255, 255, 255))

    def _render_wind_indicators(self):
        if self.wind_vector.length_squared() > 0:
            for x in range(0, self.WIDTH, 40):
                for y in range(0, self.HEIGHT, 40):
                    start = Vec2(x, y)
                    end = start + self.wind_vector * 10
                    pygame.draw.line(self.screen, (255, 255, 255, 20), start, end, 1)

    def _render_droplets(self):
        for d in self.droplets:
            d.draw(self.screen)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        largest_mass = max((d.mass for d in self.droplets), default=0)
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        progress_text = self.font_main.render(f"Largest Drop: {largest_mass:.1f} / {self.WIN_MASS_TARGET}", True, self.COLOR_TEXT)
        text_rect = progress_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(progress_text, text_rect)
        steps_text = self.font_small.render(f"Step: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 45))

    def _get_info(self):
        largest_mass = max((d.mass for d in self.droplets), default=0)
        return {
            "score": self.score,
            "steps": self.steps,
            "droplet_count": len(self.droplets),
            "largest_droplet_mass": largest_mass
        }

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            vel = Vec2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
            lifetime = self.np_random.integers(15, 31)
            self.particles.append(Particle(pos, vel, color, lifetime))

    def _find_closest_point_on_network(self, pos):
        min_dist_sq = float('inf')
        closest_point_on_network = None
        for start, end in self.tubes:
            line_vec = end - start
            line_mag_sq = line_vec.length_squared()
            if line_mag_sq == 0:
                p, d_sq = start, pos.distance_squared_to(start)
            else:
                t = (pos - start).dot(line_vec) / line_mag_sq
                t = max(0, min(1, t))
                p = start + t * line_vec
                d_sq = pos.distance_squared_to(p)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                closest_point_on_network = p
        return closest_point_on_network, min_dist_sq

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    try:
        del os.environ['SDL_VIDEODRIVER']
    except KeyError:
        pass

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("HydroFlow - Manual Control")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    done = False

    print("--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset Environment")
    print("Q: Quit")
    
    while running:
        movement_action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False
        
        if not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement_action = 1
            elif keys[pygame.K_DOWN]: movement_action = 2
            elif keys[pygame.K_LEFT]: movement_action = 3
            elif keys[pygame.K_RIGHT]: movement_action = 4

            action = [movement_action, 0, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                done = True

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        if done:
            font = pygame.font.Font(None, 50)
            text = font.render("GAME OVER", True, (255, 255, 255))
            text_rect = text.get_rect(center=(GameEnv.WIDTH/2, GameEnv.HEIGHT/2 - 20))
            screen.blit(text, text_rect)
            
            font_small = pygame.font.Font(None, 30)
            score_text = font_small.render(f"Final Score: {total_reward:.2f}", True, (255, 255, 255))
            score_rect = score_text.get_rect(center=(GameEnv.WIDTH/2, GameEnv.HEIGHT/2 + 20))
            screen.blit(score_text, score_rect)

        pygame.display.flip()
        clock.tick(GameEnv.metadata["render_fps"])

    env.close()