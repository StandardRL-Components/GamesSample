import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:07:55.633424
# Source Brief: brief_00280.md
# Brief Index: 280
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a retro arcade game.
    The player pilots a spaceship to collect fuel cells in an asteroid field
    before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a spaceship through a dangerous asteroid field to collect all the fuel cells before time runs out."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to pilot your spaceship."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_FUEL = (100, 255, 100)
    COLOR_FUEL_GLOW = (80, 200, 80)
    COLOR_ASTEROID = (255, 100, 100)
    COLOR_ASTEROID_GLOW = (200, 80, 80)
    COLOR_PARTICLE = (255, 200, 150)
    COLOR_UI_TEXT = (240, 240, 240)

    # Player
    PLAYER_SIZE = 12
    PLAYER_SPEED = 5.0

    # Fuel
    TOTAL_FUEL_CELLS = 15
    FUEL_CELL_SIZE = 10

    # Asteroids
    INITIAL_ASTEROID_SPAWN_RATE = 1.0  # per second
    ASTEROID_SPAWN_INCREASE_INTERVAL = 10 * FPS # every 10 seconds
    ASTEROID_SPAWN_INCREASE_AMOUNT = 1.0
    ASTEROID_MIN_SPEED, ASTEROID_MAX_SPEED = 0.5, 2.5
    ASTEROID_MIN_SIZE, ASTEROID_MAX_SIZE = 10, 25

    # Particles
    PARTICLE_LIFETIME = 20 # in steps
    PARTICLE_SPEED = 2.0
    PARTICLE_COUNT = 3

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)

        self.render_mode = render_mode
        self.stars = []
        self.player_pos = np.array([0.0, 0.0])
        self.fuel_cells = []
        self.asteroids = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.fuel_collected = 0
        self.game_over = False
        self.asteroid_spawn_timer = 0.0
        self.asteroid_spawn_rate = 0.0
        self.last_dist_to_fuel = float('inf')
        self.last_dist_to_asteroid = float('inf')
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.fuel_collected = 0
        self.game_over = False

        self.player_pos = np.array([self.WIDTH / 2.0, self.HEIGHT / 2.0])
        
        self._spawn_stars(100)
        self._spawn_fuel_cells()
        
        self.asteroids = []
        self.particles = []
        self.asteroid_spawn_timer = 0.0
        self.asteroid_spawn_rate = self.INITIAL_ASTEROID_SPAWN_RATE

        self.last_dist_to_fuel = self._get_closest_entity_dist(self.fuel_cells)
        self.last_dist_to_asteroid = self._get_closest_entity_dist(self.asteroids)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        reward = 0
        
        # --- Pre-move distance calculations for rewards ---
        dist_fuel_before = self._get_closest_entity_dist(self.fuel_cells)
        dist_asteroid_before = self._get_closest_entity_dist(self.asteroids)

        # --- Update Game State ---
        self._update_player(movement)
        self._update_asteroids()
        self._update_particles()
        self._spawn_new_asteroids()

        # --- Post-move distance calculations ---
        dist_fuel_after = self._get_closest_entity_dist(self.fuel_cells)
        dist_asteroid_after = self._get_closest_entity_dist(self.asteroids)
        
        # --- Distance-based rewards ---
        if dist_fuel_after < dist_fuel_before:
            reward += 0.1 # Moved closer to fuel
        if dist_asteroid_after < dist_asteroid_before:
             reward -= 0.5 # Moved closer to an asteroid
        
        # --- Handle Collisions & Events ---
        collision_reward, collision_termination = self._handle_collisions()
        reward += collision_reward
        
        self.steps += 1
        
        # --- Check Termination Conditions ---
        time_up = self.steps >= self.MAX_STEPS
        all_fuel_collected = self.fuel_collected >= self.TOTAL_FUEL_CELLS
        
        terminated = collision_termination or time_up or all_fuel_collected
        self.game_over = terminated

        # --- Goal-based rewards ---
        if all_fuel_collected:
            reward += 50 # Base reward for winning
            if not time_up:
                reward += 50 # Bonus for winning before time runs out
        elif collision_termination:
            reward = -100 # Override other rewards on death

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        self._render_fuel_cells()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel_collected": self.fuel_collected,
        }

    # --- Helper Methods: Spawning ---

    def _spawn_stars(self, count):
        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.uniform(0.5, 1.5)
            ) for _ in range(count)
        ]

    def _spawn_fuel_cells(self):
        self.fuel_cells = []
        min_dist_from_player = 100
        for _ in range(self.TOTAL_FUEL_CELLS):
            while True:
                pos = np.array([
                    self.np_random.uniform(self.FUEL_CELL_SIZE, self.WIDTH - self.FUEL_CELL_SIZE),
                    self.np_random.uniform(self.FUEL_CELL_SIZE, self.HEIGHT - self.FUEL_CELL_SIZE)
                ])
                if np.linalg.norm(pos - self.player_pos) > min_dist_from_player:
                    self.fuel_cells.append(pos)
                    break

    def _spawn_asteroid(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -self.ASTEROID_MAX_SIZE])
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ASTEROID_MAX_SIZE])
        elif edge == 2: # Left
            pos = np.array([-self.ASTEROID_MAX_SIZE, self.np_random.uniform(0, self.HEIGHT)])
        else: # Right
            pos = np.array([self.WIDTH + self.ASTEROID_MAX_SIZE, self.np_random.uniform(0, self.HEIGHT)])

        target = np.array([
            self.np_random.uniform(self.WIDTH * 0.25, self.WIDTH * 0.75),
            self.np_random.uniform(self.HEIGHT * 0.25, self.HEIGHT * 0.75)
        ])
        direction = (target - pos) / np.linalg.norm(target - pos)
        speed = self.np_random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
        velocity = direction * speed
        size = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        
        self.asteroids.append([pos, velocity, size])

    def _spawn_new_asteroids(self):
        self.asteroid_spawn_timer += 1 / self.FPS
        if self.asteroid_spawn_timer > 1.0 / self.asteroid_spawn_rate:
            self._spawn_asteroid()
            self.asteroid_spawn_timer = 0.0

        if self.steps > 0 and self.steps % self.ASTEROID_SPAWN_INCREASE_INTERVAL == 0:
            self.asteroid_spawn_rate += self.ASTEROID_SPAWN_INCREASE_AMOUNT

    # --- Helper Methods: Game Logic Updates ---

    def _update_player(self, movement):
        moved = True
        direction_vector = np.array([0.0, 0.0])
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
            direction_vector[1] = 1
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
            direction_vector[1] = -1
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
            direction_vector[0] = 1
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
            direction_vector[0] = -1
        else:
            moved = False

        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        if moved:
            # Sfx: player_thruster_sound.play()
            for _ in range(self.PARTICLE_COUNT):
                offset = self.np_random.uniform(-self.PLAYER_SIZE/2, self.PLAYER_SIZE/2, size=2)
                pos = self.player_pos.copy() + offset
                vel = direction_vector * self.PARTICLE_SPEED + self.np_random.uniform(-0.5, 0.5, size=2)
                self.particles.append([pos, vel, self.PARTICLE_LIFETIME])

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid[0] += asteroid[1] # pos += vel
            # Screen wrap
            if asteroid[0][0] < -self.ASTEROID_MAX_SIZE: asteroid[0][0] = self.WIDTH + self.ASTEROID_MAX_SIZE
            if asteroid[0][0] > self.WIDTH + self.ASTEROID_MAX_SIZE: asteroid[0][0] = -self.ASTEROID_MAX_SIZE
            if asteroid[0][1] < -self.ASTEROID_MAX_SIZE: asteroid[0][1] = self.HEIGHT + self.ASTEROID_MAX_SIZE
            if asteroid[0][1] > self.HEIGHT + self.ASTEROID_MAX_SIZE: asteroid[0][1] = -self.ASTEROID_MAX_SIZE

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[2] -= 1 # lifetime--

    def _handle_collisions(self):
        reward = 0
        terminated = False

        # Player vs Fuel Cells
        for i in range(len(self.fuel_cells) - 1, -1, -1):
            fuel_pos = self.fuel_cells[i]
            dist = np.linalg.norm(self.player_pos - fuel_pos)
            if dist < self.PLAYER_SIZE + self.FUEL_CELL_SIZE:
                # Sfx: fuel_collect_sound.play()
                self.fuel_cells.pop(i)
                self.fuel_collected += 1
                self.score += 10
                reward += 10

        # Player vs Asteroids
        for asteroid in self.asteroids:
            asteroid_pos, _, asteroid_size = asteroid
            dist = np.linalg.norm(self.player_pos - asteroid_pos)
            if dist < self.PLAYER_SIZE + asteroid_size:
                # Sfx: explosion_sound.play()
                terminated = True
                break # Only one collision needed
        
        return reward, terminated

    # --- Helper Methods: Rendering ---

    def _render_stars(self):
        for x, y, size in self.stars:
            self.screen.set_at((x, y), (int(size*50), int(size*50), int(size*70)))

    def _render_player(self):
        pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Glow effect
        glow_radius = int(self.PLAYER_SIZE * 2.5)
        glow_center = pos_int
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.COLOR_PLAYER_GLOW, 30), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (glow_center[0] - glow_radius, glow_center[1] - glow_radius))

        # Main body (triangle)
        p1 = (pos_int[0], pos_int[1] - self.PLAYER_SIZE)
        p2 = (pos_int[0] - self.PLAYER_SIZE / 1.5, pos_int[1] + self.PLAYER_SIZE / 2)
        p3 = (pos_int[0] + self.PLAYER_SIZE / 1.5, pos_int[1] + self.PLAYER_SIZE / 2)
        pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)

    def _render_fuel_cells(self):
        blink_alpha = 128 + 127 * math.sin(self.steps * 0.1)
        for pos in self.fuel_cells:
            pos_int = (int(pos[0]), int(pos[1]))
            size = self.FUEL_CELL_SIZE
            rect = pygame.Rect(pos_int[0] - size, pos_int[1] - size, size*2, size*2)
            
            # Glow
            pygame.gfxdraw.box(self.screen, rect, (*self.COLOR_FUEL_GLOW, int(blink_alpha/2)))
            # Core
            pygame.gfxdraw.box(self.screen, rect.inflate(-4, -4), (*self.COLOR_FUEL, int(blink_alpha)))
            
    def _render_asteroids(self):
        for pos, _, size in self.asteroids:
            pos_int = (int(pos[0]), int(pos[1]))
            size_int = int(size)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], size_int, self.COLOR_ASTEROID_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], size_int, self.COLOR_ASTEROID_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], max(0, size_int-2), self.COLOR_ASTEROID)

    def _render_particles(self):
        for pos, _, lifetime in self.particles:
            alpha = int(255 * (lifetime / self.PARTICLE_LIFETIME))
            size = int(5 * (lifetime / self.PARTICLE_LIFETIME))
            if size > 0:
                color = (*self.COLOR_PARTICLE, alpha)
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(pos[0]) - size, int(pos[1]) - size))

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        
        fuel_text = f"FUEL: {self.fuel_collected}/{self.TOTAL_FUEL_CELLS}"
        time_text = f"TIME: {time_left:.1f}"
        score_text = f"SCORE: {self.score}"

        self._render_text(fuel_text, (10, 10), self.font_small, self.COLOR_UI_TEXT, align="topleft")
        self._render_text(time_text, (self.WIDTH - 10, 10), self.font_small, self.COLOR_UI_TEXT, align="topright")
        self._render_text(score_text, (self.WIDTH / 2, self.HEIGHT - 40), self.font_large, self.COLOR_UI_TEXT, align="center")

    def _render_text(self, text, position, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = position
        elif align == "topleft":
            text_rect.topleft = position
        elif align == "topright":
            text_rect.topright = position
        self.screen.blit(text_surface, text_rect)

    # --- Utility Methods ---

    def _get_closest_entity_dist(self, entity_list):
        if not entity_list:
            return float('inf')
        
        min_dist = float('inf')
        for entity in entity_list:
            # Assumes entity is either a numpy array (fuel) or a list where the first element is the pos (asteroid)
            pos = entity if isinstance(entity, np.ndarray) else entity[0]
            dist = np.linalg.norm(self.player_pos - pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage ---
    # The original code had a validation check that is not part of the Gymnasium API
    # and caused issues when running locally. It has been removed from the main
    # class definition and the __main__ block has been updated for standard usage.
    
    # Set the SDL_VIDEODRIVER to a real driver for local rendering
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Manual play loop
    pygame.display.set_caption("Asteroid Collector")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    while running:
        action = np.array([0, 0, 0]) # Default: no-op
        
        # Check for events first
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        # Get key presses for continuous movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        if terminated:
            print("--- GAME OVER ---")
            print(f"Final Score: {info['score']}, Fuel: {info['fuel_collected']}/{GameEnv.TOTAL_FUEL_CELLS}")
            # Optional: wait for a key press to reset
            # running = False # Or just end the loop
            obs, info = env.reset()

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()