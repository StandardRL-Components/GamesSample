import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:49:26.677504
# Source Brief: brief_00720.md
# Brief Index: 720
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a momentum-based rover collects energy crystals.
    The rover navigates a field with magnetic zones that increase crystal spawn rates.
    The goal is to collect 50 crystals.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- Class Attributes for Gymnasium API ---
    game_description = "Pilot a momentum-based rover to collect energy crystals in a field with magnetic zones."
    user_guide = "Use the arrow keys (↑↓←→) to apply thrust and navigate the rover."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30

    # Colors
    COLOR_BG = (18, 22, 33)
    COLOR_FIELD = (40, 45, 60)
    COLOR_FIELD_GLOW = (50, 55, 75)
    COLOR_ROVER = (255, 255, 255)
    COLOR_ROVER_GLOW = (200, 200, 255, 50)
    COLOR_CRYSTAL = (0, 190, 255)
    COLOR_CRYSTAL_GLOW = (100, 220, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PARTICLE = (255, 255, 255)

    # Game Parameters
    ROVER_SIZE = 12
    ROVER_ACCELERATION = 0.4
    ROVER_DRAG = 0.96
    ROVER_MAX_SPEED = 8.0
    CRYSTAL_SIZE = 8
    CRYSTAL_TARGET = 50
    MAX_STEPS = 2000
    MAX_CRYSTALS = 30
    FIELD_COUNT = 5
    FIELD_RADIUS = 50
    FIELD_INFLUENCE_RADIUS = 150
    BASE_SPAWN_CHANCE = 0.03

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rover_pos = pygame.Vector2(0, 0)
        self.rover_vel = pygame.Vector2(0, 0)
        self.crystals = []
        self.magnetic_fields = []
        self.particles = []
        self.last_dist_to_crystal = float('inf')
        self._last_score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Rover state
        self.rover_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.rover_vel = pygame.Vector2(0, 0)

        # Environment state
        self.crystals = []
        self.particles = []
        
        # Generate fixed magnetic fields for the episode
        self.magnetic_fields = []
        for _ in range(self.FIELD_COUNT):
            padding = self.FIELD_RADIUS + 20
            pos = pygame.Vector2(
                self.np_random.uniform(padding, self.SCREEN_WIDTH - padding),
                self.np_random.uniform(padding, self.SCREEN_HEIGHT - padding)
            )
            self.magnetic_fields.append(pos)

        # Initial crystal spawn
        for _ in range(10):
            self._spawn_crystal()
            
        self.last_dist_to_crystal = self._get_dist_to_nearest_crystal()
        self._last_score = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- Action Handling ---
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        self._update_rover(movement)
        self._update_crystals()
        self._update_particles()
        
        # --- Reward Calculation ---
        reward = self._calculate_reward()

        # --- Termination Check ---
        terminated = self.score >= self.CRYSTAL_TARGET
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True
            if terminated:
                reward += 100.0 # Goal-oriented reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_rover(self, movement):
        # Apply acceleration based on action
        accel = pygame.Vector2(0, 0)
        if movement == 1: accel.y = -self.ROVER_ACCELERATION # Up
        elif movement == 2: accel.y = self.ROVER_ACCELERATION # Down
        elif movement == 3: accel.x = -self.ROVER_ACCELERATION # Left
        elif movement == 4: accel.x = self.ROVER_ACCELERATION # Right

        self.rover_vel += accel
        
        # Limit speed
        speed = self.rover_vel.length()
        if speed > self.ROVER_MAX_SPEED:
            self.rover_vel.scale_to_length(self.ROVER_MAX_SPEED)

        # Apply drag and update position
        self.rover_vel *= self.ROVER_DRAG
        self.rover_pos += self.rover_vel

        # Boundary checks
        self.rover_pos.x = np.clip(self.rover_pos.x, self.ROVER_SIZE / 2, self.SCREEN_WIDTH - self.ROVER_SIZE / 2)
        self.rover_pos.y = np.clip(self.rover_pos.y, self.ROVER_SIZE / 2, self.SCREEN_HEIGHT - self.ROVER_SIZE / 2)
        
        # Add trail particles
        if self.rover_vel.length_squared() > 1:
            self._create_particle(self.rover_pos, count=1, speed_min=0.1, speed_max=0.5, life=10, color=(150,150,180))


    def _update_crystals(self):
        # Check for collection
        collected_indices = []
        for i, crystal_pos in enumerate(self.crystals):
            if self.rover_pos.distance_to(crystal_pos) < (self.ROVER_SIZE / 2 + self.CRYSTAL_SIZE / 2):
                collected_indices.append(i)
                self.score += 1
                self._create_particle(crystal_pos, count=20, speed_min=1, speed_max=4, life=25, color=self.COLOR_CRYSTAL)

        # Remove collected crystals
        for i in sorted(collected_indices, reverse=True):
            del self.crystals[i]
        
        # Spawn new crystals
        spawn_chance = self.BASE_SPAWN_CHANCE
        for field_pos in self.magnetic_fields:
            dist = self.rover_pos.distance_to(field_pos)
            if dist < self.FIELD_INFLUENCE_RADIUS:
                boost = 2.0 * (1.0 - dist / self.FIELD_INFLUENCE_RADIUS)
                spawn_chance += self.BASE_SPAWN_CHANCE * boost
        
        if len(self.crystals) < self.MAX_CRYSTALS and self.np_random.random() < spawn_chance:
            self._spawn_crystal()
            
    def _spawn_crystal(self):
        field_idx = self.np_random.integers(len(self.magnetic_fields))
        field_pos = self.magnetic_fields[field_idx]
        angle = self.np_random.uniform(0, 2 * math.pi)
        radius = self.np_random.uniform(0, self.FIELD_RADIUS)
        pos = pygame.Vector2(
            field_pos.x + math.cos(angle) * radius,
            field_pos.y + math.sin(angle) * radius
        )
        pos.x = np.clip(pos.x, self.CRYSTAL_SIZE, self.SCREEN_WIDTH - self.CRYSTAL_SIZE)
        pos.y = np.clip(pos.y, self.CRYSTAL_SIZE, self.SCREEN_HEIGHT - self.CRYSTAL_SIZE)
        self.crystals.append(pos)

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_particle(self, pos, count, speed_min, speed_max, life, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(speed_min, speed_max)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': life,
                'color': color
            })
            
    def _calculate_reward(self):
        reward = 0.0
        
        if self.score > self._last_score:
            reward += 1.0 * (self.score - self._last_score)
        self._last_score = self.score

        dist_to_crystal = self._get_dist_to_nearest_crystal()
        if dist_to_crystal < self.last_dist_to_crystal:
            reward += 0.1
        else:
            reward -= 0.01
        self.last_dist_to_crystal = dist_to_crystal
        
        return reward
        
    def _get_dist_to_nearest_crystal(self):
        if not self.crystals:
            return float('inf')
        
        min_dist_sq = float('inf')
        for crystal_pos in self.crystals:
            dist_sq = self.rover_pos.distance_squared_to(crystal_pos)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        return math.sqrt(min_dist_sq)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for field_pos in self.magnetic_fields:
            pos_int = (int(field_pos.x), int(field_pos.y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.FIELD_RADIUS, self.COLOR_FIELD)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.FIELD_RADIUS, self.COLOR_FIELD_GLOW)

        for p in self.particles:
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            alpha = max(0, 255 * (p['life'] / 25.0))
            color_with_alpha = (*p['color'], alpha)
            size = max(1, int(2 * (p['life'] / 25.0)))
            rect = pygame.Rect(pos_int[0] - size//2, pos_int[1] - size//2, size, size)
            temp_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            temp_surf.fill(color_with_alpha)
            self.screen.blit(temp_surf, rect.topleft)

        for crystal_pos in self.crystals:
            pos_int = (int(crystal_pos.x), int(crystal_pos.y))
            points = [
                (pos_int[0], pos_int[1] - self.CRYSTAL_SIZE),
                (pos_int[0] - self.CRYSTAL_SIZE / 2, pos_int[1] + self.CRYSTAL_SIZE / 2),
                (pos_int[0] + self.CRYSTAL_SIZE / 2, pos_int[1] + self.CRYSTAL_SIZE / 2),
            ]
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_CRYSTAL)
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_CRYSTAL_GLOW)

        rover_pos_int = (int(self.rover_pos.x), int(self.rover_pos.y))
        
        glow_surface = pygame.Surface((self.ROVER_SIZE * 4, self.ROVER_SIZE * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_ROVER_GLOW, (self.ROVER_SIZE * 2, self.ROVER_SIZE * 2), self.ROVER_SIZE * 1.5)
        self.screen.blit(glow_surface, (rover_pos_int[0] - self.ROVER_SIZE * 2, rover_pos_int[1] - self.ROVER_SIZE * 2))

        rover_rect = pygame.Rect(
            rover_pos_int[0] - self.ROVER_SIZE / 2,
            rover_pos_int[1] - self.ROVER_SIZE / 2,
            self.ROVER_SIZE,
            self.ROVER_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_ROVER, rover_rect, border_radius=2)

    def _render_ui(self):
        score_text = self.font.render(f"Crystals: {self.score}/{self.CRYSTAL_TARGET}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font.render(f"Time: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rover_pos": (self.rover_pos.x, self.rover_pos.y),
            "rover_vel": (self.rover_vel.x, self.rover_vel.y),
            "crystals_remaining": len(self.crystals)
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Momentum Rover")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if keys[pygame.K_r]:
            obs, info = env.reset()
            total_reward = 0
            print("--- Game Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.TARGET_FPS)
        
    env.close()