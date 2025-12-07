import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:28:37.656093
# Source Brief: brief_01007.md
# Brief Index: 1007
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a transforming vehicle.
    The goal is to collect 5 fuel cells within a 30-second time limit.
    The vehicle can switch between car, boat, and plane modes to navigate
    different terrains (land and water).
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Control a transforming vehicle to collect fuel cells across land and water before time runs out."
    user_guide = "Use the arrow keys (↑↓←→) to move. Press space to cycle between car, boat, and plane modes."
    auto_advance = True


    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    TIME_LIMIT_SECONDS = 30
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_WATER = (50, 100, 200)
    COLOR_LAND = (100, 80, 70)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_UI_BG = (0, 0, 0, 128)

    # Vehicle Modes & Colors
    MODE_CAR, MODE_BOAT, MODE_PLANE = 0, 1, 2
    VEHICLE_COLORS = {
        MODE_CAR: (255, 70, 70),   # Red
        MODE_BOAT: (70, 150, 255), # Blue
        MODE_PLANE: (255, 220, 70) # Yellow
    }
    VEHICLE_GLOW_COLORS = {
        MODE_CAR: (255, 100, 100, 50),
        MODE_BOAT: (100, 180, 255, 50),
        MODE_PLANE: (255, 230, 100, 50)
    }

    # Fuel & Obstacles
    FUEL_COLOR = (80, 255, 80)
    OBSTACLE_COLOR = (120, 130, 140)
    NUM_FUEL = 5
    NUM_OBSTACLES = 10

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game State ---
        self.player_pos = np.array([0.0, 0.0])
        self.player_mode = self.MODE_CAR
        self.player_size = 12
        self.player_speed = 4.0

        self.fuel_positions = []
        self.fuel_collected = 0
        self.fuel_size = 8

        self.obstacles = []
        self.obstacle_size = 20

        self.particles = []
        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.game_over = False
        self.prev_space_held = False
        self.start_pos = np.array([50.0, self.SCREEN_HEIGHT / 4.0])

        # --- Terrain Definition ---
        self.water_rect = pygame.Rect(0, 150, self.SCREEN_WIDTH, 100)
        
        # --- Finalize Init ---
        if render_mode == "human":
            pygame.display.set_caption("Transform Racer")
            self.human_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TIME_LIMIT_SECONDS
        self.fuel_collected = 0
        self.prev_space_held = False

        self.player_pos = self.start_pos.copy()
        self.player_mode = self.MODE_CAR

        self._spawn_entities()
        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement, space_held, _ = action
        space_held = space_held == 1

        # --- Update Time and Steps ---
        self.steps += 1
        self.timer = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        
        # --- Store state for reward calculation ---
        dist_before = self._get_dist_to_nearest_fuel()

        # --- Handle Player Actions ---
        self._handle_movement(movement)
        self._handle_mode_switch(space_held)
        self.prev_space_held = space_held

        # --- Game Logic ---
        self._update_particles()
        if self.steps % 7 == 0: # Dynamic obstacles
            self._update_obstacles()
        
        # --- Collisions and Rewards ---
        reward = 0
        
        # Fuel collection
        collected_fuel = self._check_fuel_collision()
        if collected_fuel:
            self.fuel_collected += 1
            self.score += 1
            reward += 1.0
            # sfx: fuel_pickup.wav
            self._spawn_particles(self.player_pos, self.FUEL_COLOR, 20, 3)

        # Obstacle collision
        if self._check_obstacle_collision():
            reward -= 5.0 # Penalty for collision
            # sfx: crash.wav
            self._spawn_particles(self.player_pos, self.OBSTACLE_COLOR, 30, 2)
            self.player_pos = self.start_pos.copy()

        # Proximity reward
        dist_after = self._get_dist_to_nearest_fuel()
        if dist_after is not None and dist_before is not None:
             reward += 0.01 * (dist_before - dist_after)

        # --- Termination ---
        terminated = False
        if self.fuel_collected >= self.NUM_FUEL:
            terminated = True
            self.game_over = True
            reward += 100.0 # Victory bonus
            # sfx: win_jingle.wav
        elif self.timer <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            reward -= 100.0 # Timeout penalty
            # sfx: lose_buzzer.wav
        
        truncated = False # This environment does not truncate based on step limit
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Spawning and Updates ---
    def _spawn_entities(self):
        self.fuel_positions.clear()
        self.obstacles.clear()

        # Spawn fuel
        for i in range(self.NUM_FUEL):
            is_on_water = i < 2 # Place 2 fuel cells on water
            while True:
                pos = self._get_random_pos()
                terrain = self._get_terrain_at(pos)
                if (is_on_water and terrain == self.MODE_BOAT) or \
                   (not is_on_water and terrain == self.MODE_CAR):
                    if np.linalg.norm(pos - self.player_pos) > 100:
                        self.fuel_positions.append(pos)
                        break
        
        # Spawn obstacles
        for _ in range(self.NUM_OBSTACLES):
            self.obstacles.append(self._get_random_pos())

    def _update_obstacles(self):
        if self.obstacles and random.random() < 0.5:
            self.obstacles.pop(random.randrange(len(self.obstacles)))
            self.obstacles.append(self._get_random_pos())

    def _get_random_pos(self):
        return np.array([
            random.uniform(self.obstacle_size, self.SCREEN_WIDTH - self.obstacle_size),
            random.uniform(self.obstacle_size, self.SCREEN_HEIGHT - self.obstacle_size)
        ])

    # --- Player Actions ---
    def _handle_movement(self, movement_action):
        move_vec = np.array([0.0, 0.0])
        if movement_action == 1: move_vec[1] = -1 # Up
        elif movement_action == 2: move_vec[1] = 1  # Down
        elif movement_action == 3: move_vec[0] = -1 # Left
        elif movement_action == 4: move_vec[0] = 1  # Right
        
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)
        
        new_pos = self.player_pos + move_vec * self.player_speed
        
        # Terrain check
        current_terrain = self._get_terrain_at(new_pos)
        if self.player_mode == self.MODE_PLANE or self.player_mode == current_terrain:
            self.player_pos = new_pos
        
        # World wrapping
        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] %= self.SCREEN_HEIGHT

    def _handle_mode_switch(self, space_held):
        if space_held and not self.prev_space_held:
            self.player_mode = (self.player_mode + 1) % 3
            # sfx: mode_switch.wav
            self._spawn_particles(self.player_pos, self.VEHICLE_COLORS[self.player_mode], 30, 4)

    # --- Collision and State Checks ---
    def _check_fuel_collision(self):
        for i, pos in enumerate(self.fuel_positions):
            if np.linalg.norm(self.player_pos - pos) < self.player_size + self.fuel_size:
                self.fuel_positions.pop(i)
                return True
        return False

    def _check_obstacle_collision(self):
        if self.player_mode == self.MODE_PLANE:
            return False
            
        player_terrain = self._get_terrain_at(self.player_pos)
        for pos in self.obstacles:
            obstacle_terrain = self._get_terrain_at(pos)
            # Collide if obstacle is on the same terrain type as the player
            if player_terrain == obstacle_terrain:
                 if np.linalg.norm(self.player_pos - pos) < self.player_size + self.obstacle_size / 2:
                    return True
        return False

    def _get_terrain_at(self, pos):
        if self.water_rect.collidepoint(pos[0], pos[1]):
            return self.MODE_BOAT
        return self.MODE_CAR

    def _get_dist_to_nearest_fuel(self):
        if not self.fuel_positions:
            return None
        distances = [np.linalg.norm(self.player_pos - f_pos) for f_pos in self.fuel_positions]
        return min(distances)

    # --- Particle System ---
    def _spawn_particles(self, pos, color, count, speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * random.uniform(0.5, 1) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "color": color,
                "life": random.randint(15, 25)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["life"] -= 1

    # --- Rendering ---
    def _render_background(self):
        self.screen.fill(self.COLOR_LAND)
        pygame.draw.rect(self.screen, self.COLOR_WATER, self.water_rect)

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        color = self.VEHICLE_COLORS[self.player_mode]
        glow_color = self.VEHICLE_GLOW_COLORS[self.player_mode]

        # Glow effect
        glow_radius = int(self.player_size * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Vehicle shape
        if self.player_mode == self.MODE_CAR:
            pygame.gfxdraw.box(self.screen, (x - self.player_size, y - self.player_size//2, self.player_size*2, self.player_size), color)
        elif self.player_mode == self.MODE_BOAT:
            points = [(x, y - self.player_size), (x - self.player_size, y + self.player_size//2), (x + self.player_size, y + self.player_size//2)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif self.player_mode == self.MODE_PLANE:
            s = self.player_size
            body = [(x, y - s), (x - s//2, y + s//2), (x + s//2, y + s//2)]
            wing = [(x - s*1.5, y), (x + s*1.5, y), (x, y + s//2)]
            pygame.gfxdraw.aapolygon(self.screen, wing, color)
            pygame.gfxdraw.filled_polygon(self.screen, wing, color)
            pygame.gfxdraw.aapolygon(self.screen, body, color)
            pygame.gfxdraw.filled_polygon(self.screen, body, color)

    def _render_fuel(self):
        angle = (self.steps * 4) % 360
        rad = math.radians(angle)
        for pos in self.fuel_positions:
            s = self.fuel_size
            points = [
                (pos[0] + s * math.cos(rad), pos[1] + s * math.sin(rad)),
                (pos[0] + s * math.cos(rad + math.pi/2), pos[1] + s * math.sin(rad + math.pi/2)),
                (pos[0] + s * math.cos(rad + math.pi), pos[1] + s * math.sin(rad + math.pi)),
                (pos[0] + s * math.cos(rad + 3*math.pi/2), pos[1] + s * math.sin(rad + 3*math.pi/2)),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.FUEL_COLOR)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.FUEL_COLOR)

    def _render_obstacles(self):
        for pos in self.obstacles:
            x, y = int(pos[0]), int(pos[1])
            s = int(self.obstacle_size / 2)
            rect = (x - s, y - s, s*2, s*2)
            pygame.gfxdraw.box(self.screen, rect, self.OBSTACLE_COLOR)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 25.0))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = int(max(1, p["life"] / 5))
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (pos[0] - size, pos[1] - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Fuel collected
        fuel_text = self.font_ui.render(f"FUEL: {self.fuel_collected}/{self.NUM_FUEL}", True, self.FUEL_COLOR)
        self.screen.blit(fuel_text, (10, 10))

        # Timer
        timer_str = f"{self.timer:.1f}"
        timer_color = (255, 100, 100) if self.timer < 5 else self.COLOR_UI_TEXT
        timer_text = self.font_timer.render(timer_str, True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(timer_text, timer_rect)

    # --- Core Gym Methods ---
    def _get_observation(self):
        self._render_background()
        self._render_obstacles()
        self._render_fuel()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        obs = np.transpose(arr, (1, 0, 2)).astype(np.uint8)

        if self.render_mode == "human":
            self.human_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.FPS)

        return obs

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "fuel_collected": self.fuel_collected,
            "player_mode": self.player_mode,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # Control mapping for human play
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Main game loop for human play
    while not done:
        # Default action: no movement, buttons released
        movement = 0
        space_held = 0
        shift_held = 0

        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Get keyboard state
        keys = pygame.key.get_pressed()
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")
            # Wait a bit before resetting to show final state
            pygame.time.wait(2000)
            obs, info = env.reset()

    env.close()