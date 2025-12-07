import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:44:52.585041
# Source Brief: brief_00610.md
# Brief Index: 610
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a portal to collect artifacts
    and rescue souls in a vibrant, procedurally generated ocean.

    The core gameplay loop involves exploring a biome, collecting magnetic artifacts
    to charge the portal, and then using that energy to rescue trapped souls.
    Rescuing all souls in a biome allows progression to the next, more challenging one.

    Visuals are a key focus, with bioluminescent colors, smooth animations, and
    particle effects creating an immersive experience.
    """
    game_description = (
        "Control a magical portal to collect energy artifacts and rescue trapped souls "
        "in a vibrant, procedurally generated ocean."
    )
    user_guide = "Controls: Use arrow keys (↑↓←→) to move the portal. Press space to attempt to rescue a nearby soul."
    auto_advance = True

    metadata = {"render_modes": ["rgb_array"]}

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    PLAYER_SPEED = 15
    PLAYER_RADIUS = 20
    ARTIFACT_RADIUS = 6
    SOUL_RADIUS = 10
    MAX_ENERGY = 100.0
    INITIAL_ENERGY = 50.0

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_WHITE = (240, 240, 240)
    COLOR_UI_BG = (20, 30, 50, 200)
    COLOR_UI_TEXT = (200, 220, 255)
    COLOR_ENERGY_HIGH = (60, 220, 180)
    COLOR_ENERGY_MED = (240, 200, 80)
    COLOR_ENERGY_LOW = (255, 90, 90)

    BIOME_CONFIGS = [
        {
            "name": "Deep Sea Abyss",
            "bg_color": (10, 20, 40),
            "primary_color": (0, 180, 220),
            "soul_color": (180, 240, 255),
            "num_souls": 2,
            "num_artifacts": 10,
            "rescue_cost": 20,
            "bg_particles": 50,
        },
        {
            "name": "Volcanic Vents",
            "bg_color": (30, 10, 10),
            "primary_color": (255, 120, 50),
            "soul_color": (255, 200, 180),
            "num_souls": 3,
            "num_artifacts": 8,
            "rescue_cost": 30,
            "bg_particles": 30,
        },
        {
            "name": "Crystal Caves",
            "bg_color": (40, 15, 50),
            "primary_color": (220, 100, 255),
            "soul_color": (255, 210, 255),
            "num_souls": 4,
            "num_artifacts": 7,
            "rescue_cost": 40,
            "bg_particles": 40,
        }
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans", 16)
        self.font_large = pygame.font.SysFont("sans", 24)

        # --- State Variables ---
        self.player_pos = pygame.Vector2(0, 0)
        self.player_target_pos = pygame.Vector2(0, 0)
        self.artifacts = []
        self.souls = []
        self.particles = []
        self.bg_elements = []
        
        self.energy = 0.0
        self.score = 0
        self.steps = 0
        self.biome_index = 0
        self.total_souls_rescued = 0
        self.game_over = False
        self.space_was_held = False

        # self.reset() # No need to call reset in init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy = self.INITIAL_ENERGY
        self.biome_index = 0
        self.total_souls_rescued = 0
        self.particles.clear()
        
        self.player_pos.update(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_target_pos.update(self.player_pos)

        self._setup_biome()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- Handle Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if movement == 1: self.player_target_pos.y -= self.PLAYER_SPEED
        elif movement == 2: self.player_target_pos.y += self.PLAYER_SPEED
        elif movement == 3: self.player_target_pos.x -= self.PLAYER_SPEED
        elif movement == 4: self.player_target_pos.x += self.PLAYER_SPEED
        
        self.player_target_pos.x = np.clip(self.player_target_pos.x, 0, self.SCREEN_WIDTH)
        self.player_target_pos.y = np.clip(self.player_target_pos.y, 0, self.SCREEN_HEIGHT)

        # --- Update Game Logic ---
        self._update_player()
        reward += self._update_artifacts()
        self._update_particles()
        
        # Handle rescue attempt on space press (not hold)
        if space_held and not self.space_was_held:
            reward += self._attempt_rescue()
        self.space_was_held = space_held

        # --- Check for Biome Completion ---
        if not self.souls:
            self.biome_index += 1
            if self.biome_index >= len(self.BIOME_CONFIGS):
                self.game_over = True
                reward += 100 # Victory
            else:
                self._setup_biome()
                # Play sound: biome_transition.wav

        # --- Check Termination Conditions ---
        terminated = False
        if self.energy <= 0:
            terminated = True
            reward -= 100 # Defeat
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.game_over = terminated or truncated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _setup_biome(self):
        config = self.BIOME_CONFIGS[self.biome_index]
        self.souls.clear()
        self.artifacts.clear()
        self.bg_elements.clear()

        # Spawn souls
        for _ in range(config["num_souls"]):
            self.souls.append({
                "pos": self._get_random_pos(margin=50),
                "pulse": random.uniform(0, math.pi * 2)
            })

        # Spawn artifacts
        for _ in range(config["num_artifacts"]):
            self.artifacts.append({
                "pos": self._get_random_pos(margin=20),
                "vel": pygame.Vector2(0, 0)
            })
        
        # Spawn background elements
        for _ in range(config["bg_particles"]):
            self.bg_elements.append({
                "pos": pygame.Vector2(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                "size": random.randint(1, 3),
                "speed": random.uniform(0.1, 0.3)
            })
            
    def _update_player(self):
        # Smooth movement via interpolation
        self.player_pos.x = self.player_pos.x * 0.8 + self.player_target_pos.x * 0.2
        self.player_pos.y = self.player_pos.y * 0.8 + self.player_target_pos.y * 0.2
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

    def _update_artifacts(self):
        collected_reward = 0
        for artifact in self.artifacts[:]:
            dist_vec = self.player_pos - artifact["pos"]
            dist = dist_vec.length()

            if dist < 100: # Magnetic attraction range
                # Inverse square attraction force
                force = dist_vec.normalize() * (300 / max(dist*dist, 25))
                artifact["vel"] += force
            
            # Damping
            artifact["vel"] *= 0.85
            artifact["pos"] += artifact["vel"]
            
            # Clamp to screen
            artifact["pos"].x = np.clip(artifact["pos"].x, 0, self.SCREEN_WIDTH)
            artifact["pos"].y = np.clip(artifact["pos"].y, 0, self.SCREEN_HEIGHT)

            if self.player_pos.distance_to(artifact["pos"]) < self.PLAYER_RADIUS + self.ARTIFACT_RADIUS:
                self.artifacts.remove(artifact)
                self.energy = min(self.MAX_ENERGY, self.energy + 10)
                collected_reward += 0.1
                # Play sound: artifact_collect.wav
                self._create_particles(artifact["pos"], self.BIOME_CONFIGS[self.biome_index]["primary_color"], 15, 2, 4)
        return collected_reward
        
    def _attempt_rescue(self):
        reward = 0
        config = self.BIOME_CONFIGS[self.biome_index]
        rescue_cost = config["rescue_cost"]

        if self.energy < rescue_cost:
            # Play sound: rescue_fail.wav
            self._create_particles(self.player_pos, self.COLOR_ENERGY_LOW, 10, 1, 2, speed=1)
            return 0
        
        closest_soul = None
        min_dist = float('inf')

        for soul in self.souls:
            dist = self.player_pos.distance_to(soul["pos"])
            if dist < min_dist:
                min_dist = dist
                closest_soul = soul

        if closest_soul and min_dist < self.PLAYER_RADIUS + self.SOUL_RADIUS:
            self.souls.remove(closest_soul)
            self.energy -= rescue_cost
            self.total_souls_rescued += 1
            reward += 1.0
            # Play sound: soul_rescue.wav
            self._create_particles(closest_soul["pos"], config["soul_color"], 50, 3, 6)
        else:
            # Play sound: rescue_fail.wav
            self._create_particles(self.player_pos, self.COLOR_ENERGY_LOW, 10, 1, 2, speed=1)

        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # --- Clear screen with biome background ---
        config = self.BIOME_CONFIGS[self.biome_index]
        self.screen.fill(config["bg_color"])

        # --- Render all game elements ---
        self._render_background_elements(config)
        self._render_artifacts(config)
        self._render_souls(config)
        self._render_particles()
        self._render_player(config)
        
        # --- Render UI overlay ---
        self._render_ui(config)
        
        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_elements(self, config):
        for element in self.bg_elements:
            element["pos"].y += element["speed"]
            if element["pos"].y > self.SCREEN_HEIGHT + element["size"]:
                element["pos"].y = -element["size"]
                element["pos"].x = random.randint(0, self.SCREEN_WIDTH)
            
            color = pygame.Color(config["primary_color"])
            color.a = 50
            pygame.gfxdraw.filled_circle(
                self.screen, int(element["pos"].x), int(element["pos"].y), element["size"], color
            )

    def _render_artifacts(self, config):
        color = config["primary_color"]
        for artifact in self.artifacts:
            pos = (int(artifact["pos"].x), int(artifact["pos"].y))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ARTIFACT_RADIUS + 3, (*color, 50))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ARTIFACT_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ARTIFACT_RADIUS, color)

    def _render_souls(self, config):
        color = config["soul_color"]
        for soul in self.souls:
            soul["pulse"] += 0.05
            pulse_size = int(math.sin(soul["pulse"]) * 3)
            radius = self.SOUL_RADIUS + pulse_size
            pos = (int(soul["pos"].x), int(soul["pos"].y))
            
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 5, (*color, 60))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = (*p["color"], max(0, min(255, alpha)))
            size = int(p["size"] * (p["lifespan"] / p["max_lifespan"]))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), size, color)

    def _render_player(self, config):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        color = config["primary_color"]
        
        # Glow effect
        for i in range(4):
            alpha = 40 - i * 8
            radius = self.PLAYER_RADIUS + i * 5
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*color, alpha))
        
        # Swirling core
        for i in range(5):
            angle = (self.steps * 0.02 + i * (2 * math.pi / 5))
            start_pos = pygame.Vector2(pos)
            end_pos = start_pos + pygame.Vector2(self.PLAYER_RADIUS, 0).rotate_rad(angle)
            pygame.draw.aaline(self.screen, color, pos, end_pos, 2)
        
        # Core circle
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, color)

    def _render_ui(self, config):
        # --- UI Background ---
        ui_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH, 40)
        pygame.gfxdraw.box(self.screen, ui_rect, self.COLOR_UI_BG)

        # --- Energy Bar ---
        energy_pct = self.energy / self.MAX_ENERGY
        energy_color = self.COLOR_ENERGY_LOW
        if energy_pct > 0.66: energy_color = self.COLOR_ENERGY_HIGH
        elif energy_pct > 0.33: energy_color = self.COLOR_ENERGY_MED
        
        bar_width = 200
        bar_height = 12
        bar_x = 10
        bar_y = self.SCREEN_HEIGHT - 26
        
        pygame.draw.rect(self.screen, (30, 50, 80), (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, energy_color, (bar_x, bar_y, int(bar_width * energy_pct), bar_height), border_radius=3)

        # --- Text Info ---
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 32))

        souls_text = self.font_small.render(f"Souls Rescued: {self.total_souls_rescued}", True, self.COLOR_UI_TEXT)
        self.screen.blit(souls_text, (220, self.SCREEN_HEIGHT - 28))

        biome_text = self.font_small.render(f"Biome: {config['name']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(biome_text, (400, self.SCREEN_HEIGHT - 28))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "biome": self.biome_index + 1,
            "souls_rescued": self.total_souls_rescued,
        }

    def _get_random_pos(self, margin=0):
        return pygame.Vector2(
            random.randint(margin, self.SCREEN_WIDTH - margin),
            random.randint(margin, self.SCREEN_HEIGHT - margin - 40) # Avoid UI area
        )
    
    def _create_particles(self, pos, color, count, min_size, max_size, speed=3):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(0.5, 1.0) * speed
            lifespan = random.randint(20, 40)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color,
                "size": random.randint(min_size, max_size)
            })

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # --- Manual Play Example ---
    # The following code is not part of the Gymnasium environment and is for testing/demonstration purposes only.
    # To use the environment, you should import it and create an instance with gym.make().
    # For example:
    #
    # import gymnasium as gym
    # import your_module_name  # The file containing the GameEnv class
    # env = gym.make("your_module_name/GameEnv-v0")
    
    # To run this example, you might need to unset the SDL_VIDEODRIVER
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Ocean Portal Rescue")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Souls: {info['souls_rescued']}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()