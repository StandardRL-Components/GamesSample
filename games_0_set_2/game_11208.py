import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:37:07.458865
# Source Brief: brief_01208.md
# Brief Index: 1208
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a cloud to water a field of flowers.

    **Objective:** Make all 15 flowers bloom.
    **Losing Condition:** 3 or more flowers wilt.
    **Actions:** Move the cloud and rain.

    **Visuals:**
    - A bright, cartoonish style.
    - The player-controlled cloud is white and animated.
    - Flowers visually progress from seedlings to vibrant blooms or wilted husks.
    - Rain is represented by a particle system.
    - UI elements clearly display the game state.

    **Gameplay:**
    - The agent moves the cloud over flowers and holds the 'rain' action.
    - Raining increases a flower's water level.
    - Watered flowers slowly dry out over time (evaporation).
    - A flower blooms when its water level reaches the maximum.
    - A flower wilts if its water level drops to zero after being watered.
    - Rain on one flower causes a smaller "splash" effect on adjacent flowers.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a cloud to water a field of flowers. Make all the flowers bloom before too many wilt from drying out."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move the cloud. Hold space to rain on the flowers below."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (130, 203, 115) # Light green field
    COLOR_BG_DARK = (110, 180, 95)
    COLOR_CLOUD = (240, 245, 255)
    COLOR_CLOUD_SHADOW = (180, 190, 210)
    COLOR_RAIN = (173, 216, 230)
    COLOR_STEM = (50, 150, 50)
    COLOR_WILTED = (139, 69, 19)
    COLOR_WATER_BAR_BG = (60, 60, 60, 180)
    COLOR_WATER_BAR_FILL = (30, 144, 255)
    FLOWER_COLORS = [
        (255, 105, 180), # Pink
        (255, 215, 0),   # Gold
        (230, 100, 100), # Red
        (138, 43, 226),  # BlueViolet
        (255, 165, 0),   # Orange
    ]

    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Game Parameters
    MAX_STEPS = 1500
    NUM_FLOWERS = 15
    FLOWER_ROWS = 3
    FLOWER_COLS = 5
    FLOWER_MAX_WATER = 20
    WILT_THRESHOLD = 3
    CLOUD_SPEED = 6.0
    RAIN_RATE = 0.5  # Water per frame when raining
    RAIN_COOLDOWN = 3 # Frames between rain ticks
    EVAPORATION_RATE = 0.02 # Water lost per frame
    SPLASH_FACTOR = 0.1 # Water splashed to neighbors

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cloud_pos = None
        self.cloud_render_pos = None
        self.cloud_bob_angle = 0
        self.flowers = []
        self.flower_grid = []
        self.flower_neighbors = {}
        self.wilted_count = 0
        self.rain_cooldown_timer = 0
        self.particles = deque()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wilted_count = 0

        self.cloud_pos = pygame.Vector2(self.WIDTH / 2, 80)
        self.cloud_render_pos = self.cloud_pos.copy()
        self.cloud_bob_angle = random.uniform(0, 2 * math.pi)

        self.rain_cooldown_timer = 0
        self.particles.clear()

        self._initialize_flowers()

        return self._get_observation(), self._get_info()

    def _initialize_flowers(self):
        self.flowers.clear()
        self.flower_grid = [([None] * self.FLOWER_COLS) for _ in range(self.FLOWER_ROWS)]
        
        x_padding = self.WIDTH * 0.15
        y_padding = self.HEIGHT * 0.2
        x_spacing = (self.WIDTH - 2 * x_padding) / (self.FLOWER_COLS - 1)
        y_spacing = (self.HEIGHT - y_padding - 150) / (self.FLOWER_ROWS - 1)

        flower_id = 0
        for r in range(self.FLOWER_ROWS):
            for c in range(self.FLOWER_COLS):
                pos = pygame.Vector2(
                    x_padding + c * x_spacing,
                    150 + r * y_spacing
                )
                flower = {
                    "id": flower_id,
                    "pos": pos,
                    "rect": pygame.Rect(pos.x - 20, pos.y - 20, 40, 40),
                    "water": 0.0,
                    "state": "growing",  # growing, bloomed, wilted
                    "has_been_watered": False,
                    "bloom_color": random.choice(self.FLOWER_COLORS),
                    "bloom_animation": 0.0 # 0 to 1
                }
                self.flowers.append(flower)
                self.flower_grid[r][c] = flower
                flower_id += 1
        
        self._calculate_neighbors()

    def _calculate_neighbors(self):
        self.flower_neighbors.clear()
        for r in range(self.FLOWER_ROWS):
            for c in range(self.FLOWER_COLS):
                current_flower = self.flower_grid[r][c]
                if not current_flower: continue
                
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.FLOWER_ROWS and 0 <= nc < self.FLOWER_COLS:
                            neighbor_flower = self.flower_grid[nr][nc]
                            if neighbor_flower:
                                neighbors.append(neighbor_flower)
                self.flower_neighbors[current_flower["id"]] = neighbors


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        is_raining = action[1] == 1
        reward = 0

        self._move_cloud(movement)
        
        if is_raining:
            reward += self._handle_rain()
        
        reward += self._update_flowers()
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            bloomed_count = sum(1 for f in self.flowers if f["state"] == "bloomed")
            if bloomed_count == self.NUM_FLOWERS:
                reward += 100 # Victory bonus
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _move_cloud(self, movement):
        target_pos = self.cloud_pos.copy()
        if movement == 1: # Up
            target_pos.y -= self.CLOUD_SPEED
        elif movement == 2: # Down
            target_pos.y += self.CLOUD_SPEED
        elif movement == 3: # Left
            target_pos.x -= self.CLOUD_SPEED
        elif movement == 4: # Right
            target_pos.x += self.CLOUD_SPEED
        
        target_pos.x = np.clip(target_pos.x, 0, self.WIDTH)
        target_pos.y = np.clip(target_pos.y, 0, self.HEIGHT / 2)
        self.cloud_pos = target_pos

    def _handle_rain(self):
        reward = 0
        self.rain_cooldown_timer -= 1
        if self.rain_cooldown_timer > 0:
            return 0
        
        self.rain_cooldown_timer = self.RAIN_COOLDOWN
        
        target_flower = None
        for flower in self.flowers:
            if flower["rect"].collidepoint(self.cloud_pos) and flower["state"] == "growing":
                target_flower = flower
                break
        
        if target_flower:
            # Add water to target flower
            water_added = min(self.RAIN_RATE, self.FLOWER_MAX_WATER - target_flower["water"])
            if water_added > 0:
                target_flower["water"] += water_added
                target_flower["has_been_watered"] = True
                reward += 0.1 * water_added
                # SFX: Gentle rain sound

                # Add splash to neighbors
                splash_water = water_added * self.SPLASH_FACTOR
                neighbors = self.flower_neighbors[target_flower["id"]]
                for neighbor in neighbors:
                    if neighbor["state"] == "growing":
                        neighbor_water_added = min(splash_water, self.FLOWER_MAX_WATER - neighbor["water"])
                        neighbor["water"] += neighbor_water_added
                        neighbor["has_been_watered"] = True
                        reward += 0.1 * neighbor_water_added
            
            # Create rain particles
            for _ in range(3):
                self.particles.append({
                    "pos": self.cloud_pos + pygame.Vector2(random.uniform(-20, 20), 15),
                    "vel": pygame.Vector2(0, random.uniform(3, 5)),
                    "life": 100
                })
        return reward

    def _update_flowers(self):
        reward = 0
        self.wilted_count = 0
        
        for f in self.flowers:
            # Update bloom animation
            if f["state"] == "bloomed":
                f["bloom_animation"] = min(1.0, f["bloom_animation"] + 0.05)
            
            # Handle evaporation and state changes for growing flowers
            if f["state"] == "growing":
                if f["water"] > 0:
                    f["water"] -= self.EVAPORATION_RATE
                
                if f["water"] >= self.FLOWER_MAX_WATER:
                    f["state"] = "bloomed"
                    reward += 10
                    # SFX: Bloom pop sound
                    self._create_burst_particles(f["pos"], f["bloom_color"])
                elif f["water"] <= 0 and f["has_been_watered"]:
                    f["state"] = "wilted"
                    reward -= 10
                    # SFX: Wilt sound
                    self._create_burst_particles(f["pos"], self.COLOR_WILTED)

            if f["state"] == "wilted":
                self.wilted_count += 1
                
        return reward

    def _check_termination(self):
        if self.wilted_count >= self.WILT_THRESHOLD:
            return True
        if all(f["state"] == "bloomed" for f in self.flowers):
            return True
        return False

    def _get_observation(self):
        # Interpolate cloud position for smooth rendering
        self.cloud_render_pos.x += (self.cloud_pos.x - self.cloud_render_pos.x) * 0.2
        self.cloud_render_pos.y += (self.cloud_pos.y - self.cloud_render_pos.y) * 0.2
        self.cloud_bob_angle += 0.05

        self.screen.fill(self.COLOR_BG)
        self._render_background_details()
        self._render_flowers()
        self._render_particles()
        self._render_cloud()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_details(self):
        for i in range(20):
            # Use a seed-based random for consistent background
            r = random.Random(i)
            x = r.randint(0, self.WIDTH)
            y = r.randint(120, self.HEIGHT)
            size = r.randint(10, 30)
            pygame.draw.circle(self.screen, self.COLOR_BG_DARK, (x, y), size)

    def _render_flowers(self):
        for f in self.flowers:
            pos = f["pos"]
            water_ratio = np.clip(f["water"] / self.FLOWER_MAX_WATER, 0, 1)

            if f["state"] == "growing":
                # Stem
                stem_height = 15 + 20 * water_ratio
                pygame.draw.line(self.screen, self.COLOR_STEM, (pos.x, pos.y + 20), (pos.x, pos.y + 20 - stem_height), 3)
                # Bud
                bud_size = 3 + 7 * water_ratio
                pygame.draw.circle(self.screen, self.COLOR_STEM, (int(pos.x), int(pos.y + 20 - stem_height)), int(bud_size))

            elif f["state"] == "bloomed":
                anim_scale = 0.5 + 0.5 * f["bloom_animation"] # Ease-out effect
                # Stem
                pygame.draw.line(self.screen, self.COLOR_STEM, (pos.x, pos.y + 20), (pos.x, pos.y - 15), 4)
                # Petals
                petal_size = int(15 * anim_scale)
                for i in range(5):
                    angle = (i / 5) * 2 * math.pi + self.steps * 0.01
                    petal_pos = (
                        int(pos.x + math.cos(angle) * petal_size * 0.8),
                        int(pos.y - 15 + math.sin(angle) * petal_size * 0.8)
                    )
                    pygame.gfxdraw.filled_circle(self.screen, petal_pos[0], petal_pos[1], int(petal_size * 0.6), f["bloom_color"])
                # Center
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y - 15), int(8 * anim_scale), (255, 255, 100))
            
            elif f["state"] == "wilted":
                pygame.draw.line(self.screen, self.COLOR_WILTED, (pos.x, pos.y + 20), (pos.x+5, pos.y - 10), 3)
                pygame.draw.circle(self.screen, self.COLOR_WILTED, (int(pos.x+5), int(pos.y - 10)), 5)

            # Water bar
            bar_width = 40
            bar_height = 6
            bar_x = pos.x - bar_width / 2
            bar_y = pos.y + 25
            
            # Background
            s = pygame.Surface((bar_width, bar_height), pygame.SRCALPHA)
            s.fill(self.COLOR_WATER_BAR_BG)
            self.screen.blit(s, (bar_x, bar_y))
            # Fill
            fill_width = bar_width * water_ratio
            pygame.draw.rect(self.screen, self.COLOR_WATER_BAR_FILL, (bar_x, bar_y, fill_width, bar_height))
            # Border
            pygame.draw.rect(self.screen, (255,255,255), (bar_x, bar_y, bar_width, bar_height), 1)

    def _render_cloud(self):
        pos = self.cloud_render_pos
        bob = math.sin(self.cloud_bob_angle) * 3
        
        # Shadow
        shadow_pos = (int(pos.x), int(pos.y + 30))
        pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1], 40, 10, (0, 0, 0, 50))

        # Cloud body
        c_pos = (int(pos.x), int(pos.y + bob))
        pygame.gfxdraw.filled_circle(self.screen, c_pos[0], c_pos[1], 20, self.COLOR_CLOUD_SHADOW)
        pygame.gfxdraw.filled_circle(self.screen, c_pos[0] - 20, c_pos[1]+5, 18, self.COLOR_CLOUD_SHADOW)
        pygame.gfxdraw.filled_circle(self.screen, c_pos[0] + 20, c_pos[1]+5, 22, self.COLOR_CLOUD_SHADOW)
        
        pygame.gfxdraw.filled_circle(self.screen, c_pos[0], c_pos[1]-2, 20, self.COLOR_CLOUD)
        pygame.gfxdraw.filled_circle(self.screen, c_pos[0] - 20, c_pos[1]+3, 18, self.COLOR_CLOUD)
        pygame.gfxdraw.filled_circle(self.screen, c_pos[0] + 20, c_pos[1]+3, 22, self.COLOR_CLOUD)

    def _render_particles(self):
        # Rain and effect particles
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                if "radius" in p: # Burst particle
                    alpha = int(255 * (p["life"] / p["max_life"]))
                    color = p["color"] + (alpha,)
                    s = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
                    pygame.draw.circle(s, color, (p["radius"], p["radius"]), p["radius"])
                    self.screen.blit(s, (p["pos"].x - p["radius"], p["pos"].y - p["radius"]))
                else: # Rain particle
                    pygame.draw.line(self.screen, self.COLOR_RAIN, p["pos"], p["pos"] + pygame.Vector2(0, 5), 2)

    def _create_burst_particles(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "life": random.randint(20, 40),
                "max_life": 40,
                "radius": random.randint(2, 5),
                "color": color
            })

    def _render_ui(self):
        wilted_text = self.font_small.render(f"Wilted: {self.wilted_count} / {self.WILT_THRESHOLD}", True, (80, 20, 20))
        self.screen.blit(wilted_text, (10, 10))
        
        bloomed_count = sum(1 for f in self.flowers if f["state"] == "bloomed")
        bloomed_text = self.font_small.render(f"Bloomed: {bloomed_count} / {self.NUM_FLOWERS}", True, (20, 80, 20))
        self.screen.blit(bloomed_text, (10, 30))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))

        bloomed_count = sum(1 for f in self.flowers if f["state"] == "bloomed")
        if bloomed_count == self.NUM_FLOWERS:
            text = "FIELD IN FULL BLOOM!"
            color = (100, 255, 100)
        else:
            text = "GAME OVER"
            color = (255, 100, 100)
            
        text_surface = self.font_large.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wilted_flowers": self.wilted_count,
            "bloomed_flowers": sum(1 for f in self.flowers if f["state"] == "bloomed"),
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually
    # It will create a window and render the environment's output
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for manual play
    env = GameEnv(render_mode="rgb_array")
    
    # The validate_implementation method is not part of the standard Gym API,
    # but it's useful for debugging during development.
    # We'll call it here to ensure the environment is set up correctly.
    try:
        # A simple validation routine to check the basic API.
        # This is not a standard part of the Gym API, but it's a good practice.
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        print("✓ Basic API validation passed.")
    except Exception as e:
        print(f"✗ Basic API validation failed: {e}")

    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Flower Gardener")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement_action = 0 # No-op
        rain_action = 0 # Released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
            
        if keys[pygame.K_SPACE]:
            rain_action = 1

        action = [movement_action, rain_action, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)

        # The environment returns an RGB array, we need to convert it back to a surface
        # and display it. The observation is (H, W, C), but pygame needs (W, H) surface.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            
        clock.tick(30) # Run at 30 FPS

    env.close()