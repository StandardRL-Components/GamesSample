import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:42:57.344551
# Source Brief: brief_02959.md
# Brief Index: 2959
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Heal a blighted meadow by launching restorative seeds. "
        "Aim carefully to counter the spreading corruption and restore the land's health."
    )
    user_guide = (
        "Controls: Use ←→ arrows to aim and ↑↓ arrows to adjust power. "
        "Press space to launch a seed and shift to cycle between seed types."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.GRAVITY = 0.15
        self.PLAYER_POS = (self.WIDTH // 2, self.HEIGHT - 20)

        # --- Colors & Fonts ---
        self.COLOR_BG = (15, 30, 25)
        self.COLOR_BLIGHT = (50, 20, 40)
        self.COLOR_BLIGHT_CORE = (90, 40, 70)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_HEALTH_BAR_FG = (0, 200, 50)
        self.COLOR_TRAJECTORY = (255, 255, 255)

        self.SEED_TYPES = [
            {
                "name": "Aqua",
                "color": (0, 200, 255),
                "heal_radius": 50,
                "heal_potency": 0.5,
            },
            {
                "name": "Verdant",
                "color": (50, 255, 100),
                "heal_radius": 80,
                "heal_potency": 0.3,
            },
            {
                "name": "Sunfire",
                "color": (255, 150, 0),
                "heal_radius": 30,
                "heal_potency": 1.0,
            },
        ]
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 24)
        self.font_seed = pygame.font.Font(None, 20)

        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.meadow_health = 0.0
        self.blight_nodes = []
        self.plants = []
        self.seeds_in_flight = []
        self.particles = []
        self.aim_angle = 0.0
        self.aim_power = 0.0
        self.current_seed_index = 0
        self.plant_cooldown = 0
        self.blight_spread_rate = 0.0
        self.previous_space_state = 0
        self.previous_shift_state = 0
        self.total_meadow_area = self.WIDTH * self.HEIGHT * 0.9 # Playable area
        self.action_planted = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.blight_nodes = []
        self.plants = []
        self.seeds_in_flight = []
        self.particles = []

        self.aim_angle = -math.pi / 2
        self.aim_power = 8.0
        self.current_seed_index = 0
        self.plant_cooldown = 0
        self.blight_spread_rate = 0.01

        self.previous_space_state = 0
        self.previous_shift_state = 0
        self.action_planted = False

        # Initialize blight
        initial_blight_coverage = 0.20
        target_blight_area = self.total_meadow_area * initial_blight_coverage
        current_blight_area = 0
        while current_blight_area < target_blight_area:
            radius = self.np_random.uniform(15, 40)
            x = self.np_random.uniform(radius, self.WIDTH - radius)
            y = self.np_random.uniform(radius, self.HEIGHT * 0.7 - radius)
            self.blight_nodes.append({"pos": pygame.Vector2(x, y), "radius": radius, "max_radius": radius * 1.5})
            current_blight_area += math.pi * radius**2

        self._calculate_meadow_health()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        blight_area_before = self._get_blight_area()

        self._handle_actions(action)
        self._update_game_state()
        
        self.plant_cooldown = max(0, self.plant_cooldown - 1)
        
        blight_area_after = self._get_blight_area()
        blight_reduced_percent = (blight_area_before - blight_area_after) / self.total_meadow_area
        reward += max(0, blight_reduced_percent * 100 * 0.1) # Continuous reward for blight reduction

        self._calculate_meadow_health()
        terminated = self._check_termination()
        
        if self.action_planted:
            reward += 1 # Event-based reward for planting

        if terminated:
            self.game_over = True
            if self.meadow_health >= 100:
                reward += 100
            elif self.meadow_health <= 0:
                reward -= 100
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.action_planted = False

        # --- Aiming ---
        if movement == 1: # Up
            self.aim_power = min(12.0, self.aim_power + 0.2)
        elif movement == 2: # Down
            self.aim_power = max(4.0, self.aim_power - 0.2)
        elif movement == 3: # Left
            self.aim_angle -= 0.05
        elif movement == 4: # Right
            self.aim_angle += 0.05
        self.aim_angle = max(-math.pi + 0.1, min(-0.1, self.aim_angle))

        # --- Cycle Seed ---
        if shift_held and not self.previous_shift_state:
            self.current_seed_index = (self.current_seed_index + 1) % len(self.SEED_TYPES)
            # sfx: seed_swap
        self.previous_shift_state = shift_held

        # --- Plant Seed ---
        if space_held and not self.previous_space_state and self.plant_cooldown == 0:
            self.action_planted = True
            self.plant_cooldown = 15 # 0.5s cooldown at 30fps
            seed_type = self.SEED_TYPES[self.current_seed_index]
            vel = pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * self.aim_power
            self.seeds_in_flight.append({
                "pos": pygame.Vector2(self.PLAYER_POS),
                "vel": vel,
                "type": seed_type
            })
            # sfx: seed_throw

        self.previous_space_state = space_held
    
    def _update_game_state(self):
        # Update seeds in flight
        for seed in self.seeds_in_flight[:]:
            seed["vel"].y += self.GRAVITY
            seed["pos"] += seed["vel"]
            if seed["pos"].y > self.HEIGHT or seed["pos"].x < 0 or seed["pos"].x > self.WIDTH:
                 self.seeds_in_flight.remove(seed)
            elif seed["vel"].y > 0 and self._is_on_ground(seed["pos"]):
                self.seeds_in_flight.remove(seed)
                self._create_plant(seed["pos"], seed["type"])
                # sfx: seed_land
        
        # Update plants (grow and heal)
        for plant in self.plants[:]:
            plant["age"] += 1
            if plant["age"] < plant["growth_time"]:
                plant["current_radius"] = (plant["age"] / plant["growth_time"]) * plant["type"]["heal_radius"]
            else:
                plant["current_radius"] = plant["type"]["heal_radius"]
                # Healing pulse
                if plant["age"] % 10 == 0:
                    healed_something = False
                    for blight in self.blight_nodes[:]:
                        dist = plant["pos"].distance_to(blight["pos"])
                        if dist < plant["current_radius"] + blight["radius"]:
                            blight["radius"] -= plant["type"]["heal_potency"]
                            healed_something = True
                            if blight["radius"] <= 1:
                                self.blight_nodes.remove(blight)
                                self._spawn_particles(blight["pos"], 10, (150, 255, 200))
                    if healed_something:
                        # sfx: heal_pulse
                        self._spawn_particles(plant["pos"], 5, plant["type"]["color"], is_ring=True)

            if plant["age"] > plant["lifespan"]:
                self.plants.remove(plant)

        # Update blight (spread)
        if self.steps % 200 == 0 and self.steps > 0:
            self.blight_spread_rate += 0.005

        if self.np_random.random() < self.blight_spread_rate and self.blight_nodes:
            parent_blight = self.np_random.choice(self.blight_nodes)
            angle = self.np_random.uniform(0, 2 * math.pi)
            dist = parent_blight["radius"] + self.np_random.uniform(5, 20)
            new_pos = parent_blight["pos"] + pygame.Vector2(math.cos(angle), math.sin(angle)) * dist
            
            if 0 < new_pos.x < self.WIDTH and 0 < new_pos.y < self.HEIGHT * 0.8:
                new_radius = self.np_random.uniform(5, 15)
                self.blight_nodes.append({"pos": new_pos, "radius": new_radius, "max_radius": new_radius * self.np_random.uniform(1.5, 3.0)})
        
        # Grow existing blight
        for blight in self.blight_nodes:
            if blight["radius"] < blight["max_radius"]:
                blight["radius"] += 0.01

        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _is_on_ground(self, pos):
        # Simple ground check, can be made more complex (e.g., terrain height map)
        return pos.y >= self.HEIGHT - 40

    def _create_plant(self, pos, seed_type):
        self.plants.append({
            "pos": pos,
            "type": seed_type,
            "age": 0,
            "growth_time": 60,
            "lifespan": 300,
            "current_radius": 0,
        })
        self._spawn_particles(pos, 20, seed_type["color"])

    def _spawn_particles(self, pos, count, color, is_ring=False):
        for _ in range(count):
            if is_ring:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 2)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(0.5, 3)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            
            lifespan = self.np_random.integers(15, 31)
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifespan": lifespan, "max_life": lifespan, "color": color})

    def _get_blight_area(self):
        # This is an approximation and doesn't handle overlaps, but is sufficient for game logic
        return sum(math.pi * b["radius"]**2 for b in self.blight_nodes)

    def _calculate_meadow_health(self):
        blight_area = self._get_blight_area()
        blight_ratio = min(1.0, blight_area / self.total_meadow_area)
        self.meadow_health = 100 * (1 - blight_ratio)

    def _check_termination(self):
        return (
            self.meadow_health <= 0 or
            self.meadow_health >= 100 or
            self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render blight
        for blight in self.blight_nodes:
            pos = (int(blight["pos"].x), int(blight["pos"].y))
            radius = int(blight["radius"])
            if radius > 0:
                # Pulsating effect
                pulse_factor = 0.9 + 0.1 * math.sin(pygame.time.get_ticks() * 0.002 + blight["pos"].x)
                core_radius = int(radius * 0.5 * pulse_factor)
                
                # Outer transparent circle
                surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(surface, self.COLOR_BLIGHT + (100,), (radius, radius), radius)
                self.screen.blit(surface, (pos[0] - radius, pos[1] - radius))
                
                # Inner core
                if core_radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], core_radius, self.COLOR_BLIGHT_CORE)
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], core_radius, self.COLOR_BLIGHT_CORE)

        # Render plants
        for plant in self.plants:
            pos = (int(plant["pos"].x), int(plant["pos"].y))
            radius = int(plant["current_radius"])
            color = plant["type"]["color"]
            if radius > 0:
                # Base circle
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*color, 50))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*color, 100))
                
                # Pulsating core
                pulse = 0.8 + 0.2 * math.sin(pygame.time.get_ticks() * 0.005 + plant["pos"].y)
                core_radius = int(radius * 0.3 * pulse)
                if core_radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], core_radius, color)
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], core_radius, color)

        # Render seeds in flight
        for seed in self.seeds_in_flight:
            pos = (int(seed["pos"].x), int(seed["pos"].y))
            color = seed["type"]["color"]
            pygame.draw.circle(self.screen, color, pos, 4)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 4, 1)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_life"]))
            color = (*p["color"], alpha)
            pos = (int(p["pos"].x), int(p["pos"].y))
            surface = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(surface, color, (2, 2), 2)
            self.screen.blit(surface, (pos[0]-2, pos[1]-2))

        # Render aiming trajectory
        if not self.game_over:
            pos = pygame.Vector2(self.PLAYER_POS)
            vel = pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * self.aim_power
            for _ in range(30):
                vel.y += self.GRAVITY
                pos += vel
                if _ % 3 == 0:
                    pygame.draw.circle(self.screen, self.COLOR_TRAJECTORY, (int(pos.x), int(pos.y)), 1)
    
    def _render_ui(self):
        # Health Bar
        health_percent = max(0, self.meadow_health / 100.0)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, bar_width * health_percent, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, bar_height), 1)
        health_text = self.font_main.render(f"Meadow Health: {int(self.meadow_health)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 40))

        # Current Seed Type
        seed_type = self.SEED_TYPES[self.current_seed_index]
        seed_box_rect = pygame.Rect(self.WIDTH - 110, self.HEIGHT - 40, 100, 30)
        pygame.draw.rect(self.screen, (0,0,0,150), seed_box_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, seed_box_rect, 1, border_radius=5)
        
        pygame.draw.circle(self.screen, seed_type["color"], (seed_box_rect.left + 15, seed_box_rect.centery), 8)
        seed_name_text = self.font_seed.render(seed_type["name"], True, self.COLOR_UI_TEXT)
        self.screen.blit(seed_name_text, (seed_box_rect.left + 30, seed_box_rect.centery - 8))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "meadow_health": self.meadow_health,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    # The main loop is for human play and visualization.
    # It will not work with SDL_VIDEODRIVER="dummy".
    # To run, unset the environment variable:
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Blighted Meadow")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True

    action = [0, 0, 0] # no-op, released, released

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()