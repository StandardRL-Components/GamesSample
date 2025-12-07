import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:21:06.760494
# Source Brief: brief_02731.md
# Brief Index: 2731
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend and cultivate a system of planets against encroaching cosmic radiation. "
        "Use your energy to shoot down threats and terraform worlds to help them grow."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the reticle. Press space to shoot. "
        "Press shift to select a planet. Hold shift and press space to terraform."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_WHITE = (240, 240, 240)
    COLOR_TEXT = (220, 220, 255)
    COLOR_RETICLE = (255, 255, 0)
    COLOR_ENERGY = (255, 200, 0)
    COLOR_ENERGY_BG = (60, 50, 20)
    COLOR_RADIATION = (255, 50, 100)
    
    # Planet Types
    PLANET_TYPES = {
        "TERRAN": {
            "color": (50, 220, 120), "growth_rate": 0.05, "resistance": 1.0, "name": "Terran"
        },
        "OCEANIC": {
            "color": (80, 150, 255), "growth_rate": 0.03, "resistance": 1.5, "name": "Oceanic"
        },
        "VOLCANIC": {
            "color": (255, 120, 50), "growth_rate": 0.08, "resistance": 0.7, "name": "Volcanic"
        },
    }

    # Gameplay
    RETICLE_SPEED = 15
    INITIAL_ENERGY = 1000
    MAX_ENERGY = 1000
    ENERGY_REGEN_RATE = 2.0  # per second
    SHOT_COST = 50
    SHOT_COOLDOWN = 0.2  # seconds
    SHOT_RADIUS = 30
    TERRAFORM_COST = 250
    TERRAFORM_COOLDOWN = 1.0 # seconds

    RADIATION_BASE_SPEED = 100 # pixels per second
    RADIATION_DAMAGE = 10
    
    WIN_CONDITION_SIZE = 1000
    MAX_STEPS = 5000

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.nebula_bg = None
        self.stars = []
        
        # State variables are initialized in reset()
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.energy = self.INITIAL_ENERGY
        self.reticle_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        
        self.planets = [
            {
                "pos": pygame.Vector2(self.WIDTH * 0.3, self.HEIGHT * 0.5),
                "size": 30.0,
                "type": "TERRAN",
                "id": 0,
                "hit_timer": 0,
            },
            {
                "pos": pygame.Vector2(self.WIDTH * 0.7, self.HEIGHT * 0.5),
                "size": 25.0,
                "type": "TERRAN",
                "id": 1,
                "hit_timer": 0,
            },
        ]
        self.radiation = []
        self.visual_effects = []
        
        self.selected_planet_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.shoot_cooldown_timer = 0
        self.terraform_cooldown_timer = 0
        
        self.unlocked_planet_types = ["TERRAN"]
        self.terraforming_unlocked = False
        self.difficulty_spawn_rate = 0.2
        self.time_since_last_spawn = 0.0

        if self.nebula_bg is None:
            self.nebula_bg = self._create_nebula_background()
            self.stars = self._create_stars()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        dt = self.clock.tick(self.FPS) / 1000.0
        
        self._handle_input(action, dt)
        self._update_game_state(dt)
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        reward = self.reward_this_step

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action, dt):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Reticle Movement ---
        if movement == 1: self.reticle_pos.y -= self.RETICLE_SPEED * (dt * self.FPS)
        if movement == 2: self.reticle_pos.y += self.RETICLE_SPEED * (dt * self.FPS)
        if movement == 3: self.reticle_pos.x -= self.RETICLE_SPEED * (dt * self.FPS)
        if movement == 4: self.reticle_pos.x += self.RETICLE_SPEED * (dt * self.FPS)
        self.reticle_pos.x = np.clip(self.reticle_pos.x, 0, self.WIDTH)
        self.reticle_pos.y = np.clip(self.reticle_pos.y, 0, self.HEIGHT)

        # --- Cooldowns ---
        if self.shoot_cooldown_timer > 0: self.shoot_cooldown_timer -= dt
        if self.terraform_cooldown_timer > 0: self.terraform_cooldown_timer -= dt

        # --- Planet Selection (Shift Press) ---
        shift_pressed = shift_held and not self.last_shift_held
        if shift_pressed and self.planets:
            self.selected_planet_idx = (self.selected_planet_idx + 1) % len(self.planets)
            # Sound: UI_Select.wav

        # --- Terraforming (Shift + Space) ---
        if shift_held and space_held and self.terraforming_unlocked and self.planets:
            selected_planet = self.planets[self.selected_planet_idx]
            dist = self.reticle_pos.distance_to(selected_planet["pos"])
            if dist < selected_planet["size"] and self.energy >= self.TERRAFORM_COST and self.terraform_cooldown_timer <= 0:
                self._terraform_planet(selected_planet)
                
        # --- Shooting (Space) ---
        elif space_held:
            if self.energy >= self.SHOT_COST and self.shoot_cooldown_timer <= 0:
                self._fire_shot()
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
    
    def _fire_shot(self):
        self.energy -= self.SHOT_COST
        self.shoot_cooldown_timer = self.SHOT_COOLDOWN
        # Sound: Laser_Shoot.wav
        
        # Visual effect for shot
        self.visual_effects.append({"type": "shot", "pos": self.reticle_pos, "radius": self.SHOT_RADIUS, "timer": 0.15})
        
        # Check for hits on radiation
        hit_radiation = []
        for rad in self.radiation:
            if self.reticle_pos.distance_to(rad["pos"]) < self.SHOT_RADIUS:
                hit_radiation.append(rad)
        
        for rad in hit_radiation:
            self.radiation.remove(rad)
            self.score += 5
            self.reward_this_step += 0.1
            # Sound: Explosion.wav
            self.visual_effects.append({"type": "explosion", "pos": rad["pos"], "radius": 15, "timer": 0.2})

    def _terraform_planet(self, planet):
        self.energy -= self.TERRAFORM_COST
        self.terraform_cooldown_timer = self.TERRAFORM_COOLDOWN
        
        current_type_idx = self.unlocked_planet_types.index(planet["type"])
        next_type_idx = (current_type_idx + 1) % len(self.unlocked_planet_types)
        planet["type"] = self.unlocked_planet_types[next_type_idx]
        
        self.score += 20
        self.reward_this_step += 5
        # Sound: Terraform_Success.wav
        self.visual_effects.append({"type": "terraform", "pos": planet["pos"], "radius": planet["size"], "timer": 0.5})

    def _update_game_state(self, dt):
        # --- Energy Regen ---
        self.energy = min(self.MAX_ENERGY, self.energy + self.ENERGY_REGEN_RATE * dt * self.FPS)

        # --- Planet Growth & Energy Consumption ---
        total_growth_this_step = 0
        for p in self.planets:
            planet_info = self.PLANET_TYPES[p["type"]]
            growth = planet_info["growth_rate"] * dt * self.FPS
            energy_cost = growth * 5 # Growth costs energy
            if self.energy > energy_cost:
                self.energy -= energy_cost
                p["size"] += growth
                total_growth_this_step += growth
            if p["hit_timer"] > 0:
                p["hit_timer"] -= dt

        self.reward_this_step += 0.01 * total_growth_this_step
        self.score += total_growth_this_step / 10.0

        # --- Update & Move Radiation ---
        for rad in self.radiation:
            rad["pos"] += rad["vel"] * dt
            # Screen wrap-around
            if rad["pos"].x < 0: rad["pos"].x = self.WIDTH
            if rad["pos"].x > self.WIDTH: rad["pos"].x = 0
            if rad["pos"].y < 0: rad["pos"].y = self.HEIGHT
            if rad["pos"].y > self.HEIGHT: rad["pos"].y = 0
            
            # Update trail
            rad["trail"].append(rad["pos"].copy())
            if len(rad["trail"]) > 10:
                rad["trail"].pop(0)

        # --- Radiation Collision with Planets ---
        destroyed_planets = []
        for rad in self.radiation[:]:
            for p in self.planets:
                if rad["pos"].distance_to(p["pos"]) < p["size"]:
                    self.radiation.remove(rad)
                    planet_info = self.PLANET_TYPES[p["type"]]
                    damage = self.RADIATION_DAMAGE / planet_info["resistance"]
                    p["size"] = max(0, p["size"] - damage)
                    p["hit_timer"] = 0.2
                    # Sound: Planet_Hit.wav
                    if p["size"] <= 0:
                        if p not in destroyed_planets:
                            destroyed_planets.append(p)
                    break 

        for p in destroyed_planets:
            self.planets.remove(p)
            # Sound: Planet_Destroyed.wav
            self.visual_effects.append({"type": "explosion", "pos": p["pos"], "radius": 40, "timer": 0.4})
            if self.planets and self.selected_planet_idx >= len(self.planets):
                self.selected_planet_idx = 0

        # --- Spawn New Radiation ---
        self.time_since_last_spawn += dt
        spawn_interval = 1.0 / self.difficulty_spawn_rate
        if self.time_since_last_spawn > spawn_interval:
            self.time_since_last_spawn -= spawn_interval
            self._spawn_radiation()

        # --- Update Unlocks & Difficulty ---
        if not self.terraforming_unlocked and self.score >= 100:
            self.terraforming_unlocked = True
            self.reward_this_step += 1 # Small reward for unlocking a mechanic
        
        new_unlocks = False
        if self.score >= 50 and "OCEANIC" not in self.unlocked_planet_types:
            self.unlocked_planet_types.append("OCEANIC")
            new_unlocks = True
        if self.score >= 150 and "VOLCANIC" not in self.unlocked_planet_types:
            self.unlocked_planet_types.append("VOLCANIC")
            new_unlocks = True
        
        if new_unlocks:
            self.reward_this_step += 1
        
        self.difficulty_spawn_rate = 0.2 + (self.score // 100) * 0.05
        
        # --- Update Visual Effects ---
        for effect in self.visual_effects[:]:
            effect["timer"] -= dt
            if effect["timer"] <= 0:
                self.visual_effects.remove(effect)

    def _spawn_radiation(self):
        edge = random.randint(0, 3)
        if edge == 0: # top
            pos = pygame.Vector2(random.uniform(0, self.WIDTH), -10)
        elif edge == 1: # bottom
            pos = pygame.Vector2(random.uniform(0, self.WIDTH), self.HEIGHT + 10)
        elif edge == 2: # left
            pos = pygame.Vector2(-10, random.uniform(0, self.HEIGHT))
        else: # right
            pos = pygame.Vector2(self.WIDTH + 10, random.uniform(0, self.HEIGHT))
        
        target = pygame.Vector2(random.uniform(self.WIDTH*0.2, self.WIDTH*0.8), random.uniform(self.HEIGHT*0.2, self.HEIGHT*0.8))
        vel = (target - pos).normalize() * self.RADIATION_BASE_SPEED
        self.radiation.append({"pos": pos, "vel": vel, "trail": []})

    def _check_termination(self):
        if not self.planets:
            self.game_over = True
            self.reward_this_step -= 100
            return True
        
        total_size = sum(p["size"] for p in self.planets)
        if total_size >= self.WIN_CONDITION_SIZE:
            self.game_over = True
            self.win_condition_met = True
            self.reward_this_step += 100
            return True
            
        return False

    def _get_observation(self):
        self.screen.blit(self.nebula_bg, (0, 0))
        self._render_stars()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Radiation
        for rad in self.radiation:
            # Trail
            if "trail" in rad and len(rad["trail"]) > 1:
                for i, p in enumerate(rad["trail"]):
                    alpha = int(255 * (i / len(rad["trail"])))
                    color = (*self.COLOR_RADIATION, alpha)
                    pygame.draw.circle(self.screen, color, (int(p.x), int(p.y)), 2)
            # Head
            pygame.gfxdraw.filled_circle(self.screen, int(rad["pos"].x), int(rad["pos"].y), 4, self.COLOR_RADIATION)
            pygame.gfxdraw.aacircle(self.screen, int(rad["pos"].x), int(rad["pos"].y), 4, self.COLOR_RADIATION)

        # Render Planets
        for i, p in enumerate(self.planets):
            pos_int = (int(p["pos"].x), int(p["pos"].y))
            size_int = int(p["size"])
            if size_int <= 0: continue
            
            planet_info = self.PLANET_TYPES[p["type"]]
            color = planet_info["color"]
            
            # Hit flash effect
            if p["hit_timer"] > 0:
                flash_alpha = int(255 * (p["hit_timer"] / 0.2))
                flash_color = (*self.COLOR_WHITE, flash_alpha)
                self._draw_glow_circle(pos_int, size_int, flash_color, 3)

            # Main planet body and glow
            self._draw_glow_circle(pos_int, size_int, color, 5, glow_factor=1.5)

            # Selection indicator
            if i == self.selected_planet_idx:
                num_points = 20
                angle_step = 360 / num_points
                points = []
                radius = size_int + 10 + 3 * math.sin(pygame.time.get_ticks() * 0.005)
                for j in range(num_points):
                    angle = math.radians(j * angle_step + pygame.time.get_ticks() * 0.05)
                    x = pos_int[0] + radius * math.cos(angle)
                    y = pos_int[1] + radius * math.sin(angle)
                    points.append((int(x), int(y)))
                pygame.draw.aalines(self.screen, self.COLOR_WHITE, True, points)

        # Render Visual Effects
        for effect in self.visual_effects:
            pos_int = (int(effect["pos"].x), int(effect["pos"].y))
            if effect["type"] == "shot":
                alpha = int(255 * (effect["timer"] / 0.15))
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(effect["radius"]), (*self.COLOR_ENERGY, alpha))
            elif effect["type"] == "explosion":
                alpha = int(255 * (effect["timer"] / 0.4))
                self._draw_glow_circle(pos_int, int(effect["radius"] * (1 - effect["timer"]/0.4)), (*self.COLOR_RADIATION, alpha), 3)
            elif effect["type"] == "terraform":
                alpha = int(255 * (effect["timer"] / 0.5))
                radius = int(effect["radius"] + 20 * (1 - effect["timer"]/0.5))
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, (*self.COLOR_WHITE, alpha))
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius-2, (*self.COLOR_WHITE, alpha))

        # Render Reticle
        pos = (int(self.reticle_pos.x), int(self.reticle_pos.y))
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (pos[0] - 10, pos[1]), (pos[0] + 10, pos[1]), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (pos[0], pos[1] - 10), (pos[0], pos[1] + 10), 2)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, self.COLOR_RETICLE)

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # --- Energy Bar ---
        bar_width, bar_height = 200, 20
        bar_x, bar_y = (self.WIDTH - bar_width) // 2, self.HEIGHT - bar_height - 10
        energy_ratio = np.clip(self.energy / self.MAX_ENERGY, 0, 1)
        
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (bar_x, bar_y, int(bar_width * energy_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)
        
        energy_text = self.font_small.render("ENERGY", True, self.COLOR_TEXT)
        self.screen.blit(energy_text, (bar_x - energy_text.get_width() - 10, bar_y + 2))

        # --- Selected Planet Info ---
        if self.planets:
            p = self.planets[self.selected_planet_idx]
            p_info = self.PLANET_TYPES[p["type"]]
            info_text = f"Selected: {p_info['name']} Planet | Size: {int(p['size'])}"
            info_surf = self.font_small.render(info_text, True, self.COLOR_TEXT)
            self.screen.blit(info_surf, (10, self.HEIGHT - info_surf.get_height() - 10))
        
        # --- Game Over / Win Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.win_condition_met else "ALL PLANETS LOST"
            text_surf = self.font_large.render(message, True, self.COLOR_WHITE)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "planet_count": len(self.planets),
            "total_planet_size": sum(p["size"] for p in self.planets) if self.planets else 0
        }

    def _create_nebula_background(self):
        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        surface.fill(self.COLOR_BG)
        for _ in range(50):
            color = random.choice([(40, 20, 80, 20), (20, 40, 80, 20), (80, 20, 40, 20)])
            pos = (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT))
            radius = random.randint(50, 200)
            
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            surface.blit(temp_surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)
        return surface

    def _create_stars(self):
        stars = []
        for _ in range(200):
            stars.append({
                "pos": pygame.Vector2(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)),
                "brightness": random.uniform(50, 200),
                "period": random.uniform(2, 5)
            })
        return stars

    def _render_stars(self):
        for star in self.stars:
            brightness = star["brightness"] * (0.75 + 0.25 * math.sin(pygame.time.get_ticks() * 0.001 * star["period"]))
            color = (int(brightness), int(brightness), int(brightness))
            pygame.draw.circle(self.screen, color, star["pos"], 1)

    def _draw_glow_circle(self, pos, radius, color, num_layers=5, glow_factor=1.5):
        if radius <= 0: return
        max_glow_radius = int(radius * glow_factor)
        for i in range(num_layers):
            alpha = int(color[3] * (1 - i / num_layers)**2) if len(color) == 4 else int(255 * (1 - i / num_layers)**2)
            current_radius = radius + (max_glow_radius - radius) * (i / (num_layers - 1)) if num_layers > 1 else radius
            
            if current_radius <= 0: continue
            
            temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*color[:3], alpha), (current_radius, current_radius), current_radius)
            self.screen.blit(temp_surf, (pos[0] - current_radius, pos[1] - current_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def close(self):
        pygame.quit()
        
# Example of how to run the environment
if __name__ == '__main__':
    # To run with display, comment out the os.environ line at the top of the file
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Running in headless mode. No display will be shown.")
        print("Comment out 'os.environ.setdefault(\"SDL_VIDEODRIVER\", \"dummy\")' at the top of the file to run with a window.")
        
        # Run a short episode without rendering to screen
        env = GameEnv()
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < 500:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
        print("Headless test run finished.")
        env.close()

    else:
        env = GameEnv(render_mode="rgb_array")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Planet Cultivator")
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        done = False
        
        while not done:
            # Action mapping for human player
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset() # Reset on 'R' key
        
        env.close()