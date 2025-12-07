import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:18:05.976236
# Source Brief: brief_01601.md
# Brief Index: 1601
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your central base from waves of predators by placing offensive vines. "
        "Manage resources and choose the right vine type to survive the onslaught."
    )
    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press space to build a vine and shift to cycle between vine types."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 3000

    # Colors
    COLOR_BG = (15, 25, 30)
    COLOR_BG_LAYER_2 = (20, 35, 40)
    COLOR_BG_LAYER_3 = (25, 45, 50)
    
    COLOR_BASE = (40, 200, 120)
    COLOR_BASE_GLOW = (40, 200, 120, 50)
    COLOR_HEALTH_HIGH = (80, 220, 140)
    COLOR_HEALTH_LOW = (220, 80, 80)
    COLOR_HEALTH_BG = (50, 50, 50)

    COLOR_PREDATOR = (255, 80, 80)
    COLOR_PREDATOR_GLOW = (255, 80, 80, 60)

    COLOR_PORTAL = (100, 150, 255)
    COLOR_PORTAL_GLOW = (100, 150, 255, 70)

    COLOR_PIERCER_VINE = (200, 255, 100)
    COLOR_BURST_VINE = (255, 180, 100)
    
    COLOR_PIERCER_PROJ = (220, 255, 150)
    COLOR_BURST_PROJ = (255, 200, 150)

    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_RESOURCE = (255, 220, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_small = pygame.font.Font(None, 20)
            self.font_medium = pygame.font.Font(None, 28)
            self.font_large = pygame.font.Font(None, 48)
        except (FileNotFoundError, pygame.error):
            # Fallback to a generic font if the default is not found
            font_name = pygame.font.get_default_font()
            self.font_small = pygame.font.Font(font_name, 20)
            self.font_medium = pygame.font.Font(font_name, 28)
            self.font_large = pygame.font.Font(font_name, 48)

        self.VINE_TYPES = [
            {
                "name": "Piercer", "cost": 50, "range": 180, "cooldown": 45, "color": self.COLOR_PIERCER_VINE,
                "attack": "projectile", "damage": 10, "proj_speed": 6
            },
            {
                "name": "Burst", "cost": 75, "range": 90, "cooldown": 90, "color": self.COLOR_BURST_VINE,
                "attack": "aoe", "damage": 5, "aoe_radius": 50
            },
        ]

        self.WAVE_DEFINITIONS = [
            {"count": 5, "health": 20, "speed": 0.8, "interval": 90},
            {"count": 8, "health": 20, "speed": 0.9, "interval": 75},
            {"count": 10, "health": 30, "speed": 0.9, "interval": 60},
            {"count": 12, "health": 30, "speed": 1.0, "interval": 50},
            {"count": 15, "health": 40, "speed": 1.1, "interval": 40},
            {"count": 20, "health": 40, "speed": 1.2, "interval": 30},
            {"count": 1, "health": 500, "speed": 0.7, "interval": 0}, # Boss
        ]

        # Initialize state variables
        self.cursor_pos = None
        self.base_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        self.background_elements = self._generate_background()
        
        # Run validation
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = 100.0
        self.max_base_health = 100.0
        
        self.resources = 120
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 100.0])
        
        self.vines = []
        self.predators = []
        self.projectiles = []
        self.particles = []
        
        self.wave_index = 0
        self.wave_spawn_timer = 120 # Initial delay
        self.wave_spawns_left = 0
        
        self.selected_vine_type_idx = 0
        self.prev_space_held = 0
        self.prev_shift_held = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1], action[2]
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1

        # --- Handle Input and Actions ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Update Game State ---
        self._update_vines()
        reward += self._update_projectiles()
        reward += self._update_predators()
        self._update_particles()
        reward += self._update_waves()

        # --- Calculate Rewards ---
        # Continuous negative reward for time passing to encourage efficiency
        reward -= 0.005 

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.base_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
            # sfx: game_over_sound
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True # Use truncated for time limit
            reward -= 50 # Penalize timeout
        elif self.game_won:
            self.game_over = True
            terminated = True
            reward += 100

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Update Logic ---
    
    def _handle_input(self, movement, space_held, shift_held):
        # Cursor movement
        cursor_speed = 6
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # Cycle vine type on shift press
        if shift_held and not self.prev_shift_held:
            self.selected_vine_type_idx = (self.selected_vine_type_idx + 1) % len(self.VINE_TYPES)
            # sfx: ui_switch_sound

        # Place vine on space press
        if space_held and not self.prev_space_held:
            self._try_place_vine()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _try_place_vine(self):
        vine_spec = self.VINE_TYPES[self.selected_vine_type_idx]
        if self.resources >= vine_spec["cost"]:
            # Check if too close to another vine
            min_dist = 30
            is_valid_placement = True
            for vine in self.vines:
                if np.linalg.norm(self.cursor_pos - vine["pos"]) < min_dist:
                    is_valid_placement = False
                    break
            
            if is_valid_placement:
                self.resources -= vine_spec["cost"]
                self.vines.append({
                    "pos": self.cursor_pos.copy(),
                    "spec": vine_spec,
                    "cooldown": 0,
                    "growth": 0.0, # Animation progress
                })
                # sfx: place_vine_sound
                # Spawn placement particles
                for _ in range(20):
                    self._create_particle(
                        self.cursor_pos, color=vine_spec["color"], lifetime=20, size=random.uniform(2, 5)
                    )

    def _update_vines(self):
        for vine in self.vines:
            # Animate growth
            if vine["growth"] < 1.0:
                vine["growth"] = min(1.0, vine["growth"] + 0.05)
                continue # Can't attack while growing

            vine["cooldown"] = max(0, vine["cooldown"] - 1)
            if vine["cooldown"] > 0:
                continue

            # Find target
            target = None
            min_dist = float('inf')
            for predator in self.predators:
                dist = np.linalg.norm(vine["pos"] - predator["pos"])
                if dist < vine["spec"]["range"] and dist < min_dist:
                    min_dist = dist
                    target = predator
            
            if target:
                vine["cooldown"] = vine["spec"]["cooldown"]
                # sfx: vine_attack_sound
                if vine["spec"]["attack"] == "projectile":
                    self.projectiles.append({
                        "pos": vine["pos"].copy(),
                        "target_pos": target["pos"].copy(),
                        "spec": vine["spec"],
                    })
                elif vine["spec"]["attack"] == "aoe":
                    self._create_aoe_blast(vine)

    def _create_aoe_blast(self, vine):
        radius = vine["spec"]["aoe_radius"]
        # Visual effect
        self.particles.append({
            "pos": vine["pos"].copy(), "vel": np.zeros(2), "lifetime": 15, "max_lifetime": 15,
            "color": vine["spec"]["color"], "size": 1, "type": "shockwave"
        })
        # Damage predators in radius
        for predator in self.predators:
            if np.linalg.norm(vine["pos"] - predator["pos"]) < radius:
                predator["health"] -= vine["spec"]["damage"]
                # Create hit particles
                for _ in range(3):
                    self._create_particle(
                        predator["pos"], color=(255,255,255), lifetime=10, size=random.uniform(1,3)
                    )

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            direction = proj["target_pos"] - proj["pos"]
            dist = np.linalg.norm(direction)
            if dist < proj["spec"]["proj_speed"]:
                proj["pos"] = proj["target_pos"]
            else:
                proj["pos"] += (direction / dist) * proj["spec"]["proj_speed"]
            
            # Collision detection
            hit = False
            for predator in self.predators:
                if np.linalg.norm(proj["pos"] - predator["pos"]) < 10: # Hitbox
                    predator["health"] -= proj["spec"]["damage"]
                    reward += 0.1 # Reward for hitting
                    hit = True
                    # sfx: hit_sound
                    for _ in range(5):
                        self._create_particle(proj["pos"], color=(255,255,255), lifetime=10, size=random.uniform(1,3))
                    break
            
            if hit or dist < 1:
                self.projectiles.remove(proj)
        return reward

    def _update_predators(self):
        reward = 0
        for predator in self.predators[:]:
            # Health check
            if predator["health"] <= 0:
                self.predators.remove(predator)
                reward += 5 # Reward for kill
                self.resources += 15
                # sfx: predator_death_sound
                for _ in range(30):
                    self._create_particle(predator["pos"], color=self.COLOR_PREDATOR, lifetime=25, size=random.uniform(1, 4))
                continue
            
            # Movement
            direction = self.base_pos - predator["pos"]
            dist_to_base = np.linalg.norm(direction)
            
            if dist_to_base < 15: # Reached base
                damage = 10
                self.base_health -= damage
                reward -= 0.5 * damage # Penalty for base damage
                self.predators.remove(predator)
                # sfx: base_damage_sound
                for _ in range(40):
                    self._create_particle(self.base_pos, color=self.COLOR_HEALTH_LOW, lifetime=30, size=random.uniform(2, 5))
                continue

            predator["pos"] += (direction / dist_to_base) * predator["speed"]
            predator["anim_offset"] += 1
        return reward

    def _update_waves(self):
        reward = 0
        if self.wave_spawns_left > 0:
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer <= 0:
                wave_spec = self.WAVE_DEFINITIONS[self.wave_index]
                self._spawn_predator(wave_spec["health"], wave_spec["speed"])
                self.wave_spawns_left -= 1
                self.wave_spawn_timer = wave_spec["interval"]
        elif not self.predators and self.wave_index < len(self.WAVE_DEFINITIONS): # Wave cleared
            reward += 20 # Wave clear reward
            self.wave_index += 1
            if self.wave_index < len(self.WAVE_DEFINITIONS):
                wave_spec = self.WAVE_DEFINITIONS[self.wave_index]
                self.wave_spawns_left = wave_spec["count"]
                self.wave_spawn_timer = 180 # Time between waves
                # sfx: wave_cleared_sound
            else:
                self.game_won = True
        return reward

    def _spawn_predator(self, health, speed):
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -10.0])
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10.0])
        elif edge == 2: # Left
            pos = np.array([-10.0, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
        else: # Right
            pos = np.array([self.SCREEN_WIDTH + 10.0, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
        
        self.predators.append({
            "pos": pos,
            "health": health,
            "max_health": health,
            "speed": speed,
            "anim_offset": self.np_random.integers(0, 50)
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_index + 1,
            "vines_active": len(self.vines),
            "predators_active": len(self.predators),
        }

    # --- Rendering ---
    
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _generate_background(self):
        elements = []
        for _ in range(20): # Deepest layer
            elements.append((self.COLOR_BG_LAYER_3, pygame.Rect(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT), random.randint(50, 150), random.randint(50, 150)), 0.1))
        for _ in range(15): # Mid layer
            elements.append((self.COLOR_BG_LAYER_2, pygame.Rect(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT), random.randint(40, 120), random.randint(40, 120)), 0.2))
        return elements

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Parallax background
        center_offset = (self.base_pos - self.cursor_pos)
        for color, rect, depth in self.background_elements:
            # Create a copy to move for parallax effect
            r = rect.copy()
            r.x += int(center_offset[0] * depth)
            r.y += int(center_offset[1] * depth)
            pygame.draw.ellipse(self.screen, color, r)
            
    def _render_game(self):
        self._render_particles()
        self._render_base()
        self._render_vines()
        self._render_projectiles()
        self._render_predators()
        self._render_cursor()

    def _render_base(self):
        x, y = int(self.base_pos[0]), int(self.base_pos[1])
        # Glow
        glow_radius = int(30 + 5 * math.sin(self.steps * 0.05))
        self._draw_glow(self.screen, self.COLOR_BASE_GLOW, (x, y), glow_radius)
        # Main base
        pygame.draw.circle(self.screen, self.COLOR_BASE, (x, y), 20)
        pygame.gfxdraw.aacircle(self.screen, x, y, 20, self.COLOR_BASE)

    def _render_vines(self):
        for vine in self.vines:
            x, y = int(vine["pos"][0]), int(vine["pos"][1])
            growth_factor = vine["growth"]
            
            # Portal
            portal_radius = int(15 * growth_factor)
            self._draw_glow(self.screen, self.COLOR_PORTAL_GLOW, (x, y), portal_radius + 5)
            pygame.draw.circle(self.screen, self.COLOR_PORTAL, (x, y), portal_radius)
            pygame.gfxdraw.aacircle(self.screen, x, y, portal_radius, self.COLOR_PORTAL)

            # Vine body
            vine_radius = int(8 * growth_factor)
            pygame.draw.circle(self.screen, vine["spec"]["color"], (x, y), vine_radius)
            pygame.gfxdraw.aacircle(self.screen, x, y, vine_radius, vine["spec"]["color"])

            # Range indicator
            if np.linalg.norm(self.cursor_pos - vine["pos"]) < 20:
                pygame.gfxdraw.aacircle(self.screen, x, y, int(vine["spec"]["range"]), (*vine["spec"]["color"], 100))

    def _render_projectiles(self):
        for proj in self.projectiles:
            x, y = int(proj["pos"][0]), int(proj["pos"][1])
            color = self.COLOR_PIERCER_PROJ
            pygame.draw.circle(self.screen, color, (x, y), 4)
            self._draw_glow(self.screen, (*color, 100), (x, y), 8)

    def _render_predators(self):
        for predator in self.predators:
            x, y = int(predator["pos"][0]), int(predator["pos"][1])
            wobble = math.sin(predator["anim_offset"] * 0.1) * 2
            radius = int(8 + wobble)
            
            # Glow
            self._draw_glow(self.screen, self.COLOR_PREDATOR_GLOW, (x, y), radius + 5)
            # Body
            pygame.draw.circle(self.screen, self.COLOR_PREDATOR, (x, y), radius)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_PREDATOR)
            
            # Health bar
            self._draw_health_bar(predator["pos"] - [12, 18], 24, 4, predator["health"] / predator["max_health"])

    def _render_cursor(self):
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x - 8, y), (x + 8, y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y - 8), (x, y + 8), 2)
        
        # Range indicator for selected vine
        vine_spec = self.VINE_TYPES[self.selected_vine_type_idx]
        color = (*vine_spec["color"], 50)
        pygame.gfxdraw.aacircle(self.screen, x, y, vine_spec["range"], color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, vine_spec["range"], color)

    def _render_particles(self):
        for p in self.particles:
            x, y = int(p["pos"][0]), int(p["pos"][1])
            alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
            color = (*p["color"], alpha)
            
            if p.get("type") == "shockwave":
                radius = int(self.VINE_TYPES[1]["aoe_radius"] * (1 - (p["lifetime"] / p["max_lifetime"])))
                pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
            else:
                size = int(p["size"] * (p["lifetime"] / p["max_lifetime"]))
                if size > 0:
                    surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, color, (size, size), size)
                    self.screen.blit(surf, (x - size, y - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Base Health Bar
        self._draw_health_bar(np.array([10, 10]), 200, 20, self.base_health / self.max_base_health)
        health_text = self.font_small.render(f"{int(self.base_health)} / {int(self.max_base_health)}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Resources
        res_text = self.font_medium.render(f"{self.resources}", True, self.COLOR_RESOURCE)
        pygame.draw.circle(self.screen, self.COLOR_RESOURCE, (15, 50), 10) # Resource icon
        pygame.gfxdraw.aacircle(self.screen, 15, 50, 10, self.COLOR_RESOURCE)
        self.screen.blit(res_text, (30, 40))

        # Wave info
        wave_str = f"Wave: {self.wave_index + 1} / {len(self.WAVE_DEFINITIONS)}"
        if self.game_won: wave_str = "VICTORY!"
        wave_text = self.font_medium.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 40))

        # Selected Vine UI
        vine_spec = self.VINE_TYPES[self.selected_vine_type_idx]
        vine_name_text = self.font_medium.render(vine_spec["name"], True, vine_spec["color"])
        vine_cost_text = self.font_medium.render(f"{vine_spec['cost']}", True, self.COLOR_RESOURCE)
        
        ui_pos_x = self.SCREEN_WIDTH - 150
        self.screen.blit(vine_name_text, (ui_pos_x, self.SCREEN_HEIGHT - 40))
        pygame.draw.circle(self.screen, self.COLOR_RESOURCE, (ui_pos_x + vine_name_text.get_width() + 15, self.SCREEN_HEIGHT - 30), 10)
        self.screen.blit(vine_cost_text, (ui_pos_x + vine_name_text.get_width() + 30, self.SCREEN_HEIGHT - 40))

        # Game Over/Win Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            msg = "VICTORY" if self.game_won else "GAME OVER"
            color = self.COLOR_HEALTH_HIGH if self.game_won else self.COLOR_HEALTH_LOW
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    # --- Helper Functions ---

    def _create_particle(self, pos, color, lifetime, size):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, 2.5)
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        self.particles.append({
            "pos": pos.copy(), "vel": vel, "lifetime": lifetime, "max_lifetime": lifetime,
            "color": color, "size": size
        })

    def _draw_health_bar(self, pos, width, height, progress):
        progress = np.clip(progress, 0, 1)
        bg_rect = pygame.Rect(int(pos[0]), int(pos[1]), width, height)
        fg_rect = pygame.Rect(int(pos[0]), int(pos[1]), int(width * progress), height)
        
        # Interpolate color from green to red
        r = self.COLOR_HEALTH_LOW[0] + (self.COLOR_HEALTH_HIGH[0] - self.COLOR_HEALTH_LOW[0]) * progress
        g = self.COLOR_HEALTH_LOW[1] + (self.COLOR_HEALTH_HIGH[1] - self.COLOR_HEALTH_LOW[1]) * progress
        b = self.COLOR_HEALTH_LOW[2] + (self.COLOR_HEALTH_HIGH[2] - self.COLOR_HEALTH_LOW[2]) * progress
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect)
        pygame.draw.rect(self.screen, (r,g,b), fg_rect)

    def _draw_glow(self, surface, color, pos, radius):
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (radius, radius), radius)
        surface.blit(surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You need to unset the headless mode for this to work
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Jungle Portal Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_score = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward

        # Transpose obs for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {total_score:.2f}, Info: {info}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_score = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()