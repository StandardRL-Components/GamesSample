import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:52:00.096732
# Source Brief: brief_00659.md
# Brief Index: 659
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

def draw_regular_polygon(surface, color, num_sides, radius, position, width=0, rotation=0):
    """
    Draws a regular polygon on a Pygame surface.
    
    Args:
        surface: The pygame.Surface to draw on.
        color: The color of the polygon.
        num_sides: The number of sides of the polygon.
        radius: The distance from the center to each vertex.
        position: The (x, y) center of the polygon.
        width: The thickness of the lines (0 for a filled polygon).
        rotation: The rotation of the polygon in radians.
    """
    if num_sides < 3:
        return

    points = []
    for i in range(num_sides):
        angle = rotation + (2 * math.pi * i / num_sides)
        x = position[0] + radius * math.cos(angle)
        y = position[1] + radius * math.sin(angle)
        points.append((int(x), int(y)))

    if width == 0:
        pygame.gfxdraw.filled_polygon(surface, points, color)
    
    pygame.gfxdraw.aapolygon(surface, points, color)

def draw_glow_circle(surface, color, center, radius, glow_strength=1):
    """
    Draws a glowing circle effect.
    """
    if radius <= 0: return
    center_x, center_y = int(center[0]), int(center[1])
    base_alpha = color[3] if len(color) == 4 else 50

    for i in range(glow_strength):
        # Calculate alpha for this layer of glow
        alpha = base_alpha * (1 - (i / glow_strength))**2
        if alpha < 1: continue
        
        # Calculate radius for this layer
        current_radius = int(radius + i * 2)

        # Create a temporary surface for the glow circle
        temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            temp_surf,
            (color[0], color[1], color[2], int(alpha)),
            (current_radius, current_radius),
            current_radius
        )
        
        # Blit the glow surface, centered
        surface.blit(temp_surf, (center_x - current_radius, center_y - current_radius), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your core from waves of geometric enemies by teleporting guardians to strategic locations. "
        "Use one guardian for damage pulses and another to create slowing fields."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the reticle. Press space to teleport the selected "
        "guardian and activate its ability. Press shift to switch between guardians."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 20 * self.FPS # 20 seconds per wave approx
        self.TOTAL_WAVES = 20

        # --- Visuals ---
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_CORE = (50, 255, 150)
        self.COLOR_CORE_GLOW = (50, 255, 150, 50)
        self.COLOR_GUARDIAN_1 = (0, 191, 255)  # Deep Sky Blue
        self.COLOR_GUARDIAN_2 = (255, 0, 255)  # Magenta
        self.COLOR_ENEMY = (255, 60, 60)
        self.COLOR_RETICLE = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (0, 255, 0)
        self.COLOR_HEALTH_BAR_BG = (100, 100, 100)

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # --- State Variables ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.core = None
        self.guardians = None
        self.enemies = None
        self.particles = None
        self.active_effects = None
        self.reticle_pos = None
        self.selected_guardian_idx = None
        self.wave_number = None
        self.wave_cooldown = None
        self.last_space_press = False
        self.last_shift_press = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.wave_cooldown = self.FPS * 3  # 3 seconds before first wave

        self.core = {
            "pos": np.array([self.WIDTH / 2, self.HEIGHT / 2]),
            "size": 25, "health": 100, "max_health": 100
        }

        self.guardians = [
            {"pos": np.array([50.0, self.HEIGHT / 2 - 60]), "spawn_pos": np.array([50.0, self.HEIGHT / 2 - 60]), "type": "damage", "cooldown": 0, "max_cooldown": self.FPS * 3, "color": self.COLOR_GUARDIAN_1},
            {"pos": np.array([50.0, self.HEIGHT / 2 + 60]), "spawn_pos": np.array([50.0, self.HEIGHT / 2 + 60]), "type": "slow", "cooldown": 0, "max_cooldown": self.FPS * 5, "color": self.COLOR_GUARDIAN_2}
        ]

        self.enemies = []
        self.particles = []
        self.active_effects = []
        self.reticle_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.selected_guardian_idx = 0
        self.last_space_press = False
        self.last_shift_press = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._handle_input(movement, space_held, shift_held)
        
        self._update_guardians()
        self._update_enemies()
        reward += self._update_collisions()
        self._update_effects()
        self._update_particles()
        self._update_wave_system()
        
        # Small penalty to encourage action
        reward -= 0.001 

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.core["health"] <= 0:
                reward = -100.0  # Use large terminal rewards
            elif self.wave_number > self.TOTAL_WAVES:
                reward = 100.0

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reticle_speed = 10.0
        if movement == 1: self.reticle_pos[1] -= reticle_speed
        elif movement == 2: self.reticle_pos[1] += reticle_speed
        elif movement == 3: self.reticle_pos[0] -= reticle_speed
        elif movement == 4: self.reticle_pos[0] += reticle_speed
        self.reticle_pos = np.clip(self.reticle_pos, [0, 0], [self.WIDTH, self.HEIGHT])

        # Rising-edge detection for discrete actions
        if shift_held and not self.last_shift_press:
            self.selected_guardian_idx = (self.selected_guardian_idx + 1) % len(self.guardians)
            # sfx: UI_Select
        self.last_shift_press = shift_held

        if space_held and not self.last_space_press:
            guardian = self.guardians[self.selected_guardian_idx]
            if guardian["cooldown"] <= 0:
                old_pos = guardian["pos"].copy()
                guardian["pos"] = self.reticle_pos.copy()
                guardian["cooldown"] = guardian["max_cooldown"]
                # sfx: Teleport
                self._create_teleport_particles(old_pos)
                self._create_teleport_particles(guardian["pos"])
                self._activate_guardian_ability(guardian)
        self.last_space_press = space_held

    def _activate_guardian_ability(self, guardian):
        if guardian["type"] == "damage":
            # sfx: Damage_Pulse
            self.active_effects.append({"type": "damage_pulse", "pos": guardian["pos"].copy(), "radius": 10, "max_radius": 80, "duration": self.FPS * 0.5, "color": guardian["color"], "hit_enemies": set()})
        elif guardian["type"] == "slow":
            # sfx: Slow_Field
            self.active_effects.append({"type": "slow_field", "pos": guardian["pos"].copy(), "radius": 100, "duration": self.FPS * 3, "color": guardian["color"]})

    def _update_guardians(self):
        for g in self.guardians:
            g["cooldown"] = max(0, g["cooldown"] - 1)

    def _update_enemies(self):
        for enemy in self.enemies:
            slow_factor = 1.0
            for effect in self.active_effects:
                if effect["type"] == "slow_field":
                    if np.linalg.norm(enemy["pos"] - effect["pos"]) < effect["radius"]:
                        slow_factor = 0.3
                        break
            
            direction = self.core["pos"] - enemy["pos"]
            dist_to_core = np.linalg.norm(direction)
            if dist_to_core > 1:
                direction /= dist_to_core
            enemy["pos"] += direction * enemy["speed"] * slow_factor
            enemy["rotation"] += 0.05

    def _update_collisions(self):
        reward = 0
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            if i in enemies_to_remove: continue

            if np.linalg.norm(enemy["pos"] - self.core["pos"]) < self.core["size"] + enemy["size"]:
                damage = enemy["damage"]
                self.core["health"] -= damage
                reward -= 0.1 * damage
                self._create_explosion(enemy["pos"], self.COLOR_CORE)
                enemies_to_remove.append(i)
                # sfx: Core_Hit
                continue

            for effect in self.active_effects:
                if effect["type"] == "damage_pulse":
                    if np.linalg.norm(enemy["pos"] - effect["pos"]) < effect["radius"] and i not in effect["hit_enemies"]:
                        enemy["health"] -= 2
                        reward += 0.1
                        effect["hit_enemies"].add(i)
                        self._create_hit_particles(enemy["pos"])
                        # sfx: Enemy_Hit

            if enemy["health"] <= 0:
                reward += 1.0
                self.score += 10
                self._create_explosion(enemy["pos"], self.COLOR_ENEMY)
                enemies_to_remove.append(i)
                # sfx: Enemy_Explode

        for i in sorted(list(set(enemies_to_remove)), reverse=True):
            del self.enemies[i]
        
        self.core["health"] = max(0, self.core["health"])
        return reward

    def _update_effects(self):
        for effect in self.active_effects:
            effect["duration"] -= 1
            if effect["type"] == "damage_pulse":
                effect["radius"] += (effect["max_radius"] - 10) / (self.FPS * 0.5)
        self.active_effects = [e for e in self.active_effects if e["duration"] > 0]

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _update_wave_system(self):
        if not self.enemies and self.wave_number <= self.TOTAL_WAVES:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                self.wave_number += 1
                if self.wave_number <= self.TOTAL_WAVES:
                    self._spawn_wave()
                    self.wave_cooldown = self.FPS * 3

    def _spawn_wave(self):
        num_enemies = 3 + self.wave_number
        enemy_speed = 1.0 + self.wave_number * 0.05
        enemy_health = 3 + (self.wave_number // 5)
        enemy_damage = 5

        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0: pos = np.array([self.np_random.uniform(0, self.WIDTH), -20.0])
            elif side == 1: pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20.0])
            elif side == 2: pos = np.array([-20.0, self.np_random.uniform(0, self.HEIGHT)])
            else: pos = np.array([self.WIDTH + 20.0, self.np_random.uniform(0, self.HEIGHT)])
            
            self.enemies.append({"pos": pos.astype(np.float32), "size": self.np_random.uniform(10, 15), "health": enemy_health, "max_health": enemy_health, "speed": enemy_speed, "damage": enemy_damage, "sides": self.np_random.integers(4, 7), "rotation": self.np_random.uniform(0, 2 * math.pi)})

    def _check_termination(self):
        return (self.core["health"] <= 0 or (self.wave_number > self.TOTAL_WAVES and not self.enemies))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for effect in self.active_effects:
            alpha_mult = effect["duration"] / (self.FPS * (0.5 if effect['type'] == 'damage_pulse' else 3))
            if effect["type"] == "damage_pulse":
                color = (*effect["color"], int(200 * alpha_mult))
                pygame.gfxdraw.aacircle(self.screen, int(effect["pos"][0]), int(effect["pos"][1]), int(effect["radius"]), color)
            elif effect["type"] == "slow_field":
                color = (*effect["color"], int(80 * alpha_mult))
                draw_glow_circle(self.screen, color, effect["pos"], effect["radius"], 2)

        core_pos = (int(self.core["pos"][0]), int(self.core["pos"][1]))
        core_rect = pygame.Rect(core_pos[0] - self.core["size"], core_pos[1] - self.core["size"], self.core["size"]*2, self.core["size"]*2)
        pygame.draw.rect(self.screen, self.COLOR_CORE, core_rect, 0, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_CORE_GLOW, core_rect.inflate(10, 10), 0, border_radius=5)

        for i, g in enumerate(self.guardians):
            draw_regular_polygon(self.screen, g["color"], 3, 12, g["pos"], width=2, rotation=-math.pi/2)
            if i == self.selected_guardian_idx:
                draw_glow_circle(self.screen, (*g["color"], 100), g["pos"], 20, 2)
            pygame.draw.aaline(self.screen, (*g["color"], 50), g["pos"], g["spawn_pos"])
            pygame.gfxdraw.aacircle(self.screen, int(g["spawn_pos"][0]), int(g["spawn_pos"][1]), 8, g["color"])
            pygame.gfxdraw.filled_circle(self.screen, int(g["spawn_pos"][0]), int(g["spawn_pos"][1]), 8, (*g["color"], 80))

        for enemy in self.enemies:
            draw_regular_polygon(self.screen, self.COLOR_ENEMY, enemy["sides"], enemy["size"], enemy["pos"], width=2, rotation=enemy["rotation"])
            if enemy["health"] < enemy["max_health"]:
                hp_pos_y = enemy["pos"][1] - enemy["size"] - 10
                hp_rect = pygame.Rect(enemy["pos"][0] - 10, hp_pos_y, 20, 3)
                hp_fill = hp_rect.width * (enemy["health"] / enemy["max_health"])
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, hp_rect)
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (hp_rect.x, hp_rect.y, hp_fill, hp_rect.height))

        for p in self.particles:
            alpha = max(0, 255 * (p["lifetime"] / p["max_lifetime"]))
            color = (*p["color"], int(alpha))
            s = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(s, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])))

        if not self.game_over:
            r_pos = (int(self.reticle_pos[0]), int(self.reticle_pos[1]))
            pygame.draw.aaline(self.screen, self.COLOR_RETICLE, (r_pos[0]-10, r_pos[1]), (r_pos[0]+10, r_pos[1]))
            pygame.draw.aaline(self.screen, self.COLOR_RETICLE, (r_pos[0], r_pos[1]-10), (r_pos[0], r_pos[1]+10))

    def _render_ui(self):
        wave_text = f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}" if self.wave_number <= self.TOTAL_WAVES else "WAVE: COMPLETE"
        self.screen.blit(self.font_ui.render(wave_text, True, self.COLOR_UI_TEXT), (10, 10))
        self.screen.blit(self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT), (10, 35))

        health_text_surf = self.font_ui.render("CORE HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text_surf, health_text_surf.get_rect(topright=(self.WIDTH - 10, 10)))
        bar_rect = pygame.Rect(self.WIDTH - 160, 35, 150, 15)
        health_fill = bar_rect.width * (self.core["health"] / self.core["max_health"])
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bar_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_rect.x, bar_rect.y, health_fill, bar_rect.height), border_radius=2)

        for g in self.guardians:
            cd_pct = 1.0 - (g["cooldown"] / g["max_cooldown"])
            bar_rect = pygame.Rect(g["spawn_pos"][0] - 25, g["spawn_pos"][1] + 15, 50, 5)
            cd_fill = bar_rect.width * cd_pct
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bar_rect, border_radius=2)
            pygame.draw.rect(self.screen, g["color"], (bar_rect.x, bar_rect.y, cd_fill, bar_rect.height), border_radius=2)

        if self.game_over:
            msg, color = ("VICTORY", self.COLOR_CORE) if self.core["health"] > 0 else ("CORE DESTROYED", self.COLOR_ENEMY)
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            bg_surf = pygame.Surface(text_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, text_rect.topleft)
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "core_health": self.core["health"], "enemies_remaining": len(self.enemies)}

    def _create_explosion(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifetime": lifetime, "max_lifetime": lifetime, "size": self.np_random.uniform(1, 4), "color": color})

    def _create_teleport_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifetime = self.np_random.integers(10, 20)
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifetime": lifetime, "max_lifetime": lifetime, "size": self.np_random.uniform(1, 3), "color": (255, 255, 0)})

    def _create_hit_particles(self, pos):
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifetime = self.np_random.integers(5, 10)
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifetime": lifetime, "max_lifetime": lifetime, "size": self.np_random.uniform(1, 2), "color": (255, 255, 255)})

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv()
    
    # --- Manual Play Example ---
    # Controls:
    # Arrows: Move reticle
    # Space: Teleport/Activate ability
    # Left Shift: Cycle selected guardian
    # Q: Quit
    
    obs, info = env.reset()
    done = False
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Guardian Teleport Defense")
    clock = pygame.time.Clock()
    
    # Action state
    action = [0, 0, 0] # [movement, space, shift]
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Get key presses
        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0]
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Buttons
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        if keys[pygame.K_q]: done = True
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()