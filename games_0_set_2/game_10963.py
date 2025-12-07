import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:23:15.885659
# Source Brief: brief_00963.md
# Brief Index: 963
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a tower defense game.

    The player defends a city by switching between four elemental guardians
    to counter waves of elemental enemies. The player can spend resources
    earned from defeating enemies to upgrade their guardians.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement):
        - If Upgrade Menu is OFF: Select active guardian (1:Fire, 2:Water, 3:Nature, 4:Lightning).
        - If Upgrade Menu is ON: Navigate highlighted upgrade (1:Up, 2:Down).
    - `actions[1]` (Space):
        - If Upgrade Menu is ON: Purchase the highlighted upgrade.
    - `actions[2]` (Shift):
        - Toggles the Upgrade Menu ON/OFF.

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +0.1 for each enemy defeated.
    - -0.5 for each point of city health lost.
    - +1.0 for surviving a wave.
    - +100 for reaching wave 50.
    - -100 (terminal) if the city is destroyed.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your city from waves of elemental enemies by switching between four powerful guardians. "
        "Upgrade your guardians to survive as long as you can."
    )
    user_guide = (
        "Controls: Use arrow keys (1-4) to select a guardian. Press Shift to toggle the upgrade menu, "
        "navigate with ↑/↓, and press Space to purchase upgrades."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CITY_Y = 350
    MAX_STEPS = 30000 # Increased for longer gameplay potential

    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_UI_BG = (30, 20, 60, 200)
    COLOR_UI_BORDER = (150, 140, 200)
    COLOR_UI_TEXT = (230, 230, 255)
    COLOR_UI_HIGHLIGHT = (255, 255, 100)
    COLOR_CITY = (100, 200, 255)
    COLOR_HEALTH_BAR = (40, 200, 80)
    COLOR_HEALTH_BAR_BG = (100, 20, 20)
    
    ELEMENTS = {
        "fire":    {"color": (255, 80, 20),   "strong_vs": "nature", "weak_vs": "water"},
        "water":   {"color": (60, 150, 255),  "strong_vs": "fire",   "weak_vs": "nature"},
        "nature":  {"color": (80, 220, 80),   "strong_vs": "water",  "weak_vs": "fire"},
        "lightning": {"color": (255, 255, 100), "strong_vs": "water",  "weak_vs": "none"}
    }
    GUARDIAN_ELEMENTS = ["fire", "water", "nature", "lightning"]

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
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.wave_cooldown = 0
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        self.city_health = 0
        self.max_city_health = 100
        self.resources = 0
        
        self.guardians = []
        self.active_guardian_idx = 0
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.upgrade_menu_open = False
        self.highlighted_upgrade_idx = 0
        self.upgrades = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.city_health = self.max_city_health
        self.resources = 50

        self.guardians = self._initialize_guardians()
        self.active_guardian_idx = 0
        
        self.enemies.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.upgrade_menu_open = False
        self.highlighted_upgrade_idx = 0
        self.upgrades = self._get_available_upgrades()

        self.prev_space_held = False
        self.prev_shift_held = False

        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1

        self._handle_input(action)

        if not self.upgrade_menu_open:
            reward += self._update_game_logic()
        
        # Check for wave completion
        if self.enemies_spawned >= self.enemies_in_wave and not self.enemies:
            if self.wave_cooldown == 0: # Check to only give reward once
                reward += 1.0 # Wave survived reward
                self.score += 100
                self.resources += 25 + self.wave_number * 5
                if self.wave_number == 50:
                    reward += 100.0
                self._start_new_wave()
        
        terminated = self.city_health <= 0 or self.steps >= self.MAX_STEPS
        if self.city_health <= 0 and not self.game_over:
            reward = -100.0
            self.game_over = True
            # sfx: game_over_sound

        # The render call is external now, but we still need to update game state for observation
        self._update_ui_state()

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is false
            self._get_info()
        )

    def _update_ui_state(self):
        """Updates timers and other non-physics state for rendering."""
        if self.wave_cooldown > 0:
            self.wave_cooldown -= 1
        elif self.enemies_spawned < self.enemies_in_wave:
            self.enemy_spawn_cooldown -= 1
            if self.enemy_spawn_cooldown <= 0:
                self._spawn_enemy()
                self.enemy_spawn_cooldown = max(10, 45 - self.wave_number)

    def _initialize_guardians(self):
        guardians = []
        num_guardians = len(self.GUARDIAN_ELEMENTS)
        for i, element in enumerate(self.GUARDIAN_ELEMENTS):
            angle = math.pi / 2 + (math.pi / (num_guardians + 1)) * (i - (num_guardians - 1) / 2)
            x = self.SCREEN_WIDTH / 2 + 150 * math.cos(angle)
            y = self.CITY_Y - 100 + 60 * math.sin(angle)
            guardians.append({
                "element": element,
                "pos": pygame.Vector2(x, y),
                "damage": 10,
                "attack_speed": 1.5, # attacks per second
                "cooldown": 0,
                "level": 1
            })
        return guardians

    def _handle_input(self, action):
        movement, space_action, shift_action = action[0], action[1], action[2]

        # Use rising edge detection for single-press actions
        space_pressed = space_action == 1 and not self.prev_space_held
        shift_pressed = shift_action == 1 and not self.prev_shift_held
        self.prev_space_held = space_action == 1
        self.prev_shift_held = shift_action == 1

        if shift_pressed:
            self.upgrade_menu_open = not self.upgrade_menu_open
            # sfx: menu_toggle

        if self.upgrade_menu_open:
            if movement == 1: # Up
                self.highlighted_upgrade_idx = (self.highlighted_upgrade_idx - 1) % len(self.upgrades)
            elif movement == 2: # Down
                self.highlighted_upgrade_idx = (self.highlighted_upgrade_idx + 1) % len(self.upgrades)
            
            if space_pressed:
                self._purchase_upgrade()
        else:
            if 1 <= movement <= 4:
                new_idx = movement - 1
                if new_idx < len(self.guardians):
                    self.active_guardian_idx = new_idx
                    # sfx: guardian_switch

    def _update_game_logic(self):
        step_reward = 0

        # Update Guardians (attack)
        active_guardian = self.guardians[self.active_guardian_idx]
        active_guardian["cooldown"] = max(0, active_guardian["cooldown"] - 1)
        if active_guardian["cooldown"] == 0 and self.enemies:
            self._guardian_attack(active_guardian)
            active_guardian["cooldown"] = 60 / active_guardian["attack_speed"]

        # Update Projectiles
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.projectiles.remove(p)
                continue
        
        # Update Enemies
        for e in self.enemies[:]:
            e["pos"].y += e["speed"]
            if e["pos"].y > self.CITY_Y:
                damage = int(e["health"] / 5) # Damage based on remaining health
                self.city_health -= damage
                step_reward -= 0.5 * damage
                self._create_particles(e["pos"], self.ELEMENTS[e["element"]]["color"], 30, 3.0)
                self.enemies.remove(e)
                # sfx: city_damage
                continue

        # Update Particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

        # Collision Detection (Projectile <-> Enemy)
        for p in self.projectiles[:]:
            for e in self.enemies[:]:
                if p["pos"].distance_to(e["pos"]) < e["size"]:
                    damage = self._calculate_damage(p["damage"], p["element"], e["element"])
                    e["health"] -= damage
                    self._create_particles(p["pos"], self.ELEMENTS[p["element"]]["color"], 10, 2.0)
                    if p in self.projectiles: self.projectiles.remove(p)
                    # sfx: enemy_hit
                    if e["health"] <= 0:
                        self._create_particles(e["pos"], (200, 200, 200), 50, 4.0)
                        self.enemies.remove(e)
                        self.score += 10
                        self.resources += 5
                        step_reward += 0.1
                        # sfx: enemy_death
                    break
        
        return step_reward

    def _guardian_attack(self, guardian):
        # Find closest enemy
        if not self.enemies: return
        closest_enemy = min(self.enemies, key=lambda e: guardian["pos"].distance_to(e["pos"]))
        
        direction = (closest_enemy["pos"] - guardian["pos"]).normalize()
        
        self.projectiles.append({
            "pos": guardian["pos"].copy(),
            "vel": direction * 8,
            "element": guardian["element"],
            "damage": guardian["damage"],
            "lifespan": 100
        })
        # sfx: projectile_fire

    def _calculate_damage(self, base_damage, projectile_element, enemy_element):
        element_data = self.ELEMENTS[projectile_element]
        if element_data["strong_vs"] == enemy_element:
            return base_damage * 1.75
        if element_data["weak_vs"] == enemy_element:
            return base_damage * 0.5
        return base_damage

    def _start_new_wave(self):
        self.wave_number += 1
        self.wave_cooldown = 180 # 3 seconds at 60fps
        self.enemies_in_wave = 5 + self.wave_number * 2
        self.enemies_spawned = 0
        self.enemy_spawn_cooldown = 0
        
        # Scale enemy stats
        self.base_enemy_health = 20 * (1.05 ** (self.wave_number - 1))
        self.base_enemy_speed = 0.5 * (1.02 ** (self.wave_number - 1))

    def _spawn_enemy(self):
        if self.enemies_spawned < self.enemies_in_wave:
            self.enemies_spawned += 1
            element = random.choice(list(self.ELEMENTS.keys()))
            self.enemies.append({
                "pos": pygame.Vector2(random.randint(50, self.SCREEN_WIDTH - 50), -20),
                "element": element,
                "health": self.base_enemy_health * random.uniform(0.9, 1.1),
                "max_health": self.base_enemy_health,
                "speed": self.base_enemy_speed * random.uniform(0.9, 1.1),
                "size": 12
            })

    def _get_available_upgrades(self):
        upgrades = []
        for i, g in enumerate(self.guardians):
            upgrades.append({
                "name": f"{g['element'].capitalize()} Dmg ({g['level']})",
                "cost": 20 * g["level"],
                "action": lambda i=i: self._upgrade_stat(i, "damage", 5)
            })
            upgrades.append({
                "name": f"{g['element'].capitalize()} Spd ({g['level']})",
                "cost": 25 * g["level"],
                "action": lambda i=i: self._upgrade_stat(i, "attack_speed", 0.2)
            })
        return upgrades

    def _upgrade_stat(self, guardian_idx, stat, amount):
        self.guardians[guardian_idx][stat] += amount
        self.guardians[guardian_idx]["level"] += 1
        # sfx: upgrade_success

    def _purchase_upgrade(self):
        if not self.upgrades: return
        upgrade = self.upgrades[self.highlighted_upgrade_idx]
        if self.resources >= upgrade["cost"]:
            self.resources -= upgrade["cost"]
            upgrade["action"]()
            self.upgrades = self._get_available_upgrades() # Refresh upgrades list
            self.highlighted_upgrade_idx = min(self.highlighted_upgrade_idx, len(self.upgrades) - 1)
        else:
            # sfx: upgrade_fail
            pass

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": random.randint(15, 40),
                "color": color,
                "radius": random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "city_health": self.city_health,
            "resources": self.resources,
            "active_guardian": self.guardians[self.active_guardian_idx]["element"]
        }

    def _render_game(self):
        # Draw background stars
        for i in range(100):
            # Use a deterministic seed based on 'i' so stars don't flicker
            random.seed(i)
            color_val = random.randint(50, 150)
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            pygame.gfxdraw.pixel(self.screen, x, y, (color_val, color_val, color_val))
        random.seed() # Reset seed

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 40))
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p["color"], alpha), (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (p["pos"][0] - p["radius"], p["pos"][1] - p["radius"]))


        # Draw City
        pygame.draw.circle(self.screen, self.COLOR_CITY, (self.SCREEN_WIDTH // 2, self.CITY_Y), 30)
        self._draw_circle_glow(self.screen, self.COLOR_CITY, (self.SCREEN_WIDTH // 2, self.CITY_Y), 30, 15)

        # Draw Guardian Platforms and Guardians
        for i, g in enumerate(self.guardians):
            color = self.ELEMENTS[g["element"]]["color"]
            platform_color = (60, 50, 90)
            pygame.draw.circle(self.screen, platform_color, g["pos"], 20, 2)
            
            radius = 12
            if i == self.active_guardian_idx:
                radius = 18
                self._draw_circle_glow(self.screen, color, g["pos"], radius, 10)
            
            pygame.draw.circle(self.screen, color, g["pos"], radius)

        # Draw Projectiles
        for p in self.projectiles:
            color = self.ELEMENTS[p["element"]]["color"]
            end_pos = p["pos"] - p["vel"].normalize() * 10
            pygame.draw.line(self.screen, color, p["pos"], end_pos, 4)
            self._draw_circle_glow(self.screen, color, p["pos"], 5, 5)

        # Draw Enemies
        for e in self.enemies:
            color = self.ELEMENTS[e["element"]]["color"]
            pygame.draw.polygon(self.screen, color, [
                (e["pos"].x, e["pos"].y - e["size"]),
                (e["pos"].x - e["size"], e["pos"].y + e["size"]),
                (e["pos"].x + e["size"], e["pos"].y + e["size"]),
            ])
            # Health bar
            health_ratio = e["health"] / e["max_health"]
            bar_width = e["size"] * 2
            pygame.draw.rect(self.screen, (100, 0, 0), (e["pos"].x - bar_width/2, e["pos"].y - e["size"] - 8, bar_width, 5))
            pygame.draw.rect(self.screen, (0, 200, 0), (e["pos"].x - bar_width/2, e["pos"].y - e["size"] - 8, bar_width * health_ratio, 5))

    def _render_ui(self):
        # Top-left UI
        self._render_text(f"Wave: {self.wave_number}", (10, 10), self.font_medium)
        self._render_text(f"Score: {self.score}", (10, 40), self.font_small)

        # Top-right UI
        self._render_text(f"Resources: {self.resources}", (self.SCREEN_WIDTH - 150, 40), self.font_small, align="right")
        # City Health Bar
        health_bar_width = 140
        health_ratio = max(0, self.city_health / self.max_city_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (self.SCREEN_WIDTH - health_bar_width - 10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (self.SCREEN_WIDTH - health_bar_width - 10, 10, health_bar_width * health_ratio, 20))
        self._render_text("City Health", (self.SCREEN_WIDTH - 10, 10), self.font_small, align="right")

        # Wave transition text
        if self.wave_cooldown > 0 and not self.enemies: # Only show if wave is truly over
            alpha = min(255, int(255 * (self.wave_cooldown / 60)))
            text_surf = self.font_large.render(f"Wave {self.wave_number} starting...", True, self.COLOR_UI_TEXT)
            text_surf.set_alpha(alpha)
            pos = (self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - 50)
            self.screen.blit(text_surf, pos)

        # Upgrade Menu
        if self.upgrade_menu_open:
            self._render_upgrade_menu()
            
        # Game Over Text
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            self._render_text("GAME OVER", (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 - 30), self.font_large, align="center")
            self._render_text(f"Final Score: {self.score}", (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 + 20), self.font_medium, align="center")


    def _render_upgrade_menu(self):
        menu_width, menu_height = 400, 300
        menu_x = (self.SCREEN_WIDTH - menu_width) // 2
        menu_y = (self.SCREEN_HEIGHT - menu_height) // 2
        
        # Draw background
        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        pygame.gfxdraw.box(self.screen, menu_rect, self.COLOR_UI_BG)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, menu_rect, 2)
        
        # Title
        self._render_text("Upgrades", (menu_x + menu_width // 2, menu_y + 20), self.font_medium, align="center")
        
        # List upgrades
        for i, upgrade in enumerate(self.upgrades):
            y_pos = menu_y + 60 + i * 25
            color = self.COLOR_UI_TEXT
            if i == self.highlighted_upgrade_idx:
                color = self.COLOR_UI_HIGHLIGHT
                
                highlight_surf = pygame.Surface((menu_width-10, 22), pygame.SRCALPHA)
                highlight_surf.fill((255,255,100,50))
                self.screen.blit(highlight_surf, (menu_x+5, y_pos-2))

            text = f"{upgrade['name']:<25} Cost: {upgrade['cost']}"
            self._render_text(text, (menu_x + 20, y_pos), self.font_small, color=color)

    def _render_text(self, text, pos, font, color=COLOR_UI_TEXT, align="left"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "right":
            text_rect.topright = pos
        elif align == "center":
            text_rect.midtop = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _draw_circle_glow(self, surface, color, center, radius, glow_size):
        for i in range(glow_size, 0, -1):
            alpha = int(150 * (1 - i / glow_size))
            
            glow_surf = pygame.Surface(( (radius+i)*2, (radius+i)*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, alpha), (radius+i, radius+i), radius+i)
            surface.blit(glow_surf, (center[0] - (radius+i), center[1] - (radius+i)))


    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the test suite.
    os.environ['SDL_VIDEODRIVER'] = 'x11' # or 'windows', 'macOS'
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Use Arrow Keys to select guardian/navigate menu
    # Use Shift to toggle upgrade menu
    # Use Space to buy upgrades
    # ---
    
    running = True
    terminated = False
    
    # Pygame setup for rendering
    pygame.display.set_caption("Guardian Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # [movement, space, shift]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0]
        
        # Map keys to actions
        if keys[pygame.K_1] or keys[pygame.K_KP_1]:
            action[0] = 1
        elif keys[pygame.K_2] or keys[pygame.K_KP_2]:
            action[0] = 2
        elif keys[pygame.K_3] or keys[pygame.K_KP_3]:
            action[0] = 3
        elif keys[pygame.K_4] or keys[pygame.K_KP_4]:
            action[0] = 4
        
        # Alternative mapping for menu navigation
        if env.upgrade_menu_open:
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
        else: # Guardian selection with arrows
             if keys[pygame.K_UP]: action[0] = 1
             if keys[pygame.K_DOWN]: action[0] = 2
             if keys[pygame.K_LEFT]: action[0] = 3
             if keys[pygame.K_RIGHT]: action[0] = 4

        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Run at 60 FPS

    env.close()