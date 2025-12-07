from gymnasium.spaces import MultiDiscrete
import os
import pygame


import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math

# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import Box, MultiDiscrete
import numpy as np


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = "Unlock and upgrade deep-sea portals to harvest valuable minerals from the abyss in this idle resource management game."
    user_guide = "Use ↑ and ↓ arrow keys to select a portal. Press space to upgrade the selected portal or unlock a new one."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    TARGET_FPS = 30

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_BG_BUBBLE = (30, 50, 80, 50)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_SHADOW = (5, 10, 20)
    COLOR_SELECTION = (255, 255, 255)
    COLOR_LOCKED = (80, 80, 90)
    COLOR_LOCKED_FRAME = (50, 50, 60)

    PORTAL_CONFIG = [
        {"name": "Surface Zone", "depth": 100, "unlock_cost": 0, "upgrade_base": 20, "upgrade_factor": 1.2, "mineral_value": 1, "color": (255, 100, 255), "mineral_color": (255, 180, 255)},
        {"name": "Twilight Zone", "depth": 1500, "unlock_cost": 200, "upgrade_base": 150, "upgrade_factor": 1.25, "mineral_value": 5, "color": (100, 255, 255), "mineral_color": (180, 255, 255)},
        {"name": "Abyssal Zone", "depth": 4000, "unlock_cost": 2500, "upgrade_base": 1200, "upgrade_factor": 1.3, "mineral_value": 25, "color": (255, 150, 50), "mineral_color": (255, 200, 150)},
        {"name": "Mariana Trench", "depth": 10984, "unlock_cost": 30000, "upgrade_base": 10000, "upgrade_factor": 1.35, "mineral_value": 150, "color": (255, 50, 100), "mineral_color": (255, 150, 180)},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.SysFont("monospace", 16)
        self.font_title = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 12)

        # State variables initialized in reset()
        self.steps = 0
        self.total_reward = 0.0
        self.game_over = False
        self.victory = False
        self.resources = 0.0
        self.portals = []
        self.particles = []
        self.bubbles = []
        self.effects = []
        self.selected_portal_idx = 0
        self.resources_for_reward = 0.0
        self.new_resources_this_step = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.total_reward = 0.0
        self.game_over = False
        self.victory = False
        self.resources = 0.0
        self.resources_for_reward = 0.0
        self.selected_portal_idx = 0

        self.portals = []
        y_pos = 70
        for i, config in enumerate(self.PORTAL_CONFIG):
            portal_state = config.copy()
            portal_state.update({
                "level": 1 if i == 0 else 0,
                "unlocked": i == 0,
                "pos": (self.SCREEN_WIDTH // 2, y_pos),
                "id": i
            })
            self.portals.append(portal_state)
            y_pos += 80

        self.particles = []
        self.effects = []
        self.bubbles = [self._create_bubble() for _ in range(30)]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- 1. Handle Actions ---
        if movement == 1: # Up
            self.selected_portal_idx = max(0, self.selected_portal_idx - 1)
        elif movement == 2: # Down
            self.selected_portal_idx = min(len(self.portals) - 1, self.selected_portal_idx + 1)

        # Space press (upgrade)
        if space_held:
            reward += self._attempt_upgrade()

        # --- 2. Update Game State ---
        self.steps += 1
        
        # Generate resources from portals
        self._generate_minerals()
        
        # Update particles
        self._update_particles()

        # Update bubbles and effects
        self._update_bubbles()
        self._update_effects()

        # Check for portal unlocks
        unlock_reward = self._check_unlocks()
        reward += unlock_reward

        # --- 3. Calculate Rewards ---
        self.resources_for_reward += self.new_resources_this_step
        if self.resources_for_reward >= 10:
            num_chunks = math.floor(self.resources_for_reward / 10)
            reward += num_chunks * 0.1
            self.resources_for_reward -= num_chunks * 10
        
        # --- 4. Check Termination ---
        terminated = False
        truncated = False
        if self.victory:
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.game_over = terminated or truncated
        self.total_reward += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _attempt_upgrade(self):
        portal = self.portals[self.selected_portal_idx]
        if portal["unlocked"]:
            upgrade_cost = self._get_upgrade_cost(portal)
            if self.resources >= upgrade_cost:
                self.resources -= upgrade_cost
                portal["level"] += 1
                self.effects.append({"type": "upgrade", "pos": portal["pos"], "radius": 20, "max_radius": 60, "life": 1.0})
                return 0.2
        return -0.01

    def _generate_minerals(self):
        self.new_resources_this_step = 0
        for portal in self.portals:
            if portal["unlocked"]:
                spawn_chance = (0.5 + 0.2 * portal["level"]) / self.TARGET_FPS
                if self.np_random.random() < spawn_chance:
                    start_x = self.np_random.uniform(0, self.SCREEN_WIDTH)
                    start_y = portal["pos"][1] + self.np_random.uniform(-10, 10)
                    self.particles.append({
                        "pos": np.array([start_x, start_y], dtype=float),
                        "target_idx": portal["id"],
                        "value": portal["mineral_value"],
                        "color": portal["mineral_color"],
                        "size": self.np_random.uniform(2, 4)
                    })

    def _update_particles(self):
        for p in self.particles[:]:
            target_pos = self.portals[p["target_idx"]]["pos"]
            direction = target_pos - p["pos"]
            dist = np.linalg.norm(direction)

            if dist < 5:
                self.resources += p["value"]
                self.new_resources_this_step += p["value"]
                self.particles.remove(p)
                self.effects.append({"type": "collect", "pos": target_pos, "radius": 0, "max_radius": 15, "life": 1.0, "color": p["color"]})
            else:
                p["pos"] += direction * 0.1

    def _check_unlocks(self):
        reward = 0
        for portal in self.portals:
            if not portal["unlocked"] and self.resources >= portal["unlock_cost"]:
                portal["unlocked"] = True
                self.resources -= portal["unlock_cost"]
                reward += 1.0
                self.effects.append({"type": "unlock", "pos": portal["pos"], "radius": 0, "max_radius": 80, "life": 1.0})
                
                if portal["id"] == len(self.PORTAL_CONFIG) - 1:
                    self.victory = True
        return reward

    def _get_upgrade_cost(self, portal):
        return portal["upgrade_base"] * (portal["upgrade_factor"] ** (portal["level"] - 1))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.total_reward,
            "steps": self.steps,
            "resources": self.resources,
            "victory": self.victory,
        }

    def render(self):
        return self._get_observation()

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            color_val = 10 + 30 * (y / self.SCREEN_HEIGHT)
            pygame.draw.line(self.screen, (10, 20, int(color_val)), (0, y), (self.SCREEN_WIDTH, y))
        
        for bubble in self.bubbles:
            pos = (int(bubble['pos'][0]), int(bubble['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(bubble['size']), self.COLOR_BG_BUBBLE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(bubble['size']), self.COLOR_BG_BUBBLE)

    def _render_game(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            self._draw_glow_circle(self.screen, pos, p["size"], p["color"])
            
        for i, portal in enumerate(self.portals):
            pos = portal["pos"]
            if portal["unlocked"]:
                glow_factor = min(1.0 + portal["level"] * 0.1, 3.0)
                self._draw_glow_circle(self.screen, pos, 20 * glow_factor, portal["color"], 0.5)
                self._draw_glow_circle(self.screen, pos, 15, portal["color"])
            else:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, self.COLOR_LOCKED_FRAME)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, self.COLOR_LOCKED)

        for fx in self.effects:
            alpha = 255
            if fx["type"] == "upgrade":
                alpha = int(255 * fx["life"])
                color = (*self.COLOR_SELECTION, alpha)
                pygame.gfxdraw.aacircle(self.screen, fx["pos"][0], fx["pos"][1], int(fx["radius"]), color)
            elif fx["type"] == "collect":
                alpha = int(150 * fx["life"])
                color = (*fx["color"], alpha)
                self._draw_glow_circle(self.screen, fx["pos"], fx["radius"], color, 0.8)
            elif fx["type"] == "unlock":
                alpha = int(200 * fx["life"])
                color = (*self.COLOR_SELECTION, alpha)
                for i in range(3):
                    pygame.gfxdraw.aacircle(self.screen, fx["pos"][0], fx["pos"][1], int(fx["radius"] * (1 - i*0.2)), color)

    def _render_ui(self):
        res_text = f"MINERALS: {int(self.resources):,}"
        self._draw_text(res_text, (10, 10), self.font_title)
        
        step_text = f"STEP: {self.steps}/{self.MAX_STEPS}"
        self._draw_text(step_text, (self.SCREEN_WIDTH - 180, 10), self.font_main)

        for i, portal in enumerate(self.portals):
            pos = portal["pos"]
            is_selected = (i == self.selected_portal_idx)
            
            if is_selected:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                alpha = 100 + 155 * pulse
                rect = pygame.Rect(pos[0] - 150, pos[1] - 25, 300, 50)
                self._draw_selection_box(rect, (*self.COLOR_SELECTION, int(alpha)))

            if portal["unlocked"]:
                name_text = f"{portal['name']} (Lv. {portal['level']})"
                self._draw_text(name_text, (pos[0] - 140, pos[1] - 20), self.font_main)
                
                cost = self._get_upgrade_cost(portal)
                cost_text = f"Upgrade: {int(cost):,}"
                color = self.COLOR_TEXT if self.resources >= cost else (255, 100, 100)
                self._draw_text(cost_text, (pos[0] - 140, pos[1]), self.font_small, color=color)

                depth_text = f"Depth: {portal['depth']}m"
                self._draw_text(depth_text, (pos[0] + 40, pos[1] - 20), self.font_small)

                val_text = f"Value: {portal['mineral_value']}"
                self._draw_text(val_text, (pos[0] + 40, pos[1]), self.font_small)
            else:
                self._draw_text(f"{portal['name']} [LOCKED]", (pos[0] - 140, pos[1] - 20), self.font_main, color=self.COLOR_LOCKED)
                
                unlock_cost = portal['unlock_cost']
                cost_text = f"Unlock: {int(unlock_cost):,}"
                color = self.COLOR_LOCKED if self.resources < unlock_cost else (180, 255, 180)
                self._draw_text(cost_text, (pos[0] - 140, pos[1]), self.font_small, color=color)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 1, pos[1] + 1))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _draw_glow_circle(self, surface, pos, radius, color, intensity=1.0):
        if radius <= 0: return
        pos = (int(pos[0]), int(pos[1]))
        
        rgb_color = color[:3]

        num_layers = max(1, int(radius * 0.5))
        for i in range(num_layers):
            layer_radius = int(radius - i * (radius / num_layers))
            alpha = int(255 * (1 - i / num_layers)**2 * 0.3 * intensity)
            layer_color = (*rgb_color, alpha)
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], layer_radius, layer_color)
        
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius), color)
        
    def _draw_selection_box(self, rect, color):
        pygame.draw.rect(self.screen, color, rect, width=2, border_radius=5)

    def _create_bubble(self):
        return {
            'pos': np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)], dtype=float),
            'size': self.np_random.uniform(1, 5),
            'speed': self.np_random.uniform(0.2, 1.0)
        }

    def _update_bubbles(self):
        for bubble in self.bubbles:
            bubble['pos'][1] -= bubble['speed']
            if bubble['pos'][1] < -bubble['size']:
                bubble['pos'][0] = self.np_random.uniform(0, self.SCREEN_WIDTH)
                bubble['pos'][1] = self.SCREEN_HEIGHT + bubble['size']

    def _update_effects(self):
        for fx in self.effects[:]:
            fx["life"] -= 1.0 / self.TARGET_FPS
            if fx["type"] in ["upgrade", "collect", "unlock"]:
                fx["radius"] += (fx["max_radius"] - fx["radius"]) * 0.15
            if fx["life"] <= 0:
                self.effects.remove(fx)

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # To run with rendering, disable the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Portal Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    movement = 0
    space_held = 0
    shift_held = 0

    print("\n--- Manual Control ---")
    print("W/S or Up/Down Arrow: Select Portal")
    print("Spacebar: Upgrade Selected Portal")
    print("Q: Quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key in [pygame.K_w, pygame.K_UP]:
                    movement = 1
                if event.key in [pygame.K_s, pygame.K_DOWN]:
                    movement = 2
                if event.key == pygame.K_SPACE:
                    space_held = 1
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_UP, pygame.K_s, pygame.K_DOWN]:
                    movement = 0
                if event.key == pygame.K_SPACE:
                    space_held = 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(GameEnv.TARGET_FPS)
        
        movement = 0

    env.close()