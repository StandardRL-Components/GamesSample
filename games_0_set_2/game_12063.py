import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:15:46.048343
# Source Brief: brief_02063.md
# Brief Index: 2063
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

def draw_text(surface, text, size, x, y, color, font, align="topleft"):
    """Helper function to draw text with alignment."""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if align == "topleft":
        text_rect.topleft = (x, y)
    elif align == "topright":
        text_rect.topright = (x, y)
    elif align == "bottomleft":
        text_rect.bottomleft = (x, y)
    elif align == "bottomright":
        text_rect.bottomright = (x, y)
    elif align == "center":
        text_rect.center = (x, y)
    surface.blit(text_surface, text_rect)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Build and manage a network of orbital stations to harvest energy from a star. "
        "Optimize your solar panels and craft upgrades to expand your cosmic empire."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to select a planet or station. Use ←→ to rotate a station's solar panels. "
        "Press space to build a new station on a selected habitable planet. Hold shift to craft available modules or boost panel rotation."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CENTER_X, self.CENTER_Y = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 30
        self.WIN_SCORE = 10000
        self.MAX_STEPS = 5000

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_STAR = (255, 255, 200)
        self.COLOR_STAR_GLOW = (255, 240, 180)
        self.COLOR_ORBIT = (255, 255, 255, 30)
        self.COLOR_HABITABLE = (120, 220, 120)
        self.COLOR_UNINHABITABLE = (160, 160, 160)
        self.COLOR_STATION = (240, 240, 255)
        self.COLOR_PANEL = (100, 150, 255)
        self.COLOR_ENERGY = (255, 220, 50)
        self.COLOR_SELECTION = (50, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (20, 30, 60, 180)

        # Rewards
        self.REWARD_PER_ENERGY = 0.1
        self.REWARD_CRAFT_MODULE = 10.0
        self.REWARD_BUILD_STATION = 20.0
        self.REWARD_WIN = 100.0

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
        try:
            self.font_s = pygame.font.SysFont("bahnschrift", 14)
            self.font_m = pygame.font.SysFont("bahnschrift", 18)
            self.font_l = pygame.font.SysFont("bahnschrift", 28)
        except pygame.error:
            self.font_s = pygame.font.SysFont("sans-serif", 14)
            self.font_m = pygame.font.SysFont("sans-serif", 18)
            self.font_l = pygame.font.SysFont("sans-serif", 28)

        # --- State Variables ---
        self.steps = 0
        self.total_accumulated_energy = 0.0
        self.game_over = False
        self.star = {}
        self.starfield = []
        self.planets = []
        self.stations = []
        self.particles = []
        self.interactables = []
        self.selection_index = 0
        self.modules = []
        self.can_build_station = False
        self.energy_efficiency = 1.0
        self.last_selection_action_step = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.total_accumulated_energy = 0.0
        self.game_over = False
        self.particles = []
        self.last_selection_action_step = -1
        self.can_build_station = False
        self.energy_efficiency = 1.0

        self._generate_starfield()
        self._generate_star_system()
        self._initialize_modules()

        self.stations = []
        first_habitable_idx = next((i for i, p in enumerate(self.planets) if p['habitable']), None)
        if first_habitable_idx is not None:
            self._create_station(first_habitable_idx)

        self._update_interactables()
        self.selection_index = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.game_over = False

        # --- Handle Actions ---
        if self.interactables:
            selected_obj = self.interactables[self.selection_index]

            # 1. Selection (Up/Down)
            if movement in [1, 2] and self.steps > self.last_selection_action_step:
                self.last_selection_action_step = self.steps
                if movement == 1: # Up
                    self.selection_index = (self.selection_index - 1) % len(self.interactables)
                elif movement == 2: # Down
                    self.selection_index = (self.selection_index + 1) % len(self.interactables)

            # 2. Panel Rotation (Left/Right)
            if movement in [3, 4] and selected_obj['type'] == 'station':
                rotation_speed = 10 if shift_held else 5 # Shift boosts rotation
                if movement == 3: # Left
                    selected_obj['panel_angle'] -= rotation_speed
                elif movement == 4: # Right
                    selected_obj['panel_angle'] += rotation_speed
                selected_obj['panel_angle'] %= 360

            # 3. Build Station (Space)
            if space_held and selected_obj['type'] == 'planet' and self.can_build_station:
                planet_idx = selected_obj['id']
                if not any(s['planet_idx'] == planet_idx for s in self.stations):
                    self._create_station(planet_idx)
                    self._update_interactables()
                    self.can_build_station = False
                    reward += self.REWARD_BUILD_STATION
                    # SFX: build_complete.wav

            # 4. Craft Module (Shift) - Note: Shift is also a modifier for rotation
            if shift_held:
                for module in self.modules:
                    if module['unlocked'] and not module['crafted']:
                        module['crafted'] = True
                        self._apply_module_effect(module['name'])
                        reward += self.REWARD_CRAFT_MODULE
                        # SFX: module_crafted.wav
                        break # Craft one at a time

        # --- Update Game State ---
        self._update_planets()
        self._update_stations()

        generated_energy = self._calculate_energy()
        self.total_accumulated_energy += generated_energy
        reward += generated_energy * self.REWARD_PER_ENERGY

        self._update_module_unlocks()
        self._update_particles(generated_energy)

        # --- Termination ---
        self.steps += 1
        terminated = False
        if self.total_accumulated_energy >= self.WIN_SCORE:
            terminated = True
            reward += self.REWARD_WIN
            # SFX: victory.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            # SFX: game_over.wav
        
        truncated = False # This environment does not truncate based on time limits

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.total_accumulated_energy,
            "steps": self.steps,
            "stations": len(self.stations),
            "modules_crafted": sum(1 for m in self.modules if m['crafted']),
        }

    # --- Private Helper Methods ---

    def _generate_starfield(self):
        self.starfield = []
        for _ in range(200):
            self.starfield.append({
                'pos': (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'size': self.np_random.uniform(0.5, 1.5),
                'brightness': self.np_random.integers(50, 150)
            })

    def _generate_star_system(self):
        self.star = {'pos': (self.CENTER_X, self.CENTER_Y), 'size': 20}
        self.planets = []
        num_planets = self.np_random.integers(3, 6)
        orbit_radii = sorted([self.np_random.uniform(60, 180) for _ in range(num_planets)])

        habitable_chosen = False
        for i in range(num_planets):
            is_habitable = not habitable_chosen and self.np_random.random() > 0.3
            if i == num_planets - 1 and not any(p.get('habitable', False) for p in self.planets):
                is_habitable = True # Ensure at least one
            if is_habitable:
                habitable_chosen = True

            self.planets.append({
                'id': i,
                'type': 'planet',
                'orbit_r': orbit_radii[i],
                'angle': self.np_random.uniform(0, 360),
                'speed': self.np_random.uniform(0.2, 0.8) / (orbit_radii[i] / 100),
                'size': self.np_random.uniform(5, 12),
                'habitable': is_habitable,
                'color': self.COLOR_HABITABLE if is_habitable else self.COLOR_UNINHABITABLE,
                'pos': (0, 0) # Calculated in update
            })

    def _initialize_modules(self):
        self.modules = [
            {'name': 'Efficient Panels', 'cost': 500, 'unlocked': False, 'crafted': False},
            {'name': 'Expansion Bay', 'cost': 2000, 'unlocked': False, 'crafted': False},
            {'name': 'Energy Core', 'cost': 5000, 'unlocked': False, 'crafted': False},
        ]

    def _create_station(self, planet_idx):
        station_id = len(self.stations)
        self.stations.append({
            'id': station_id,
            'type': 'station',
            'planet_idx': planet_idx,
            'panel_angle': self.np_random.uniform(0, 360),
            'pos': (0, 0) # Calculated in update
        })

    def _update_interactables(self):
        self.interactables = []
        stationed_planets = {s['planet_idx'] for s in self.stations}
        
        # Add stations first
        for s in self.stations:
            self.interactables.append(s)

        # Add buildable planets
        for p in self.planets:
            if p['habitable'] and p['id'] not in stationed_planets:
                self.interactables.append(p)
        
        if self.selection_index >= len(self.interactables) and self.interactables:
            self.selection_index = len(self.interactables) - 1


    def _apply_module_effect(self, name):
        if name == 'Efficient Panels':
            self.energy_efficiency *= 1.5
        elif name == 'Expansion Bay':
            self.can_build_station = True
        elif name == 'Energy Core':
            self.energy_efficiency *= 2.0 # Renamed effect to fit efficiency model

    def _update_planets(self):
        for p in self.planets:
            p['angle'] = (p['angle'] + p['speed']) % 360
            rad = math.radians(p['angle'])
            p['pos'] = (
                self.CENTER_X + p['orbit_r'] * math.cos(rad),
                self.CENTER_Y + p['orbit_r'] * math.sin(rad)
            )

    def _update_stations(self):
        for s in self.stations:
            s['pos'] = self.planets[s['planet_idx']]['pos']

    def _calculate_energy(self):
        total_energy = 0
        for station in self.stations:
            station_pos = np.array(station['pos'])
            star_pos = np.array(self.star['pos'])
            
            vec_to_star = star_pos - station_pos
            dist_to_star = np.linalg.norm(vec_to_star)
            if dist_to_star == 0: continue

            angle_to_star = math.degrees(math.atan2(vec_to_star[1], vec_to_star[0]))
            
            panel_angle = station['panel_angle']
            angle_diff = abs((panel_angle - angle_to_star + 180) % 360 - 180)

            # Energy = efficiency * (1/dist^2) * angle_factor
            distance_factor = 10000 / max(1, dist_to_star**2)
            angle_factor = max(0, math.cos(math.radians(angle_diff)))**2 # Sharper falloff
            
            energy = self.energy_efficiency * distance_factor * angle_factor
            total_energy += energy
            
            # Spawn particles based on generation
            if energy > 0.1:
                # SFX: solar_panel_active.wav (loop)
                num_particles = int(energy * 0.5)
                for _ in range(num_particles):
                    self._spawn_particle(station)
        return total_energy

    def _update_module_unlocks(self):
        for module in self.modules:
            if not module['unlocked'] and self.total_accumulated_energy >= module['cost']:
                module['unlocked'] = True
                # SFX: module_unlocked.wav

    def _spawn_particle(self, station):
        angle_rad = math.radians(station['panel_angle'])
        offset_dist = 15
        start_pos_1 = (station['pos'][0] + offset_dist * math.cos(angle_rad), station['pos'][1] + offset_dist * math.sin(angle_rad))
        start_pos_2 = (station['pos'][0] - offset_dist * math.cos(angle_rad), station['pos'][1] - offset_dist * math.sin(angle_rad))
        
        start_pos = random.choice([start_pos_1, start_pos_2])
        
        self.particles.append({
            'pos': list(start_pos),
            'vel': [(station['pos'][0] - start_pos[0]) / 20, (station['pos'][1] - start_pos[1]) / 20],
            'life': 20,
            'size': self.np_random.uniform(1, 2.5)
        })

    def _update_particles(self, energy_generated):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_game(self):
        # Starfield
        for star in self.starfield:
            c = int(star['brightness'] + 20 * math.sin(self.steps / 20 + star['pos'][0]))
            c = max(50, min(150, c))
            pygame.draw.circle(self.screen, (c,c,c), star['pos'], star['size'])

        # Orbits
        for p in self.planets:
            pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, int(p['orbit_r']), self.COLOR_ORBIT)

        # Star
        star_glow_size = int(self.star['size'] * 2.5 + 5 * math.sin(self.steps / 15))
        pygame.gfxdraw.filled_circle(self.screen, self.star['pos'][0], self.star['pos'][1], star_glow_size, (*self.COLOR_STAR_GLOW, 50))
        pygame.gfxdraw.filled_circle(self.screen, self.star['pos'][0], self.star['pos'][1], self.star['size'], self.COLOR_STAR)
        pygame.gfxdraw.aacircle(self.screen, self.star['pos'][0], self.star['pos'][1], self.star['size'], self.COLOR_STAR)

        # Planets
        for p in self.planets:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), p['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['size']), p['color'])

        # Stations
        for s in self.stations:
            pos = (int(s['pos'][0]), int(s['pos'][1]))
            angle_rad = math.radians(s['panel_angle'])
            
            # Solar Panels
            panel_w, panel_h = 20, 6
            for sign in [-1, 1]:
                panel_center_x = pos[0] + sign * 10 * math.cos(angle_rad)
                panel_center_y = pos[1] + sign * 10 * math.sin(angle_rad)
                points = [
                    (panel_center_x + panel_w/2 * math.cos(angle_rad) - panel_h/2 * math.sin(angle_rad), panel_center_y + panel_w/2 * math.sin(angle_rad) + panel_h/2 * math.cos(angle_rad)),
                    (panel_center_x - panel_w/2 * math.cos(angle_rad) - panel_h/2 * math.sin(angle_rad), panel_center_y - panel_w/2 * math.sin(angle_rad) + panel_h/2 * math.cos(angle_rad)),
                    (panel_center_x - panel_w/2 * math.cos(angle_rad) + panel_h/2 * math.sin(angle_rad), panel_center_y - panel_w/2 * math.sin(angle_rad) - panel_h/2 * math.cos(angle_rad)),
                    (panel_center_x + panel_w/2 * math.cos(angle_rad) + panel_h/2 * math.sin(angle_rad), panel_center_y + panel_w/2 * math.sin(angle_rad) - panel_h/2 * math.cos(angle_rad)),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PANEL)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PANEL)

            # Station Core
            core_size = 6 + sum(1 for m in self.modules if m['crafted']) # Grows with modules
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], core_size, self.COLOR_STATION)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], core_size, self.COLOR_STATION)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 12)))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), (*self.COLOR_ENERGY, alpha))

        # Selection Highlight
        if self.interactables:
            selected = self.interactables[self.selection_index]
            pos = (int(selected['pos'][0]), int(selected['pos'][1]))
            size = selected.get('size', 10) + 8
            alpha = 128 + 127 * math.sin(self.steps * 0.2)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, (*self.COLOR_SELECTION, int(alpha)))


    def _render_ui(self):
        # Energy Counter
        ui_panel = pygame.Surface((200, 50), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (10, 10))
        draw_text(self.screen, "TOTAL ENERGY", 14, 20, 18, self.COLOR_TEXT, self.font_s)
        draw_text(self.screen, f"{int(self.total_accumulated_energy):,}", 28, 20, 30, self.COLOR_ENERGY, self.font_l)

        # Module Status
        mod_panel = pygame.Surface((220, 100), pygame.SRCALPHA)
        mod_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(mod_panel, (self.WIDTH - 230, self.HEIGHT - 110))
        draw_text(self.screen, "MODULES", 18, self.WIDTH - 220, self.HEIGHT - 105, self.COLOR_TEXT, self.font_m)
        y_offset = self.HEIGHT - 80
        for module in self.modules:
            status_text = ""
            color = (100, 100, 120)
            if module['crafted']:
                status_text = "[CRAFTED]"
                color = (150, 255, 150)
            elif module['unlocked']:
                status_text = "[READY]"
                color = (255, 255, 150)
            elif self.total_accumulated_energy > 0:
                progress = int(100 * self.total_accumulated_energy / module['cost'])
                status_text = f"({progress}%)"
                color = (200, 200, 220)
            
            draw_text(self.screen, f"{module['name']}", 14, self.WIDTH - 220, y_offset, color, self.font_s)
            draw_text(self.screen, status_text, 14, self.WIDTH - 20, y_offset, color, self.font_s, align="topright")
            y_offset += 20
        if self.can_build_station:
            draw_text(self.screen, "Expansion Bay Ready", 14, self.WIDTH-125, self.HEIGHT-15, self.COLOR_SELECTION, self.font_s, align="center")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    
    # Switch to a real video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.quit() # Quit the dummy instance
    pygame.init() # Re-init with the new driver

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cosmic Architect")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if keys[pygame.K_r]:
            obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    pygame.quit()