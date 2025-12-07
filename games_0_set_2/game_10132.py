import gymnasium as gym
import os
import pygame
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A tactical tower-defense game where you deploy marines to defend against waves of pathogens. "
        "Match energy cells to unleash powerful special attacks."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place a unit or select an energy cell. "
        "Press Shift to cycle through available marine types."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 20, 12
    CELL_SIZE = 32
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_TEXT = (220, 230, 255)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_INVALID = (255, 50, 50)

    MARINE_COLORS = {
        "RIFLE": (0, 255, 128),
        "SNIPER": (0, 180, 255),
        "HEALER": (255, 100, 255),
    }
    PATHOGEN_COLOR = (255, 60, 60)
    ENERGY_COLORS = {
        1: (0, 150, 255),   # Blue
        2: (255, 200, 0),   # Yellow
        3: (200, 50, 255),  # Purple
    }

    # Game Parameters
    MAX_STEPS = 2500
    MAX_WAVES = 10
    FPS = 30

    # Marine Stats: cost, hp, range(grid), damage/heal, cooldown(steps)
    MARINE_STATS = {
        "RIFLE": {"cost": 25, "hp": 100, "range": 4, "damage": 8, "cooldown": 20},
        "SNIPER": {"cost": 50, "hp": 75, "range": 8, "damage": 25, "cooldown": 50},
        "HEALER": {"cost": 40, "hp": 120, "range": 3, "heal": 10, "cooldown": 30},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        self.unlocked_marine_types = ["RIFLE"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.wave = 1
        self.biomass = 50  # Starting resource

        self.marines = []
        self.pathogens = []
        self.energy_cells = []
        self.projectiles = []
        self.particles = []
        self.selected_cells = []

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_unit_idx = 0

        self.last_space_press = False
        self.last_shift_press = False

        self.grid = [[None for _ in range(self.GRID_ROWS)] for _ in range(self.GRID_COLS)]

        self._spawn_initial_marines()
        self._spawn_energy_cells(15)
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        step_reward = 0

        if self.game_over:
            return self._get_observation(), 0, True, self.steps >= self.MAX_STEPS, self._get_info()

        self.steps += 1

        # --- Handle Input (as press events) ---
        space_press = space_held and not self.last_space_press
        shift_press = shift_held and not self.last_shift_press
        self.last_space_press = space_held
        self.last_shift_press = shift_held

        step_reward += self._handle_input(movement, space_press, shift_press)

        # --- Update Game Logic ---
        step_reward += self._update_marines()
        step_reward += self._update_pathogens()
        step_reward += self._update_projectiles()
        self._update_particles()

        # --- Check Wave Completion ---
        if not self.pathogens and not self.game_over:
            step_reward += 5.0
            self.wave += 1
            if self.wave > self.MAX_WAVES:
                self.win = True
                self.game_over = True
            else:
                self._spawn_wave()
                self._spawn_energy_cells(5)  # Add more cells between waves
                if self.wave == 3 and "SNIPER" not in self.unlocked_marine_types:
                    self.unlocked_marine_types.append("SNIPER")
                if self.wave == 5 and "HEALER" not in self.unlocked_marine_types:
                    self.unlocked_marine_types.append("HEALER")

        # --- Check Termination Conditions ---
        if not self.marines and not self.game_over:
            self.game_over = True
            step_reward -= 100.0  # Loss penalty
        if self.win:
            step_reward += 100.0  # Win bonus

        self.score += step_reward

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_energy_cells()
        self._render_pathogens()
        self._render_marines()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "biomass": self.biomass}

    # --- GAME LOGIC ---

    def _handle_input(self, movement, space_press, shift_press):
        # Move cursor
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        # Cycle selected unit
        if shift_press:
            self.selected_unit_idx = (self.selected_unit_idx + 1) % len(self.unlocked_marine_types)

        # Primary action: Place unit or select cell
        if space_press:
            cx, cy = self.cursor_pos
            target_obj = self.grid[cx][cy]

            if isinstance(target_obj, dict) and 'cell_type' in target_obj:
                return self._handle_cell_selection(target_obj)
            elif target_obj is None:
                return self._handle_unit_placement(cx, cy)
        return 0

    def _handle_cell_selection(self, cell):
        reward = 0
        if cell in self.selected_cells:
            self.selected_cells.remove(cell)
        else:
            self.selected_cells.append(cell)

        if len(self.selected_cells) == 3:
            first_type = self.selected_cells[0]['cell_type']
            if all(c['cell_type'] == first_type for c in self.selected_cells):
                reward += 0.5
                for c in self.selected_cells:
                    self.energy_cells.remove(c)
                    self.grid[c['pos'][0]][c['pos'][1]] = None
                    self._create_particles(self._grid_to_pixel(c['pos']), 10, self.ENERGY_COLORS[c['cell_type']])

                for marine in self.marines:
                    if marine['type'] != "HEALER":
                        marine['special_shot'] = True

                self._spawn_energy_cells(3)
            self.selected_cells.clear()
        return reward

    def _handle_unit_placement(self, x, y):
        unit_type = self.unlocked_marine_types[self.selected_unit_idx]
        stats = self.MARINE_STATS[unit_type]
        if self.biomass >= stats['cost']:
            self.biomass -= stats['cost']
            new_marine = {
                "type": unit_type, "pos": [x, y], "hp": stats['hp'], "max_hp": stats['hp'],
                "cooldown": 0, "target": None, "special_shot": False
            }
            self.marines.append(new_marine)
            self.grid[x][y] = new_marine
            self._create_particles(self._grid_to_pixel([x, y]), 15, self.MARINE_COLORS[unit_type])
            return 0.1
        return 0

    def _update_marines(self):
        for marine in self.marines:
            marine['cooldown'] = max(0, marine['cooldown'] - 1)
            stats = self.MARINE_STATS[marine['type']]

            if marine['type'] == "HEALER":
                if marine['cooldown'] == 0:
                    target = self._find_lowest_hp_marine_in_range(marine['pos'], stats['range'])
                    if target:
                        marine['cooldown'] = stats['cooldown']
                        self.projectiles.append(self._create_projectile(marine, target, is_heal=True))
            else:  # Attacking units
                if marine['cooldown'] == 0 or marine['special_shot']:
                    target = self._find_closest_pathogen_in_range(marine['pos'], stats['range'])
                    if target:
                        if marine['special_shot']:
                            marine['special_shot'] = False
                        else:
                            marine['cooldown'] = stats['cooldown']
                        self.projectiles.append(self._create_projectile(marine, target))
        return 0

    def _update_pathogens(self):
        for pathogen in self.pathogens:
            target_marine = self._find_closest_marine(pathogen['pixel_pos'])
            if not target_marine:
                continue

            target_pixel_pos = self._grid_to_pixel(target_marine['pos'])
            dist = math.hypot(target_pixel_pos[0] - pathogen['pixel_pos'][0], target_pixel_pos[1] - pathogen['pixel_pos'][1])

            if dist < self.CELL_SIZE * 0.8:  # Attack range
                pathogen['attack_cooldown'] -= 1
                if pathogen['attack_cooldown'] <= 0:
                    pathogen['attack_cooldown'] = pathogen['max_attack_cooldown']
                    target_marine['hp'] -= pathogen['damage']
                    self._create_particles(target_pixel_pos, 5, self.PATHOGEN_COLOR)
                    if target_marine['hp'] <= 0:
                        self.grid[target_marine['pos'][0]][target_marine['pos'][1]] = None
                        self.marines.remove(target_marine)
                        self._create_particles(target_pixel_pos, 20, (200, 200, 220), 2.0)
            else:  # Move towards target
                angle = math.atan2(target_pixel_pos[1] - pathogen['pixel_pos'][1], target_pixel_pos[0] - pathogen['pixel_pos'][0])
                pathogen['pixel_pos'][0] += math.cos(angle) * pathogen['speed']
                pathogen['pixel_pos'][1] += math.sin(angle) * pathogen['speed']
        return 0

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['is_heal']:
                if proj['target'] not in self.marines:
                    self.projectiles.remove(proj)
                    continue
                target_pos = self._grid_to_pixel(proj['target']['pos'])
            else:  # Damage
                if proj['target'] not in self.pathogens:
                    self.projectiles.remove(proj)
                    continue
                target_pos = proj['target']['pixel_pos']

            dist_to_target = math.hypot(target_pos[0] - proj['pos'][0], target_pos[1] - proj['pos'][1])

            if dist_to_target < proj['speed']:
                # Hit
                if proj['is_heal']:
                    proj['target']['hp'] = min(proj['target']['max_hp'], proj['target']['hp'] + proj['power'])
                    self._create_particles(target_pos, 8, proj['color'])
                else:  # Damage
                    proj['target']['hp'] -= proj['power']
                    reward += 0.1
                    self._create_particles(target_pos, 5, proj['color'])
                    if proj['target']['hp'] <= 0:
                        self.pathogens.remove(proj['target'])
                        reward += 1.0
                        self.biomass += 10
                        self._create_particles(target_pos, 30, (255, 150, 50), 2.5)
                self.projectiles.remove(proj)
            else:
                # Move
                angle = math.atan2(target_pos[1] - proj['pos'][1], target_pos[0] - proj['pos'][0])
                proj['pos'][0] += math.cos(angle) * proj['speed']
                proj['pos'][1] += math.sin(angle) * proj['speed']
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    # --- SPAWNING ---

    def _spawn_initial_marines(self):
        pos = [self.GRID_COLS // 2 - 1, self.GRID_ROWS // 2]
        stats = self.MARINE_STATS["RIFLE"]
        marine = {
            "type": "RIFLE", "pos": pos, "hp": stats['hp'], "max_hp": stats['hp'],
            "cooldown": 0, "target": None, "special_shot": False
        }
        self.marines.append(marine)
        self.grid[pos[0]][pos[1]] = marine

    def _spawn_wave(self):
        num_pathogens = 2 + self.wave * 2
        for _ in range(num_pathogens):
            side = self.np_random.integers(4)
            if side == 0: x, y = -self.CELL_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT)
            elif side == 1: x, y = self.SCREEN_WIDTH + self.CELL_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT)
            elif side == 2: x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), -self.CELL_SIZE
            else: x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.CELL_SIZE

            scale_factor = 1 + (self.wave - 1) * 0.05
            self.pathogens.append({
                "pixel_pos": [x, y],
                "hp": 30 * scale_factor,
                "max_hp": 30 * scale_factor,
                "speed": self.np_random.uniform(0.8, 1.2) * scale_factor,
                "damage": 5 * scale_factor,
                "attack_cooldown": 0,
                "max_attack_cooldown": 60,
            })

    def _spawn_energy_cells(self, count):
        for _ in range(count):
            for _ in range(100):  # Max 100 tries to find empty spot
                x, y = self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)
                if self.grid[x][y] is None:
                    cell_type = self.np_random.integers(1, 4)
                    cell = {"pos": [x, y], "cell_type": cell_type}
                    self.energy_cells.append(cell)
                    self.grid[x][y] = cell
                    break

    # --- HELPERS ---

    def _create_projectile(self, owner, target, is_heal=False):
        stats = self.MARINE_STATS[owner['type']]
        color = self.MARINE_COLORS[owner['type']]
        power = stats['heal'] if is_heal else stats['damage']

        if owner['special_shot'] and not is_heal:
            power *= 2.5
            color = (255, 255, 255)

        return {
            "pos": self._grid_to_pixel(owner['pos']), "target": target, "speed": 8,
            "power": power, "color": color, "is_heal": is_heal,
        }

    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": self.np_random.integers(15, 30),
                "color": color,
                "size": self.np_random.uniform(2, 5)
            })

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return [x, y]

    def _find_closest_pathogen_in_range(self, marine_pos, max_range):
        closest, min_dist = None, float('inf')
        for p in self.pathogens:
            p_grid_pos = [
                (p['pixel_pos'][0] - self.GRID_X_OFFSET) // self.CELL_SIZE,
                (p['pixel_pos'][1] - self.GRID_Y_OFFSET) // self.CELL_SIZE
            ]
            dist = math.hypot(marine_pos[0] - p_grid_pos[0], marine_pos[1] - p_grid_pos[1])
            if dist <= max_range and dist < min_dist:
                min_dist, closest = dist, p
        return closest

    def _find_lowest_hp_marine_in_range(self, healer_pos, max_range):
        lowest_hp_target, min_hp_perc = None, 1.0
        for m in self.marines:
            if m['hp'] < m['max_hp']:
                dist = math.hypot(healer_pos[0] - m['pos'][0], healer_pos[1] - m['pos'][1])
                hp_perc = m['hp'] / m['max_hp']
                if dist <= max_range and hp_perc < min_hp_perc:
                    min_hp_perc, lowest_hp_target = hp_perc, m
        return lowest_hp_target

    def _find_closest_marine(self, pathogen_pixel_pos):
        closest, min_dist_sq = None, float('inf')
        for m in self.marines:
            m_pixel_pos = self._grid_to_pixel(m['pos'])
            dist_sq = (m_pixel_pos[0] - pathogen_pixel_pos[0])**2 + (m_pixel_pos[1] - pathogen_pixel_pos[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq, closest = dist_sq, m
        return closest

    # --- RENDERING ---

    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT))

    def _render_energy_cells(self):
        for cell in self.energy_cells:
            pixel_pos = self._grid_to_pixel(cell['pos'])
            color = self.ENERGY_COLORS[cell['cell_type']]
            radius = int(self.CELL_SIZE * 0.3)

            if cell in self.selected_cells:
                self._draw_glow(pixel_pos, radius * 1.8, color)

            pygame.gfxdraw.aacircle(self.screen, int(pixel_pos[0]), int(pixel_pos[1]), radius, color)
            pygame.gfxdraw.filled_circle(self.screen, int(pixel_pos[0]), int(pixel_pos[1]), radius, color)

    def _render_marines(self):
        for marine in self.marines:
            pixel_pos = self._grid_to_pixel(marine['pos'])
            color = self.MARINE_COLORS[marine['type']]
            radius = int(self.CELL_SIZE * 0.4)

            self._draw_glow(pixel_pos, radius * 1.5, color)
            pygame.gfxdraw.aacircle(self.screen, int(pixel_pos[0]), int(pixel_pos[1]), radius, color)
            pygame.gfxdraw.filled_circle(self.screen, int(pixel_pos[0]), int(pixel_pos[1]), radius, color)

            self._draw_health_bar(pixel_pos, marine['hp'], marine['max_hp'], self.CELL_SIZE * 0.8, (0, 255, 0))

    def _render_pathogens(self):
        for p in self.pathogens:
            pos = p['pixel_pos']
            radius = int(self.CELL_SIZE * 0.35)
            self._draw_glow(pos, radius * 1.5, self.PATHOGEN_COLOR)

            angle_to_closest_marine = 0
            closest_m = self._find_closest_marine(pos)
            if closest_m:
                m_pos = self._grid_to_pixel(closest_m['pos'])
                angle_to_closest_marine = math.atan2(m_pos[1] - pos[1], m_pos[0] - pos[0])

            points = []
            for i in range(3):
                angle = angle_to_closest_marine + i * 2 * math.pi / 3
                points.append((pos[0] + math.cos(angle) * radius, pos[1] + math.sin(angle) * radius))

            pygame.gfxdraw.aapolygon(self.screen, points, self.PATHOGEN_COLOR)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.PATHOGEN_COLOR)

            self._draw_health_bar(pos, p['hp'], p['max_hp'], self.CELL_SIZE * 0.8, self.PATHOGEN_COLOR)

    def _render_projectiles(self):
        for proj in self.projectiles:
            self._draw_glow(proj['pos'], 8, proj['color'])
            vel = [proj['pos'][0] - self._grid_to_pixel(proj['target']['pos'])[0] if proj['is_heal'] else proj['pos'][0] - proj['target']['pixel_pos'][0],
                   proj['pos'][1] - self._grid_to_pixel(proj['target']['pos'])[1] if proj['is_heal'] else proj['pos'][1] - proj['target']['pixel_pos'][1]]
            end_pos = (proj['pos'][0] - vel[0] * 0.05, proj['pos'][1] - vel[1] * 0.05)
            pygame.draw.line(self.screen, proj['color'], proj['pos'], end_pos, 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

    def _render_cursor(self):
        pixel_pos = self._grid_to_pixel(self.cursor_pos)
        cx, cy = self.cursor_pos
        is_valid = self.grid[cx][cy] is None and self.biomass >= self.MARINE_STATS[self.unlocked_marine_types[self.selected_unit_idx]]['cost']
        color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID

        rect = pygame.Rect(pixel_pos[0] - self.CELL_SIZE // 2, pixel_pos[1] - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect, 2)

    def _render_ui(self):
        self._draw_text(f"WAVE: {self.wave}/{self.MAX_WAVES}", (10, 10), self.font_medium)
        self._draw_text(f"SCORE: {int(self.score)}", (self.SCREEN_WIDTH - 150, 10), self.font_medium)
        self._draw_text(f"BIOMASS: {self.biomass}", (10, 35), self.font_medium, color=(140, 255, 140))

        selected_type = self.unlocked_marine_types[self.selected_unit_idx]
        stats = self.MARINE_STATS[selected_type]
        color = self.MARINE_COLORS[selected_type]
        self._draw_text(f"PLACE: {selected_type} (Cost: {stats['cost']})", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 25), self.font_medium, center=True, color=color)

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))

        message = "VICTORY" if self.win else "DEFENSES FAILED"
        color = (100, 255, 100) if self.win else (255, 100, 100)
        self._draw_text(message, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20), self.font_large, center=True, color=color)
        self._draw_text(f"Final Score: {int(self.score)}", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 30), self.font_medium, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _draw_health_bar(self, pixel_pos, hp, max_hp, width, color):
        if hp < max_hp:
            hp_perc = hp / max_hp
            bar_x = pixel_pos[0] - width / 2
            bar_y = pixel_pos[1] - self.CELL_SIZE * 0.6

            pygame.draw.rect(self.screen, (80, 0, 0), (bar_x - 1, bar_y - 1, width + 2, 7))
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, width * hp_perc, 5))

    def _draw_glow(self, pos, radius, color):
        alpha = 90
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*color, alpha), (radius, radius), radius)
        self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

if __name__ == '__main__':
    # This block is for human play and debugging.
    # It will not be executed by the test environment.
    # We re-enable the display for this mode.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Micro Marines")
    
    while not done:
        movement = 0
        space_held = False
        shift_held = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    pygame.quit()
    print(f"Game Over. Final Info: {info}")