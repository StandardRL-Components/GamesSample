import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers. "
        "Manage your resources to survive the onslaught and defeat the final boss."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to place the selected tower "
        "and shift to cycle between available towers."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    TILE_SIZE = 40
    FPS = 30
    MAX_STEPS = 30000

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_PATH = (25, 30, 50)
    COLOR_BASE = (255, 200, 0)
    COLOR_BASE_GLOW = (255, 200, 0, 50)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_ENEMY_GLOW = (220, 50, 50, 70)
    COLOR_RESOURCE = (50, 150, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CURSOR_INVALID = (255, 0, 0)
    COLOR_HEALTH_BAR_BG = (50, 50, 50)
    COLOR_HEALTH_BAR_FG = (0, 200, 100)
    COLOR_HEALTH_BAR_BASE_FG = (200, 180, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Consolas', 16)
        self.font_medium = pygame.font.SysFont('Consolas', 24)
        self.font_large = pygame.font.SysFont('Consolas', 48)

        # --- Game Data Definitions ---
        self.TOWER_DATA = {
            "Shooter": {"cost": 100, "range": 120, "cooldown": 30, "projectile_speed": 8, "damage": 10, "color": (0, 255, 128)},
            "Slower": {"cost": 75, "range": 80, "cooldown": 10, "slow_factor": 0.5, "damage": 0, "color": (0, 180, 255)},
            "Cannon": {"cost": 250, "range": 160, "cooldown": 90, "projectile_speed": 4, "damage": 50, "aoe_radius": 40, "color": (255, 128, 0)},
        }
        self.ENEMY_DATA = {
            "Grunt": {"health": 100, "speed": 0.8, "reward": 10},
            "Swarm": {"health": 40, "speed": 1.5, "reward": 5},
            "Tank": {"health": 500, "speed": 0.5, "reward": 50},
            "Boss": {"health": 2000, "speed": 0.4, "reward": 200},
        }

        self.ENEMY_WAVE_SCHEDULE = [
            {"type": "Grunt", "count": 5, "interval": 45},
            {"type": "Grunt", "count": 8, "interval": 40},
            {"type": "Grunt", "count": 12, "interval": 35},
            {"type": "Swarm", "count": 10, "interval": 20},
            {"type": "Grunt", "count": 10, "interval": 30},
            {"type": "Swarm", "count": 15, "interval": 15},
            {"type": "Tank", "count": 2, "interval": 120},
            {"type": "Grunt", "count": 15, "interval": 25},
            {"type": "Swarm", "count": 20, "interval": 10},
            {"type": "Boss", "count": 1, "interval": 0},
        ]
        self.FINAL_WAVE = len(self.ENEMY_WAVE_SCHEDULE)

        self.path = [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (7, 4), (7, 3), (6, 3), (5, 3), (4, 3), (3, 3), (2, 3), (2, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1)]
        self.path_pixels = [self._grid_to_pixel(p, center=True) for p in self.path]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        self.game_won = False

        self.base_max_health = 100
        self.base_health = self.base_max_health
        self.resources = 250

        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.unlocked_towers = ["Shooter"]
        self.selected_tower_idx = 0

        self.last_action_state = [0, 0, 0]

        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_cooldown = self.FPS * 5 # Time before first wave
        self.spawn_queue = []
        self.spawn_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.steps += 1

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            self.score += self.reward_this_step

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not truncated:
            if self.game_won:
                self.reward_this_step += 100
            else:
                self.reward_this_step -= 100

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_action_state[1]
        shift_press = shift_held and not self.last_action_state[2]

        # --- Move cursor ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # --- Cycle tower ---
        if shift_press and self.unlocked_towers:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.unlocked_towers)

        # --- Place tower ---
        if space_press:
            self._place_tower()

        self.last_action_state = [movement, space_held, shift_held]

    def _place_tower(self):
        cx, cy = self.cursor_pos
        selected_type = self.unlocked_towers[self.selected_tower_idx]
        cost = self.TOWER_DATA[selected_type]["cost"]

        if self.grid[cy][cx] is None and tuple(self.cursor_pos) not in self.path and self.resources >= cost:
            self.resources -= cost
            tower_obj = {
                "type": selected_type,
                "pos": [cx, cy],
                "cooldown_timer": 0,
                **self.TOWER_DATA[selected_type]
            }
            self.grid[cy][cx] = tower_obj
            self.towers.append(tower_obj)
            self._create_explosion(self._grid_to_pixel(self.cursor_pos, center=True), 30, 20, tower_obj["color"])

    def _update_game_state(self):
        if not self.wave_in_progress:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                self._start_next_wave()
        else:
            self._update_spawner()

        self._update_towers()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()

        if self.wave_in_progress and not self.enemies and not self.spawn_queue:
            self._end_wave()

        if self.base_health <= 0 and not self.game_over:
            self.game_over = True

        if self.game_won and not self.game_over:
            self.game_over = True

    def _start_next_wave(self):
        self.wave_in_progress = True
        self.wave_number += 1

        if self.wave_number > self.FINAL_WAVE:
            self.game_won = True
            return

        wave_data = self.ENEMY_WAVE_SCHEDULE[self.wave_number - 1]
        difficulty_mod = 1 + (self.wave_number - 1) * 0.05

        for _ in range(wave_data["count"]):
            enemy_type = wave_data["type"]
            self.spawn_queue.append({
                "type": enemy_type,
                "health": self.ENEMY_DATA[enemy_type]["health"] * difficulty_mod,
                "max_health": self.ENEMY_DATA[enemy_type]["health"] * difficulty_mod,
                "speed": self.ENEMY_DATA[enemy_type]["speed"] * (1 + (self.wave_number-1)*0.01),
                "reward": self.ENEMY_DATA[enemy_type]["reward"],
                "pos": list(self.path_pixels[0]),
                "path_index": 0,
                "slow_timer": 0,
            })
        self.spawn_timer = wave_data["interval"]

    def _end_wave(self):
        self.wave_in_progress = False
        self.wave_cooldown = self.FPS * 10
        self.reward_this_step += 1.0

        if self.wave_number % 3 == 0 and "Slower" not in self.unlocked_towers:
            self.unlocked_towers.append("Slower")
        if self.wave_number % 6 == 0 and "Cannon" not in self.unlocked_towers:
            self.unlocked_towers.append("Cannon")

        if self.wave_number % 10 == 0: # Boss wave
            self.reward_this_step += 5.0

        if self.wave_number >= self.FINAL_WAVE:
            self.game_won = True

    def _update_spawner(self):
        if self.spawn_timer > 0:
            self.spawn_timer -= 1
        elif self.spawn_queue:
            self.enemies.append(self.spawn_queue.pop(0))
            wave_data = self.ENEMY_WAVE_SCHEDULE[self.wave_number - 1]
            self.spawn_timer = wave_data["interval"]

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown_timer"] > 0:
                tower["cooldown_timer"] -= 1
                continue

            tower_pos_px = self._grid_to_pixel(tower["pos"], center=True)

            if tower["type"] == "Slower":
                for enemy in self.enemies:
                    dist = math.hypot(enemy["pos"][0] - tower_pos_px[0], enemy["pos"][1] - tower_pos_px[1])
                    if dist <= tower["range"]:
                        enemy["slow_timer"] = max(enemy["slow_timer"], 2)
                tower["cooldown_timer"] = tower["cooldown"]
                continue

            target = None
            min_dist = float('inf')
            for enemy in self.enemies:
                dist = math.hypot(enemy["pos"][0] - tower_pos_px[0], enemy["pos"][1] - tower_pos_px[1])
                if dist <= tower["range"] and dist < min_dist:
                    min_dist = dist
                    target = enemy

            if target:
                tower["cooldown_timer"] = tower["cooldown"]
                self.projectiles.append({
                    "pos": list(tower_pos_px),
                    "target": target,
                    "type": tower["type"],
                    "speed": tower["projectile_speed"],
                    "damage": tower["damage"],
                    "aoe_radius": tower.get("aoe_radius"),
                    "color": tower["color"]
                })

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            current_speed = enemy["speed"]
            if enemy["slow_timer"] > 0:
                enemy["slow_timer"] -= 1
                current_speed *= self.TOWER_DATA["Slower"]["slow_factor"]

            if enemy["path_index"] < len(self.path_pixels) - 1:
                target_pos = self.path_pixels[enemy["path_index"] + 1]
                dx = target_pos[0] - enemy["pos"][0]
                dy = target_pos[1] - enemy["pos"][1]
                dist = math.hypot(dx, dy)

                if dist < current_speed:
                    enemy["path_index"] += 1
                    enemy["pos"] = list(target_pos)
                else:
                    enemy["pos"][0] += (dx / dist) * current_speed
                    enemy["pos"][1] += (dy / dist) * current_speed
            else: # Reached base
                self.enemies.remove(enemy)
                self.base_health -= 10
                self.reward_this_step -= 0.1
                self._create_explosion(self._grid_to_pixel((self.GRID_COLS - 1, self.path[-1][1]), center=True), 40, 30, self.COLOR_ENEMY)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            target = proj["target"]
            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue

            dx = target["pos"][0] - proj["pos"][0]
            dy = target["pos"][1] - proj["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < proj["speed"]:
                self.projectiles.remove(proj)
                if proj["type"] == "Cannon":
                    self._create_explosion(proj["pos"], proj["aoe_radius"], 40, proj["color"])
                    for enemy in self.enemies[:]:
                        if math.hypot(enemy["pos"][0] - proj["pos"][0], enemy["pos"][1] - proj["pos"][1]) <= proj["aoe_radius"]:
                            self._damage_enemy(enemy, proj["damage"])
                else:
                    self._damage_enemy(target, proj["damage"])
                    self._create_explosion(proj["pos"], 10, 5, proj["color"])
            else:
                proj["pos"][0] += (dx / dist) * proj["speed"]
                proj["pos"][1] += (dy / dist) * proj["speed"]

    def _damage_enemy(self, enemy, damage):
        enemy["health"] -= damage
        if enemy["health"] <= 0:
            if enemy in self.enemies:
                self.enemies.remove(enemy)
                self.resources += enemy["reward"]
                self.reward_this_step += 0.1
                self.reward_this_step += 0.05 * enemy["reward"]
                self._create_explosion(enemy["pos"], 20, 15, self.COLOR_ENEMY)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, radius, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 31),
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        if not self.game_over:
            self._render_cursor()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def _render_background(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = pygame.Rect(c * self.TILE_SIZE, r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        if len(self.path_pixels) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_PATH, False, self.path_pixels, 10)

    def _render_base(self):
        base_pos = self._grid_to_pixel((self.GRID_COLS - 0.5, self.path[-1][1]), center=True)
        points = []
        for i in range(7):
            angle = i / 7 * 2 * math.pi
            points.append((base_pos[0] + math.cos(angle) * 18, base_pos[1] + math.sin(angle) * 18))
        self._draw_glowing_polygon(self.screen, points, self.COLOR_BASE, self.COLOR_BASE_GLOW, 10)

        bar_width = 40
        bar_height = 5
        bar_x = base_pos[0] - bar_width // 2
        bar_y = base_pos[1] + 25
        health_ratio = max(0, self.base_health / self.base_max_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BASE_FG, (bar_x, bar_y, bar_width * health_ratio, bar_height))

    def _render_towers(self):
        for tower in self.towers:
            pos = self._grid_to_pixel(tower["pos"], center=True)
            color = tower["color"]
            glow_color = (*color, 60)

            if tower["type"] == "Shooter":
                points = [(pos[0], pos[1] - 12), (pos[0] - 10, pos[1] + 8), (pos[0] + 10, pos[1] + 8)]
                self._draw_glowing_polygon(self.screen, points, color, glow_color, 8)
            elif tower["type"] == "Slower":
                rect = pygame.Rect(pos[0] - 10, pos[1] - 10, 20, 20)
                glow_rect = pygame.Rect(pos[0] - 14, pos[1] - 14, 28, 28)
                pygame.draw.rect(self.screen, glow_color, glow_rect, border_radius=4)
                pygame.draw.rect(self.screen, color, rect, border_radius=3)
                if tower["cooldown_timer"] > tower["cooldown"] - 5:
                    pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(tower["range"]), (*color, 80))
            elif tower["type"] == "Cannon":
                points = []
                for i in range(6):
                    angle = i / 6 * 2 * math.pi
                    points.append((pos[0] + math.cos(angle) * 12, pos[1] + math.sin(angle) * 12))
                self._draw_glowing_polygon(self.screen, points, color, glow_color, 8)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            glow_color = self.COLOR_ENEMY_GLOW

            if enemy["slow_timer"] > 0:
                glow_color = (*self.TOWER_DATA["Slower"]["color"], 150)

            if enemy["type"] == "Grunt":
                self._draw_glowing_circle(self.screen, pos, 8, self.COLOR_ENEMY, glow_color, 5)
            elif enemy["type"] == "Swarm":
                self._draw_glowing_circle(self.screen, pos, 5, self.COLOR_ENEMY, glow_color, 3)
            elif enemy["type"] == "Tank":
                points = []
                for i in range(5):
                    angle = i / 5 * 2 * math.pi
                    points.append((pos[0] + math.cos(angle) * 12, pos[1] + math.sin(angle) * 12))
                self._draw_glowing_polygon(self.screen, points, self.COLOR_ENEMY, glow_color, 8)
            elif enemy["type"] == "Boss":
                points = []
                for i in range(8):
                    angle = i / 8 * 2 * math.pi + self.steps * 0.02
                    r = 18 + 4 * math.sin(self.steps * 0.1 + i)
                    points.append((pos[0] + math.cos(angle) * r, pos[1] + math.sin(angle) * r))
                self._draw_glowing_polygon(self.screen, points, self.COLOR_ENEMY, glow_color, 15)

            bar_width = 25
            bar_height = 4
            bar_x = pos[0] - bar_width // 2
            bar_y = pos[1] - 20
            health_ratio = max(0, enemy["health"] / enemy["max_health"])
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (bar_x, bar_y, bar_width * health_ratio, bar_height))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, proj["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, proj["color"])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p["radius"]), color)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        rect = pygame.Rect(cx * self.TILE_SIZE, cy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)

        selected_type = self.unlocked_towers[self.selected_tower_idx]
        cost = self.TOWER_DATA[selected_type]["cost"]
        is_valid = self.grid[cy][cx] is None and tuple(self.cursor_pos) not in self.path and self.resources >= cost

        color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID

        pygame.draw.rect(self.screen, (*color, 50), rect)
        pygame.draw.rect(self.screen, color, rect, 2)

        if self.TOWER_DATA[selected_type]["range"] > 0:
            center_px = self._grid_to_pixel(self.cursor_pos, center=True)
            pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], int(self.TOWER_DATA[selected_type]["range"]), (*color, 50))

    def _render_ui(self):
        self._draw_text(f"RESOURCES: {self.resources}", (10, 10), self.font_medium, self.COLOR_RESOURCE)
        self._draw_text(f"WAVE: {self.wave_number}/{self.FINAL_WAVE}", (self.SCREEN_WIDTH - 10, 10), self.font_medium, self.COLOR_TEXT, align="topright")

        selected_type = self.unlocked_towers[self.selected_tower_idx]
        cost = self.TOWER_DATA[selected_type]["cost"]
        color = self.COLOR_TEXT if self.resources >= cost else self.COLOR_CURSOR_INVALID
        self._draw_text(f"SELECTED: {selected_type}", (10, self.SCREEN_HEIGHT - 30), self.font_medium, self.COLOR_TEXT)
        self._draw_text(f"COST: {cost}", (10, self.SCREEN_HEIGHT - 55), self.font_small, color)

        if self.game_over:
            msg = "VICTORY!" if self.game_won else "GAME OVER"
            color = self.COLOR_BASE if self.game_won else self.COLOR_ENEMY
            self._draw_text(msg, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_large, color, align="center")
        elif not self.wave_in_progress and self.wave_cooldown > 0:
            msg = f"WAVE {self.wave_number + 1} STARTING IN {self.wave_cooldown / self.FPS:.1f}"
            self._draw_text(msg, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_medium, self.COLOR_TEXT, align="center")

    def _grid_to_pixel(self, grid_pos, center=False):
        x, y = grid_pos
        px = x * self.TILE_SIZE
        py = y * self.TILE_SIZE
        if center:
            px += self.TILE_SIZE // 2
            py += self.TILE_SIZE // 2
        return int(px), int(py)

    def _draw_text(self, text, pos, font, color, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        setattr(text_rect, align, pos)
        self.screen.blit(text_surface, text_rect)

    def _draw_glowing_polygon(self, surface, points, color, glow_color, glow_radius):
        pygame.gfxdraw.aapolygon(surface, points, glow_color)
        pygame.gfxdraw.filled_polygon(surface, points, glow_color)
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_color, glow_radius):
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius + glow_radius, glow_color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius + glow_radius, glow_color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)

if __name__ == '__main__':
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    terminated = False
    truncated = False
    total_reward = 0

    while not terminated and not truncated:
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4

        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']:.2f}, Waves Survived: {info['wave']}")
            pygame.time.wait(3000)

    env.close()
    pygame.quit()