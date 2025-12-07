import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. SHIFT to cycle tower types. SPACE to place a tower or start the wave."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this minimalist isometric TD game."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 24, 14
        self.TILE_W, self.TILE_H = 28, 14
        self.MAX_WAVES = 20
        self.MAX_STEPS = 30 * 180  # 3 minutes at 30fps
        self.STARTING_GOLD = 150
        self.STARTING_HEALTH = 100
        self.GOLD_PER_KILL = 10
        self.GOLD_PER_WAVE = 50

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PATH = (60, 70, 80)
        self.COLOR_BASE = (0, 200, 150)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_GOLD = (255, 200, 0)
        self.COLOR_UI_BG = (35, 45, 55)
        self.COLOR_HEALTH_BAR_BG = (80, 20, 20)
        self.COLOR_HEALTH_BAR_FG = (20, 180, 20)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_m = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 32, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.base_health = 0
        self.gold = 0
        self.wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_idx = 0
        self.game_phase = "PREP"
        self.prev_space_held = False
        self.prev_shift_held = False
        self.wave_spawner = None
        self.spawn_timer = 0
        self.win_condition = False

        # Game assets (no external files)
        self.path = self._generate_path()
        self.tower_types = self._define_towers()
        self.grid_offset = self._calculate_grid_offset()
        self.start_button_rect = pygame.Rect(self.WIDTH - 150, self.HEIGHT - 45, 140, 35)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.base_health = self.STARTING_HEALTH
        self.gold = self.STARTING_GOLD
        self.wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._handle_input(movement, space_held, shift_held)

        if self.game_phase == "WAVE":
            self._update_spawner()
            self._update_towers()
            reward += self._update_projectiles()
            reward += self._update_enemies()

            if not self.enemies and self.wave_spawner is None:
                reward += 1.0
                self.score += 10
                if self.wave >= self.MAX_WAVES:
                    self.win_condition = True
                else:
                    self._start_next_wave()

        self._update_particles()
        self.steps += 1

        terminated = self._check_termination()
        if terminated:
            if self.win_condition:
                reward += 100.0
                self.score += 1000
            else:
                reward -= 100.0
                self.score -= 1000

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Allow input only during preparation phase
        if self.game_phase == "PREP":
            # --- Movement ---
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)

            # --- Cycle Tower (on press) ---
            if shift_held and not self.prev_shift_held:
                self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.tower_types)
                # sfx: UI_cycle

            # --- Place Tower / Start Wave (on press) ---
            if space_held and not self.prev_space_held:
                cursor_screen_pos = self._iso_to_screen(self.cursor_pos)
                if self.start_button_rect.collidepoint(cursor_screen_pos):
                    self._begin_wave()
                else:
                    self._place_tower()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _place_tower(self):
        tower_type = self.tower_types[self.selected_tower_idx]
        if self.gold >= tower_type["cost"] and self._is_valid_placement(self.cursor_pos):
            self.gold -= tower_type["cost"]
            self.towers.append({
                "pos": list(self.cursor_pos),
                "type": tower_type,
                "cooldown": 0,
                "angle": 0
            })
            self._create_particles(self._iso_to_screen(self.cursor_pos), 10, tower_type["color"])
            # sfx: place_tower

    def _begin_wave(self):
        self.game_phase = "WAVE"
        self.wave_spawner = self._get_wave_spawner()
        self.spawn_timer = 0
        # sfx: wave_start

    def _start_next_wave(self):
        self.game_phase = "PREP"
        self.wave += 1
        if self.wave > 1:
            self.gold += self.GOLD_PER_WAVE
        self.wave_spawner = None

    def _update_spawner(self):
        if self.wave_spawner is None: return
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            try:
                delay, enemy_type = next(self.wave_spawner)
                self.enemies.append(enemy_type)
                self.spawn_timer = delay
            except StopIteration:
                self.wave_spawner = None

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = self._find_target(tower)
                if target:
                    # sfx: tower_shoot
                    tower['cooldown'] = tower['type']['rate']
                    start_pos = self._iso_to_screen(tower['pos'])
                    self.projectiles.append({
                        "pos": list(start_pos),
                        "target": target,
                        "speed": tower['type']['proj_speed'],
                        "damage": tower['type']['damage'],
                        "color": tower['type']['proj_color']
                    })
                    # Muzzle flash
                    self._create_particles(start_pos, 5, tower['type']['proj_color'], 1, 3)

    def _find_target(self, tower):
        tower_screen_pos = self._iso_to_screen(tower['pos'])
        for enemy in self.enemies:
            dist = math.hypot(enemy['pos'][0] - tower_screen_pos[0], enemy['pos'][1] - tower_screen_pos[1])
            if dist <= tower['type']['range']:
                return enemy
        return None

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj['target']['pos']
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < proj['speed']:
                proj['target']['health'] -= proj['damage']
                self._create_particles(proj['pos'], 8, proj['color'], 2, 5)
                self.projectiles.remove(proj)
                # sfx: enemy_hit
                if proj['target']['health'] <= 0:
                    reward += 0.1  # Small reward for kill
                    self.gold += self.GOLD_PER_KILL
                    # sfx: enemy_die
                    self._create_particles(proj['target']['pos'], 20, self.COLOR_ENEMY, 3, 8)
                    self.enemies.remove(proj['target'])
            else:
                proj['pos'][0] += (dx / dist) * proj['speed']
                proj['pos'][1] += (dy / dist) * proj['speed']
        return reward

    def _update_enemies(self):
        damage_to_base = 0
        for enemy in self.enemies[:]:
            path_idx = enemy['path_idx']
            target_node = self.path[path_idx]
            target_pos = self._iso_to_screen(target_node)

            dx = target_pos[0] - enemy['pos'][0]
            dy = target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < enemy['speed']:
                enemy['path_idx'] += 1
                if enemy['path_idx'] >= len(self.path):
                    damage_to_base += enemy['damage']
                    self.enemies.remove(enemy)
                    self._create_particles(self.base_screen_pos, 20, self.COLOR_BASE, 5, 10)
                    # sfx: base_hit
                else:
                    enemy['pos'] = list(target_pos)
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']

        if damage_to_base > 0:
            self.base_health = max(0, self.base_health - damage_to_base)
            return -0.01 * damage_to_base
        return 0

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.base_health <= 0 or self.win_condition or self.steps >= self.MAX_STEPS

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
            "wave": self.wave,
            "gold": self.gold,
            "base_health": self.base_health,
            "phase": self.game_phase,
        }

    # Rendering methods
    def _render_game(self):
        self._draw_grid()
        self._draw_path()
        self._draw_base()
        for tower in self.towers: self._draw_tower(tower)
        for enemy in self.enemies: self._draw_enemy(enemy)
        for proj in self.projectiles: self._draw_projectile(proj)
        for particle in self.particles: self._draw_particle(particle)
        if self.game_phase == "PREP": self._draw_cursor()

    def _draw_grid(self):
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                if (c, r) not in self.path_set and (c, r) != self.base_pos:
                    pos = self._iso_to_screen((c, r))
                    points = [
                        (pos[0], pos[1] - self.TILE_H / 2),
                        (pos[0] + self.TILE_W / 2, pos[1]),
                        (pos[0], pos[1] + self.TILE_H / 2),
                        (pos[0] - self.TILE_W / 2, pos[1]),
                    ]
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

    def _draw_path(self):
        for node in self.path:
            pos = self._iso_to_screen(node)
            self._draw_iso_rect(pos, self.TILE_W, self.TILE_H, self.COLOR_PATH)

    def _draw_base(self):
        pos = self._iso_to_screen(self.base_pos)
        size_w, size_h = self.TILE_W * 1.5, self.TILE_H * 1.5
        color = self.COLOR_BASE

        # Pulsing effect
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        glow_color = tuple(min(255, c + int(pulse * 40)) for c in color)

        self._draw_iso_rect(pos, size_w, size_h, glow_color)
        self._draw_iso_rect(pos, self.TILE_W, self.TILE_H, color)
        self.base_screen_pos = pos

    def _draw_tower(self, tower):
        pos = self._iso_to_screen(tower['pos'])
        ttype = tower['type']
        self._draw_iso_rect(pos, self.TILE_W * 0.8, self.TILE_H * 0.8, ttype['color'])

        # Simple turret shape
        gun_len = self.TILE_W * 0.5
        target = self._find_target(tower)
        if target:
            angle = math.atan2(target['pos'][1] - pos[1], target['pos'][0] - pos[0])
            tower['angle'] = angle  # Store angle for smooth rotation

        end_x = pos[0] + gun_len * math.cos(tower['angle'])
        end_y = pos[1] + gun_len * math.sin(tower['angle'])
        pygame.draw.line(self.screen, (200, 200, 200), pos, (end_x, end_y), 3)

    def _draw_enemy(self, enemy):
        # Bobbing motion
        bob = math.sin(self.steps * 0.2 + enemy['id'] * 0.5) * 2
        pos = (enemy['pos'][0], enemy['pos'][1] - bob)

        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 6, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 6, self.COLOR_ENEMY)

        # Health bar
        bar_w = 12
        health_pct = enemy['health'] / enemy['max_health']
        pygame.draw.rect(self.screen, (80, 0, 0), (pos[0] - bar_w / 2, pos[1] - 12, bar_w, 3))
        pygame.draw.rect(self.screen, (0, 200, 0), (pos[0] - bar_w / 2, pos[1] - 12, bar_w * health_pct, 3))

    def _draw_projectile(self, proj):
        pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 3, proj['color'])
        pygame.gfxdraw.aacircle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 3, proj['color'])

    def _draw_particle(self, p):
        size = p['life'] / p['max_life'] * p['size']
        if size > 0:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(size), p['color'])

    def _draw_cursor(self):
        pos = self._iso_to_screen(self.cursor_pos)
        tower_type = self.tower_types[self.selected_tower_idx]

        # Range indicator
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], tower_type['range'], (255, 255, 255, 50))

        # Placement validity indicator
        is_valid = self.gold >= tower_type["cost"] and self._is_valid_placement(self.cursor_pos)
        cursor_color = (0, 255, 0, 100) if is_valid else (255, 0, 0, 100)
        self._draw_iso_rect(pos, self.TILE_W, self.TILE_H, cursor_color, filled=False)

    def _render_ui(self):
        # Bottom bar
        ui_rect = pygame.Rect(0, self.HEIGHT - 60, self.WIDTH, 60)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.HEIGHT - 60), (self.WIDTH, self.HEIGHT - 60), 2)

        # Health Bar
        health_pct = self.base_health / self.STARTING_HEALTH
        health_bar_rect = pygame.Rect(10, self.HEIGHT - 45, 150, 15)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG,
                         (health_bar_rect.x, health_bar_rect.y, health_bar_rect.w * health_pct, health_bar_rect.h))
        self._draw_text(f"BASE: {self.base_health}/{self.STARTING_HEALTH}", (15, self.HEIGHT - 46), self.font_s,
                        self.COLOR_TEXT)

        # Gold and Wave
        self._draw_text(f"GOLD: {self.gold}", (10, self.HEIGHT - 25), self.font_s, self.COLOR_GOLD)
        self._draw_text(f"WAVE: {self.wave}/{self.MAX_WAVES}", (100, self.HEIGHT - 25), self.font_s, self.COLOR_TEXT)

        # Tower selection UI
        for i, ttype in enumerate(self.tower_types):
            x_offset = 200 + i * 120
            box_rect = pygame.Rect(x_offset, self.HEIGHT - 50, 110, 40)

            border_color = self.COLOR_GOLD if i == self.selected_tower_idx else self.COLOR_GRID
            pygame.draw.rect(self.screen, self.COLOR_BG, box_rect)
            pygame.draw.rect(self.screen, border_color, box_rect, 2)

            self._draw_iso_rect((x_offset + 20, self.HEIGHT - 30), self.TILE_W * 0.6, self.TILE_H * 0.6, ttype['color'])
            self._draw_text(ttype['name'], (x_offset + 40, self.HEIGHT - 45), self.font_s, self.COLOR_TEXT)
            self._draw_text(f"${ttype['cost']}", (x_offset + 40, self.HEIGHT - 30), self.font_s, self.COLOR_GOLD)

        # Game Phase Indicator
        if self.game_phase == "PREP":
            cursor_screen_pos = self._iso_to_screen(self.cursor_pos)
            is_hover = self.start_button_rect.collidepoint(cursor_screen_pos)
            btn_color = (0, 255, 0) if is_hover else (0, 200, 0)
            pygame.draw.rect(self.screen, btn_color, self.start_button_rect)
            self._draw_text("START WAVE", self.start_button_rect.center, self.font_m, (0, 0, 0), align="center")
        elif self.game_phase == "WAVE":
            self._draw_text("WAVE IN PROGRESS", (self.WIDTH - 80, self.HEIGHT - 30), self.font_m, self.COLOR_ENEMY,
                            align="center")

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "VICTORY!" if self.win_condition else "GAME OVER"
            color = self.COLOR_BASE if self.win_condition else self.COLOR_ENEMY
            self._draw_text(msg, (self.WIDTH / 2, self.HEIGHT / 2 - 20), self.font_l, color, align="center")
            self._draw_text(f"Final Score: {int(self.score)}", (self.WIDTH / 2, self.HEIGHT / 2 + 20), self.font_m,
                            self.COLOR_TEXT, align="center")

    # Helper methods
    def _generate_path(self):
        path = []
        path.extend([(i, 2) for i in range(self.GRID_W - 3)])
        path.extend([(self.GRID_W - 4, i) for i in range(2, self.GRID_H - 2)])
        path.extend([(i, self.GRID_H - 3) for i in range(self.GRID_W - 4, 2, -1)])
        path.extend([(3, i) for i in range(self.GRID_H - 3, 5, -1)])
        path.extend([(i, 6) for i in range(3, self.GRID_W - 8)])
        self.path_set = set(path)
        self.base_pos = (self.GRID_W - 9, 6)
        path.append(self.base_pos)
        return path

    def _define_towers(self):
        return [
            {"name": "Gun", "cost": 50, "damage": 10, "range": 80, "rate": 20, "color": (0, 150, 255),
             "proj_color": (100, 200, 255), "proj_speed": 8},
            {"name": "Cannon", "cost": 120, "damage": 45, "range": 100, "rate": 60, "color": (255, 150, 0),
             "proj_color": (255, 180, 50), "proj_speed": 6},
            {"name": "Sniper", "cost": 150, "damage": 100, "range": 200, "rate": 120, "color": (200, 0, 200),
             "proj_color": (255, 100, 255), "proj_speed": 15},
        ]

    def _get_wave_spawner(self):
        num_enemies = 3 + self.wave * 2
        health = 20 * (1.05 ** self.wave)
        speed = 1.0 * (1.02 ** self.wave)

        for i in range(num_enemies):
            enemy_data = {
                "id": self.np_random.integers(10000),
                "pos": list(self._iso_to_screen(self.path[0])),
                "path_idx": 1,
                "health": health,
                "max_health": health,
                "speed": speed + self.np_random.uniform(-0.1, 0.1),
                "damage": 5,
            }
            yield (15, enemy_data)  # 15 frames between spawns

    def _is_valid_placement(self, pos):
        return tuple(pos) not in self.path_set and all(tuple(pos) != t['pos'] for t in self.towers)

    def _calculate_grid_offset(self):
        total_w = (self.GRID_W + self.GRID_H) * self.TILE_W / 2
        total_h = (self.GRID_W + self.GRID_H) * self.TILE_H / 2
        offset_x = (self.WIDTH - total_w) / 2 + self.GRID_W * self.TILE_W / 2
        offset_y = (self.HEIGHT - total_h) / 2 + self.TILE_H * 2  # Shift down a bit
        return offset_x, offset_y

    def _iso_to_screen(self, pos):
        iso_x, iso_y = pos
        screen_x = self.grid_offset[0] + (iso_x - iso_y) * (self.TILE_W / 2)
        screen_y = self.grid_offset[1] + (iso_x + iso_y) * (self.TILE_H / 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_rect(self, pos, width, height, color, filled=True):
        points = [
            (pos[0], pos[1] - height / 2),
            (pos[0] + width / 2, pos[1]),
            (pos[0], pos[1] + height / 2),
            (pos[0] - width / 2, pos[1]),
        ]
        if filled:
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_text(self, text, pos, font, color, align="left"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "right":
            text_rect.topright = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, pos, count, color, min_speed=1, max_speed=3, life=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'size': self.np_random.uniform(2, 5),
                'color': color
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Re-enable the normal video driver for local play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    running = True
    terminated = False

    # Create a window to display the game
    pygame.display.set_caption("Tiny Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while running:
        action = [0, 0, 0]  # Default no-op action

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        env.clock.tick(30)  # Run at 30 FPS

    env.close()