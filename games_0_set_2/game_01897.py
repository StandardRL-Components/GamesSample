# Generated: 2025-08-28T03:03:12.522768
# Source Brief: brief_01897.md
# Brief Index: 1897


import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the placement cursor. Press space to build a tower. "
        "Defend your base from enemy waves."
    )

    game_description = (
        "A strategic tower defense game. Place towers on the grid to destroy enemies before they reach your base. "
        "Survive all 10 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    WIDTH, HEIGHT = 640, 400

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 50, 60)
    COLOR_PATH = (60, 70, 80)
    COLOR_BASE = (0, 150, 255)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TOWER = (0, 200, 200)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_CURSOR_VALID = (0, 255, 0, 100)
    COLOR_CURSOR_INVALID = (255, 0, 0, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    # Game Grid
    GRID_SIZE_X, GRID_SIZE_Y = 22, 14
    TILE_WIDTH, TILE_HEIGHT = 40, 20
    ORIGIN_X, ORIGIN_Y = WIDTH // 2, 80

    # Game Mechanics
    MAX_STEPS = 30 * 120  # 2 minutes at 30fps
    TOTAL_WAVES = 10
    BASE_MAX_HEALTH = 100
    WAVE_PREP_TIME = 150  # frames
    TOWER_COOLDOWN = 30  # frames
    TOWER_RANGE = 4.5  # grid units

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

        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        self.path_grid_coords = [
            (-1, 6), (3, 6), (3, 2), (7, 2), (7, 10), (12, 10),
            (12, 5), (18, 5), (18, 8), (22, 8)
        ]
        self.path_pixels = [self._cart_to_iso(x, y) for x, y in self.path_grid_coords]

        self.grid_cells = {}
        for x in range(self.GRID_SIZE_X):
            for y in range(self.GRID_SIZE_Y):
                self.grid_cells[(x, y)] = {'valid': True}
        self._mark_path_invalid()

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = self.BASE_MAX_HEALTH
        self.cursor_pos = [self.GRID_SIZE_X // 2, self.GRID_SIZE_Y // 2]

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 0
        self.wave_timer = self.WAVE_PREP_TIME
        self.spawning_info = {'index': 0, 'timer': 0}
        self.wave_cleared_bonus_given = True  # Prevent bonus at start
        self.step_reward = 0

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_reward = 0
        self.steps += 1

        if not self.game_over:
            # --- Handle Player Action ---
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            self._handle_input(movement, space_held)

            # --- Update Game State ---
            self._update_waves()
            self._update_towers()
            self._update_projectiles()
            self._update_enemies()
            self._update_particles()

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        self.score += self.step_reward

        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE_X - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE_Y - 1)

        if space_held:
            self._place_tower()

    def _update_waves(self):
        if self.current_wave > self.TOTAL_WAVES:
            return

        if len(self.enemies) == 0 and self.spawning_info['index'] >= self._get_wave_enemy_count():
            if not self.wave_cleared_bonus_given:
                self.step_reward += 10
                self.wave_cleared_bonus_given = True
                if self.current_wave == self.TOTAL_WAVES:
                    self.win = True

            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()

        if self.spawning_info['index'] < self._get_wave_enemy_count():
            self.spawning_info['timer'] -= 1
            if self.spawning_info['timer'] <= 0:
                self._spawn_enemy()
                self.spawning_info['index'] += 1
                self.spawning_info['timer'] = 30 - self.current_wave  # Spawn faster on later waves

    def _start_next_wave(self):
        if self.current_wave < self.TOTAL_WAVES:
            self.current_wave += 1
            self.wave_timer = self.WAVE_PREP_TIME
            self.spawning_info = {'index': 0, 'timer': 0}
            self.wave_cleared_bonus_given = False

    def _get_wave_enemy_count(self):
        return 2 + self.current_wave * 2

    def _spawn_enemy(self):
        health = 10 * (1 + (self.current_wave - 1) * 0.25)
        speed = 0.8 * (1 + (self.current_wave - 1) * 0.05)
        self.enemies.append({
            'pos': list(self.path_grid_coords[0]),
            'pixel_pos': list(self.path_pixels[0]),
            'health': health,
            'max_health': health,
            'speed': speed,
            'path_index': 0,
        })

    def _update_enemies(self):
        is_base_under_attack = False
        for enemy in reversed(self.enemies):
            if enemy['path_index'] >= len(self.path_grid_coords) - 1:
                self.base_health -= enemy['health'] / 2
                self._create_particles(self._cart_to_iso(20, 7), self.COLOR_BASE_DMG, 15)
                self.enemies.remove(enemy)
                continue

            target_pos = self.path_pixels[enemy['path_index'] + 1]
            direction = np.array(target_pos) - np.array(enemy['pixel_pos'])
            distance = np.linalg.norm(direction)

            if distance < enemy['speed']:
                enemy['path_index'] += 1
            else:
                move = direction / distance * enemy['speed']
                enemy['pixel_pos'][0] += move[0]
                enemy['pixel_pos'][1] += move[1]

        if self.base_health < self.BASE_MAX_HEALTH and any(e['path_index'] >= len(self.path_grid_coords) - 1 for e in self.enemies):
            is_base_under_attack = True

        if is_base_under_attack:
            self.step_reward -= 0.01

    def _place_tower(self):
        x, y = self.cursor_pos
        if self.grid_cells.get((x, y), {'valid': False})['valid']:
            self.towers.append({
                'pos': [x, y],
                'cooldown': 0,
                'range_sq': self.TOWER_RANGE ** 2,
            })
            self.grid_cells[(x, y)]['valid'] = False
            # Sound: Tower place sfx

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] > 0:
                continue

            target = None
            min_dist_sq = tower['range_sq']

            tower_cart_pos = np.array(tower['pos'])

            for enemy in self.enemies:
                # Approximate enemy grid position for distance check
                # This is a simplification; true inverse iso is more complex
                iso_x, iso_y = enemy['pixel_pos']
                iso_x -= self.ORIGIN_X
                iso_y -= self.ORIGIN_Y
                cart_y = (2 * iso_y - iso_x) / self.TILE_HEIGHT
                cart_x = iso_x / self.TILE_WIDTH + cart_y

                enemy_cart_pos = np.array([cart_x, cart_y])
                dist_sq = np.sum((tower_cart_pos - enemy_cart_pos) ** 2)

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    target = enemy

            if target:
                tower['cooldown'] = self.TOWER_COOLDOWN
                start_pos = self._cart_to_iso(tower['pos'][0], tower['pos'][1])
                self.projectiles.append({
                    'pos': list(start_pos),
                    'target': target,
                    'speed': 8,
                })
                # Sound: Tower fire sfx

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj['target']['pixel_pos']
            direction = np.array(target_pos) - np.array(proj['pos'])
            distance = np.linalg.norm(direction)

            if distance < proj['speed']:
                # Hit
                proj['target']['health'] -= 10
                self.step_reward += 0.1
                self._create_particles(proj['pos'], self.COLOR_PROJECTILE, 5)
                # Sound: Enemy hit sfx
                if proj['target']['health'] <= 0:
                    self.step_reward += 1
                    self._create_particles(proj['target']['pixel_pos'], self.COLOR_ENEMY, 20)
                    self.enemies.remove(proj['target'])
                    # Sound: Enemy death sfx
                self.projectiles.remove(proj)
            else:
                move = direction / distance * proj['speed']
                proj['pos'][0] += move[0]
                proj['pos'][1] += move[1]

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.game_over:
            return True

        if self.base_health <= 0:
            self.game_over = True
            self.step_reward -= 100
            self.base_health = 0
            return True

        if self.win:
            self.game_over = True
            self.step_reward += 100
            return True

        return False

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
            "base_health": self.base_health,
            "wave": self.current_wave,
            "enemies_left": len(self.enemies),
        }

    # --- Rendering ---
    def _render_game(self):
        self._render_grid()
        self._render_path()
        self._render_base()

        # Sort entities by Y-pos for correct isometric layering
        render_queue = []
        for t in self.towers:
            render_queue.append(('tower', t))
        for e in self.enemies:
            render_queue.append(('enemy', e))

        render_queue.sort(key=lambda item: (item[1]['pos'][0] + item[1]['pos'][1]) if item[0] == 'tower' else (item[1]['pixel_pos'][1]))

        for item_type, item_data in render_queue:
            if item_type == 'tower':
                self._render_tower(item_data)
            elif item_type == 'enemy':
                self._render_enemy(item_data)

        self._render_projectiles()
        self._render_cursor()
        self._render_particles()

    def _render_text(self, text, font, pos, color=COLOR_TEXT, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        # Base Health Bar
        health_ratio = self.base_health / self.BASE_MAX_HEALTH
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_TEXT_SHADOW, (self.WIDTH - bar_width - 18, 18, bar_width + 4, 24))
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, (self.WIDTH - bar_width - 20, 20, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.WIDTH - bar_width - 20, 20, bar_width * health_ratio, 20))
        self._render_text(f"Base HP: {int(self.base_health)}", self.font_small, (self.WIDTH - bar_width - 15, 22))

        # Score
        self._render_text(f"Score: {int(self.score)}", self.font_medium, (20, 20))

        # Wave Info
        wave_text = f"Wave: {self.current_wave}/{self.TOTAL_WAVES}"
        if len(self.enemies) == 0 and not self.win:
            next_wave_in = max(0, self.wave_timer // 30)
            wave_text += f" (Next in {next_wave_in}s)"
        self._render_text(wave_text, self.font_medium, (20, 50))

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            self._render_text(msg, self.font_large, (self.WIDTH / 2 - self.font_large.size(msg)[0] / 2, self.HEIGHT / 2 - 50))

    def _render_grid(self):
        for y in range(self.GRID_SIZE_Y):
            for x in range(self.GRID_SIZE_X):
                p1 = self._cart_to_iso(x, y)
                p2 = self._cart_to_iso(x + 1, y)
                p3 = self._cart_to_iso(x + 1, y + 1)
                p4 = self._cart_to_iso(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

    def _render_path(self):
        for i in range(len(self.path_pixels) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_pixels[i], self.path_pixels[i + 1], self.TILE_HEIGHT)

    def _render_base(self):
        pos = self._cart_to_iso(20, 7)
        points = [
            (pos[0], pos[1] - 15),
            (pos[0] + 25, pos[1]),
            (pos[0], pos[1] + 15),
            (pos[0] - 25, pos[1]),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BASE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BASE)

    def _render_tower(self, tower):
        pos = self._cart_to_iso(tower['pos'][0], tower['pos'][1])
        base_points = [
            (pos[0], pos[1] + 5),
            (pos[0] + 10, pos[1]),
            (pos[0], pos[1] - 5),
            (pos[0] - 10, pos[1]),
        ]
        pygame.gfxdraw.aapolygon(self.screen, base_points, self.COLOR_TOWER)
        pygame.gfxdraw.filled_polygon(self.screen, base_points, self.COLOR_TOWER)
        pygame.draw.circle(self.screen, (200, 255, 255), (int(pos[0]), int(pos[1] - 8)), 4)

    def _render_enemy(self, enemy):
        pos = enemy['pixel_pos']
        size = 8
        points = [
            (pos[0], pos[1] - size),
            (pos[0] + size, pos[1]),
            (pos[0], pos[1] + size),
            (pos[0] - size, pos[1]),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
        # Health bar
        health_ratio = enemy['health'] / enemy['max_health']
        bar_w = 20
        pygame.draw.rect(self.screen, (50, 0, 0), (pos[0] - bar_w / 2, pos[1] - 18, bar_w, 4))
        pygame.draw.rect(self.screen, (0, 255, 0), (pos[0] - bar_w / 2, pos[1] - 18, bar_w * health_ratio, 4))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

    def _render_cursor(self):
        x, y = self.cursor_pos
        is_valid = self.grid_cells.get((x, y), {'valid': False})['valid']
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID

        p1 = self._cart_to_iso(x, y)
        p2 = self._cart_to_iso(x + 1, y)
        p3 = self._cart_to_iso(x + 1, y + 1)
        p4 = self._cart_to_iso(x, y + 1)

        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(s, [p1, p2, p3, p4], color)
        self.screen.blit(s, (0, 0))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (p['pos'][0] - p['size'], p['pos'][1] - p['size']), special_flags=pygame.BLEND_RGBA_ADD)

    # --- Helpers ---
    def _cart_to_iso(self, x, y):
        iso_x = self.ORIGIN_X + (x - y) * (self.TILE_WIDTH / 2)
        iso_y = self.ORIGIN_Y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(iso_x), int(iso_y)

    def _mark_path_invalid(self):
        for i in range(len(self.path_grid_coords) - 1):
            p1 = self.path_grid_coords[i]
            p2 = self.path_grid_coords[i + 1]
            dx = np.sign(p2[0] - p1[0])
            dy = np.sign(p2[1] - p1[1])

            curr = list(p1)
            while tuple(curr) != p2:
                if (curr[0], curr[1]) in self.grid_cells:
                    self.grid_cells[(curr[0], curr[1])]['valid'] = False
                if dx != 0: curr[0] += dx
                if dy != 0: curr[1] += dy
            if (p2[0], p2[1]) in self.grid_cells:
                self.grid_cells[(p2[0], p2[1])]['valid'] = False

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.5)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.randint(2, 5)
            })

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()

    # --- Manual Play ---
    # Use arrow keys to move, space to place tower

    running = True
    terminated = False
    truncated = False

    # Set up a window to display the game
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))

    while running:
        action = [0, 0, 0]  # no-op

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

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}, Truncated: {truncated}")

        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)  # Limit to 30 FPS

        if terminated or truncated:
            print("Game Over! Resetting in 3 seconds...")
            pygame.time.wait(3000)
            obs, info = env.reset()
            terminated = False
            truncated = False

    env.close()