# Generated: 2025-08-28T04:31:36.304293
# Source Brief: brief_02339.md
# Brief Index: 2339


import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of invading aliens by strategically placing defensive towers on a grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 16, 10
    CELL_SIZE = 40
    MAX_STEPS = 5000
    MAX_WAVES = 50

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_PATH = (25, 25, 40)
    COLOR_BASE = pygame.Color(0, 100, 200)
    COLOR_BASE_DAMAGED = pygame.Color(200, 100, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_RESOURCES = (255, 220, 0)
    COLOR_WAVE = (200, 50, 150)
    COLOR_HEALTH_HIGH = pygame.Color(0, 200, 50)
    COLOR_HEALTH_LOW = pygame.Color(200, 50, 0)

    TOWER_SPECS = [
        {'name': 'Cannon', 'cost': 50, 'range': 80, 'damage': 12, 'fire_rate': 1.5, 'color': (0, 180, 255)},
        {'name': 'Missile', 'cost': 120, 'range': 150, 'damage': 40, 'fire_rate': 0.5, 'color': (255, 150, 0)},
        {'name': 'Laser', 'cost': 200, 'range': 100, 'damage': 3, 'fire_rate': 10.0, 'color': (255, 0, 255)}
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Etc...
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.time_to_next_wave = 0.0
        self.wave_in_progress = False
        self.aliens_to_spawn = 0
        self.aliens_spawned_this_wave = 0
        self.time_since_last_spawn = 0.0
        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.available_tower_types = 1

        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False

        self.path_waypoints = self._generate_path()

        # Initialize state variables
        # A seed is not passed here, as it will be passed in the public reset() call
        self.reset()


    def _generate_path(self):
        path = []
        for i in range(self.GRID_W + 1):
            x = i * self.CELL_SIZE
            y = self.HEIGHT // 2 + int(math.sin(i / 2.5) * self.HEIGHT / 3.5)
            path.append(pygame.Vector2(x, y))
        return path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 100
        self.resources = 100
        self.wave_number = 0
        self.time_to_next_wave = 5.0 # Time before first wave
        self.wave_in_progress = False
        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.grid = [[0 for _ in range(self.GRID_W)] for _ in range(self.GRID_H)]
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 0
        self.available_tower_types = 1

        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False

        # Mark path cells as non-buildable
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                cell_center = pygame.Vector2(x * self.CELL_SIZE + self.CELL_SIZE / 2, y * self.CELL_SIZE + self.CELL_SIZE / 2)
                for i in range(len(self.path_waypoints) - 1):
                    p1, p2 = self.path_waypoints[i], self.path_waypoints[i+1]
                    if self._point_segment_distance(cell_center, p1, p2) < self.CELL_SIZE * 0.75:
                        self.grid[y][x] = -1 # Path is unbuildable
                        break

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        dt = 1 / 30.0 # Fixed delta time for auto_advance=True
        reward = -0.001 # Small penalty for time passing

        # Update game logic
        self.steps += 1

        # Handle Actions
        reward += self._handle_actions(movement, space_held, shift_held)

        # Update Game State
        self._update_waves(dt)
        self._update_aliens(dt)
        reward += self._update_towers_and_projectiles(dt)
        self._update_particles(dt)

        # Check for Alien Reaching Base
        for alien in self.aliens[:]:
            if alien['pos'].x >= self.WIDTH - self.CELL_SIZE / 2:
                self.base_health -= alien['damage']
                self.aliens.remove(alien)
                # sfx: base_damage
                self._create_particles(pygame.Vector2(self.WIDTH, alien['pos'].y), 20, self.COLOR_BASE_DAMAGED)

        terminated = self._check_termination()
        if terminated:
            if self.base_health <= 0: reward = -10.0
            elif self.wave_number > self.MAX_WAVES: reward = 100.0

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_actions(self, movement, space_held, shift_held):
        reward = 0

        # Cursor Movement (once per step)
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)

        # Place Tower (Space) - rising edge detection
        if space_held and not self.space_pressed_last_frame:
            x, y = self.cursor_pos
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.grid[y][x] == 0 and self.resources >= spec['cost']:
                self.resources -= spec['cost']
                self.grid[y][x] = 1
                self.towers.append({
                    'pos': pygame.Vector2(x * self.CELL_SIZE + self.CELL_SIZE/2, y * self.CELL_SIZE + self.CELL_SIZE/2),
                    'type': self.selected_tower_type,
                    'cooldown': 0, 'target': None
                })
                reward += 0.5 # Small reward for placing a tower
                # sfx: place_tower
        self.space_pressed_last_frame = space_held

        # Cycle Tower Type (Shift) - rising edge detection
        if shift_held and not self.shift_pressed_last_frame:
            self.selected_tower_type = (self.selected_tower_type + 1) % self.available_tower_types
            # sfx: cycle_weapon
        self.shift_pressed_last_frame = shift_held

        return reward

    def _check_termination(self):
        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        if self.wave_number > self.MAX_WAVES and not self.aliens and not self.wave_in_progress:
            self.game_over = True
            return True
        return False

    def _update_waves(self, dt):
        if self.wave_in_progress:
            self.time_since_last_spawn += dt
            spawn_interval = 1.0
            if self.time_since_last_spawn > spawn_interval and self.aliens_spawned_this_wave < self.aliens_to_spawn:
                self.time_since_last_spawn = 0
                self.aliens_spawned_this_wave += 1

                health_mult = 1 + (self.wave_number // 10) * 0.1
                speed_mult = 1 + (self.wave_number // 10) * 0.05
                damage_mult = 1 + (self.wave_number // 10) * 0.05

                self.aliens.append({
                    'pos': self.path_waypoints[0].copy(), 'path_index': 0,
                    'health': 10 * health_mult * (1 + self.wave_number * 0.1),
                    'max_health': 10 * health_mult * (1 + self.wave_number * 0.1),
                    'speed': 30 * speed_mult, 'damage': 5 * damage_mult
                })
                # sfx: alien_spawn

            if self.aliens_spawned_this_wave == self.aliens_to_spawn and not self.aliens:
                self.wave_in_progress = False
                self.time_to_next_wave = 10.0
                self.resources += 50 + self.wave_number * 5
        elif self.wave_number < self.MAX_WAVES:
            self.time_to_next_wave -= dt
            if self.time_to_next_wave <= 0:
                self._start_new_wave()

    def _start_new_wave(self):
        self.wave_number += 1
        self.wave_in_progress = True
        self.aliens_to_spawn = 5 + self.wave_number * 2
        self.aliens_spawned_this_wave = 0
        self.time_since_last_spawn = 0

        if self.wave_number >= 20 and self.available_tower_types < 3: self.available_tower_types = 3
        elif self.wave_number >= 10 and self.available_tower_types < 2: self.available_tower_types = 2

    def _update_aliens(self, dt):
        for alien in self.aliens:
            if alien['path_index'] < len(self.path_waypoints) - 1:
                target_pos = self.path_waypoints[alien['path_index'] + 1]
                direction_vec = target_pos - alien['pos']
                if direction_vec.length() > 0:
                    direction = direction_vec.normalize()
                    alien['pos'] += direction * alien['speed'] * dt
                if alien['pos'].distance_to(target_pos) < 5:
                    alien['path_index'] += 1

    def _update_towers_and_projectiles(self, dt):
        reward = 0
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            tower['cooldown'] = max(0, tower['cooldown'] - dt)

            if tower['cooldown'] <= 0:
                target_alien = None
                min_dist = spec['range']
                for alien in self.aliens:
                    dist = tower['pos'].distance_to(alien['pos'])
                    if dist < min_dist:
                        min_dist, target_alien = dist, alien

                if target_alien:
                    tower['cooldown'] = 1.0 / spec['fire_rate']
                    self.projectiles.append({
                        'pos': tower['pos'].copy(), 'target': target_alien,
                        'type': tower['type'], 'speed': 300 if tower['type'] != 2 else 1000
                    })
                    # sfx: tower_fire

        for proj in self.projectiles[:]:
            if proj['target'] not in self.aliens:
                self.projectiles.remove(proj)
                continue

            spec = self.TOWER_SPECS[proj['type']]
            direction = proj['target']['pos'] - proj['pos']

            if direction.length() < 10:
                proj['target']['health'] -= spec['damage']
                reward += 0.1 # Reward for damage
                self._create_particles(proj['pos'], 5, spec['color'])
                if proj['target']['health'] <= 0:
                    reward += 1.0 # Reward for kill
                    self.score += 1
                    self.resources += 5
                    self._create_particles(proj['target']['pos'], 15, (255, 255, 100))
                    # sfx: alien_die
                    self.aliens.remove(proj['target'])
                self.projectiles.remove(proj)
            else:
                proj['pos'] += direction.normalize() * proj['speed'] * dt
        return reward

    def _update_particles(self, dt):
        for p in self.particles[:]:
            p['pos'] += p['vel'] * dt
            p['vel'] *= 0.95
            p['life'] -= dt
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(20, 100)
            self.particles.append({
                'pos': pos.copy(), 'life': self.np_random.uniform(0.3, 0.8),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_and_path()
        self._render_base()
        self._render_towers()
        self._render_aliens()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()

    def _render_grid_and_path(self):
        for i in range(len(self.path_waypoints) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_waypoints[i], self.path_waypoints[i+1], self.CELL_SIZE)
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_base(self):
        base_rect = pygame.Rect(self.WIDTH - self.CELL_SIZE / 2, 0, self.CELL_SIZE / 2, self.HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        health_color = self.COLOR_BASE_DAMAGED.lerp(self.COLOR_BASE, self.base_health / 100.0)
        pygame.draw.rect(self.screen, health_color, base_rect, 5)

    def _render_towers(self):
        for tower in self.towers:
            spec, pos = self.TOWER_SPECS[tower['type']], (int(tower['pos'].x), int(tower['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CELL_SIZE // 3, spec['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CELL_SIZE // 3, spec['color'])

    def _render_aliens(self):
        for alien in self.aliens:
            pos, size = (int(alien['pos'].x), int(alien['pos'].y)), 10
            pygame.gfxdraw.filled_trigon(self.screen, pos[0], pos[1]-size, pos[0]-size, pos[1]+size, pos[0]+size, pos[1]+size, self.COLOR_HEALTH_LOW)
            health_pct = alien['health'] / alien['max_health']
            bar_w, bar_h, bar_x, bar_y = 20, 4, pos[0] - 10, pos[1] - size - 8
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
            health_color = self.COLOR_HEALTH_LOW.lerp(self.COLOR_HEALTH_HIGH, health_pct)
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, bar_w * health_pct, bar_h))

    def _render_projectiles(self):
        for proj in self.projectiles:
            spec, pos = self.TOWER_SPECS[proj['type']], (int(proj['pos'].x), int(proj['pos'].y))
            if proj['type'] == 2:
                target_pos = (int(proj['target']['pos'].x), int(proj['target']['pos'].y))
                pygame.draw.aaline(self.screen, spec['color'], pos, target_pos, 2)
            else:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, spec['color'])

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 0.8))))
            size = int(max(1, 5 * p['life']))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p['color'], alpha), (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'].x - size), int(p['pos'].y - size)))

    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        can_build = self.grid[y][x] == 0 and self.resources >= self.TOWER_SPECS[self.selected_tower_type]['cost']
        color = (0, 255, 0) if can_build else (255, 0, 0)
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, 60), (0, 0, self.CELL_SIZE, self.CELL_SIZE))
        pygame.draw.rect(s, (*color, 200), (2, 2, self.CELL_SIZE-4, self.CELL_SIZE-4), 2)
        self.screen.blit(s, rect.topleft)
        spec = self.TOWER_SPECS[self.selected_tower_type]
        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, int(spec['range']), (*color, 100))

    def _render_ui(self):
        ui_panel = pygame.Surface((self.WIDTH, 30), pygame.SRCALPHA)
        ui_panel.fill((20, 20, 30, 180))
        self.screen.blit(ui_panel, (0, 0))

        self.screen.blit(self.font_small.render(f"Base: {int(self.base_health)}/100", 1, self.COLOR_TEXT), (10, 8))
        self.screen.blit(self.font_small.render(f"$: {int(self.resources)}", 1, self.COLOR_RESOURCES), (150, 8))
        wave_str = f"Wave: {self.wave_number}/{self.MAX_WAVES}" if self.wave_in_progress else f"Next: {int(self.time_to_next_wave)}s"
        self.screen.blit(self.font_small.render(wave_str, 1, self.COLOR_WAVE), (260, 8))
        self.screen.blit(self.font_small.render(f"Score: {self.score}", 1, self.COLOR_TEXT), (550, 8))

        spec = self.TOWER_SPECS[self.selected_tower_type]
        self.screen.blit(self.font_small.render(f"Tower: {spec['name']} | Cost: {spec['cost']}", 1, self.COLOR_TEXT), (10, self.HEIGHT - 22))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            end_text, color = ("VICTORY", (0, 255, 0)) if self.wave_number > self.MAX_WAVES else ("GAME OVER", (255, 0, 0))
            text_surf = self.font_large.render(end_text, 1, color)
            self.screen.blit(text_surf, text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _point_segment_distance(self, p, a, b):
        if a == b: return p.distance_to(a)
        l2 = a.distance_squared_to(b)
        if l2 == 0.0: return p.distance_to(a)
        t = max(0, min(1, (p - a).dot(b - a) / l2))
        return p.distance_to(a + t * (b - a))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        assert self._get_observation().shape == (self.HEIGHT, self.WIDTH, 3)
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(info, dict)
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(reward, float) and isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # The __main__ block is for local testing and is not used by the evaluation system.
    # It will be ignored, so you can leave it as is.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for local testing
    
    env = GameEnv()
    obs, info = env.reset()

    running = True
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    while running:
        movement = 0
        space_pressed = False
        shift_pressed = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_pressed = keys[pygame.K_SPACE]
        shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, space_pressed, shift_pressed]
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(30)
    env.close()