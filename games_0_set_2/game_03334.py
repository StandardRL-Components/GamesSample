
# Generated: 2025-08-27T23:03:33.160850
# Source Brief: brief_03334.md
# Brief Index: 3334

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press 'Shift' to cycle tower types. Press 'Space' to build a tower at the cursor."
    )

    game_description = (
        "Defend your base from waves of invading enemies by strategically placing "
        "defensive towers in a top-down, procedurally generated tactical arena."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.CELL_SIZE = self.WIDTH // self.GRID_W

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PATH = (60, 70, 80)
        self.COLOR_BASE = (0, 255, 128)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (0, 200, 100)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        
        # Tower definitions: [cost, range (grid units), fire_rate (steps), damage, color, projectile_speed]
        self.TOWER_SPECS = {
            0: {"name": "Gatling", "cost": 50, "range": 4, "fire_rate": 10, "damage": 5, "color": (0, 255, 255), "proj_speed": 15},
            1: {"name": "Cannon", "cost": 120, "range": 6, "fire_rate": 40, "damage": 25, "color": (255, 165, 0), "proj_speed": 10},
        }

        # Game parameters
        self.MAX_STEPS = 2000
        self.MAX_WAVES = 10
        self.STARTING_RESOURCES = 150
        self.STARTING_BASE_HEALTH = 100
        self.WAVE_PREP_TIME = 150 # Steps between waves

        # --- Gym Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_m = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.wave_timer = 0
        self.wave_active = False
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        
        self.path_waypoints = []
        self.path_coords = set()
        self.base_pos = (0,0)
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self._last_shift_state = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = self.STARTING_BASE_HEALTH
        self.resources = self.STARTING_RESOURCES
        self.current_wave = 0
        self.wave_timer = self.WAVE_PREP_TIME
        self.wave_active = False
        self.enemies_in_wave = 0
        self.enemies_spawned = 0

        self._generate_path()
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 0
        self._last_shift_state = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        self._handle_input(movement, space_held, shift_held)
        
        # --- Game Logic Update ---
        if not self.wave_active:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()
                if not self.game_won:
                    reward += 10.0 # Wave survival bonus
        else:
            self._spawn_enemies()
        
        hit_reward, kill_reward, resource_gain = self._update_projectiles()
        reward += hit_reward + kill_reward
        self.resources += resource_gain
        self.score += kill_reward # Score only for kills

        damage_to_base = self._update_enemies()
        if damage_to_base > 0:
            self.base_health -= damage_to_base
            reward -= 5.0 * (damage_to_base / 10) # Penalty for base damage
            
        self._update_towers()
        self._update_particles()
        
        # Check for wave completion
        if self.wave_active and self.enemies_spawned == self.enemies_in_wave and not self.enemies:
            self.wave_active = False
            self.wave_timer = self.WAVE_PREP_TIME
            if self.current_wave >= self.MAX_WAVES:
                self.game_won = True

        # --- Termination ---
        self.steps += 1
        terminated = False
        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
            terminated = True
            reward -= 100.0
        elif self.game_won:
            self.game_over = True
            terminated = True
            reward += 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # Cycle tower type on shift press (rising edge)
        if shift_held and not self._last_shift_state:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self._last_shift_state = shift_held

        # Place tower on space press
        if space_held:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.resources >= spec["cost"]:
                is_valid_pos = tuple(self.cursor_pos) not in self.path_coords
                for t in self.towers:
                    if t['pos'] == self.cursor_pos:
                        is_valid_pos = False; break
                
                if is_valid_pos:
                    self.resources -= spec["cost"]
                    self.towers.append({
                        "pos": list(self.cursor_pos), "type": self.selected_tower_type,
                        "cooldown": 0, "spec": spec
                    })
                    # sfx: tower_place.wav
                    for _ in range(20):
                        self._add_particle(self._grid_to_screen(self.cursor_pos), color=spec['color'], life=20)


    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            self.game_won = True
            return
            
        self.wave_active = True
        self.enemies_in_wave = 5 + self.current_wave * 2
        self.enemies_spawned = 0
        # sfx: wave_start.wav

    def _spawn_enemies(self):
        spawn_interval = max(5, 30 - self.current_wave * 2)
        if self.steps % spawn_interval == 0 and self.enemies_spawned < self.enemies_in_wave:
            self.enemies_spawned += 1
            start_pos = self._grid_to_screen(self.path_waypoints[0])
            speed = 1.0 + self.current_wave * 0.1
            health = 20 + self.current_wave * 10
            self.enemies.append({
                "pos": list(start_pos), "health": health, "max_health": health,
                "speed": speed, "waypoint_idx": 1, "id": self.np_random.integers(1, 1e9)
            })

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = None
                min_dist = tower['spec']['range'] * self.CELL_SIZE
                tower_screen_pos = self._grid_to_screen(tower['pos'])
                
                for enemy in self.enemies:
                    dist = math.hypot(enemy['pos'][0] - tower_screen_pos[0], enemy['pos'][1] - tower_screen_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    tower['cooldown'] = tower['spec']['fire_rate']
                    self.projectiles.append({
                        "pos": list(tower_screen_pos), "target_id": target['id'],
                        "spec": tower['spec']
                    })
                    # sfx: shoot.wav
                    self._add_particle(tower_screen_pos, color=tower['spec']['color'], life=5, count=3)

    def _update_projectiles(self):
        hit_reward = 0.0
        kill_reward = 0.0
        resource_gain = 0
        
        for proj in self.projectiles[:]:
            target_enemy = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            
            if not target_enemy:
                self.projectiles.remove(proj)
                continue

            # Move projectile towards target
            target_pos = target_enemy['pos']
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj['spec']['proj_speed']:
                # Hit
                target_enemy['health'] -= proj['spec']['damage']
                hit_reward += 0.1
                self.projectiles.remove(proj)
                # sfx: hit_enemy.wav
                for _ in range(10): self._add_particle(target_enemy['pos'], color=self.COLOR_ENEMY, life=15)
                
                if target_enemy['health'] <= 0:
                    kill_reward += 1.0
                    resource_gain += 5 + self.current_wave
                    # sfx: enemy_die.wav
                    for _ in range(30): self._add_particle(target_enemy['pos'], color=self.COLOR_ENEMY, life=30, speed_mult=2.0)
                    self.enemies.remove(target_enemy)
            else:
                proj['pos'][0] += (dx / dist) * proj['spec']['proj_speed']
                proj['pos'][1] += (dy / dist) * proj['spec']['proj_speed']
                
        return hit_reward, kill_reward, resource_gain

    def _update_enemies(self):
        damage_to_base = 0
        for enemy in self.enemies[:]:
            if enemy['waypoint_idx'] >= len(self.path_waypoints):
                damage_to_base += 10 # Each enemy deals 10 damage
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                self._add_particle(self._grid_to_screen(self.base_pos), color=self.COLOR_BASE, count=50, life=40, speed_mult=3.0)
                continue
            
            target_pos = self._grid_to_screen(self.path_waypoints[enemy['waypoint_idx']])
            dx = target_pos[0] - enemy['pos'][0]
            dy = target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < enemy['speed']:
                enemy['waypoint_idx'] += 1
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']
        return damage_to_base

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_path()
        
        # Base
        base_px = self._grid_to_screen(self.base_pos)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (*base_px, self.CELL_SIZE, self.CELL_SIZE))

        # Towers
        for tower in self.towers:
            pos = self._grid_to_screen(tower['pos'])
            p1 = (pos[0] + self.CELL_SIZE // 2, pos[1])
            p2 = (pos[0], pos[1] + self.CELL_SIZE)
            p3 = (pos[0] + self.CELL_SIZE, pos[1] + self.CELL_SIZE)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), tower['spec']['color'])
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), tower['spec']['color'])

        # Projectiles
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, proj['spec']['color'], (int(proj['pos'][0]), int(proj['pos'][1])), 3)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            radius = self.CELL_SIZE // 2
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_w = int(radius * 2 * health_ratio)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (pos[0] - radius, pos[1] - radius - 5, bar_w, 3))
            
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            surf = pygame.Surface((2, 2), pygame.SRCALPHA)
            surf.fill(color)
            self.screen.blit(surf, (int(p['pos'][0]), int(p['pos'][1])))

        # Cursor
        spec = self.TOWER_SPECS[self.selected_tower_type]
        cursor_px = self._grid_to_screen(self.cursor_pos)
        # Range indicator
        range_px = spec['range'] * self.CELL_SIZE
        range_surface = pygame.Surface((range_px * 2, range_px * 2), pygame.SRCALPHA)
        pygame.draw.circle(range_surface, (*spec['color'], 50), (range_px, range_px), range_px)
        pygame.draw.circle(range_surface, (*spec['color'], 100), (range_px, range_px), range_px, 1)
        self.screen.blit(range_surface, (cursor_px[0] + self.CELL_SIZE//2 - range_px, cursor_px[1] + self.CELL_SIZE//2 - range_px))
        # Cursor box
        pygame.draw.rect(self.screen, spec['color'], (*cursor_px, self.CELL_SIZE, self.CELL_SIZE), 2)


    def _render_ui(self):
        # Health Bar
        health_ratio = self.base_health / self.STARTING_BASE_HEALTH
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_s.render(f"Base: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Resources
        res_text = self.font_s.render(f"$: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (220, 12))

        # Wave Info
        wave_str = f"Wave: {self.current_wave}/{self.MAX_WAVES}" if self.current_wave > 0 else "Wave: 0/10"
        wave_text = self.font_s.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - 150, 12))

        # Inter-wave timer
        if not self.wave_active and not self.game_over:
            timer_text = self.font_m.render(f"Next wave in: {math.ceil(self.wave_timer / 30)}s", True, self.COLOR_TEXT)
            text_rect = timer_text.get_rect(center=(self.WIDTH/2, 30))
            self.screen.blit(timer_text, text_rect)
        
        # Selected Tower Info
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_info_text = self.font_s.render(f"Build: {spec['name']} (Cost: {spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_info_text, (10, self.HEIGHT - 25))

        # Game Over / Win
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            msg = "YOU WON" if self.game_won else "GAME OVER"
            end_text = self.font_l.render(msg, True, self.COLOR_BASE if self.game_won else self.COLOR_ENEMY)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score, "steps": self.steps, "wave": self.current_wave,
            "base_health": self.base_health, "resources": self.resources,
        }
        
    def _generate_path(self):
        self.path_waypoints = []
        self.path_coords.clear()
        
        # Start on left, move towards right half
        pos = [0, self.np_random.integers(3, self.GRID_H - 4)]
        self.path_waypoints.append(list(pos))

        while pos[0] < self.GRID_W - 5:
            move_x = self.np_random.integers(3, 7)
            pos[0] += move_x
            
            # Add intermediate points for smooth path drawing
            for i in range(1, move_x + 1):
                self.path_coords.add((self.path_waypoints[-1][0] + i, self.path_waypoints[-1][1]))
            self.path_waypoints.append(list(pos))
            
            if pos[0] >= self.GRID_W - 5: break

            move_y_dir = 1 if pos[1] < self.GRID_H / 2 else -1
            if self.np_random.random() < 0.3: move_y_dir *= -1 # chance to go wrong way
            move_y = self.np_random.integers(2, 5) * move_y_dir
            
            new_y = np.clip(pos[1] + move_y, 1, self.GRID_H - 2)
            
            for i in range(1, abs(new_y - pos[1]) + 1):
                self.path_coords.add((pos[0], pos[1] + i * np.sign(move_y)))
            
            pos[1] = new_y
            self.path_waypoints.append(list(pos))

        self.base_pos = tuple(self.path_waypoints[-1])
        self.path_coords.add(self.base_pos)

    def _draw_grid(self):
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
    def _draw_path(self):
        if len(self.path_waypoints) < 2: return
        path_px = [self._grid_to_screen(p) for p in self.path_waypoints]
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, path_px, self.CELL_SIZE)
        
    def _grid_to_screen(self, grid_pos):
        return [grid_pos[0] * self.CELL_SIZE, grid_pos[1] * self.CELL_SIZE]
        
    def _add_particle(self, pos, color=(255,255,255), life=20, count=1, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2.0 * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Tower Defense Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # no-op, release, release
        
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
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # Blit the observation from the environment to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print("Game Over! Resetting...")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Limit manual play to 30 FPS

    env.close()