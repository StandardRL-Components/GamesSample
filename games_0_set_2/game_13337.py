import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:49:40.029759
# Source Brief: brief_03337.md
# Brief Index: 3337
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An Oracle Bone Tower Defense game.

    The player matches ancient oracle bone symbols to summon mythical beasts.
    These beasts are then teleported onto the battlefield to defend a city
    against waves of invading armies. The game is presented in a stylized
    2D aesthetic inspired by the Chinese Shang Dynasty.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement/Pause (0=pause, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Match/Summon (0=released, 1=pressed) - Space Bar
    - actions[2]: Teleport Beast (0=released, 1=pressed) - Shift Key

    Observation Space: A 640x400 RGB image of the game state.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Match ancient oracle bone symbols to summon mythical beasts and defend your city from "
        "invading armies in this tower defense game."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select symbols. Press space to attempt a match and "
        "summon a beast. Press shift to cycle the teleport location for the last-summoned beast."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # === Game Constants ===
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 3000
        self.NUM_WAVES = 10
        self.CITY_HEALTH_MAX = 100
        self.MANA_MAX = 100
        self.MANA_REGEN_RATE = 0.05  # Mana per step

        # === Colors (High Contrast & Thematic) ===
        self.COLOR_BG = (20, 15, 10)
        self.COLOR_UI_BG = (30, 25, 20)
        self.COLOR_PATH = (50, 40, 30)
        self.COLOR_CITY = (100, 80, 60)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_BEAST_PRIMARY = (50, 150, 250)
        self.COLOR_BEAST_SECONDARY = (100, 200, 255)
        self.COLOR_HEALTH = (50, 200, 50)
        self.COLOR_MANA = (250, 180, 50)
        self.COLOR_TEXT = (230, 220, 200)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TELEPORT_MARKER = (255, 255, 0, 128) # RGBA for transparency

        # === Gymnasium Spaces ===
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # === Pygame Setup ===
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("sans-serif", 16)
        self.font_m = pygame.font.SysFont("sans-serif", 24)
        self.font_l = pygame.font.SysFont("sans-serif", 48)

        # === Game Entity Definitions ===
        self.path_points = [
            (-20, 150), (100, 150), (100, 250), (350, 250),
            (350, 100), (500, 100), (500, 300), (600, 300)
        ]
        self.city_rect = pygame.Rect(580, 280, 60, 40)
        self.teleport_locations = [(225, 175), (425, 175), (300, 320)]
        
        self.oracle_grid_size = (4, 2)
        self.oracle_symbols_defs = {
            1: [((0, -1), (0, 1))],  # 'Man'
            2: [((-1, 0), (1, 0)), ((0, -1), (0, 1))], # 'Field'
            3: [((-1, -1), (1, 1)), ((-1, 1), (1, -1))], # 'Mountain'
            4: [((0, -1), (0, 0)), ((-0.5, 0.5), (0, 0)), ((0.5, 0.5), (0, 0))], # 'Water'
            5: [((0, -1), (0, 1)), ((-1, 0), (-0.5, 0)), ((1, 0), (0.5, 0))] # 'Tree'
        }
        
        self.beast_recipes = {
            (1, 2): {'name': 'Taotie', 'cost': 30, 'hp': 100, 'dmg': 15, 'range': 80, 'rate': 60},
            (4, 5, 1): {'name': 'Qilin', 'cost': 50, 'hp': 150, 'dmg': 5, 'range': 60, 'rate': 15},
            (3, 3, 3): {'name': 'Fenghuang', 'cost': 80, 'hp': 80, 'dmg': 25, 'range': 100, 'rate': 90}
        }
        self.unlocked_beasts = [(1, 2)]

        # === State variables (initialized in reset) ===
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.is_paused = True
        self.last_space_action = 0
        self.last_shift_action = 0
        self.city_health = 0
        self.mana = 0
        self.wave_number = 0
        self.enemies = []
        self.beasts = []
        self.particles = []
        self.oracle_grid = []
        self.cursor_pos = [0, 0]
        self.match_selection = []
        self.teleport_target_idx = 0
        self.last_summoned_beast = None
        self.enemies_spawned_this_wave = 0
        self.enemies_to_spawn = 0
        self.last_spawn_step = 0
        self.next_entity_id = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.is_paused = True
        self.last_space_action = 0
        self.last_shift_action = 0
        
        self.city_health = self.CITY_HEALTH_MAX
        self.mana = self.MANA_MAX / 2
        
        self.wave_number = 1
        self.enemies = []
        self.beasts = []
        self.particles = []
        
        self.oracle_grid = [
            [self.np_random.integers(1, len(self.oracle_symbols_defs) + 1) for _ in range(self.oracle_grid_size[0])]
            for _ in range(self.oracle_grid_size[1])
        ]
        self.cursor_pos = [0, 0]
        self.match_selection = []
        
        self.teleport_target_idx = 0
        self.last_summoned_beast = None
        self.unlocked_beasts = [(1, 2)]
        
        self.next_entity_id = 0
        self._setup_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement, space_action, shift_action = action
        
        # --- Handle Actions ---
        self.is_paused = (movement == 0)
        
        if not self.is_paused:
            self._handle_cursor_move(movement)
        
        if space_action and not self.last_space_action: # Rising edge for space
            match_reward = self._handle_match_attempt()
            reward += match_reward
        self.last_space_action = space_action
        
        if shift_action and not self.last_shift_action: # Rising edge for shift
            self._handle_teleport_cycle()
        self.last_shift_action = shift_action

        # --- Update Game State (if not paused) ---
        if not self.is_paused:
            self.steps += 1
            self.mana = min(self.MANA_MAX, self.mana + self.MANA_REGEN_RATE)

            health_lost, enemies_killed_reward = self._update_enemies()
            reward -= health_lost * 0.5 # Penalty for city damage
            reward += enemies_killed_reward
            self.city_health -= health_lost

            damage_dealt_reward = self._update_beasts()
            reward += damage_dealt_reward * 0.01 # Small reward for dealing damage

            self._update_particles()
            
            if self._check_wave_completion():
                reward += 5 # Wave clear bonus
                self.wave_number += 1
                if self.wave_number > self.NUM_WAVES:
                    self.game_over = True # Victory
                else:
                    self._setup_wave()
                    if self.wave_number == 3: self.unlocked_beasts.append((4, 5, 1))
                    if self.wave_number == 6: self.unlocked_beasts.append((3, 3, 3))

        # --- Check Termination ---
        terminated = False
        if self.city_health <= 0:
            reward -= 100 # Loss penalty
            self.game_over = terminated = True
        elif self.game_over: # Victory case
            reward += 100 # Win bonus
            terminated = True
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Game Logic Helpers ---
    def _get_unique_id(self):
        self.next_entity_id += 1
        return self.next_entity_id

    def _setup_wave(self):
        self.enemies_to_spawn = 5 + self.wave_number * 2
        self.enemies_spawned_this_wave = 0
        self.last_spawn_step = self.steps
        self.spawn_interval = max(30, 90 - self.wave_number * 5)

    def _handle_cursor_move(self, movement):
        if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.oracle_grid_size[1]
        if movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.oracle_grid_size[1]
        if movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.oracle_grid_size[0]
        if movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.oracle_grid_size[0]
        
        # Add selected symbol to match list
        symbol = self.oracle_grid[self.cursor_pos[1]][self.cursor_pos[0]]
        self.match_selection.append(symbol)
        if len(self.match_selection) > 5: # Limit selection length
            self.match_selection.pop(0)

    def _handle_match_attempt(self):
        selection_tuple = tuple(self.match_selection)
        
        # Check recipes from longest to shortest
        for length in range(len(selection_tuple), 1, -1):
            sub_selection = selection_tuple[-length:]
            if sub_selection in self.beast_recipes and sub_selection in self.unlocked_beasts:
                recipe = self.beast_recipes[sub_selection]
                if self.mana >= recipe['cost']:
                    self.mana -= recipe['cost']
                    self._summon_beast(recipe)
                    # sfx: summon_success
                    self.match_selection = []
                    return 1.0 # Reward for successful summon
        
        # sfx: summon_fail
        self.match_selection = []
        return -0.1 # Small penalty for failed attempt

    def _summon_beast(self, recipe):
        beast = {
            'id': self._get_unique_id(),
            'pos': list(self.teleport_locations[self.teleport_target_idx]),
            'recipe': recipe,
            'hp': recipe['hp'],
            'max_hp': recipe['hp'],
            'cooldown': 0,
            'target_id': None
        }
        self.beasts.append(beast)
        self.last_summoned_beast = beast
        self._create_particle_burst(beast['pos'], self.COLOR_BEAST_PRIMARY, 30)

    def _handle_teleport_cycle(self):
        if self.last_summoned_beast:
            self.teleport_target_idx = (self.teleport_target_idx + 1) % len(self.teleport_locations)
            new_pos = self.teleport_locations[self.teleport_target_idx]
            
            # sfx: teleport
            self._create_particle_burst(self.last_summoned_beast['pos'], (255,255,255), 15, 'implode')
            self.last_summoned_beast['pos'] = list(new_pos)
            self._create_particle_burst(new_pos, (255,255,255), 25, 'explode')

    def _update_enemies(self):
        health_lost = 0
        kill_reward = 0
        
        # Spawn new enemies
        if self.enemies_spawned_this_wave < self.enemies_to_spawn and (self.steps - self.last_spawn_step) > self.spawn_interval:
            self.last_spawn_step = self.steps
            self.enemies_spawned_this_wave += 1
            
            base_hp = 20 + self.wave_number * 10
            base_speed = 0.5 + self.wave_number * 0.1
            
            enemy = {
                'id': self._get_unique_id(),
                'pos': list(self.path_points[0]),
                'hp': base_hp,
                'max_hp': base_hp,
                'speed': base_speed,
                'path_idx': 1
            }
            self.enemies.append(enemy)

        # Move existing enemies
        for enemy in self.enemies[:]:
            if enemy['path_idx'] >= len(self.path_points):
                health_lost += 10
                self.enemies.remove(enemy)
                self._create_particle_burst(enemy['pos'], self.COLOR_ENEMY, 20)
                # sfx: city_damage
                continue

            target_pos = self.path_points[enemy['path_idx']]
            direction = np.array(target_pos) - np.array(enemy['pos'])
            distance = np.linalg.norm(direction)
            
            if distance < enemy['speed']:
                enemy['path_idx'] += 1
            else:
                move_vec = (direction / distance) * enemy['speed']
                enemy['pos'][0] += move_vec[0]
                enemy['pos'][1] += move_vec[1]
        
        # Check for deaths (handled in _update_beasts after damage)
        enemies_alive_before = len(self.enemies)
        self.enemies = [e for e in self.enemies if e['hp'] > 0]
        enemies_killed = enemies_alive_before - len(self.enemies)
        if enemies_killed > 0:
            kill_reward = 2.0 * enemies_killed
            # sfx: enemy_death

        return health_lost, kill_reward

    def _update_beasts(self):
        total_damage_dealt = 0
        for beast in self.beasts:
            beast['cooldown'] = max(0, beast['cooldown'] - 1)
            
            if beast['cooldown'] > 0:
                continue

            # Find target
            target = None
            min_dist = float('inf')
            for enemy in self.enemies:
                dist = np.linalg.norm(np.array(beast['pos']) - np.array(enemy['pos']))
                if dist < beast['recipe']['range'] and dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                beast['cooldown'] = beast['recipe']['rate']
                damage = beast['recipe']['dmg']
                target['hp'] -= damage
                total_damage_dealt += damage
                # sfx: beast_attack
                
                # Visual effect for attack
                self._create_particle_line(beast['pos'], target['pos'], self.COLOR_BEAST_SECONDARY, 5)
                if target['hp'] <= 0:
                    self._create_particle_burst(target['pos'], self.COLOR_ENEMY, 15)

        return total_damage_dealt

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            p['radius'] *= 0.98
            if p['lifetime'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)

    def _check_wave_completion(self):
        return self.enemies_spawned_this_wave == self.enemies_to_spawn and not self.enemies

    def _create_particle_burst(self, pos, color, count, mode='explode'):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4) if mode == 'explode' else random.uniform(0.5, 2)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            if mode == 'implode': vel = [-v for v in vel]
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'color': color,
                'radius': random.uniform(2, 5), 'lifetime': random.randint(15, 30)
            })

    def _create_particle_line(self, start, end, color, count):
        for i in range(count):
            t = i / (count -1) if count > 1 else 0.5
            pos = [start[0] * (1-t) + end[0] * t, start[1] * (1-t) + end[1] * t]
            self.particles.append({
                'pos': pos, 'vel': [0,0], 'color': color,
                'radius': random.uniform(1, 3), 'lifetime': 5
            })

    # --- Rendering Methods ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        if len(self.path_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_points, 30)
        
        # Draw city
        pygame.draw.rect(self.screen, self.COLOR_CITY, self.city_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, self.city_rect, 1)

        # Draw teleport marker
        if self.last_summoned_beast:
            pos = self.teleport_locations[self.teleport_target_idx]
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 20, self.COLOR_TELEPORT_MARKER)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 20, self.COLOR_CURSOR)

        # Draw beasts
        for beast in self.beasts:
            pos = (int(beast['pos'][0]), int(beast['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_BEAST_PRIMARY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, self.COLOR_BEAST_SECONDARY)
            self._draw_bar(beast['pos'], beast['hp'], beast['max_hp'], 20, self.COLOR_HEALTH)

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.gfxdraw.filled_trigon(self.screen, pos[0], pos[1]-8, pos[0]-7, pos[1]+6, pos[0]+7, pos[1]+6, self.COLOR_ENEMY)
            self._draw_bar(enemy['pos'], enemy['hp'], enemy['max_hp'], 16, self.COLOR_ENEMY)
        
        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            rad = int(p['radius'])
            if rad > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad, p['color'])

    def _render_ui(self):
        ui_panel_rect = pygame.Rect(0, self.HEIGHT - 80, self.WIDTH, 80)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_panel_rect)
        pygame.draw.line(self.screen, self.COLOR_TEXT, (0, self.HEIGHT-80), (self.WIDTH, self.HEIGHT-80))
        
        # Draw Oracle Grid
        grid_x_start, grid_y_start = 20, self.HEIGHT - 60
        cell_size, padding = 30, 5
        for r, row in enumerate(self.oracle_grid):
            for c, symbol_id in enumerate(row):
                x = grid_x_start + c * (cell_size + padding)
                y = grid_y_start + r * (cell_size + padding)
                rect = pygame.Rect(x, y, cell_size, cell_size)
                pygame.draw.rect(self.screen, self.COLOR_PATH, rect)
                self._draw_oracle_symbol(symbol_id, (x + cell_size/2, y + cell_size/2), 10)
        
        # Draw Cursor
        c, r = self.cursor_pos
        x = grid_x_start + c * (cell_size + padding)
        y = grid_y_start + r * (cell_size + padding)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (x, y, cell_size, cell_size), 2)

        # Draw Match Selection
        sel_x_start = grid_x_start + (self.oracle_grid_size[0] + 1) * (cell_size + padding)
        self._draw_text("Match:", (sel_x_start, self.HEIGHT - 75), self.font_s)
        for i, symbol_id in enumerate(self.match_selection):
            pos = (sel_x_start + 15 + i * 20, self.HEIGHT - 45)
            self._draw_oracle_symbol(symbol_id, pos, 8)

        # Draw Stats
        stats_x = 420
        self._draw_text(f"Wave: {self.wave_number}/{self.NUM_WAVES}", (stats_x, self.HEIGHT - 70), self.font_m)
        self._draw_text(f"Score: {int(self.score)}", (stats_x, self.HEIGHT - 40), self.font_m)
        
        # Health and Mana Bars
        self._draw_bar((550, self.HEIGHT - 65), self.city_health, self.CITY_HEALTH_MAX, 120, self.COLOR_HEALTH, "City")
        self._draw_bar((550, self.HEIGHT - 35), self.mana, self.MANA_MAX, 120, self.COLOR_MANA, "Mana")

        # Pause/Game Over Overlay
        if self.is_paused and not self.game_over:
            self._draw_text("PAUSED", (self.WIDTH/2, self.HEIGHT/2), self.font_l, center=True)
        if self.game_over:
            msg = "VICTORY" if self.city_health > 0 else "DEFEAT"
            self._draw_text(msg, (self.WIDTH/2, self.HEIGHT/2), self.font_l, center=True)

    def _draw_oracle_symbol(self, symbol_id, pos, size):
        if symbol_id not in self.oracle_symbols_defs: return
        for p1, p2 in self.oracle_symbols_defs[symbol_id]:
            start = (pos[0] + p1[0] * size, pos[1] + p1[1] * size)
            end = (pos[0] + p2[0] * size, pos[1] + p2[1] * size)
            pygame.draw.aaline(self.screen, self.COLOR_TEXT, start, end)

    def _draw_text(self, text, pos, font, color=None, center=False):
        if color is None: color = self.COLOR_TEXT
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surf, text_rect)

    def _draw_bar(self, pos, current, maximum, length, color, label=None):
        if maximum == 0: return
        fill_ratio = max(0, min(1, current / maximum))
        
        if label: # Horizontal bar with label on the left
            label_pos = (pos[0] - length/2 - 35, pos[1])
            self._draw_text(label, label_pos, self.font_s)
            bar_rect = pygame.Rect(pos[0] - length/2, pos[1]-5, length, 10)
            fill_rect = pygame.Rect(pos[0] - length/2, pos[1]-5, length * fill_ratio, 10)
        else: # Vertical bar above a position
            bar_rect = pygame.Rect(pos[0] - length/2, pos[1] - 20, length, 5)
            fill_rect = pygame.Rect(pos[0] - length/2, pos[1] - 20, length * fill_ratio, 5)

        pygame.draw.rect(self.screen, self.COLOR_PATH, bar_rect)
        pygame.draw.rect(self.screen, color, fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, bar_rect, 1)

    # --- Gymnasium Interface ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "city_health": self.city_health,
            "mana": self.mana,
            "beasts_on_field": len(self.beasts),
            "enemies_on_field": len(self.enemies),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed and will open a window.
    # The environment itself runs headlessly.
    os.environ.unsetenv("SDL_VIDEODRIVER")
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Oracle Beast Defense")
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_pressed = 0
    shift_pressed = 0

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Any other key: Pause game")
    print("----------------------\n")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Handle key presses for continuous actions
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_pressed = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_pressed = 1
                else: movement = 0 # Pause on any other key press
            
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0 # Pause when movement key is released
                if event.key == pygame.K_SPACE:
                    space_pressed = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_pressed = 0

        action = [movement, space_pressed, shift_pressed]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            
        clock.tick(env.metadata["render_fps"])

    print("\nGame Over!")
    print(f"Final Info: {info}")
    env.close()