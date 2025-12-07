import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the placement cursor. "
        "Space to build the selected tower. Shift to cycle tower types."
    )

    game_description = (
        "An isometric tower defense game. Place towers to defend your base "
        "from waves of enemies. Earn resources by defeating enemies and "
        "survive all waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 20, 10
    ORIGIN_X, ORIGIN_Y = SCREEN_WIDTH // 2, 60
    
    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (45, 50, 62)
    COLOR_PATH = (70, 78, 95)
    COLOR_BASE = (0, 150, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (40, 44, 52, 200)

    TOWER_SPECS = {
        "Gatling": {
            "cost": 50, "damage": 2, "range": 4.5, "fire_rate": 5, 
            "color": (0, 255, 150), "projectile_speed": 15, "projectile_color": (150, 255, 200)
        },
        "Cannon": {
            "cost": 120, "damage": 25, "range": 6, "fire_rate": 45, 
            "color": (255, 150, 0), "projectile_speed": 10, "projectile_color": (255, 200, 150)
        }
    }

    WAVE_DEFINITIONS = [
        {"count": 10, "health": 20, "speed": 1.0, "spawn_delay": 45, "reward": 10},
        {"count": 15, "health": 30, "speed": 1.1, "spawn_delay": 30, "reward": 20},
        {"count": 20, "health": 45, "speed": 1.2, "spawn_delay": 25, "reward": 30},
        {"count": 25, "health": 60, "speed": 1.3, "spawn_delay": 20, "reward": 40},
        {"count": 30, "health": 80, "speed": 1.4, "spawn_delay": 15, "reward": 50},
    ]

    MAX_STEPS = 15000

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
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.path_waypoints = self._define_path()
        self.buildable_tiles = self._get_buildable_tiles()
        self.tower_types = list(self.TOWER_SPECS.keys())

        # All state variables are initialized in reset()
        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        self.base_health = 100
        self.resources = 150
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = deque(maxlen=200)

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_idx = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        # Wave management
        self.current_wave_idx = -1
        self.wave_cooldown = 150 # Time before first wave
        self.wave_spawning = False
        self.spawn_timer = 0
        self.enemies_spawned_in_wave = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(60) # Run at 60 FPS for smooth logic/rendering
        self.steps += 1
        
        reward = 0.0

        # 1. Handle player input
        self._handle_input(action)

        # 2. Update game state
        self._update_waves()
        reward += self._update_enemies()
        self._update_towers()
        self._update_projectiles()
        self._update_particles()
        
        # 3. Check for termination
        terminated = False
        if self.base_health <= 0 and not self.game_over:
            self.game_over = True
            self.win_condition = False
            reward -= 100.0
            terminated = True
        
        if self.win_condition and not self.game_over:
            self.game_over = True
            reward += 100.0
            terminated = True

        if self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_over: # Timed out
                reward -= 50.0

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        if movement == 2: self.cursor_pos[1] += 1 # Down
        if movement == 3: self.cursor_pos[0] -= 1 # Left
        if movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Cycle Tower (Shift Press) ---
        if shift_held and not self.last_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.tower_types)
            # sfx: ui_cycle.wav

        # --- Place Tower (Space Press) ---
        if space_held and not self.last_space_held:
            self._try_place_tower()
            # sfx: ui_confirm.wav or ui_error.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _try_place_tower(self):
        pos = tuple(self.cursor_pos)
        if pos not in self.buildable_tiles: return
        if any(t['pos'] == pos for t in self.towers): return

        spec = self.TOWER_SPECS[self.tower_types[self.selected_tower_idx]]
        if self.resources >= spec['cost']:
            self.resources -= spec['cost']
            self.towers.append({
                'pos': pos,
                'type': self.tower_types[self.selected_tower_idx],
                'spec': spec,
                'cooldown': 0,
                'angle': -math.pi/2
            })
            # sfx: tower_place.wav
            self._create_particles(self._iso_to_screen(pos[0], pos[1]), 20, spec['color'])

    def _update_waves(self):
        if self.win_condition: return

        # Check if wave is complete
        if self.wave_spawning and self.enemies_spawned_in_wave >= self.WAVE_DEFINITIONS[self.current_wave_idx]['count'] and not self.enemies:
            self.wave_spawning = False
            self.wave_cooldown = 300 # 5 seconds
            
            # Check for win condition
            if self.current_wave_idx >= len(self.WAVE_DEFINITIONS) - 1:
                self.win_condition = True
                return

        # Countdown to next wave
        if not self.wave_spawning:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                self.current_wave_idx += 1
                self.wave_spawning = True
                self.enemies_spawned_in_wave = 0
                self.spawn_timer = 0
    
        # Spawn enemies
        if self.wave_spawning:
            self.spawn_timer -= 1
            wave_spec = self.WAVE_DEFINITIONS[self.current_wave_idx]
            if self.spawn_timer <= 0 and self.enemies_spawned_in_wave < wave_spec['count']:
                self.spawn_timer = wave_spec['spawn_delay']
                self.enemies_spawned_in_wave += 1
                start_pos = self.path_waypoints[0]
                self.enemies.append({
                    'pos': list(start_pos),
                    'health': wave_spec['health'],
                    'max_health': wave_spec['health'],
                    'speed': wave_spec['speed'],
                    'path_idx': 1,
                    'id': self.np_random.random()
                })
                # sfx: enemy_spawn.wav

    def _update_enemies(self):
        reward = 0.0
        for enemy in self.enemies[:]:
            target_waypoint = self.path_waypoints[enemy['path_idx']]
            direction = np.array(target_waypoint) - np.array(enemy['pos'])
            distance = np.linalg.norm(direction)

            if distance < 0.1:
                enemy['path_idx'] += 1
                if enemy['path_idx'] >= len(self.path_waypoints):
                    self.enemies.remove(enemy)
                    self.base_health = max(0, self.base_health - 10)
                    reward -= 1.0 # Significant penalty for letting one through
                    # sfx: base_damage.wav
                    continue
            else:
                move_vec = direction / distance * enemy['speed'] * 0.05
                enemy['pos'][0] += move_vec[0]
                enemy['pos'][1] += move_vec[1]
        return reward

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            target = None
            min_dist = tower['spec']['range']
            
            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower['cooldown'] = tower['spec']['fire_rate']
                start_pos = self._iso_to_screen(tower['pos'][0], tower['pos'][1])
                self.projectiles.append({
                    'start_pos': start_pos,
                    'pos': list(start_pos),
                    'target': target,
                    'spec': tower['spec']
                })
                # sfx: tower_shoot.wav
                
                # Update tower angle to face target
                dx = target['pos'][0] - tower['pos'][0]
                dy = target['pos'][1] - tower['pos'][1]
                tower['angle'] = math.atan2(dy, dx)


    def _update_projectiles(self):
        for p in self.projectiles[:]:
            if p['target'] not in self.enemies:
                self.projectiles.remove(p)
                continue

            target_screen_pos = self._iso_to_screen(p['target']['pos'][0], p['target']['pos'][1])
            direction = np.array(target_screen_pos) - np.array(p['pos'])
            distance = np.linalg.norm(direction)

            if distance < p['spec']['projectile_speed']:
                # Hit
                p['target']['health'] -= p['spec']['damage']
                self._create_particles(target_screen_pos, 5, p['spec']['projectile_color'])
                self.projectiles.remove(p)
                # sfx: projectile_hit.wav
                if p['target']['health'] <= 0:
                    self.resources += 5
                    self.score += 0.1 # Small reward for kill
                    self._create_particles(target_screen_pos, 15, self.COLOR_ENEMY, 1.5)
                    self.enemies.remove(p['target'])
                    # sfx: enemy_die.wav
            else:
                move_vec = direction / distance * p['spec']['projectile_speed']
                p['pos'][0] += move_vec[0]
                p['pos'][1] += move_vec[1]
    
    def _update_particles(self):
        for p in list(self.particles):
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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave_idx + 1,
            "enemies": len(self.enemies),
        }

    def _render_game(self):
        # 1. Draw grid and path
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                is_path = any(math.hypot(x - p[0], y - p[1]) < 1.5 for p in self.path_waypoints)
                color = self.COLOR_PATH if (x,y) in self.path_waypoints else self.COLOR_GRID
                self._draw_iso_tile(x, y, color)

        # 2. Collect and sort all dynamic objects for correct draw order
        render_queue = []
        for tower in self.towers:
            render_queue.append({'type': 'tower', 'obj': tower, 'sort_key': tower['pos'][0] + tower['pos'][1]})
        for enemy in self.enemies:
            render_queue.append({'type': 'enemy', 'obj': enemy, 'sort_key': enemy['pos'][0] + enemy['pos'][1]})
        
        render_queue.sort(key=lambda item: item['sort_key'])

        # 3. Draw sorted objects
        for item in render_queue:
            if item['type'] == 'tower': self._draw_tower(item['obj'])
            elif item['type'] == 'enemy': self._draw_enemy(item['obj'])

        # 4. Draw projectiles and particles on top
        for p in self.projectiles:
            pygame.draw.circle(self.screen, p['spec']['projectile_color'], (int(p['pos'][0]), int(p['pos'][1])), 3)
        
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'] + (alpha,)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # 5. Draw cursor
        self._draw_cursor()

    def _render_ui(self):
        # --- Base Health ---
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, 200, 20))
        health_width = int(196 * (self.base_health / 100.0))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (12, 12, health_width, 16))
        self._draw_text(f"BASE HEALTH: {self.base_health}", (110, 20), self.font_small, center=True)

        # --- Wave Info ---
        wave_text = f"WAVE: {self.current_wave_idx + 1}/{len(self.WAVE_DEFINITIONS)}" if self.current_wave_idx >= 0 else "WAVE: 0/5"
        if self.win_condition: wave_text = "VICTORY!"
        self._draw_text(wave_text, (self.SCREEN_WIDTH - 10, 20), self.font_small, align='right')

        # --- Resources & Tower Selection ---
        ui_panel_rect = (self.SCREEN_WIDTH // 2 - 150, self.SCREEN_HEIGHT - 50, 300, 45)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_panel_rect, border_radius=5)
        
        res_text = f"R: {self.resources}"
        self._draw_text(res_text, (ui_panel_rect[0] + 15, ui_panel_rect[1] + 22), self.font_small, align='left')
        
        tower_type = self.tower_types[self.selected_tower_idx]
        spec = self.TOWER_SPECS[tower_type]
        tower_text = f"Build: {tower_type} (Cost: {spec['cost']})"
        self._draw_text(tower_text, (ui_panel_rect[0] + 150, ui_panel_rect[1] + 22), self.font_small, center=True, color=spec['color'])

        # --- Game Over Screen ---
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "YOU WIN!" if self.win_condition else "GAME OVER"
            color = (100, 255, 100) if self.win_condition else (255, 100, 100)
            self._draw_text(msg, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_large, center=True, color=color)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, center=False, align='left'):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center: text_rect.center = pos
        elif align == 'right': text_rect.topright = pos
        else: text_rect.midleft = pos
        self.screen.blit(text_surface, text_rect)

    def _draw_iso_tile(self, x, y, color):
        points = [
            self._iso_to_screen(x, y + 1),
            self._iso_to_screen(x + 1, y + 1),
            self._iso_to_screen(x + 1, y),
            self._iso_to_screen(x, y)
        ]
        # Ensure color is a valid tuple of integers, not a generator or floats
        int_color = tuple(int(c) for c in color)
        pygame.gfxdraw.filled_polygon(self.screen, points, int_color)
        
        # Create a slightly brighter, clamped color for the outline
        outline_color = tuple(min(255, int(c * 1.2)) for c in int_color)
        pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _draw_cursor(self):
        x, y = self.cursor_pos
        screen_pos = self._iso_to_screen(x, y)
        
        is_buildable = tuple(self.cursor_pos) in self.buildable_tiles and not any(t['pos'] == tuple(self.cursor_pos) for t in self.towers)
        spec = self.TOWER_SPECS[self.tower_types[self.selected_tower_idx]]
        has_resources = self.resources >= spec['cost']

        color = (0, 255, 0, 150) if is_buildable and has_resources else (255, 0, 0, 150)
        
        radius = self.TILE_WIDTH_HALF * spec['range']
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (radius, radius), radius)
        self.screen.blit(s, (screen_pos[0] - radius, screen_pos[1] - radius))
        
        cursor_tile_color = tuple(int(c * 0.5) for c in color)
        self._draw_iso_tile(x, y, cursor_tile_color)

    def _draw_tower(self, tower):
        screen_pos = self._iso_to_screen(tower['pos'][0], tower['pos'][1])
        spec = tower['spec']
        color = spec['color']
        
        # Base
        pygame.draw.circle(self.screen, tuple(int(c*0.6) for c in color), screen_pos, 8)
        pygame.draw.circle(self.screen, color, screen_pos, 6)
        
        # Barrel
        barrel_len = 10
        end_x = screen_pos[0] + barrel_len * math.cos(tower['angle'])
        end_y = screen_pos[1] + barrel_len * math.sin(tower['angle'])
        pygame.draw.line(self.screen, tuple(int(c*0.8) for c in color), screen_pos, (end_x, end_y), 4)

    def _draw_enemy(self, enemy):
        screen_pos = self._iso_to_screen(enemy['pos'][0], enemy['pos'][1])
        
        # Bobbing animation
        bob = math.sin(self.steps * 0.1 + enemy['id'] * 10) * 2
        pos = (int(screen_pos[0]), int(screen_pos[1] - bob - 5))
        
        # Body
        pygame.draw.circle(self.screen, tuple(int(c*0.7) for c in self.COLOR_ENEMY), pos, 7)
        pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 5)

        # Health bar
        bar_w = 16
        health_pct = enemy['health'] / enemy['max_health']
        health_w = int(bar_w * health_pct)
        pygame.draw.rect(self.screen, (50, 0, 0), (pos[0] - bar_w//2 -1, pos[1] - 15 -1, bar_w+2, 5))
        pygame.draw.rect(self.screen, (0, 200, 0), (pos[0] - bar_w//2, pos[1] - 15, health_w, 3))

    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = (self.np_random.random() * 2 + 1) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 25)
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life,
                'color': color, 'size': self.np_random.integers(1, 4)
            })

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _define_path(self):
        path = []
        path.extend([(i, 2) for i in range(-1, 8)])
        path.extend([(7, i) for i in range(3, 9)])
        path.extend([(i, 8) for i in range(8, 15)])
        path.extend([(14, i) for i in range(7, 2, -1)])
        path.extend([(i, 3) for i in range(15, 21)])
        return path

    def _get_buildable_tiles(self):
        tiles = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                is_path = False
                for px, py in self.path_waypoints:
                    if math.hypot(x-px, y-py) < 2.0: # Buffer around path
                        is_path = True
                        break
                if not is_path:
                    tiles.add((x, y))
        return tiles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-set the dummy driver to allow for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Tower Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    print(env.user_guide)
    print(env.game_description)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No movement
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Match the internal clock for smooth playback
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Resources: {info['resources']}")

    print(f"Game Over! Final Score: {info['score']:.2f}")
    env.close()