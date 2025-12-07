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


# Set the environment variable for headless Pygame operation
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to place a tower. Hold Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this isometric tower defense game."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 12
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 28, 14
    ISO_OFFSET_X, ISO_OFFSET_Y = SCREEN_WIDTH // 2, 80
    MAX_STEPS = 30 * 60 # 60 seconds at 30fps

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 45, 58)
    COLOR_PATH = (60, 68, 87)
    COLOR_BASE = (0, 200, 150)
    COLOR_BASE_STROKE = (150, 255, 230)
    COLOR_ENEMY = (230, 50, 50)
    COLOR_ENEMY_STROKE = (255, 150, 150)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (40, 45, 58, 180)
    COLOR_HEALTH_BAR_BG = (80, 40, 40)
    COLOR_HEALTH_BAR_FG = (200, 60, 60)
    
    TOWER_SPECS = {
        'GUN': {'cost': 50, 'range': 100, 'damage': 5, 'cooldown': 10, 'color': (0, 180, 255), 'proj_speed': 8},
        'CANNON': {'cost': 120, 'range': 150, 'damage': 25, 'cooldown': 45, 'color': (255, 165, 0), 'proj_speed': 5},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        # FIX: Initialize a display mode. In "dummy" mode, this creates a surface that
        # other Pygame functions (like convert_alpha) can use without opening a window.
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        self.path_grid_coords = self._define_path()
        self.path_iso_coords = [self._cart_to_iso(x, y) for x, y in self.path_grid_coords]
        self.buildable_tiles = self._get_buildable_tiles()
        
        self.tower_types = list(self.TOWER_SPECS.keys())
        
        # self.reset() is called by the gym wrapper, no need to call it here.
        
    def _define_path(self):
        path = []
        for x in range(-1, 6): path.append((x, 1))
        for y in range(1, 5): path.append((5, y))
        for x in range(5, -1, -1): path.append((x, 4))
        for y in range(4, 8): path.append((0, y))
        for x in range(0, 7): path.append((x, 7))
        for y in range(7, 10): path.append((6, y))
        path.append((6, 10)) # Base
        return path

    def _get_buildable_tiles(self):
        all_tiles = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        path_tiles = set(self.path_grid_coords)
        return all_tiles - path_tiles

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        self.win = False

        self.base_health = 100
        self.resources = 150
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_idx = 0
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.wave_number = 0
        self.wave_cooldown = 150 # 5 seconds
        self.enemies_to_spawn = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = -0.01 # Time penalty
        self.steps += 1
        
        self._handle_input(action)
        self._update_game_state()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        reward = self.reward_this_step
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Place Tower (on key press) ---
        if space_held and not self.prev_space_held:
            self._place_tower()
        self.prev_space_held = space_held

        # --- Cycle Tower (on key press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.tower_types)
        self.prev_shift_held = shift_held

    def _place_tower(self):
        pos_tuple = tuple(self.cursor_pos)
        is_buildable = pos_tuple in self.buildable_tiles
        is_occupied = any(t['grid_pos'] == pos_tuple for t in self.towers)
        
        tower_type = self.tower_types[self.selected_tower_idx]
        spec = self.TOWER_SPECS[tower_type]
        
        if is_buildable and not is_occupied and self.resources >= spec['cost']:
            self.resources -= spec['cost']
            iso_pos = self._cart_to_iso(pos_tuple[0], pos_tuple[1])
            self.towers.append({
                'grid_pos': pos_tuple,
                'iso_pos': iso_pos,
                'type': tower_type,
                'spec': spec,
                'cooldown_timer': 0,
                'target': None,
                'bob_offset': random.uniform(0, math.pi * 2)
            })

    def _update_game_state(self):
        self._update_waves()
        self._update_enemies()
        self._update_towers()
        self._update_projectiles()
        self._update_particles()
        
    def _update_waves(self):
        if not self.enemies and not self.enemies_to_spawn and self.wave_number <= 10:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0 and not self.win:
                self.wave_number += 1
                if self.wave_number > 10:
                    self.win = True
                    return
                self.wave_cooldown = 300 # 10s between waves
                self._setup_wave()
        
        if self.enemies_to_spawn and self.steps % 15 == 0: # Spawn interval
            self.enemies.append(self.enemies_to_spawn.pop(0))

    def _setup_wave(self):
        num_enemies = 5 + self.wave_number * 2
        base_health = 10 * (1.1 ** (self.wave_number - 1))
        base_speed = 1.0 * (1.05 ** (self.wave_number - 1))
        
        for i in range(num_enemies):
            start_pos = pygame.Vector2(self.path_iso_coords[0])
            self.enemies_to_spawn.append({
                'pos': start_pos,
                'path_idx': 0,
                'health': base_health,
                'max_health': base_health,
                'speed': base_speed,
                'bob_offset': random.uniform(0, math.pi * 2)
            })

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['path_idx'] < len(self.path_iso_coords) - 1:
                target_pos = pygame.Vector2(self.path_iso_coords[enemy['path_idx'] + 1])
                direction = (target_pos - enemy['pos'])
                if direction.length() < enemy['speed']:
                    enemy['pos'] = target_pos
                    enemy['path_idx'] += 1
                else:
                    enemy['pos'] += direction.normalize() * enemy['speed']
            else: # Reached base
                self.enemies.remove(enemy)
                self.base_health = max(0, self.base_health - 10)
                self.reward_this_step -= 10
                self._create_particles(enemy['pos'], self.COLOR_BASE, 20)

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown_timer'] = max(0, tower['cooldown_timer'] - 1)
            
            if tower['target'] is None or tower['target'] not in self.enemies or pygame.Vector2(tower['iso_pos']).distance_to(tower['target']['pos']) > tower['spec']['range']:
                tower['target'] = None
                closest_enemy = None
                min_dist = float('inf')
                for enemy in self.enemies:
                    dist = pygame.Vector2(tower['iso_pos']).distance_to(enemy['pos'])
                    if dist <= tower['spec']['range'] and dist < min_dist:
                        min_dist = dist
                        closest_enemy = enemy
                tower['target'] = closest_enemy

            if tower['target'] and tower['cooldown_timer'] <= 0:
                tower['cooldown_timer'] = tower['spec']['cooldown']
                self.projectiles.append({
                    'pos': pygame.Vector2(tower['iso_pos']),
                    'target': tower['target'],
                    'spec': tower['spec'],
                })

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj['target']['pos']
            direction = target_pos - proj['pos']
            
            if direction.length() < proj['spec']['proj_speed']:
                proj['target']['health'] -= proj['spec']['damage']
                self.reward_this_step += 0.1
                self._create_particles(proj['pos'], proj['spec']['color'], 5, 1, 3)
                if proj['target']['health'] <= 0:
                    self.reward_this_step += 1
                    self.resources += 5
                    self._create_particles(proj['target']['pos'], self.COLOR_ENEMY, 15)
                    self.enemies.remove(proj['target'])
                self.projectiles.remove(proj)
            else:
                proj['pos'] += direction.normalize() * proj['spec']['proj_speed']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.reward_this_step -= 100
            return True
        if self.win:
            self.game_over = True
            self.reward_this_step += 100
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_path()
        self._draw_cursor()
        self._draw_base()
        
        render_queue = self.towers + self.enemies
        render_queue.sort(key=lambda e: e['iso_pos'][1] if 'iso_pos' in e else e['pos'][1])

        for item in render_queue:
            if 'grid_pos' in item:
                self._draw_tower(item)
            else:
                self._draw_enemy(item)

        for proj in self.projectiles: self._draw_projectile(proj)
        for p in self.particles: self._draw_particle(p)

    def _cart_to_iso(self, x, y):
        iso_x = self.ISO_OFFSET_X + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.ISO_OFFSET_Y + (x + y) * self.TILE_HEIGHT_HALF
        return iso_x, iso_y

    def _draw_path(self):
        if len(self.path_iso_coords) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_iso_coords, width=self.TILE_HEIGHT_HALF*2)
            pygame.draw.lines(self.screen, self.COLOR_GRID, False, self.path_iso_coords, width=1)

    def _draw_cursor(self):
        pos_tuple = tuple(self.cursor_pos)
        is_buildable = pos_tuple in self.buildable_tiles
        is_occupied = any(t['grid_pos'] == pos_tuple for t in self.towers)
        
        tower_type = self.tower_types[self.selected_tower_idx]
        cost = self.TOWER_SPECS[tower_type]['cost']

        color = (0, 255, 0, 100)
        if not is_buildable or is_occupied or self.resources < cost:
            color = (255, 0, 0, 100)

        cx, cy = self.cursor_pos
        points = [
            self._cart_to_iso(cx, cy),
            self._cart_to_iso(cx + 1, cy),
            self._cart_to_iso(cx + 1, cy + 1),
            self._cart_to_iso(cx, cy + 1)
        ]
        
        surface = self.screen.convert_alpha()
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, (color[0], color[1], color[2], 200))
        self.screen.blit(surface, (0, 0))

    def _draw_base(self):
        base_pos = self.path_iso_coords[-1]
        x, y = int(base_pos[0]), int(base_pos[1])
        h = self.TILE_HEIGHT_HALF
        w = self.TILE_WIDTH_HALF
        
        top_points = [(x, y - h), (x + w, y), (x, y + h), (x - w, y)]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, self.COLOR_BASE)
        pygame.gfxdraw.aapolygon(self.screen, top_points, self.COLOR_BASE_STROKE)

    def _draw_tower(self, tower):
        x, y = int(tower['iso_pos'][0]), int(tower['iso_pos'][1])
        bob = math.sin(self.steps * 0.1 + tower['bob_offset']) * 2
        y += int(bob)
        
        spec = tower['spec']
        color = spec['color']
        
        base_h = 4
        base_points = [
            (x, y + base_h),
            (x + self.TILE_WIDTH_HALF-5, y - self.TILE_HEIGHT_HALF+5 + base_h),
            (x, y - self.TILE_HEIGHT_HALF*2+10 + base_h),
            (x - self.TILE_WIDTH_HALF+5, y - self.TILE_HEIGHT_HALF+5 + base_h)
        ]
        # FIX: Generator expressions are not valid colors; convert to tuple of ints.
        darker_color = tuple(int(c * 0.6) for c in color)
        pygame.gfxdraw.filled_polygon(self.screen, base_points, darker_color)
        
        pygame.gfxdraw.filled_circle(self.screen, x, y - 8, 8, color)
        # FIX: Ensure color components are ints and do not exceed 255.
        brighter_color = tuple(min(255, int(c * 1.2)) for c in color)
        pygame.gfxdraw.aacircle(self.screen, x, y - 8, 8, brighter_color)

    def _draw_enemy(self, enemy):
        x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
        bob = math.sin(self.steps * 0.2 + enemy['bob_offset']) * 3
        y += int(bob)
        
        size = 8
        top_points = [(x, y-size), (x+size, y), (x, y+size), (x-size, y)]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, self.COLOR_ENEMY)
        pygame.gfxdraw.aapolygon(self.screen, top_points, self.COLOR_ENEMY_STROKE)
        
        bar_w = 20
        bar_h = 4
        bar_x = x - bar_w // 2
        bar_y = y - size - 12
        health_pct = enemy['health'] / enemy['max_health']
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (bar_x, bar_y, int(bar_w * health_pct), bar_h))

    def _draw_projectile(self, proj):
        color = proj['spec']['color']
        pygame.draw.line(self.screen, color, proj['pos'], proj['pos'] + (proj['target']['pos']-proj['pos']).normalize()*5, 3)
        pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'].x), int(proj['pos'].y), 3, color)

    def _draw_particle(self, p):
        radius = int(p['radius'] * (p['lifespan'] / p['max_lifespan']))
        if radius > 0:
            color = p['color']
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, (*color, alpha))
            except TypeError: # Fallback if alpha is not supported
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, color)

    def _render_ui(self):
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        health_text = self.font_small.render("BASE HP", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 15))
        health_bar_w = 150
        pygame.draw.rect(self.screen, (80, 80, 80), (90, 15, health_bar_w, 20))
        health_pct = max(0, self.base_health / 100)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (90, 15, int(health_bar_w * health_pct), 20))

        res_text = self.font_small.render(f"RES: ${int(self.resources)}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (260, 15))

        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/10", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (380, 15))
        
        tower_type = self.tower_types[self.selected_tower_idx]
        cost = self.TOWER_SPECS[tower_type]['cost']
        sel_text = self.font_small.render(f"SEL: {tower_type} (${cost})", True, self.COLOR_TEXT)
        self.screen.blit(sel_text, (490, 15))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
            "enemies_left": len(self.enemies) + len(self.enemies_to_spawn)
        }

    def _create_particles(self, pos, color, count, min_speed=1, max_speed=4):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = random.randint(10, 25)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'radius': random.randint(2, 5)
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly.
    # It creates a headless environment and a separate visible Pygame window
    # to render the observations from the environment.
    
    # The main `GameEnv` is run in "rgb_array" mode (headless).
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # A separate display is created for human viewing.
    # We must unset the dummy driver environment variable to create a real window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        # Pygame event handling for the visible window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Human Input to Action Mapping
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Gym Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Rendering the observation from the headless env to the visible window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        env.clock.tick(30)

    env.close()