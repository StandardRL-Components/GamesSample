
# Generated: 2025-08-27T18:36:47.488330
# Source Brief: brief_01884.md
# Brief Index: 1884

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to place selected tower. Shift to cycle tower types."
    )

    game_description = (
        "Defend your base against waves of enemies by strategically placing procedurally generated towers in a minimalist isometric 2D world."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 22, 14
        self.TILE_WIDTH, self.TILE_HEIGHT = 32, 16
        self.ISO_ORIGIN_X = self.WIDTH // 2
        self.ISO_ORIGIN_Y = 80
        self.MAX_STEPS = 3000 # Increased from 1000 to allow for 10 waves
        self.MAX_WAVES = 10

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (40, 44, 52)
        self.COLOR_PATH = (80, 88, 104)
        self.COLOR_BASE = (80, 200, 120)
        self.COLOR_BASE_DMG = (220, 50, 50)
        self.COLOR_ENEMY = (224, 82, 99)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_GOLD = (255, 204, 0)
        self.COLOR_CURSOR_VALID = (80, 200, 120, 150)
        self.COLOR_CURSOR_INVALID = (224, 82, 99, 150)
        self.TOWER_COLORS = [
            (97, 218, 251), # Fast/Low Damage
            (255, 121, 198), # Slow/High Damage
            (255, 184, 108)  # Area of Effect
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont('sans-serif', 18, bold=True)
        self.font_m = pygame.font.SysFont('sans-serif', 24, bold=True)
        self.font_l = pygame.font.SysFont('sans-serif', 48, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.base_health = 0
        self.gold = 0
        self.current_wave_index = 0
        self.wave_timer = 0
        self.wave_in_progress = False
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.path_grid_coords = []
        self.path_waypoints = []
        self.base_grid_pos = (0,0)
        
        self.cursor_grid_pos = (0,0)
        self.selected_tower_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.tower_specs = [
            {'name': 'Gatling', 'cost': 25, 'range': 80, 'damage': 1, 'cooldown': 10, 'proj_speed': 8, 'aoe': 0},
            {'name': 'Cannon', 'cost': 75, 'range': 120, 'damage': 10, 'cooldown': 60, 'proj_speed': 6, 'aoe': 0},
            {'name': 'Mortar', 'cost': 100, 'range': 150, 'damage': 5, 'cooldown': 90, 'proj_speed': 3, 'aoe': 40}
        ]

        # Initialize state
        self.reset()
        
        # Self-validation
        self.validate_implementation()

    def _grid_to_screen(self, x, y):
        screen_x = self.ISO_ORIGIN_X + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = self.ISO_ORIGIN_Y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _generate_path(self):
        self.path_grid_coords = []
        
        # Start at a random top/left edge
        if self.np_random.random() < 0.5:
            x, y = self.np_random.integers(0, self.GRID_WIDTH // 2), 0
        else:
            x, y = 0, self.np_random.integers(0, self.GRID_HEIGHT // 2)
            
        self.path_grid_coords.append((x, y))
        
        # Simple random walk towards the opposite corner (base)
        while x < self.GRID_WIDTH - 2 and y < self.GRID_HEIGHT - 2:
            moved = False
            # Prioritize moving towards the goal
            if self.np_random.random() < 0.75: # 75% chance to move towards goal
                if x < self.GRID_WIDTH - 2 and self.np_random.random() < 0.5:
                    x += 1
                    moved = True
                elif y < self.GRID_HEIGHT - 2:
                    y += 1
                    moved = True
            else: # 25% chance to move sideways
                 if self.np_random.random() < 0.5:
                    if x < self.GRID_WIDTH - 1: x += 1
                 else:
                    if y < self.GRID_HEIGHT - 1: y += 1
                 moved = True
            
            if moved and self.path_grid_coords[-1] != (x,y):
                 self.path_grid_coords.append((x, y))

        self.base_grid_pos = (self.GRID_WIDTH - 1, self.GRID_HEIGHT - 1)
        self.path_grid_coords.append(self.base_grid_pos)
        
        self.path_waypoints = [self._grid_to_screen(gx, gy) for gx, gy in self.path_grid_coords]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.gold = 100
        self.current_wave_index = 0
        self.wave_timer = 150 # Time before first wave
        self.wave_in_progress = False
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self._generate_path()
        
        self.cursor_grid_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_tower_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_in_progress = True
        self.current_wave_index += 1
        
        num_enemies = 3 + self.current_wave_index * 2
        enemy_health = 5 + (self.current_wave_index // 2)
        enemy_speed = 0.5 + self.current_wave_index * 0.05
        
        self.enemies_to_spawn = []
        for _ in range(num_enemies):
            self.enemies_to_spawn.append({'health': enemy_health, 'speed': enemy_speed})
            
        self.spawn_timer = 0

    def step(self, action):
        reward = 0
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cursor movement
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1  # Right
        self.cursor_grid_pos = (
            max(0, min(self.GRID_WIDTH - 1, self.cursor_grid_pos[0] + dx)),
            max(0, min(self.GRID_HEIGHT - 1, self.cursor_grid_pos[1] + dy))
        )

        # Cycle tower type on button press (not hold)
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_specs)
            # sfx: UI_cycle_sound
        
        # Place tower on button press
        if space_held and not self.prev_space_held:
            spec = self.tower_specs[self.selected_tower_type]
            is_valid_pos = self.cursor_grid_pos not in self.path_grid_coords and \
                           self.cursor_grid_pos != self.base_grid_pos and \
                           all(t['grid_pos'] != self.cursor_grid_pos for t in self.towers)
            
            if self.gold >= spec['cost'] and is_valid_pos:
                self.gold -= spec['cost']
                reward -= spec['cost'] * 0.01 # Penalty for spending
                pos = self._grid_to_screen(*self.cursor_grid_pos)
                self.towers.append({
                    'grid_pos': self.cursor_grid_pos,
                    'pos': pos,
                    'type': self.selected_tower_type,
                    'cooldown': 0,
                    'target': None
                })
                # sfx: place_tower_sound
                self._create_particle_effect(pos, 20, self.TOWER_COLORS[self.selected_tower_type])

        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # --- Update Game Logic ---
        # Wave management
        if not self.wave_in_progress:
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.current_wave_index < self.MAX_WAVES:
                self._start_next_wave()
        else:
            # Spawn enemies
            if self.enemies_to_spawn:
                self.spawn_timer -= 1
                if self.spawn_timer <= 0:
                    enemy_spec = self.enemies_to_spawn.pop(0)
                    self.enemies.append({
                        'pos': list(self.path_waypoints[0]),
                        'health': enemy_spec['health'],
                        'max_health': enemy_spec['health'],
                        'speed': enemy_spec['speed'],
                        'waypoint_index': 1,
                    })
                    self.spawn_timer = 30 # Time between spawns
            elif not self.enemies: # Wave cleared
                self.wave_in_progress = False
                reward += 10 # Wave clear bonus
                self.gold += 50 + self.current_wave_index * 10
                self.wave_timer = 300 # Time before next wave
                # sfx: wave_complete_sound

        # Update Towers
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            spec = self.tower_specs[tower['type']]
            if tower['cooldown'] == 0:
                # Find target: closest enemy to the base
                best_target = None
                max_waypoint = -1
                for enemy in self.enemies:
                    dist = math.hypot(tower['pos'][0] - enemy['pos'][0], tower['pos'][1] - enemy['pos'][1])
                    if dist <= spec['range'] and enemy['waypoint_index'] > max_waypoint:
                        max_waypoint = enemy['waypoint_index']
                        best_target = enemy
                
                if best_target:
                    tower['target'] = best_target
                    tower['cooldown'] = spec['cooldown']
                    self.projectiles.append({
                        'pos': list(tower['pos']),
                        'target': best_target,
                        'spec': spec,
                        'color': self.TOWER_COLORS[tower['type']]
                    })
                    # sfx: fire_sound (type dependent)

        # Update Projectiles
        for proj in self.projectiles[:]:
            target = proj['target']
            if target not in self.enemies: # Target already dead
                self.projectiles.remove(proj)
                continue
            
            dx = target['pos'][0] - proj['pos'][0]
            dy = target['pos'][1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj['spec']['proj_speed']:
                # Hit
                # sfx: impact_sound
                if proj['spec']['aoe'] > 0: # AOE damage
                    self._create_particle_effect(target['pos'], 40, proj['color'])
                    for enemy in self.enemies:
                        aoe_dist = math.hypot(target['pos'][0] - enemy['pos'][0], target['pos'][1] - enemy['pos'][1])
                        if aoe_dist <= proj['spec']['aoe']:
                            enemy['health'] -= proj['spec']['damage']
                else: # Single target damage
                    self._create_particle_effect(target['pos'], 10, proj['color'])
                    target['health'] -= proj['spec']['damage']
                
                self.projectiles.remove(proj)
            else:
                proj['pos'][0] += (dx / dist) * proj['spec']['proj_speed']
                proj['pos'][1] += (dy / dist) * proj['spec']['proj_speed']

        # Update Enemies
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self.enemies.remove(enemy)
                self.gold += 5
                reward += 0.1
                # sfx: enemy_death_sound
                self._create_particle_effect(enemy['pos'], 15, self.COLOR_ENEMY)
                continue

            if enemy['waypoint_index'] >= len(self.path_waypoints):
                self.base_health -= 10
                reward -= 1
                self.enemies.remove(enemy)
                # sfx: base_damage_sound
                self._create_particle_effect(self._grid_to_screen(*self.base_grid_pos), 50, self.COLOR_BASE_DMG)
                continue
            
            target_pos = self.path_waypoints[enemy['waypoint_index']]
            dx = target_pos[0] - enemy['pos'][0]
            dy = target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < enemy['speed']:
                enemy['waypoint_index'] += 1
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']

        # Update Particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Check for termination
        self.steps += 1
        terminated = False
        if self.base_health <= 0:
            terminated = True
            reward -= 100
            self.game_over = "DEFEAT"
        elif self.current_wave_index > self.MAX_WAVES and not self.enemies:
            terminated = True
            reward += 100
            self.game_over = "VICTORY"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            # No extra reward/penalty for timeout

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particle_effect(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 2
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 30),
                'color': color
            })

    def _draw_iso_rect(self, surface, color, grid_pos, height=12):
        x, y = grid_pos
        top_points = [
            self._grid_to_screen(x, y),
            self._grid_to_screen(x + 1, y),
            self._grid_to_screen(x + 1, y + 1),
            self._grid_to_screen(x, y + 1),
        ]
        
        screen_x, screen_y = self._grid_to_screen(x, y)
        
        side1_points = [
            (top_points[3]),
            (top_points[2]),
            (top_points[2][0], top_points[2][1] + height),
            (top_points[3][0], top_points[3][1] + height),
        ]
        
        side2_points = [
            (top_points[2]),
            (top_points[1]),
            (top_points[1][0], top_points[1][1] + height),
            (top_points[2][0], top_points[2][1] + height),
        ]
        
        darker_color = tuple(max(0, c - 40) for c in color)
        darkest_color = tuple(max(0, c - 60) for c in color)

        pygame.gfxdraw.filled_polygon(surface, side1_points, darker_color)
        pygame.gfxdraw.aapolygon(surface, side1_points, darker_color)
        pygame.gfxdraw.filled_polygon(surface, side2_points, darkest_color)
        pygame.gfxdraw.aapolygon(surface, side2_points, darkest_color)
        pygame.gfxdraw.filled_polygon(surface, top_points, color)
        pygame.gfxdraw.aapolygon(surface, top_points, color)

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_HEIGHT):
            pygame.draw.aaline(self.screen, self.COLOR_GRID, self._grid_to_screen(0, r), self._grid_to_screen(self.GRID_WIDTH, r))
        for c in range(self.GRID_WIDTH):
            pygame.draw.aaline(self.screen, self.COLOR_GRID, self._grid_to_screen(c, 0), self._grid_to_screen(c, self.GRID_HEIGHT))

        # Draw path
        if len(self.path_waypoints) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 10)

        # Draw base
        self._draw_iso_rect(self.screen, self.COLOR_BASE, self.base_grid_pos, height=20)
        
        # Sort all drawable entities by y-position for correct layering
        draw_queue = []
        for t in self.towers:
            draw_queue.append(('tower', t))
        for e in self.enemies:
            draw_queue.append(('enemy', e))
        
        draw_queue.sort(key=lambda item: item[1]['pos'][1])

        for item_type, item in draw_queue:
            if item_type == 'tower':
                spec = self.tower_specs[item['type']]
                color = self.TOWER_COLORS[item['type']]
                self._draw_iso_rect(self.screen, color, item['grid_pos'])
            elif item_type == 'enemy':
                pos = (int(item['pos'][0]), int(item['pos'][1]))
                pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 8)
                # Health bar
                health_pct = item['health'] / item['max_health']
                pygame.draw.rect(self.screen, (50,50,50), (pos[0]-10, pos[1]-15, 20, 4))
                pygame.draw.rect(self.screen, self.COLOR_BASE, (pos[0]-10, pos[1]-15, int(20 * health_pct), 4))

        # Draw cursor
        spec = self.tower_specs[self.selected_tower_type]
        cursor_screen_pos = self._grid_to_screen(*self.cursor_grid_pos)
        is_valid_pos = self.cursor_grid_pos not in self.path_grid_coords and \
                       self.cursor_grid_pos != self.base_grid_pos and \
                       all(t['grid_pos'] != self.cursor_grid_pos for t in self.towers) and \
                       self.gold >= spec['cost']
        cursor_color = self.COLOR_CURSOR_VALID if is_valid_pos else self.COLOR_CURSOR_INVALID
        
        pygame.gfxdraw.filled_circle(self.screen, cursor_screen_pos[0], cursor_screen_pos[1], spec['range'], (cursor_color[0], cursor_color[1], cursor_color[2], 30))
        pygame.gfxdraw.aacircle(self.screen, cursor_screen_pos[0], cursor_screen_pos[1], spec['range'], cursor_color)
        self._draw_iso_rect(self.screen, tuple(c // 2 for c in self.TOWER_COLORS[self.selected_tower_type]), self.cursor_grid_pos)


        # Draw projectiles and particles on top
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.circle(self.screen, proj['color'], pos, 4)
            pygame.draw.circle(self.screen, (255,255,255), pos, 2)

        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = p['color'] + (alpha,)
            size = int(p['life'] / 10)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            if size > 0:
                pygame.draw.circle(self.screen, color, pos, size)

    def _render_ui(self):
        # Base Health
        health_text = self.font_m.render(f"Base: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        pygame.draw.rect(self.screen, (50,50,50), (10, 40, 150, 15))
        health_pct = max(0, self.base_health / 100)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (10, 40, int(150 * health_pct), 15))

        # Gold
        gold_text = self.font_m.render(f"{self.gold} G", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (self.WIDTH - gold_text.get_width() - 10, 10))
        
        # Wave Info
        if not self.wave_in_progress and self.current_wave_index < self.MAX_WAVES:
            wave_text_str = f"Wave {self.current_wave_index + 1} in {self.wave_timer // 30}s"
        elif self.wave_in_progress:
            wave_text_str = f"Wave {self.current_wave_index} / {self.MAX_WAVES}"
        else:
            wave_text_str = f"All waves cleared!"
            
        wave_text = self.font_m.render(wave_text_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH // 2 - wave_text.get_width() // 2, 10))

        # Selected Tower Info
        spec = self.tower_specs[self.selected_tower_type]
        tower_name = self.font_m.render(f"{spec['name']}", True, self.TOWER_COLORS[self.selected_tower_type])
        tower_cost = self.font_s.render(f"Cost: {spec['cost']}G", True, self.COLOR_GOLD)
        tower_dmg = self.font_s.render(f"Dmg: {spec['damage']}", True, self.COLOR_TEXT)
        tower_rng = self.font_s.render(f"Rng: {spec['range']}", True, self.COLOR_TEXT)
        
        ui_box_y = self.HEIGHT - 70
        self.screen.blit(tower_name, (10, ui_box_y))
        self.screen.blit(tower_cost, (10, ui_box_y + 25))
        self.screen.blit(tower_dmg, (10, ui_box_y + 40))
        self.screen.blit(tower_rng, (100, ui_box_y + 40))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = self.game_over
            end_color = self.COLOR_BASE if self.game_over == "VICTORY" else self.COLOR_BASE_DMG
            end_text = self.font_l.render(end_text_str, True, end_color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

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
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.current_wave_index
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to test and play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a real screen for testing
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # To store the state of held keys
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = 0
    shift_held = 0

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0
        
        # Map arrow keys to movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        else: movement = 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']:.2f}, Survived to Wave: {info['wave']}")
    pygame.time.wait(2000)
    env.close()