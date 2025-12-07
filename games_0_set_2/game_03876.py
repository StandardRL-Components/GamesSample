
# Generated: 2025-08-28T00:43:09.074308
# Source Brief: brief_03876.md
# Brief Index: 3876

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set up headless mode for Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- Helper Classes for Game Entities ---

class Particle:
    """A single particle for visual effects."""
    def __init__(self, x, y, color, life, size, angle, speed):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.initial_life = life
        self.size = size
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.life -= 1
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95
        self.vy *= 0.95
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            color = self.color + (alpha,)
            temp_surf = pygame.Surface((int(self.size)*2, int(self.size)*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (int(self.size), int(self.size)), int(self.size))
            surface.blit(temp_surf, (self.x - self.size, self.y - self.size), special_flags=pygame.BLEND_RGBA_ADD)

class Projectile:
    """A projectile fired from a tower."""
    def __init__(self, start_pos, target_enemy, damage, speed, color):
        self.x, self.y = start_pos
        self.target = target_enemy
        self.damage = damage
        self.speed = speed
        self.color = color
        self.is_active = True

    def update(self):
        if not self.target.is_active:
            self.is_active = False
            return
        
        target_x, target_y = self.target.screen_pos
        dx, dy = target_x - self.x, target_y - self.y
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.is_active = False
            return 'hit'
        else:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        return None

    def draw(self, surface):
        if self.is_active:
            pygame.draw.line(surface, self.color, (self.x, self.y), (self.x, self.y), 3)
            pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), 3, self.color)
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), 3, self.color)

class Enemy:
    """An enemy that moves along the path."""
    def __init__(self, path, path_screen_coords, speed, health, value, size, color, np_random):
        self.path = path
        self.path_screen_coords = path_screen_coords
        self.path_index = 0
        self.progress = 0.0
        self.speed = speed
        self.max_health = health
        self.health = health
        self.value = value
        self.size = size
        self.color = color
        self.screen_pos = self.path_screen_coords[0]
        self.is_active = True
        self.np_random = np_random

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.is_active = False
            return 'killed'
        return 'hit'

    def update(self):
        if not self.is_active:
            return None

        start_node = self.path_screen_coords[self.path_index]
        if self.path_index + 1 >= len(self.path_screen_coords):
            self.is_active = False
            return 'reached_base'

        end_node = self.path_screen_coords[self.path_index + 1]
        dist_to_next = math.hypot(end_node[0] - start_node[0], end_node[1] - start_node[1])
        
        if dist_to_next > 0:
            self.progress += self.speed / dist_to_next
        else:
            self.progress = 1.0

        if self.progress >= 1.0:
            self.path_index += 1
            self.progress = 0.0
            if self.path_index + 1 >= len(self.path_screen_coords):
                self.is_active = False
                return 'reached_base'

        start_node = self.path_screen_coords[self.path_index]
        end_node = self.path_screen_coords[self.path_index + 1]
        self.screen_pos = (
            start_node[0] + (end_node[0] - start_node[0]) * self.progress,
            start_node[1] + (end_node[1] - start_node[1]) * self.progress,
        )
        return None

    def draw(self, surface):
        if self.is_active:
            x, y = int(self.screen_pos[0]), int(self.screen_pos[1])
            
            # Health bar
            bar_width = self.size * 1.5
            bar_height = 4
            health_ratio = self.health / self.max_health
            pygame.draw.rect(surface, (50, 50, 50), (x - bar_width / 2, y - self.size - bar_height - 2, bar_width, bar_height))
            pygame.draw.rect(surface, (255, 0, 0), (x - bar_width / 2, y - self.size - bar_height - 2, bar_width * health_ratio, bar_height))

            # Body
            pygame.gfxdraw.filled_circle(surface, x, y, self.size, self.color)
            pygame.gfxdraw.aacircle(surface, x, y, self.size, (0,0,0))

class Tower:
    """A tower that shoots at enemies."""
    def __init__(self, pos, screen_pos, tower_type, np_random):
        self.pos = pos
        self.screen_pos = screen_pos
        self.type = tower_type
        self.range = tower_type['range']
        self.damage = tower_type['damage']
        self.fire_rate = tower_type['fire_rate']
        self.color = tower_type['color']
        self.cooldown = 0
        self.target = None
        self.np_random = np_random

    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        if self.target and self.target.is_active and self._in_range(self.target):
            # Target is still valid
            pass
        else:
            # Find a new target
            self.target = self._find_target(enemies)

        if self.target:
            self.cooldown = self.fire_rate
            # sfx: tower_shoot.wav
            return Projectile(self.screen_pos, self.target, self.damage, 10, self.type['proj_color'])
        return None

    def _in_range(self, enemy):
        dist = math.hypot(enemy.screen_pos[0] - self.screen_pos[0], enemy.screen_pos[1] - self.screen_pos[1])
        return dist <= self.range

    def _find_target(self, enemies):
        valid_targets = [e for e in enemies if self._in_range(e)]
        if not valid_targets:
            return None
        # Target enemy furthest along the path
        return max(valid_targets, key=lambda e: (e.path_index, e.progress))
        
    def draw(self, surface):
        x, y = self.screen_pos
        base_size = 12
        top_size = 8
        pygame.draw.rect(surface, (50, 50, 50), (x - base_size, y - base_size//2, base_size*2, base_size))
        pygame.draw.rect(surface, self.color, (x - top_size, y - top_size//2, top_size*2, top_size))

# --- Main Game Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to place basic towers. Space for high-damage, short-range. Shift for long-range, low-damage."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this isometric tower defense game."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.MAX_STEPS = 2500
        self.ISO_ORIGIN_X = self.WIDTH // 2
        self.ISO_ORIGIN_Y = 80
        self.TILE_WIDTH_HALF = 24
        self.TILE_HEIGHT_HALF = 12
        self.PATH_THICKNESS = 8

        self._define_colors_and_fonts()
        self._define_path()
        self._define_tower_spots()
        self._define_tower_types()
        self._define_wave_data()
        
        self.reset()
        self.validate_implementation()

    def _define_colors_and_fonts(self):
        self.COLOR_BG = (30, 40, 50)
        self.COLOR_PATH = (80, 90, 100)
        self.COLOR_PATH_SIDE = (60, 70, 80)
        self.COLOR_BASE = (0, 150, 200)
        self.COLOR_BASE_SIDE = (0, 120, 160)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_UI_SHADOW = (20, 20, 20)
        self.FONT_UI = pygame.font.SysFont("Consolas", 20, bold=True)
        self.FONT_GAMEOVER = pygame.font.SysFont("Consolas", 50, bold=True)

    def _define_path(self):
        self.path_nodes = [(0, 5), (5, 5), (5, 0), (10, 0), (10, 10), (2, 10), (2, 13), (13, 13)]
        self.path_screen_coords = [self._iso_to_screen(x, y) for x, y in self.path_nodes]

    def _define_tower_spots(self):
        # Grid positions for tower placement
        self.tower_spots = [
            {'grid_pos': (4, 4), 'occupied': False, 'type': 1, 'action_map': (0, 1)}, # Up
            {'grid_pos': (6, 6), 'occupied': False, 'type': 1, 'action_map': (0, 2)}, # Down
            {'grid_pos': (4, 1), 'occupied': False, 'type': 1, 'action_map': (0, 3)}, # Left
            {'grid_pos': (9, 1), 'occupied': False, 'type': 1, 'action_map': (0, 4)}, # Right
            {'grid_pos': (6, 9), 'occupied': False, 'type': 2, 'action_map': (1, 1)}, # Space
            {'grid_pos': (1, 8), 'occupied': False, 'type': 3, 'action_map': (2, 1)}, # Shift
        ]
        for spot in self.tower_spots:
            spot['screen_pos'] = self._iso_to_screen(*spot['grid_pos'])

    def _define_tower_types(self):
        self.tower_types = {
            1: {'name': 'Basic', 'cost': 50, 'damage': 10, 'range': 80, 'fire_rate': 30, 'color': (0, 200, 0), 'proj_color': (100, 255, 100)},
            2: {'name': 'Heavy', 'cost': 80, 'damage': 35, 'range': 60, 'fire_rate': 60, 'color': (255, 100, 0), 'proj_color': (255, 150, 50)},
            3: {'name': 'Sniper', 'cost': 70, 'damage': 15, 'range': 150, 'fire_rate': 45, 'color': (150, 50, 255), 'proj_color': (200, 150, 255)},
        }

    def _define_wave_data(self):
        # type, count, interval, health, speed, value
        self.waves = [
            [('grunt', 10, 45, 50, 1.0, 10)],
            [('grunt', 15, 30, 60, 1.1, 12)],
            [('tank', 5, 90, 200, 0.7, 30), ('grunt', 10, 30, 70, 1.2, 12)],
            [('runner', 20, 15, 40, 2.0, 8)],
            [('tank', 10, 60, 250, 0.8, 35), ('runner', 10, 30, 50, 2.2, 10)],
            [('grunt', 20, 15, 100, 1.3, 15), ('tank', 5, 120, 300, 0.9, 40)],
        ]
        self.enemy_types = {
            'grunt': {'size': 8, 'color': (220, 50, 50)},
            'tank': {'size': 12, 'color': (180, 40, 40)},
            'runner': {'size': 6, 'color': (255, 80, 80)},
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.ISO_ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ISO_ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return screen_x, screen_y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.terminal_reward_given = False

        self.base_health = 100
        self.resources = 150
        self.base_damage_flash = 0

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        for spot in self.tower_spots:
            spot['occupied'] = False

        self.current_wave_index = -1
        self.wave_spawns = []
        self.spawn_timer = 120 # Initial delay
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.001 # Small time penalty

        # 1. Handle Input (Tower Placement)
        self._handle_input(action)

        # 2. Update Towers
        new_projectiles = self._update_towers()
        self.projectiles.extend(new_projectiles)

        # 3. Update Projectiles
        reward += self._update_projectiles()

        # 4. Update Enemies
        enemy_updates = self._update_enemies()
        reward += enemy_updates['reward']
        if enemy_updates['base_hit']:
            self.base_health = max(0, self.base_health - 10)
            self.base_damage_flash = 15 # Flash for 15 frames
            # sfx: base_damage.wav
            
        # 5. Spawn new enemies
        self._spawn_enemies()

        # 6. Update Particles & Effects
        self._update_particles()
        if self.base_damage_flash > 0:
            self.base_damage_flash -= 1

        # 7. Check Termination
        terminated = self._check_termination()
        if terminated and not self.terminal_reward_given:
            if self.win:
                reward += 50
                # sfx: game_win.wav
            else:
                reward -= 50
                # sfx: game_lose.wav
            self.terminal_reward_given = True

        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space, shift = action
        
        # Priority: Space > Shift > Movement
        actions_to_try = []
        if space == 1: actions_to_try.append((1, 1))
        if shift == 1: actions_to_try.append((2, 1))
        if movement > 0: actions_to_try.append((0, movement))
        
        for action_index, action_value in actions_to_try:
            for spot in self.tower_spots:
                if spot['action_map'] == (action_index, action_value):
                    if not spot['occupied']:
                        tower_type = self.tower_types[spot['type']]
                        if self.resources >= tower_type['cost']:
                            self.resources -= tower_type['cost']
                            spot['occupied'] = True
                            new_tower = Tower(spot['grid_pos'], spot['screen_pos'], tower_type, self.np_random)
                            self.towers.append(new_tower)
                            # sfx: place_tower.wav
                            # Only place one tower per step
                            return
    
    def _update_towers(self):
        new_projectiles = []
        for tower in self.towers:
            proj = tower.update(self.enemies)
            if proj:
                new_projectiles.append(proj)
        return new_projectiles

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles:
            result = p.update()
            if result == 'hit':
                status = p.target.take_damage(p.damage)
                # sfx: enemy_hit.wav
                reward += 0.1
                if status == 'killed':
                    # sfx: enemy_die.wav
                    reward += 1.0
                    self.resources += p.target.value
                    self._create_explosion(p.target.screen_pos, p.target.color)
        self.projectiles = [p for p in self.projectiles if p.is_active]
        return reward

    def _update_enemies(self):
        reward = 0
        base_hit = False
        for e in self.enemies:
            result = e.update()
            if result == 'reached_base':
                base_hit = True
        
        self.enemies = [e for e in self.enemies if e.is_active]
        return {'reward': reward, 'base_hit': base_hit}

    def _spawn_enemies(self):
        if self.spawn_timer > 0:
            self.spawn_timer -= 1
            return

        if not self.wave_spawns:
            if self.current_wave_index + 1 < len(self.waves):
                self.current_wave_index += 1
                self.wave_spawns = self.waves[self.current_wave_index][:]
                self.spawn_timer = 180 # Delay between waves
            return

        spawn_info = self.wave_spawns[0]
        e_type, e_count, e_interval, e_health, e_speed, e_value = spawn_info
        
        self.spawn_timer = e_interval
        
        base_stats = self.enemy_types[e_type]
        new_enemy = Enemy(self.path_nodes, self.path_screen_coords, e_speed, e_health, e_value, **base_stats, np_random=self.np_random)
        self.enemies.append(new_enemy)

        self.wave_spawns[0] = (e_type, e_count - 1, *spawn_info[2:])
        if self.wave_spawns[0][1] <= 0:
            self.wave_spawns.pop(0)

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.life > 0]

    def _create_explosion(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            size = self.np_random.uniform(2, 5)
            self.particles.append(Particle(pos[0], pos[1], color, life, size, angle, speed))

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.win = False
            return True
        
        is_last_wave = self.current_wave_index >= len(self.waves) - 1
        if is_last_wave and not self.wave_spawns and not self.enemies:
            self.game_over = True
            self.win = True
            return True

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw static world elements ---
        self._draw_path()
        self._draw_base()

        # --- Draw dynamic elements, sorted by y-pos for correct isometric layering ---
        renderables = self.towers + self.enemies
        renderables.sort(key=lambda r: r.screen_pos[1])
        
        for item in renderables:
            item.draw(self.screen)
            # Draw tower range on hover (not implemented for agent, but good for debug)
            # if isinstance(item, Tower):
            #     pygame.gfxdraw.aacircle(self.screen, int(item.screen_pos[0]), int(item.screen_pos[1]), item.range, (255,255,255,50))

        for p in self.projectiles:
            p.draw(self.screen)
        
        for p in self.particles:
            p.draw(self.screen)

    def _draw_path(self):
        for i in range(len(self.path_nodes) - 1):
            p1_grid = self.path_nodes[i]
            p2_grid = self.path_nodes[i+1]
            p1 = self.path_screen_coords[i]
            p2 = self.path_screen_coords[i+1]

            # Determine direction for 3D effect
            dx = p2_grid[0] - p1_grid[0]
            dy = p2_grid[1] - p1_grid[1]

            if dx > 0: # Moving right
                p3 = (p2[0], p2[1] + self.PATH_THICKNESS)
                p4 = (p1[0], p1[1] + self.PATH_THICKNESS)
                pygame.gfxdraw.aapolygon(self.screen, [p1, (p2[0], p1[1]), p2, p4], self.COLOR_PATH)
                pygame.gfxdraw.filled_polygon(self.screen, [p1, (p2[0], p1[1]), p2, p4], self.COLOR_PATH)
                pygame.gfxdraw.filled_polygon(self.screen, [p1, p4, (p4[0]+dx*self.TILE_WIDTH_HALF*2, p4[1]), (p1[0]+dx*self.TILE_WIDTH_HALF*2, p1[1])], self.COLOR_PATH_SIDE)

            elif dy > 0: # Moving down
                p3 = (p2[0], p2[1] + self.PATH_THICKNESS)
                p4 = (p1[0], p1[1] + self.PATH_THICKNESS)
                pygame.gfxdraw.aapolygon(self.screen, [p1, (p1[0], p2[1]), p2, p4], self.COLOR_PATH)
                pygame.gfxdraw.filled_polygon(self.screen, [p1, (p1[0], p2[1]), p2, p4], self.COLOR_PATH)
        
        # Draw base platform
        base_node = self.path_nodes[-1]
        center = self._iso_to_screen(base_node[0], base_node[1])
        points = [
            self._iso_to_screen(base_node[0] - 1.5, base_node[1] - 1.5),
            self._iso_to_screen(base_node[0] + 1.5, base_node[1] - 1.5),
            self._iso_to_screen(base_node[0] + 1.5, base_node[1] + 1.5),
            self._iso_to_screen(base_node[0] - 1.5, base_node[1] + 1.5),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PATH)

    def _draw_base(self):
        base_pos = self.path_screen_coords[-1]
        color = self.COLOR_BASE
        if self.base_damage_flash > 0 and self.base_damage_flash % 4 < 2:
            color = (255, 50, 50) # Flash red
        
        # Simple block representation
        x, y = base_pos
        h = 20
        w = 30
        pygame.draw.rect(self.screen, self.COLOR_BASE_SIDE, (x - w/2, y-h/2, w, h/2))
        pygame.draw.rect(self.screen, color, (x - w/2, y-h, w, h))


    def _render_ui(self):
        # --- Helper for shadowed text ---
        def draw_text(text, font, color, shadow_color, x, y, center=False):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            text_rect = text_surf.get_rect()
            if center:
                text_rect.center = (x, y)
            else:
                text_rect.topleft = (x, y)
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)

        # --- Base Health Bar ---
        bar_width = 200
        bar_height = 20
        health_x = self.WIDTH // 2 - bar_width // 2
        health_y = 10
        health_ratio = self.base_health / 100.0
        pygame.draw.rect(self.screen, (50, 0, 0), (health_x, health_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (200, 0, 0), (health_x, health_y, bar_width * health_ratio, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (health_x, health_y, bar_width, bar_height), 2)
        draw_text(f"BASE HP: {self.base_health}", self.FONT_UI, self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW, self.WIDTH // 2, health_y + bar_height//2, center=True)

        # --- Resources and Score ---
        draw_text(f"RESOURCES: ${self.resources}", self.FONT_UI, (255, 215, 0), self.COLOR_UI_SHADOW, 10, 10)
        draw_text(f"SCORE: {int(self.score)}", self.FONT_UI, self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW, 10, 35)

        # --- Wave Info ---
        wave_text = f"WAVE: {self.current_wave_index + 1}/{len(self.waves)}"
        if self.current_wave_index < 0: wave_text = "WAVE: 1/6 (Starting...)"
        if self.win: wave_text = "ALL WAVES CLEARED"
        draw_text(wave_text, self.FONT_UI, self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW, self.WIDTH - 10 - self.FONT_UI.size(wave_text)[0], 10)

        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            draw_text(msg, self.FONT_GAMEOVER, color, self.COLOR_UI_SHADOW, self.WIDTH/2, self.HEIGHT/2, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave_index + 1
        }
    
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

    def close(self):
        pygame.quit()