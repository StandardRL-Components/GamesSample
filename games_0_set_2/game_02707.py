
# Generated: 2025-08-28T05:44:31.064542
# Source Brief: brief_02707.md
# Brief Index: 2707

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→/↑↓ to select a tower spot. Shift to cycle tower types. Space to build."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers along the path."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 24)
        self.font_huge = pygame.font.SysFont("sans-serif", 48)

        # --- Colors ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PATH = (50, 50, 65)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_BASE_DMG = (200, 50, 50)
        self.COLOR_ENEMY = (210, 40, 40)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.TOWER_COLORS = {
            "Gatling": (60, 180, 255),
            "Cannon": (255, 165, 0),
            "Slower": (180, 50, 255),
        }

        # --- Game Constants ---
        self.MAX_WAVES = 10
        self.MAX_STEPS = 4500 # ~2.5 mins at 30fps
        self.BASE_START_HEALTH = 100
        self.STARTING_RESOURCES = 100
        self.INTERWAVE_TIME = 300 # 10 seconds at 30fps
        
        # --- Game Path & Tower Spots ---
        self.PATH_WAYPOINTS = [
            (-50, 150), (100, 150), (100, 300), (300, 300),
            (300, 100), (500, 100), (500, 250), (self.WIDTH + 50, 250)
        ]
        self.TOWER_SPOTS = [
            (100, 225), (175, 300), (225, 300), (300, 225),
            (300, 175), (375, 100), (425, 100), (500, 175)
        ]
        
        # --- Tower Definitions ---
        self.TOWER_SPECS = {
            "Gatling": {"cost": 50, "range": 75, "damage": 2, "fire_rate": 5},
            "Cannon": {"cost": 120, "range": 100, "damage": 25, "fire_rate": 45},
            "Slower": {"cost": 80, "range": 60, "damage": 0.5, "fire_rate": 20, "slow": 0.5}
        }
        self.TOWER_TYPES = list(self.TOWER_SPECS.keys())

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.game_phase = ""
        self.phase_timer = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.selected_spot_idx = 0
        self.selected_tower_type_idx = 0
        self.last_action = np.array([0, 0, 0])
        
        self.base_damage_flash = 0

        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = self.BASE_START_HEALTH
        self.resources = self.STARTING_RESOURCES
        self.current_wave = 0
        self.game_phase = "interwave"
        self.phase_timer = self.INTERWAVE_TIME
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.selected_spot_idx = 0
        self.selected_tower_type_idx = 0
        self.last_action = np.array([0, 0, 0])
        self.base_damage_flash = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        self._handle_actions(action)
        
        self._update_game_phase()
        
        if self.game_phase == "wave_active":
            reward -= 0.01

        step_reward = self._update_towers()
        step_reward += self._update_projectiles()
        step_reward += self._update_enemies()
        reward += step_reward
        self._update_particles()
        
        if self.base_damage_flash > 0:
            self.base_damage_flash -= 1

        self.steps += 1
        
        # Check for wave clear reward
        if self.game_phase == "interwave" and self.phase_timer == self.INTERWAVE_TIME -1 and self.current_wave > 0:
            reward += 50 # Survived a wave

        # Check for termination conditions
        if self.base_health <= 0 and not self.game_over:
            reward -= 100
            self.game_over = True
            terminated = True
        elif self.game_won and not self.game_over:
            reward += 100
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.last_action = action

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_actions(self, action):
        movement, space_press, shift_press = action[0], action[1], action[2]
        _, last_space, last_shift = self.last_action[0], self.last_action[1], self.last_action[2]

        is_movement_pressed = movement != 0 and self.last_action[0] == 0
        if is_movement_pressed:
            if movement in [1, 3]: # Up or Left
                self.selected_spot_idx = (self.selected_spot_idx - 1) % len(self.TOWER_SPOTS)
            elif movement in [2, 4]: # Down or Right
                self.selected_spot_idx = (self.selected_spot_idx + 1) % len(self.TOWER_SPOTS)
        
        if shift_press and not last_shift:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.TOWER_TYPES)

        if space_press and not last_space:
            spot_pos = self.TOWER_SPOTS[self.selected_spot_idx]
            tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
            spec = self.TOWER_SPECS[tower_type]

            is_occupied = any(t['pos'] == spot_pos for t in self.towers)
            
            if not is_occupied and self.resources >= spec['cost']:
                self.resources -= spec['cost']
                self.towers.append({
                    "pos": spot_pos, "type": tower_type, "spec": spec,
                    "cooldown": 0, "target": None
                })
                # sfx: build_tower.wav
                self._create_particles(spot_pos, self.TOWER_COLORS[tower_type], 20, 3, 15)

    def _update_game_phase(self):
        if self.game_phase == "interwave":
            self.phase_timer -= 1
            if self.phase_timer <= 0:
                self.current_wave += 1
                if self.current_wave > self.MAX_WAVES:
                    if not self.enemies: self.game_won = True
                else:
                    self.game_phase = "wave_active"
                    self._spawn_wave()
        elif self.game_phase == "wave_active" and not self.enemies:
            self.game_phase = "interwave"
            self.phase_timer = self.INTERWAVE_TIME
            # sfx: wave_complete.wav

    def _spawn_wave(self):
        num_enemies = 2 + self.current_wave * 2
        base_health = 10 + self.current_wave * 5
        base_speed = 0.8 + self.current_wave * 0.05
        
        for i in range(num_enemies):
            speed = base_speed * self.np_random.uniform(0.9, 1.1)
            self.enemies.append({
                "pos": pygame.Vector2(self.PATH_WAYPOINTS[0][0] - i * 25, self.PATH_WAYPOINTS[0][1]),
                "health": base_health, "max_health": base_health,
                "speed": speed, "base_speed": speed,
                "waypoint_idx": 1, "value": 5 + self.current_wave,
                "slow_timer": 0
            })
            
    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            if tower['target'] is None or tower['target'] not in self.enemies:
                tower['target'] = None
                in_range_enemies = [e for e in self.enemies if pygame.Vector2(tower['pos']).distance_to(e['pos']) <= tower['spec']['range']]
                if in_range_enemies:
                    tower['target'] = max(in_range_enemies, key=lambda e: e['waypoint_idx'])
            
            if tower['target'] and pygame.Vector2(tower['pos']).distance_to(tower['target']['pos']) <= tower['spec']['range']:
                tower['cooldown'] = tower['spec']['fire_rate']
                # sfx: shoot.wav
                self.projectiles.append({
                    "pos": pygame.Vector2(tower['pos']), "target": tower['target'],
                    "speed": 8, "damage": tower['spec']['damage'], "type": tower['type'],
                    "slow": tower['spec'].get('slow', 0)
                })
        return 0

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            if p['target'] not in self.enemies:
                self.projectiles.remove(p)
                continue

            target_pos = p['target']['pos']
            direction = (target_pos - p['pos']).normalize() if (target_pos - p['pos']).length() > 0 else pygame.Vector2(0,0)
            p['pos'] += direction * p['speed']

            if p['pos'].distance_to(target_pos) < 5:
                p['target']['health'] -= p['damage']
                if p['slow'] > 0:
                    p['target']['speed'] = p['target']['base_speed'] * (1 - p['slow'])
                    p['target']['slow_timer'] = 60 # 2 seconds
                
                reward += 0.1
                self._create_particles(p['pos'], self.TOWER_COLORS[p['type']], 5, 2, 10)
                
                if p['target']['health'] <= 0:
                    # sfx: enemy_destroyed.wav
                    reward += 1.0
                    self.resources += p['target']['value']
                    self.score += p['target']['value']
                    self._create_particles(p['target']['pos'], self.COLOR_ENEMY, 30, 4, 20)
                    self.enemies.remove(p['target'])

                self.projectiles.remove(p)
        return reward

    def _update_enemies(self):
        for e in self.enemies[:]:
            if e['slow_timer'] > 0:
                e['slow_timer'] -= 1
                if e['slow_timer'] == 0:
                    e['speed'] = e['base_speed']

            if e['waypoint_idx'] >= len(self.PATH_WAYPOINTS):
                self.base_health -= 10
                self.base_damage_flash = 15
                self.enemies.remove(e)
                # sfx: base_damage.wav
                continue
            
            target_pos = pygame.Vector2(self.PATH_WAYPOINTS[e['waypoint_idx']])
            direction = (target_pos - e['pos']).normalize() if (target_pos - e['pos']).length() > 0 else pygame.Vector2(0,0)
            e['pos'] += direction * e['speed']

            if e['pos'].distance_to(target_pos) < e['speed']:
                e['waypoint_idx'] += 1
        return 0

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
    
    def _create_particles(self, pos, color, count, speed_max, life_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(5, life_max)
            self.particles.append({
                "pos": pygame.Vector2(pos), "vel": vel, "life": life, "max_life": life,
                "color": color, "size": self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.PATH_WAYPOINTS, 30)

        base_pos = self.PATH_WAYPOINTS[-1]
        base_color = self.COLOR_BASE_DMG if self.base_damage_flash > 0 else self.COLOR_BASE
        pygame.draw.rect(self.screen, base_color, (base_pos[0]-20, base_pos[1]-20, 40, 40))
        
        for i, pos in enumerate(self.TOWER_SPOTS):
            is_occupied = any(t['pos'] == pos for t in self.towers)
            color = (100, 100, 100) if is_occupied else (70, 70, 85)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 15, color)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 15, (120, 120, 135))
        
        sel_pos = self.TOWER_SPOTS[self.selected_spot_idx]
        pygame.gfxdraw.aacircle(self.screen, int(sel_pos[0]), int(sel_pos[1]), 18, self.COLOR_SELECTOR)
        pygame.gfxdraw.aacircle(self.screen, int(sel_pos[0]), int(sel_pos[1]), 19, self.COLOR_SELECTOR)

        for t in self.towers:
            pos, type = t['pos'], t['type']
            color = self.TOWER_COLORS[type]
            pygame.draw.rect(self.screen, color, (pos[0]-10, pos[1]-10, 20, 20))
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in color), (pos[0]-10, pos[1]-10, 20, 20), 2)
            if pos == self.TOWER_SPOTS[self.selected_spot_idx]:
                 pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), t['spec']['range'], (*color, 100))

        for p in self.projectiles:
            color = self.TOWER_COLORS[p['type']]
            pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), 3)

        for e in self.enemies:
            pos = (int(e['pos'].x), int(e['pos'].y))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 8)
            health_ratio = max(0, e['health'] / e['max_health'])
            pygame.draw.rect(self.screen, (50, 50, 50), (pos[0]-10, pos[1]-15, 20, 3))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (pos[0]-10, pos[1]-15, int(20 * health_ratio), 3))
            
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            color = p['color']
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, alpha), (size, size), size)
                self.screen.blit(s, (p['pos'].x - size, p['pos'].y - size))

    def _render_ui(self):
        ui_texts = [
            f"Base Health: {max(0, self.base_health)} / {self.BASE_START_HEALTH}",
            f"Resources: {self.resources}", f"Score: {self.score}",
            f"Wave: {min(self.current_wave, self.MAX_WAVES)} / {self.MAX_WAVES}",
        ]
        for i, text in enumerate(ui_texts):
            self._draw_text(text, (10, 10 + i * 22), self.font_small)

        tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type]
        cost_color = self.COLOR_TEXT if self.resources >= spec['cost'] else self.COLOR_ENEMY
        
        self._draw_text("Selected Tower:", (self.WIDTH - 160, 10), self.font_small)
        self._draw_text(f"> {tower_type}", (self.WIDTH - 150, 32), self.font_large, self.TOWER_COLORS[tower_type])
        self._draw_text(f"Cost: {spec['cost']}", (self.WIDTH - 150, 60), self.font_small, cost_color)
        self._draw_text(f"Dmg: {spec['damage']}", (self.WIDTH - 150, 78), self.font_small)
        self._draw_text(f"Range: {spec['range']}", (self.WIDTH - 150, 96), self.font_small)

        if self.game_phase == "interwave" and not self.game_won and self.current_wave < self.MAX_WAVES:
            timer_text = f"Next wave in: {math.ceil(self.phase_timer / 30)}"
            self._draw_text(timer_text, (self.WIDTH // 2, 20), self.font_large, self.COLOR_TEXT, center=True)
            
        if self.game_over:
            msg, color = ("VICTORY!", (50, 255, 50)) if self.game_won else ("GAME OVER", self.COLOR_ENEMY)
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 - 30), self.font_huge, color, center=True)

    def _draw_text(self, text, pos, font, color=None, center=False):
        color = color or self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center: text_rect.center = pos
        else: text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score, "steps": self.steps, "wave": self.current_wave,
            "resources": self.resources, "base_health": self.base_health
        }

    def close(self):
        pygame.quit()