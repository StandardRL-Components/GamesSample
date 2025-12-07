import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:12:36.464821
# Source Brief: brief_00325.md
# Brief Index: 325
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    Magnetize asteroids to build bases, clone specialized magnetic units,
    and conquer the galaxy in this real-time strategy game.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Build bases, collect resources from asteroids, create an army of units, and conquer "
        "all the star systems to win this real-time strategy game."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selector. When a base is selected, "
        "press space to create a unit and shift to upgrade the base."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CRITICAL: Spaces ---
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
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_world = pygame.font.SysFont("Consolas", 12)

        # --- Visuals & Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_LIGHT = (120, 200, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_LIGHT = (255, 150, 150)
        self.COLOR_ASTEROID = (120, 130, 140)
        self.COLOR_RESOURCE = (255, 220, 50)
        self.COLOR_NEUTRAL = (100, 100, 100)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_SELECTOR = (50, 255, 50)

        # --- Game Constants ---
        self.MAX_STEPS = 5000
        self.NUM_STAR_SYSTEMS = 3
        self.NUM_INITIAL_ASTEROIDS = 15
        self.MAX_ASTEROIDS = 25
        self.BASE_INITIAL_HP = 500
        self.BASE_UPGRADE_COST = 150
        self.BASE_UPGRADE_HP_BONUS = 250
        self.BASE_COLLECTION_RADIUS = 70
        self.UNIT_COST = 50
        self.UNIT_HP = 50
        self.UNIT_DAMAGE = 5
        self.UNIT_SPEED = 1.5
        self.UNIT_ATTACK_RADIUS = 100
        self.UNIT_SPAWN_COOLDOWN = 30 # steps
        self.ENEMY_BASE_SPAWN_PERIOD_INITIAL = 600 # steps
        self.ENEMY_UNIT_SPAWN_COOLDOWN = 90 # steps

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        self.resources = 0
        self.player_bases = []
        self.enemy_bases = []
        self.player_units = []
        self.enemy_units = []
        self.asteroids = []
        self.star_systems = []
        self.particles = []
        self.projectiles = []
        self.static_stars = []
        self.selector_pos = None
        self.selected_base_idx = -1
        self.enemy_base_spawn_timer = 0
        self.enemy_base_spawn_period = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        
        self.resources = 100
        self.player_bases = [{
            'pos': np.array([self.WIDTH * 0.2, self.HEIGHT * 0.5]),
            'hp': self.BASE_INITIAL_HP,
            'max_hp': self.BASE_INITIAL_HP,
            'level': 1,
            'unit_spawn_cooldown': 0,
        }]
        
        self.enemy_bases = []
        self.player_units = []
        self.enemy_units = []
        self.particles = []
        self.projectiles = []
        
        self.asteroids = [
            {'pos': self._get_random_pos(), 'size': self.np_random.integers(5, 11)}
            for _ in range(self.NUM_INITIAL_ASTEROIDS)
        ]
        
        self.star_systems = [{
            'pos': np.array([self.WIDTH * (0.25 * (i + 1.5)), self.HEIGHT * 0.2]),
            'control': 0.0,  # -1.0 (enemy) to +1.0 (player)
            'owner': 0 # 0=neutral, 1=player, -1=enemy
        } for i in range(self.NUM_STAR_SYSTEMS)]
        
        self.selector_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        self.selected_base_idx = -1
        
        self.enemy_base_spawn_period = self.ENEMY_BASE_SPAWN_PERIOD_INITIAL
        self.enemy_base_spawn_timer = self.enemy_base_spawn_period

        self.static_stars = [
            (self.np_random.integers(0, self.WIDTH + 1), self.np_random.integers(0, self.HEIGHT + 1), self.np_random.uniform(0.5, 1.5))
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = -0.01 # Small penalty for time passing

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        self._update_game_state()

        # --- Calculate Reward & Check Termination ---
        reward = self.reward_this_step
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            self.game_over = True
            if not self.player_bases: # Loss
                self.score -= 100
                reward -= 100
            elif all(s['owner'] == 1 for s in self.star_systems): # Win
                self.score += 100
                reward += 100

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # =================================================================================
    # --- Game Logic Update Methods ---
    # =================================================================================

    def _handle_input(self, movement, space_held, shift_held):
        # Update selector position
        move_speed = 5
        if movement == 1: self.selector_pos[1] -= move_speed
        elif movement == 2: self.selector_pos[1] += move_speed
        elif movement == 3: self.selector_pos[0] -= move_speed
        elif movement == 4: self.selector_pos[0] += move_speed
        self.selector_pos[0] = np.clip(self.selector_pos[0], 0, self.WIDTH)
        self.selector_pos[1] = np.clip(self.selector_pos[1], 0, self.HEIGHT)
        
        # Find closest player base to selector
        self.selected_base_idx = -1
        min_dist = float('inf')
        for i, base in enumerate(self.player_bases):
            dist = np.linalg.norm(self.selector_pos - base['pos'])
            if dist < 30 * base['level'] and dist < min_dist:
                min_dist = dist
                self.selected_base_idx = i

        # Handle actions on selected base
        if self.selected_base_idx != -1:
            base = self.player_bases[self.selected_base_idx]
            
            # Action: Clone Attack Unit (Space)
            if space_held and self.resources >= self.UNIT_COST and base['unit_spawn_cooldown'] <= 0:
                self.resources -= self.UNIT_COST
                spawn_angle = self.np_random.uniform(0, 2 * math.pi)
                spawn_offset = np.array([math.cos(spawn_angle), math.sin(spawn_angle)]) * 25 * base['level']
                self.player_units.append({
                    'pos': base['pos'] + spawn_offset,
                    'hp': self.UNIT_HP, 'max_hp': self.UNIT_HP,
                    'target_entity': None,
                    'attack_cooldown': 0,
                })
                base['unit_spawn_cooldown'] = self.UNIT_SPAWN_COOLDOWN
                self.reward_this_step += 0.2
                self._spawn_particle_explosion(base['pos'], 10, self.COLOR_PLAYER_LIGHT, 0.5)

            # Action: Upgrade Base (Shift)
            if shift_held and self.resources >= self.BASE_UPGRADE_COST * base['level']:
                self.resources -= self.BASE_UPGRADE_COST * base['level']
                base['level'] += 1
                base['max_hp'] += self.BASE_UPGRADE_HP_BONUS
                base['hp'] = base['max_hp'] # Full heal on upgrade
                self.reward_this_step += 0.5
                self._spawn_particle_explosion(base['pos'], 30, self.COLOR_RESOURCE, 1.0)

    def _update_game_state(self):
        # Cooldowns
        for base in self.player_bases: base['unit_spawn_cooldown'] = max(0, base['unit_spawn_cooldown'] - 1)
        for base in self.enemy_bases: base['unit_spawn_cooldown'] = max(0, base['unit_spawn_cooldown'] - 1)
        for unit in self.player_units + self.enemy_units: unit['attack_cooldown'] = max(0, unit['attack_cooldown'] - 1)

        self._update_resource_collection()
        self._update_star_system_capture()
        self._update_ai_and_movement()
        self._update_combat()
        self._update_spawners()
        self._update_particles()
        self._cleanup_dead_entities()

    def _update_resource_collection(self):
        for base in self.player_bases:
            collected_asteroids = []
            for i, asteroid in enumerate(self.asteroids):
                dist = np.linalg.norm(base['pos'] - asteroid['pos'])
                if dist < self.BASE_COLLECTION_RADIUS * base['level']:
                    self.resources += asteroid['size']
                    self.reward_this_step += 0.1
                    collected_asteroids.append(i)
                    for _ in range(asteroid['size']):
                        self.particles.append({
                            'pos': asteroid['pos'].copy(),
                            'vel': (base['pos'] - asteroid['pos']) * 0.1 + self.np_random.random(2) * 2 - 1,
                            'lifespan': 20, 'max_lifespan': 20,
                            'color': self.COLOR_RESOURCE, 'size': self.np_random.uniform(1, 3)
                        })
            self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in collected_asteroids]

    def _update_star_system_capture(self):
        for system in self.star_systems:
            player_influence = 0
            enemy_influence = 0
            capture_radius = 120

            for unit in self.player_units:
                if np.linalg.norm(unit['pos'] - system['pos']) < capture_radius:
                    player_influence += 1
            for unit in self.enemy_units:
                if np.linalg.norm(unit['pos'] - system['pos']) < capture_radius:
                    enemy_influence += 1
            
            influence_diff = player_influence - enemy_influence
            system['control'] += influence_diff * 0.005
            system['control'] = np.clip(system['control'], -1.0, 1.0)
            
            new_owner = 0
            if system['control'] >= 1.0: new_owner = 1
            elif system['control'] <= -1.0: new_owner = -1

            if new_owner != 0 and system['owner'] != new_owner:
                if new_owner == 1: self.reward_this_step += 5 # Captured system
                system['owner'] = new_owner
                self._spawn_particle_explosion(system['pos'], 50, self.COLOR_PLAYER if new_owner == 1 else self.COLOR_ENEMY, 1.5)

    def _update_ai_and_movement(self):
        # Player unit AI
        for unit in self.player_units:
            if unit['target_entity'] and unit['target_entity']['hp'] <= 0:
                unit['target_entity'] = None

            if not unit['target_entity']:
                targets = self.enemy_units + self.enemy_bases + [s for s in self.star_systems if s['owner'] <= 0]
                unit['target_entity'] = self._get_closest_entity(unit['pos'], targets)
            
            if unit['target_entity']:
                target_pos = unit['target_entity']['pos']
                dist_to_target = np.linalg.norm(target_pos - unit['pos'])
                if dist_to_target > self.UNIT_ATTACK_RADIUS * 0.8:
                    direction = (target_pos - unit['pos']) / dist_to_target
                    unit['pos'] += direction * self.UNIT_SPEED

        # Enemy unit AI
        for unit in self.enemy_units:
            if unit['target_entity'] and unit['target_entity']['hp'] <= 0:
                unit['target_entity'] = None
            if not unit['target_entity']:
                targets = self.player_bases + self.player_units
                unit['target_entity'] = self._get_closest_entity(unit['pos'], targets)
            
            if unit['target_entity']:
                target_pos = unit['target_entity']['pos']
                dist_to_target = np.linalg.norm(target_pos - unit['pos'])
                if dist_to_target > self.UNIT_ATTACK_RADIUS * 0.8:
                    direction = (target_pos - unit['pos']) / dist_to_target
                    unit['pos'] += direction * self.UNIT_SPEED

    def _update_combat(self):
        # Player units attacking
        for unit in self.player_units:
            if unit['target_entity'] and unit['attack_cooldown'] <= 0:
                target = unit['target_entity']
                if 'control' not in target: # Is a unit or base, not a star system
                    dist = np.linalg.norm(target['pos'] - unit['pos'])
                    if dist <= self.UNIT_ATTACK_RADIUS:
                        target['hp'] -= self.UNIT_DAMAGE
                        unit['attack_cooldown'] = 20 # steps
                        self.projectiles.append({'start': unit['pos'], 'end': target['pos'], 'life': 5, 'color': self.COLOR_PLAYER_LIGHT})

        # Enemy units attacking
        for unit in self.enemy_units:
            if unit['target_entity'] and unit['attack_cooldown'] <= 0:
                target = unit['target_entity']
                dist = np.linalg.norm(target['pos'] - unit['pos'])
                if dist <= self.UNIT_ATTACK_RADIUS:
                    target['hp'] -= self.UNIT_DAMAGE
                    unit['attack_cooldown'] = 20
                    self.projectiles.append({'start': unit['pos'], 'end': target['pos'], 'life': 5, 'color': self.COLOR_ENEMY_LIGHT})

    def _update_spawners(self):
        # Enemy base spawner
        self.enemy_base_spawn_timer -= 1
        if self.enemy_base_spawn_timer <= 0:
            if len(self.enemy_bases) < self.NUM_STAR_SYSTEMS + 1:
                spawn_pos = self._get_random_pos(on_edge=True)
                self.enemy_bases.append({
                    'pos': spawn_pos,
                    'hp': self.BASE_INITIAL_HP, 'max_hp': self.BASE_INITIAL_HP,
                    'level': 1, 'unit_spawn_cooldown': self.ENEMY_UNIT_SPAWN_COOLDOWN
                })
            self.enemy_base_spawn_period = max(100, self.enemy_base_spawn_period - (self.steps // 100) * 0.1)
            self.enemy_base_spawn_timer = self.enemy_base_spawn_period

        # Enemy unit spawner
        for base in self.enemy_bases:
            if base['unit_spawn_cooldown'] <= 0:
                spawn_angle = self.np_random.uniform(0, 2 * math.pi)
                spawn_offset = np.array([math.cos(spawn_angle), math.sin(spawn_angle)]) * 25
                self.enemy_units.append({
                    'pos': base['pos'] + spawn_offset,
                    'hp': self.UNIT_HP, 'max_hp': self.UNIT_HP,
                    'target_entity': None, 'attack_cooldown': 0,
                })
                base['unit_spawn_cooldown'] = self.ENEMY_UNIT_SPAWN_COOLDOWN

        # Asteroid spawner
        if self.np_random.uniform() < 0.05 and len(self.asteroids) < self.MAX_ASTEROIDS:
             self.asteroids.append(
                {'pos': self._get_random_pos(), 'size': self.np_random.integers(5, 11)}
            )

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        self.projectiles = [p for p in self.projectiles if p['life'] > 0]
        for p in self.projectiles: p['life'] -= 1

    def _cleanup_dead_entities(self):
        dead_player_units = [u for u in self.player_units if u['hp'] <= 0]
        if dead_player_units:
            self.reward_this_step -= 0.5 * len(dead_player_units)
            for u in dead_player_units: self._spawn_particle_explosion(u['pos'], 15, self.COLOR_PLAYER, 0.8)
        self.player_units = [u for u in self.player_units if u['hp'] > 0]

        dead_enemy_units = [u for u in self.enemy_units if u['hp'] <= 0]
        if dead_enemy_units:
            for u in dead_enemy_units: self._spawn_particle_explosion(u['pos'], 15, self.COLOR_ENEMY, 0.8)
        self.enemy_units = [u for u in self.enemy_units if u['hp'] > 0]

        dead_player_bases = [b for b in self.player_bases if b['hp'] <= 0]
        if dead_player_bases:
            self.reward_this_step -= 1.0 * len(dead_player_bases)
            for b in dead_player_bases: self._spawn_particle_explosion(b['pos'], 100, self.COLOR_PLAYER, 2.0)
        self.player_bases = [b for b in self.player_bases if b['hp'] > 0]
        
        dead_enemy_bases = [b for b in self.enemy_bases if b['hp'] <= 0]
        if dead_enemy_bases:
            for b in dead_enemy_bases: self._spawn_particle_explosion(b['pos'], 100, self.COLOR_ENEMY, 2.0)
        self.enemy_bases = [b for b in self.enemy_bases if b['hp'] > 0]

    # =================================================================================
    # --- Rendering Methods ---
    # =================================================================================

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background
        for x, y, size in self.static_stars:
            c = int(120 - size * 20)
            pygame.draw.circle(self.screen, (c,c,c), (x,y), size)

        # Star Systems
        for system in self.star_systems:
            color = self.COLOR_NEUTRAL
            if system['owner'] == 1: color = self.COLOR_PLAYER
            elif system['owner'] == -1: color = self.COLOR_ENEMY
            pygame.gfxdraw.aacircle(self.screen, int(system['pos'][0]), int(system['pos'][1]), 40, color)
            pygame.gfxdraw.aacircle(self.screen, int(system['pos'][0]), int(system['pos'][1]), 39, color)
            bar_width = 60
            bar_height = 5
            bar_x = system['pos'][0] - bar_width / 2
            bar_y = system['pos'][1] + 45
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, bar_width/2, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x + bar_width/2, bar_y, bar_width/2, bar_height))
            marker_x = bar_x + bar_width/2 + system['control'] * bar_width/2
            pygame.draw.rect(self.screen, self.COLOR_WHITE, (marker_x - 1, bar_y - 2, 3, bar_height + 4))

        # Asteroids
        for asteroid in self.asteroids:
            pygame.gfxdraw.filled_circle(self.screen, int(asteroid['pos'][0]), int(asteroid['pos'][1]), int(asteroid['size']), self.COLOR_ASTEROID)
        
        # Bases
        for base in self.player_bases + self.enemy_bases:
            is_player = 'unit_spawn_cooldown' in base and (not self.enemy_bases or base not in self.enemy_bases)
            color = self.COLOR_PLAYER if is_player else self.COLOR_ENEMY
            light_color = self.COLOR_PLAYER_LIGHT if is_player else self.COLOR_ENEMY_LIGHT
            radius = int(15 * base['level'])
            
            pulse = abs(math.sin(self.steps * 0.05))
            glow_radius = int(radius + 5 + pulse * 3)
            glow_alpha = int(80 + pulse * 40)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, (*light_color, glow_alpha))
            self.screen.blit(s, (int(base['pos'][0] - glow_radius), int(base['pos'][1] - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.filled_circle(self.screen, int(base['pos'][0]), int(base['pos'][1]), radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(base['pos'][0]), int(base['pos'][1]), radius, self.COLOR_WHITE)
            self._draw_health_bar(base)

        # Units
        for unit in self.player_units + self.enemy_units:
            is_player = unit in self.player_units
            color = self.COLOR_PLAYER if is_player else self.COLOR_ENEMY
            self._draw_unit_shape(unit['pos'], color)
            self._draw_health_bar(unit)

        # Projectiles
        for p in self.projectiles:
            pygame.draw.aaline(self.screen, p['color'], p['start'], p['end'], 2)
        
        # Particles
        for p in self.particles:
            alpha = p['lifespan'] / p['max_lifespan']
            color = (*p['color'], int(alpha * 255))
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

        # Selector
        if self.selected_base_idx != -1:
            base = self.player_bases[self.selected_base_idx]
            radius = int(20 * base['level'])
            pulse = abs(math.sin(self.steps * 0.1)) * 5
            pygame.gfxdraw.aacircle(self.screen, int(base['pos'][0]), int(base['pos'][1]), radius + int(pulse), self.COLOR_SELECTOR)
        else:
            pygame.draw.aaline(self.screen, self.COLOR_SELECTOR, (self.selector_pos[0]-5, self.selector_pos[1]), (self.selector_pos[0]+5, self.selector_pos[1]))
            pygame.draw.aaline(self.screen, self.COLOR_SELECTOR, (self.selector_pos[0], self.selector_pos[1]-5), (self.selector_pos[0], self.selector_pos[1]+5))

    def _render_ui(self):
        res_text = f"RESOURCES: {int(self.resources)}"
        self._draw_text(res_text, (10, 10), self.font_ui, self.COLOR_RESOURCE)
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (self.WIDTH - 150, 10), self.font_ui, self.COLOR_WHITE)
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        self._draw_text(steps_text, (self.WIDTH - 150, 30), self.font_ui, self.COLOR_WHITE)

    # =================================================================================
    # --- Helper & Utility Methods ---
    # =================================================================================

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "resources": self.resources}

    def _check_termination(self):
        win = all(s['owner'] == 1 for s in self.star_systems)
        loss = not self.player_bases
        return win or loss

    def _get_random_pos(self, on_edge=False):
        if on_edge:
            edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top': return np.array([self.np_random.uniform(0, self.WIDTH), 0])
            if edge == 'bottom': return np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT])
            if edge == 'left': return np.array([0, self.np_random.uniform(0, self.HEIGHT)])
            if edge == 'right': return np.array([self.WIDTH, self.np_random.uniform(0, self.HEIGHT)])
        else:
            return np.array([self.np_random.uniform(50, self.WIDTH - 50), self.np_random.uniform(50, self.HEIGHT - 50)])

    def _get_closest_entity(self, pos, entity_list):
        closest = None
        min_dist = float('inf')
        for entity in entity_list:
            dist = np.linalg.norm(pos - entity['pos'])
            if dist < min_dist:
                min_dist = dist
                closest = entity
        return closest
    
    def _spawn_particle_explosion(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'lifespan': 30, 'max_lifespan': 30,
                'color': color, 'size': self.np_random.uniform(1, 4)
            })

    def _draw_health_bar(self, entity):
        pos = entity['pos']
        is_base = 'level' in entity
        bar_width = 40 if is_base else 20
        offset_y = (20 * entity['level']) if is_base else 10
        
        hp_ratio = max(0, entity['hp'] / entity['max_hp'])
        bar_x = pos[0] - bar_width / 2
        bar_y = pos[1] - offset_y - 10
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, 5))
        pygame.draw.rect(self.screen, (50,200,50), (bar_x, bar_y, bar_width * hp_ratio, 5))

    def _draw_unit_shape(self, pos, color):
        points = [
            (pos[0], pos[1] - 6),
            (pos[0] - 4, pos[1] + 4),
            (pos[0] + 4, pos[1] + 4)
        ]
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)

    def _draw_text(self, text, pos, font, color):
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The main script needs a display, so we unset the dummy driver
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("RTS Galaxy Conquest")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset() # Auto-reset

        # Convert observation back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()