import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:33:08.544983
# Source Brief: brief_01125.md
# Brief Index: 1125
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Rune Citadel: A tower defense game where you defend a central citadel
    against waves of geometric enemies by placing defensive runes.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Defend your central citadel from waves of geometric enemies by placing magical runes that automatically attack."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to select a rune. Press space to place the selected rune in an empty slot."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 3000
    MAX_WAVES = 20

    # Colors
    COLOR_BG = (15, 18, 23)
    COLOR_BG_ACCENT = (30, 35, 45)
    COLOR_CITADEL = (150, 160, 180)
    COLOR_CITADEL_GLOW = (190, 200, 220)
    COLOR_HEALTH_BAR = (40, 180, 100)
    COLOR_HEALTH_DAMAGE = (220, 80, 80)
    COLOR_MANA_BAR = (60, 120, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_DIM = (120, 120, 120)

    RUNE_COLORS = {
        'damage': (255, 80, 80),
        'slow': (80, 180, 255),
        'aoe': (255, 150, 50),
    }

    ENEMY_COLORS = {
        'triangle': (220, 50, 50),
        'square': (230, 70, 70),
        'pentagon': (240, 90, 90),
    }

    # Citadel
    CITADEL_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    CITADEL_RADIUS = 40
    CITADEL_SIDES = 8
    CITADEL_SLOT_RADIUS = 55

    # Rune Config
    RUNE_CONFIG = {
        'damage': {'cost': 25, 'cooldown': 45, 'range': 150, 'power': 10},
        'slow': {'cost': 40, 'cooldown': 60, 'range': 120, 'power': 0.5, 'duration': 90},
        'aoe': {'cost': 60, 'cooldown': 120, 'range': 100, 'power': 15},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self.citadel_slot_positions = [self._get_citadel_slot_pos(i) for i in range(self.CITADEL_SIDES)]

        self.render_mode = render_mode
        # The reset call is deferred to the first call of env.reset() as per Gymnasium API
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.citadel_health = 100
        self.max_citadel_health = 100
        self.mana = 50

        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_cooldown_timer = self.FPS * 3 # 3 seconds to first wave

        self.enemies = []
        self.runes = [None] * self.CITADEL_SIDES
        self.projectiles = []
        self.particles = []
        self.effects = []

        self.unlocked_rune_types = ['damage']
        self.selected_rune_type_idx = 0

        self.prev_action = np.array([0, 0, 0])
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        step_reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self._handle_input(action)

        self._update_runes()
        self._update_projectiles()
        kill_reward, kills = self._update_enemies()
        step_reward += kill_reward
        self.score += kills * 10

        self._update_particles_and_effects()

        wave_completion_reward = self._update_wave_manager()
        step_reward += wave_completion_reward

        self.steps += 1
        terminated = self.citadel_health <= 0 or self.wave_number > self.MAX_WAVES
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True
            if self.citadel_health > 0 and self.wave_number > self.MAX_WAVES:
                step_reward += 100.0  # Victory bonus
                self.score += 10000
            elif self.citadel_health <= 0:
                step_reward += -100.0 # Loss penalty

        self.prev_action = action
        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_press, shift_press = action[0], action[1], action[2]
        prev_movement, prev_space, _ = self.prev_action

        # Cycle rune selection on left/right key press (not hold)
        if movement != prev_movement and movement in [3, 4]:
            if movement == 3: # Left
                self.selected_rune_type_idx = (self.selected_rune_type_idx - 1) % len(self.unlocked_rune_types)
                # sfx: UI_bleep_low
            elif movement == 4: # Right
                self.selected_rune_type_idx = (self.selected_rune_type_idx + 1) % len(self.unlocked_rune_types)
                # sfx: UI_bleep_high

        # Place rune on space press (not hold)
        if space_press == 1 and prev_space == 0:
            self._attempt_place_rune()

    def _attempt_place_rune(self):
        try:
            empty_slot = self.runes.index(None)
        except ValueError:
            # sfx: error_buzz
            return # No empty slots

        rune_type = self.unlocked_rune_types[self.selected_rune_type_idx]
        cost = self.RUNE_CONFIG[rune_type]['cost']

        if self.mana >= cost:
            self.mana -= cost
            self.runes[empty_slot] = {
                'type': rune_type,
                'cooldown': 0,
                'slot': empty_slot
            }
            pos = self.citadel_slot_positions[empty_slot]
            self._create_particles(pos, self.RUNE_COLORS[rune_type], 20, 2.0)
            # sfx: rune_place_success
        else:
            # sfx: insufficient_funds
            pass

    def _update_runes(self):
        for i, rune in enumerate(self.runes):
            if not rune:
                continue

            rune['cooldown'] = max(0, rune['cooldown'] - 1)
            if rune['cooldown'] == 0:
                self._activate_rune(rune)

    def _activate_rune(self, rune):
        rune_type = rune['type']
        config = self.RUNE_CONFIG[rune_type]
        pos = self.citadel_slot_positions[rune['slot']]

        # Find closest target in range
        target = None
        min_dist = float('inf')
        for enemy in self.enemies:
            dist = math.hypot(enemy['pos'][0] - pos[0], enemy['pos'][1] - pos[1])
            if dist < config['range'] and dist < min_dist:
                min_dist = dist
                target = enemy
        
        if not target and rune_type != 'aoe':
            return

        rune['cooldown'] = config['cooldown']
        # sfx: rune_fire_[type]

        if rune_type == 'damage':
            self.projectiles.append({
                'start_pos': list(pos),
                'pos': list(pos),
                'target': target,
                'speed': 6,
                'damage': config['power'],
                'color': self.RUNE_COLORS[rune_type]
            })
        elif rune_type == 'slow':
            target['effects']['slow'] = {'duration': config['duration'], 'power': config['power']}
            self.effects.append({'type': 'slow_aura', 'pos': target['pos'], 'target': target, 'lifespan': 30, 'radius': 15})
        elif rune_type == 'aoe':
            self.effects.append({'type': 'aoe_pulse', 'pos': pos, 'lifespan': 20, 'max_radius': config['range'], 'damage': config['power']})
            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - pos[0], enemy['pos'][1] - pos[1])
                if dist < config['range']:
                    enemy['health'] -= config['power']
                    self._create_particles(enemy['pos'], self.RUNE_COLORS['aoe'], 5, 1.0, 0.5)

    def _update_projectiles(self):
        remaining_projectiles = []
        for p in self.projectiles:
            if p['target']['health'] <= 0:
                self._create_particles(p['pos'], p['color'], 5, 1.0, 0.2) # Fizzle
                continue

            target_pos = p['target']['pos']
            direction = [target_pos[0] - p['pos'][0], target_pos[1] - p['pos'][1]]
            dist = math.hypot(*direction)

            if dist < p['speed']:
                p['target']['health'] -= p['damage']
                self._create_particles(target_pos, p['color'], 10, 1.5) # Impact
                # sfx: projectile_hit
                continue

            direction[0] /= dist
            direction[1] /= dist
            p['pos'][0] += direction[0] * p['speed']
            p['pos'][1] += direction[1] * p['speed']
            remaining_projectiles.append(p)
        self.projectiles = remaining_projectiles

    def _update_enemies(self):
        remaining_enemies = []
        kill_reward = 0.0
        kills = 0
        for e in self.enemies:
            if e['health'] <= 0:
                kill_reward += 0.1
                kills += 1
                self.mana = min(200, self.mana + e['mana_value'])
                self._create_particles(e['pos'], self.ENEMY_COLORS[e['type']], 30, 3.0, 1.5)
                # sfx: enemy_death
                continue

            speed_modifier = 1.0
            if 'slow' in e['effects']:
                slow = e['effects']['slow']
                speed_modifier = 1.0 - slow['power']
                slow['duration'] -= 1
                if slow['duration'] <= 0:
                    del e['effects']['slow']
            
            direction = [self.CITADEL_POS[0] - e['pos'][0], self.CITADEL_POS[1] - e['pos'][1]]
            dist = math.hypot(*direction)

            if dist < self.CITADEL_RADIUS:
                self.citadel_health -= e['damage']
                self.citadel_health = max(0, self.citadel_health)
                self._create_particles(self.CITADEL_POS, (255, 255, 255), 50, 4.0, 2.0)
                self.effects.append({'type': 'screen_shake', 'lifespan': 15, 'magnitude': 5})
                # sfx: citadel_hit
                continue

            direction[0] /= dist
            direction[1] /= dist
            e['pos'][0] += direction[0] * e['speed'] * speed_modifier
            e['pos'][1] += direction[1] * e['speed'] * speed_modifier
            remaining_enemies.append(e)
        self.enemies = remaining_enemies
        return kill_reward, kills

    def _update_particles_and_effects(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Drag
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1

        self.effects = [e for e in self.effects if e['lifespan'] > 0]
        for e in self.effects:
            e['lifespan'] -= 1
            if e['type'] == 'slow_aura' and e['target']['health'] > 0:
                e['pos'] = e['target']['pos']


    def _update_wave_manager(self):
        if self.wave_in_progress and not self.enemies:
            self.wave_in_progress = False
            self.wave_cooldown_timer = self.FPS * 5 # 5 seconds between waves
            self.score += self.wave_number * 100
            
            # Unlock new runes
            if self.wave_number == 3 and 'slow' not in self.unlocked_rune_types:
                self.unlocked_rune_types.append('slow')
            if self.wave_number == 6 and 'aoe' not in self.unlocked_rune_types:
                self.unlocked_rune_types.append('aoe')
            
            return 1.0 # Wave survival reward

        if not self.wave_in_progress:
            self.wave_cooldown_timer -= 1
            if self.wave_cooldown_timer <= 0:
                self.wave_number += 1
                if self.wave_number <= self.MAX_WAVES:
                    self._spawn_wave()
                    self.wave_in_progress = True
        return 0.0

    def _spawn_wave(self):
        num_enemies = 3 + self.wave_number
        base_health = 10 + self.wave_number * 5
        base_speed = 0.8 + (self.wave_number // 2) * 0.05
        
        for _ in range(num_enemies):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(self.SCREEN_WIDTH/2, self.SCREEN_WIDTH/2 + 50)
            pos = [
                self.CITADEL_POS[0] + distance * math.cos(angle),
                self.CITADEL_POS[1] + distance * math.sin(angle)
            ]
            
            enemy_type = random.choice(['triangle', 'square', 'pentagon'])
            health_mult = {'triangle': 0.8, 'square': 1.0, 'pentagon': 1.3}[enemy_type]
            speed_mult = {'triangle': 1.2, 'square': 1.0, 'pentagon': 0.8}[enemy_type]
            
            self.enemies.append({
                'pos': pos,
                'health': base_health * health_mult,
                'max_health': base_health * health_mult,
                'speed': base_speed * speed_mult,
                'damage': 5 + self.wave_number,
                'mana_value': 3 + self.wave_number,
                'type': enemy_type,
                'effects': {}
            })
        # sfx: new_wave_horn

    def _get_observation(self):
        offset_x, offset_y = 0, 0
        for effect in self.effects:
            if effect['type'] == 'screen_shake':
                offset_x = random.randint(-effect['magnitude'], effect['magnitude'])
                offset_y = random.randint(-effect['magnitude'], effect['magnitude'])
                break
        
        self.screen.fill(self.COLOR_BG)
        self._render_background(offset_x, offset_y)
        self._render_particles(offset_x, offset_y)
        self._render_citadel_and_runes(offset_x, offset_y)
        self._render_enemies(offset_x, offset_y)
        self._render_projectiles(offset_x, offset_y)
        self._render_effects(offset_x, offset_y)
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, ox, oy):
        for i in range(50):
            x = (hash(i * 10) % self.SCREEN_WIDTH + ox) % self.SCREEN_WIDTH
            y = (hash(i * 33) % self.SCREEN_HEIGHT + oy) % self.SCREEN_HEIGHT
            self.screen.set_at((int(x), int(y)), self.COLOR_BG_ACCENT)

    def _render_citadel_and_runes(self, ox, oy):
        # Citadel
        citadel_points = []
        for i in range(self.CITADEL_SIDES):
            angle = 2 * math.pi * i / self.CITADEL_SIDES - math.pi / self.CITADEL_SIDES
            x = self.CITADEL_POS[0] + self.CITADEL_RADIUS * math.cos(angle) + ox
            y = self.CITADEL_POS[1] + self.CITADEL_RADIUS * math.sin(angle) + oy
            citadel_points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(self.screen, citadel_points, self.COLOR_CITADEL_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, citadel_points, self.COLOR_CITADEL)

        # Runes
        for i, rune in enumerate(self.runes):
            pos = self.citadel_slot_positions[i]
            pos_off = (int(pos[0] + ox), int(pos[1] + oy))
            if rune:
                color = self.RUNE_COLORS[rune['type']]
                if rune['cooldown'] > 0:
                    color = tuple(c // 2 for c in color) # Dim if on cooldown
                self._draw_rune(rune['type'], pos_off, color, 10)
            else:
                pygame.gfxdraw.aacircle(self.screen, pos_off[0], pos_off[1], 5, self.COLOR_CITADEL)

    def _draw_rune(self, type, pos, color, size):
        half_size = size // 2
        if type == 'damage':
            points = [(pos[0], pos[1] - half_size), (pos[0] - half_size, pos[1] + half_size), (pos[0] + half_size, pos[1] + half_size)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif type == 'slow':
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], half_size, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], half_size - 3, color)
        elif type == 'aoe':
            for r in range(2, half_size, 3):
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], r, color)

    def _render_enemies(self, ox, oy):
        for e in self.enemies:
            pos = (int(e['pos'][0] + ox), int(e['pos'][1] + oy))
            color = self.ENEMY_COLORS[e['type']]
            
            if e['type'] == 'triangle':
                points = [(pos[0], pos[1]-8), (pos[0]-8, pos[1]+8), (pos[0]+8, pos[1]+8)]
            elif e['type'] == 'square':
                points = [(pos[0]-7, pos[1]-7), (pos[0]+7, pos[1]-7), (pos[0]+7, pos[1]+7), (pos[0]-7, pos[1]+7)]
            else: # Pentagon
                points = []
                for i in range(5):
                    angle = 2 * math.pi * i / 5 - math.pi / 2
                    points.append((pos[0] + 10 * math.cos(angle), pos[1] + 10 * math.sin(angle)))
            
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

            # Health bar
            if e['health'] < e['max_health']:
                bar_w = 20
                bar_h = 3
                bar_x = pos[0] - bar_w // 2
                bar_y = pos[1] - 20
                health_pct = e['health'] / e['max_health']
                pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_w * health_pct), bar_h))

    def _render_projectiles(self, ox, oy):
        for p in self.projectiles:
            pos = (int(p['pos'][0] + ox), int(p['pos'][1] + oy))
            # Trail effect
            self._create_particles(p['pos'], p['color'], 1, 0, 0.1, 10)
            pygame.draw.aaline(self.screen, p['color'], p['start_pos'], pos, 2)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, p['color'])

    def _render_particles(self, ox, oy):
        for p in self.particles:
            pos = (int(p['pos'][0] + ox), int(p['pos'][1] + oy))
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                # This is a trick to draw alpha-blended filled circles
                surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size, size), size)
                self.screen.blit(surf, (pos[0] - size, pos[1] - size))

    def _render_effects(self, ox, oy):
        for e in self.effects:
            pos = (int(e['pos'][0] + ox), int(e['pos'][1] + oy))
            progress = (e['lifespan'] / e['max_lifespan']) if 'max_lifespan' in e else (e['lifespan'] / 30)
            
            if e['type'] == 'slow_aura':
                radius = int(e['radius'] * (1 - progress))
                alpha = int(200 * progress)
                if radius > 0 and alpha > 0:
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.RUNE_COLORS['slow'], alpha))
            
            elif e['type'] == 'aoe_pulse':
                radius = int(e['max_radius'] * (1 - progress))
                alpha = int(255 * progress)
                if radius > 0 and alpha > 0:
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.RUNE_COLORS['aoe'], alpha))
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius-2, (*self.RUNE_COLORS['aoe'], alpha))

    def _render_ui(self):
        # Health Bar
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_DAMAGE, (10, 10, 200, 20))
        health_w = int(200 * (self.citadel_health / self.max_citadel_health))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, max(0, health_w), 20))
        health_text = self.font_medium.render(f"{int(self.citadel_health)} / {self.max_citadel_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Mana
        mana_text = self.font_medium.render(f"Mana: {int(self.mana)}", True, self.COLOR_MANA_BAR)
        self.screen.blit(mana_text, (220, 12))

        # Wave
        wave_str = f"Wave: {self.wave_number}/{self.MAX_WAVES}" if self.wave_in_progress else f"Next wave in {self.wave_cooldown_timer/self.FPS:.1f}s"
        wave_text = self.font_medium.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 35))

        # Rune selection UI
        ui_y = self.SCREEN_HEIGHT - 60
        total_w = len(self.unlocked_rune_types) * 80
        start_x = self.SCREEN_WIDTH // 2 - total_w // 2
        for i, r_type in enumerate(self.unlocked_rune_types):
            is_selected = i == self.selected_rune_type_idx
            box_x = start_x + i * 80
            rect = pygame.Rect(box_x, ui_y, 70, 50)
            
            border_color = self.RUNE_COLORS[r_type] if is_selected else self.COLOR_CITADEL
            bg_color = tuple(c // 4 for c in self.RUNE_COLORS[r_type]) if is_selected else self.COLOR_BG_ACCENT
            
            pygame.draw.rect(self.screen, bg_color, rect, border_radius=5)
            pygame.draw.rect(self.screen, border_color, rect, 2, 5)

            self._draw_rune(r_type, (box_x + 20, ui_y + 25), self.RUNE_COLORS[r_type], 12)

            name_text = self.font_small.render(r_type.upper(), True, self.COLOR_TEXT)
            self.screen.blit(name_text, (box_x + 35, ui_y + 10))

            cost = self.RUNE_CONFIG[r_type]['cost']
            cost_color = self.COLOR_MANA_BAR if self.mana >= cost else self.COLOR_TEXT_DIM
            cost_text = self.font_small.render(f"{cost}", True, cost_color)
            self.screen.blit(cost_text, (box_x + 35, ui_y + 28))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "mana": self.mana, "health": self.citadel_health}

    def _create_particles(self, pos, color, count, speed_mult=1.0, size=1.0, lifespan_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5) * speed_mult
            lifespan = random.randint(15, 30) * lifespan_mult
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'size': random.uniform(1, 3) * size
            })

    def _get_citadel_slot_pos(self, slot_index):
        angle = 2 * math.pi * slot_index / self.CITADEL_SIDES - math.pi / 2
        x = self.CITADEL_POS[0] + self.CITADEL_SLOT_RADIUS * math.cos(angle)
        y = self.CITADEL_POS[1] + self.CITADEL_SLOT_RADIUS * math.sin(angle)
        return (x, y)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the headless evaluation environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Rune Citadel")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                
                # Update action state on keydown
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1

            if event.type == pygame.KEYUP:
                # Reset action state on keyup
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0
                if event.key == pygame.K_SPACE:
                    space_held = 0
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    shift_held = 0

        action = np.array([movement, space_held, shift_held])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0
            
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()