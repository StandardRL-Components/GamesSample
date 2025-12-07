import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:55:40.594944
# Source Brief: brief_01294.md
# Brief Index: 1294
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your fortress from waves of digital glitches by powering up musical towers with mixtapes. "
        "Match tape types to towers to unleash powerful attacks and special abilities."
    )
    user_guide = (
        "Controls: ←→ to select a tower, ↑↓ to select a tape. "
        "Press space to load the selected tape into the tower. Press shift to activate a tower's special ability."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    
    # Colors (Retro Neon Palette)
    COLOR_BG = (13, 10, 33)
    COLOR_GRID = (33, 25, 84)
    COLOR_TEXT = (255, 255, 255)
    COLOR_SCORE = (255, 220, 0)
    COLOR_HEALTH_GOOD = (0, 255, 128)
    COLOR_HEALTH_BAD = (255, 50, 50)
    COLOR_SELECT_GLOW = (255, 255, 255)

    TOWER_COLORS = {
        'BASS': (255, 0, 128),    # Hot Pink
        'TREBLE': (0, 255, 255),  # Cyan
        'RHYTHM': (255, 255, 0),  # Yellow
        'SPECIAL': (170, 0, 255)  # Purple
    }

    # Game Parameters
    MAX_STEPS = 2500
    TOTAL_WAVES = 10
    INITIAL_FORTRESS_HEALTH = 10
    FORTRESS_Y_LINE = 100
    INVENTORY_SIZE = 4
    TAPE_REFILL_TIME = 90  # frames
    TOWER_MAX_CHARGE = 100
    TOWER_CHARGE_PER_TAPE = 40
    TOWER_CHARGE_DECAY = 0.05
    
    TOWER_SPECS = {
        'BASS': {'fire_rate': 90, 'special_cost': 100},
        'TREBLE': {'fire_rate': 45, 'special_cost': 100},
        'RHYTHM': {'fire_rate': 60, 'special_cost': 100}
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        try:
            self.font_main = pygame.font.Font("freesansbold.ttf", 24)
            self.font_small = pygame.font.Font("freesansbold.ttf", 16)
            self.font_symbol = pygame.font.Font("freesansbold.ttf", 18)
        except IOError:
            self.font_main = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 20)
            self.font_symbol = pygame.font.Font(None, 22)
            
        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # validation is done by the test suite

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fortress_health = self.INITIAL_FORTRESS_HEALTH
        
        self.wave = 0
        self.wave_timer = 0
        self.glitches_to_spawn_this_wave = 0
        self.glitches_spawned_this_wave = 0
        
        self._init_towers()
        self.glitches = []
        self.projectiles = []
        self.particles = []
        
        self.inventory = [self._generate_tape() for _ in range(self.INVENTORY_SIZE)]
        self.tape_refill_timer = 0
        
        self.selected_inventory_idx = 0
        self.selected_tower_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.screen_shake = 0
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        
        # --- Tape Deployment & Special Activation ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            reward += self._deploy_tape()
        
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed:
            reward += self._activate_special()
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Game Logic Updates ---
        self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_glitches()
        self._update_particles()
        self._update_inventory_refill()
        self._update_wave_spawner()

        if self.screen_shake > 0:
            self.screen_shake -= 1

        # --- Wave & Level Completion ---
        if not self.glitches and self.glitches_spawned_this_wave == self.glitches_to_spawn_this_wave:
            if self.wave >= self.TOTAL_WAVES:
                self.game_over = True
                reward += 100  # Victory
            else:
                reward += 5  # Wave cleared
                self._start_next_wave()
        
        # --- Termination Conditions ---
        terminated = self.game_over
        if self.fortress_health <= 0 and not self.game_over:
            reward -= 100  # Failure
            self.game_over = True
            terminated = True
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Gymnasium standard
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # region # --- Game Logic Sub-functions ---
    def _handle_input(self, movement, space_held, shift_held):
        # Using a simple cooldown to make manual play feel better
        if not hasattr(self, '_input_cooldown'): self._input_cooldown = 0
        if self._input_cooldown > 0: 
            self._input_cooldown -= 1
            return

        if movement != 0:
            if movement == 1:  # Up
                self.selected_inventory_idx = (self.selected_inventory_idx - 1) % self.INVENTORY_SIZE
            elif movement == 2:  # Down
                self.selected_inventory_idx = (self.selected_inventory_idx + 1) % self.INVENTORY_SIZE
            elif movement == 3:  # Left
                self.selected_tower_idx = (self.selected_tower_idx - 1) % len(self.towers)
            elif movement == 4:  # Right
                self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.towers)
            self._input_cooldown = 5 # 5-frame cooldown for selection

    def _init_towers(self):
        self.towers = []
        num_towers = 3
        for i in range(num_towers):
            tower_type = list(self.TOWER_SPECS.keys())[i]
            self.towers.append({
                'type': tower_type,
                'pos': (self.WIDTH // (num_towers + 1) * (i + 1), self.HEIGHT - 50),
                'charge': 0.0,
                'cooldown': 0,
                'special_active_timer': 0
            })

    def _generate_tape(self):
        # Ensure a mix of tapes, with a chance for a special one
        if self.np_random.random() < 0.15: # 15% chance for a special tape
            tape_type = 'SPECIAL'
        else:
            tape_type = self.np_random.choice(list(self.TOWER_SPECS.keys()))
        
        symbols = {'BASS': '♮', 'TREBLE': '♯', 'RHYTHM': '♭', 'SPECIAL': '★'}
        return {'type': tape_type, 'symbol': symbols[tape_type]}

    def _deploy_tape(self):
        if self.inventory[self.selected_inventory_idx] is None:
            return 0
        
        selected_tape = self.inventory[self.selected_inventory_idx]
        selected_tower = self.towers[self.selected_tower_idx]

        # Matching tape to tower
        if selected_tape['type'] == selected_tower['type']:
            selected_tower['charge'] = min(self.TOWER_MAX_CHARGE, selected_tower['charge'] + self.TOWER_CHARGE_PER_TAPE)
            self.inventory[self.selected_inventory_idx] = None
            # sound: "tape_match.wav"
            self._create_particles(selected_tower['pos'], 10, self.TOWER_COLORS[selected_tower['type']], 2, 4)
            return 1.0  # Reward for matching
        # Using a special tape
        elif selected_tape['type'] == 'SPECIAL':
            selected_tower['charge'] = self.TOWER_MAX_CHARGE
            self.inventory[self.selected_inventory_idx] = None
            # sound: "special_tape.wav"
            self._create_particles(selected_tower['pos'], 20, self.TOWER_COLORS['SPECIAL'], 3, 6)
            return 2.0 # Higher reward for using a special tape
        
        return -0.1 # Penalty for mismatch

    def _activate_special(self):
        tower = self.towers[self.selected_tower_idx]
        spec = self.TOWER_SPECS[tower['type']]
        if tower['charge'] >= spec['special_cost']:
            tower['charge'] -= spec['special_cost']
            tower['special_active_timer'] = 10 # Visual indicator
            # sound: f"special_{tower['type']}.wav"
            self._create_particles(tower['pos'], 50, (255,255,255), 5, 10)
            self.screen_shake = 15

            # --- Trigger Special Abilities ---
            if tower['type'] == 'BASS': # Screen-wide damage pulse
                for glitch in self.glitches:
                    glitch['health'] -= 20
                    self._create_particles(glitch['pos'], 5, self.TOWER_COLORS['BASS'], 2, 5)
                self.score += 20 * len(self.glitches)
            elif tower['type'] == 'TREBLE': # Piercing beam
                beam_y = tower['pos'][1] - 20
                self.projectiles.append({'type': 'TREBLE_BEAM', 'pos': [0, beam_y], 'power': 50, 'lifespan': 10})
            elif tower['type'] == 'RHYTHM': # Rapid fire volley
                for i in range(10):
                    if i < len(self.glitches):
                        target_glitch = self.glitches[i]
                        angle = math.atan2(target_glitch['pos'][1] - tower['pos'][1], target_glitch['pos'][0] - tower['pos'][0])
                        speed = 8
                        self.projectiles.append({
                            'type': 'RHYTHM', 'pos': list(tower['pos']), 'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                            'power': 15, 'lifespan': 120
                        })
            return 10.0 # Reward for special activation
        return 0

    def _update_towers(self):
        for tower in self.towers:
            tower['charge'] = max(0, tower['charge'] - self.TOWER_CHARGE_DECAY)
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
            if tower['special_active_timer'] > 0:
                tower['special_active_timer'] -= 1

            # Auto-fire logic
            if tower['charge'] > 0 and tower['cooldown'] == 0 and self.glitches:
                spec = self.TOWER_SPECS[tower['type']]
                tower['cooldown'] = spec['fire_rate']
                # sound: f"fire_{tower['type']}.wav"
                
                if tower['type'] == 'BASS':
                    self.projectiles.append({'type': 'BASS', 'pos': list(tower['pos']), 'radius': 10, 'max_radius': 60, 'power': 10, 'lifespan': 30})
                else: # TREBLE and RHYTHM are targeted
                    target_glitch = min(self.glitches, key=lambda g: g['pos'][0])
                    angle = math.atan2(target_glitch['pos'][1] - tower['pos'][1], target_glitch['pos'][0] - tower['pos'][0])
                    
                    if tower['type'] == 'TREBLE':
                        speed = 12
                        self.projectiles.append({'type': 'TREBLE', 'pos': list(tower['pos']), 'vel': [math.cos(angle)*speed, math.sin(angle)*speed], 'power': 20, 'lifespan': 80})
                    
                    elif tower['type'] == 'RHYTHM':
                        for i in range(-1, 2):
                            burst_angle = angle + math.radians(i * 15)
                            speed = 7
                            self.projectiles.append({'type': 'RHYTHM', 'pos': list(tower['pos']), 'vel': [math.cos(burst_angle)*speed, math.sin(burst_angle)*speed], 'power': 8, 'lifespan': 100})

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.projectiles.remove(p)
                continue

            if p['type'] == 'BASS':
                p['radius'] += (p['max_radius'] - 10) / p['lifespan'] if p['lifespan'] > 0 else 0
                for g in self.glitches[:]:
                    dist = math.hypot(p['pos'][0] - g['pos'][0], p['pos'][1] - g['pos'][1])
                    if dist < p['radius'] + g['size']/2:
                        reward += self._damage_glitch(g, p['power'])
                        # Bass projectile hits multiple targets but is consumed at end of life
            elif p['type'] == 'TREBLE_BEAM':
                 for g in self.glitches[:]:
                    if g['pos'][1] > p['pos'][1] and g['pos'][1] < p['pos'][1] + 10:
                        reward += self._damage_glitch(g, p['power'])
            else: # Targeted projectiles
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                for g in self.glitches[:]:
                    if math.hypot(p['pos'][0] - g['pos'][0], p['pos'][1] - g['pos'][1]) < g['size'] / 2:
                        reward += self._damage_glitch(g, p['power'])
                        self.projectiles.remove(p)
                        break
        return reward

    def _damage_glitch(self, glitch, power):
        glitch['health'] -= power
        self.score += power
        # sound: "hit.wav"
        self._create_particles(glitch['pos'], 3, self.COLOR_HEALTH_BAD, 1, 3)
        return 0.1 # Reward for hitting a glitch

    def _update_glitches(self):
        breach_reward_penalty = 0
        for g in self.glitches[:]:
            if g['health'] <= 0:
                # sound: "glitch_destroy.wav"
                self._create_particles(g['pos'], 30, (200, 200, 255), 2, 8)
                self.score += g['max_health']
                self.glitches.remove(g)
                continue
            
            g['pos'][0] -= g['speed']
            if g['pos'][0] < self.FORTRESS_Y_LINE:
                self.fortress_health -= 1
                breach_reward_penalty -= 10.0 # Significant penalty
                # sound: "fortress_hit.wav"
                self.screen_shake = 10
                self.glitches.remove(g)
        return breach_reward_penalty

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _update_inventory_refill(self):
        self.tape_refill_timer -= 1
        if self.tape_refill_timer <= 0:
            for i in range(self.INVENTORY_SIZE):
                if self.inventory[i] is None:
                    self.inventory[i] = self._generate_tape()
                    # sound: "tape_refill.wav"
                    self.tape_refill_timer = self.TAPE_REFILL_TIME
                    break

    def _start_next_wave(self):
        self.wave += 1
        self.wave_timer = 0
        self.glitches_spawned_this_wave = 0
        self.glitches_to_spawn_this_wave = 5 + self.wave * 2
        
    def _update_wave_spawner(self):
        if self.glitches_spawned_this_wave < self.glitches_to_spawn_this_wave:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._spawn_glitch()
                self.glitches_spawned_this_wave += 1
                self.wave_timer = max(10, 60 - self.wave * 3) # Spawn faster in later waves

    def _spawn_glitch(self):
        health = 15 + (self.wave // 2) * 5
        speed = 0.8 + (self.wave // 5) * 0.2
        size = self.np_random.integers(18, 29)
        pos = [self.WIDTH + size, self.np_random.integers(int(size/2), self.HEIGHT - 150)]
        self.glitches.append({'pos': pos, 'health': health, 'max_health': health, 'speed': speed, 'size': size, 'seed': self.np_random.integers(0, 10001)})
    
    def _create_particles(self, pos, count, color, speed_range, size_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(speed_range/2, speed_range)
            size = self.np_random.uniform(size_range/2, size_range)
            lifespan = self.np_random.integers(15, 41)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'size': size
            })
    # endregion

    # region # --- Rendering Sub-functions ---
    def _get_observation(self):
        render_offset = [0, 0]
        if self.screen_shake > 0:
            render_offset[0] = self.np_random.integers(-4, 5)
            render_offset[1] = self.np_random.integers(-4, 5)

        # Create a temporary surface to apply the shake
        temp_surf = self.screen.copy()
        temp_surf.fill(self.COLOR_BG)

        self._render_background(temp_surf)
        self._render_towers(temp_surf)
        self._render_glitches(temp_surf)
        self._render_projectiles(temp_surf)
        self._render_particles(temp_surf)
        self._render_ui(temp_surf)

        self.screen.blit(temp_surf, render_offset)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, surface):
        # Grid lines
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(surface, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(surface, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)
        # Fortress line
        pygame.draw.line(surface, self.COLOR_HEALTH_BAD, (self.FORTRESS_Y_LINE, 0), (self.FORTRESS_Y_LINE, self.HEIGHT - 120), 3)

    def _render_towers(self, surface):
        for i, tower in enumerate(self.towers):
            pos = tower['pos']
            color = self.TOWER_COLORS[tower['type']]
            
            # Base
            base_rect = pygame.Rect(pos[0] - 25, pos[1], 50, 20)
            pygame.draw.rect(surface, (80, 80, 100), base_rect)
            pygame.draw.rect(surface, (120, 120, 140), base_rect, 2)
            
            # Charge indicator
            charge_ratio = tower['charge'] / self.TOWER_MAX_CHARGE
            glow_size = int(5 + 15 * charge_ratio)
            
            # Glow
            if glow_size > 5:
                s = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, 60), (glow_size, glow_size), glow_size)
                surface.blit(s, (pos[0] - glow_size, pos[1] - 30 - glow_size))
            
            # Crystal
            crystal_points = [
                (pos[0], pos[1] - 30 - glow_size),
                (pos[0] - glow_size/2, pos[1] - 30),
                (pos[0] + glow_size/2, pos[1] - 30)
            ]
            pygame.gfxdraw.aapolygon(surface, crystal_points, color)
            pygame.gfxdraw.filled_polygon(surface, crystal_points, color)

            # Special active glow
            if tower['special_active_timer'] > 0:
                r = 30 + (10 - tower['special_active_timer']) * 2
                alpha = 150 * (tower['special_active_timer'] / 10)
                pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]-15), r, (*self.COLOR_SELECT_GLOW, int(alpha)))

            # Selection highlight
            if i == self.selected_tower_idx:
                pygame.draw.rect(surface, self.COLOR_SELECT_GLOW, (pos[0] - 30, pos[1] - 45, 60, 70), 2, border_radius=5)

    def _render_glitches(self, surface):
        for g in self.glitches:
            pos_x, pos_y = int(g['pos'][0]), int(g['pos'][1])
            size = int(g['size'])
            
            # Body
            rect = pygame.Rect(pos_x - size/2, pos_y - size/2, size, size)
            pygame.draw.rect(surface, (50,0,0), rect)

            # Static effect
            glitch_rng = np.random.default_rng(g['seed'] + self.steps)
            for _ in range(5):
                sub_x = pos_x - size/2 + glitch_rng.integers(0, int(size-3))
                sub_y = pos_y - size/2 + glitch_rng.integers(0, int(size-3))
                sub_w = glitch_rng.integers(2, 7)
                sub_h = glitch_rng.integers(2, 7)
                sub_c_idx = glitch_rng.integers(0,3)
                sub_c = [self.COLOR_HEALTH_BAD, self.COLOR_GRID, (255,100,100)][sub_c_idx]
                pygame.draw.rect(surface, sub_c, (sub_x, sub_y, sub_w, sub_h))
            
            pygame.draw.rect(surface, self.COLOR_HEALTH_BAD, rect, 2)

            # Health bar
            health_ratio = g['health'] / g['max_health']
            bar_w = size
            bar_h = 5
            pygame.draw.rect(surface, (50,50,50), (pos_x - bar_w/2, pos_y - size/2 - 10, bar_w, bar_h))
            pygame.draw.rect(surface, self.COLOR_HEALTH_GOOD, (pos_x - bar_w/2, pos_y - size/2 - 10, bar_w * health_ratio, bar_h))

    def _render_projectiles(self, surface):
        for p in self.projectiles:
            pos_x, pos_y = int(p['pos'][0]), int(p['pos'][1])
            if p['type'] == 'BASS':
                color = (*self.TOWER_COLORS['BASS'], 150)
                pygame.gfxdraw.aacircle(surface, pos_x, pos_y, int(p['radius']), color)
                pygame.gfxdraw.filled_circle(surface, pos_x, pos_y, int(p['radius']), color)
            elif p['type'] == 'TREBLE':
                end_pos = (pos_x + p['vel'][0] * 0.5, pos_y + p['vel'][1] * 0.5)
                pygame.draw.aaline(surface, self.TOWER_COLORS['TREBLE'], (pos_x, pos_y), end_pos, 3)
            elif p['type'] == 'RHYTHM':
                pygame.gfxdraw.filled_circle(surface, pos_x, pos_y, 4, self.TOWER_COLORS['RHYTHM'])
            elif p['type'] == 'TREBLE_BEAM':
                rect = pygame.Rect(0, p['pos'][1], self.WIDTH, 10)
                alpha = 255 * (p['lifespan'] / 10)
                s = pygame.Surface((self.WIDTH, 10), pygame.SRCALPHA)
                s.fill((*self.TOWER_COLORS['TREBLE'], int(alpha)))
                surface.blit(s, (0, p['pos'][1]))

    def _render_particles(self, surface):
        for p in self.particles:
            life_ratio = p['lifespan'] / p['max_lifespan']
            color = (*p['color'], int(255 * life_ratio))
            size = int(p['size'] * life_ratio)
            if size > 0:
                pygame.gfxdraw.filled_circle(surface, int(p['pos'][0]), int(p['pos'][1]), size, color)

    def _render_ui(self, surface):
        # Bottom panel
        panel_rect = pygame.Rect(0, self.HEIGHT - 120, self.WIDTH, 120)
        pygame.gfxdraw.box(surface, panel_rect, (10, 8, 26, 200))
        pygame.draw.line(surface, self.COLOR_GRID, (0, self.HEIGHT - 120), (self.WIDTH, self.HEIGHT - 120), 2)
        
        # Score, Wave, Health
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        surface.blit(score_text, (20, 20))
        wave_text = self.font_main.render(f"WAVE: {self.wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        surface.blit(wave_text, (self.WIDTH - 180, 20))
        health_text = self.font_main.render(f"FORTRESS: {self.fortress_health}", True, self.COLOR_HEALTH_GOOD)
        surface.blit(health_text, (self.WIDTH / 2 - 100, 20))

        # Inventory
        inv_title = self.font_small.render("TAPES", True, self.COLOR_TEXT)
        surface.blit(inv_title, (self.WIDTH - 110, self.HEIGHT - 115))
        for i in range(self.INVENTORY_SIZE):
            slot_rect = pygame.Rect(self.WIDTH - 120, self.HEIGHT - 95 + i * 22, 110, 20)
            if self.inventory[i] is None:
                pygame.draw.rect(surface, (50, 50, 70), slot_rect, 1)
                continue
            
            tape = self.inventory[i]
            color = self.TOWER_COLORS[tape['type']]
            pygame.draw.rect(surface, color, slot_rect)
            pygame.draw.rect(surface, tuple(c*0.5 for c in color), slot_rect.inflate(-4, -4))
            
            symbol_text = self.font_symbol.render(tape['symbol'], True, self.COLOR_TEXT)
            surface.blit(symbol_text, (slot_rect.x + 5, slot_rect.y))

            if i == self.selected_inventory_idx:
                pygame.draw.rect(surface, self.COLOR_SELECT_GLOW, slot_rect, 2)
    # endregion

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.fortress_health,
        }
        
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not be run by the testing environment
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # To display the game, we need to unset the dummy video driver
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Mixtape Marauders")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset(seed=43)
            total_reward = 0
            # A small pause on game over
            pygame.time.wait(2000)

        clock.tick(30) # Run at 30 FPS
        
    env.close()