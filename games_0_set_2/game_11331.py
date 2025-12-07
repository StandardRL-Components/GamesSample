import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:01:00.414783
# Source Brief: brief_01331.md
# Brief Index: 1331
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from itertools import combinations

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a musical-themed tower defense game.

    The player places and tunes frequency towers to create harmonic fields that
    destroy incoming waves of 'dissonance' (enemies). The goal is to survive
    for a set number of steps.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Movement): 0=none, 1=up, 2=down, 3=left, 4=right. Controls the cursor.
    - `action[1]` (Space): 0=released, 1=held. Places a new tower or upgrades an existing one.
    - `action[2]` (Shift): 0=released, 1=held. Cycles the frequency of the tower under the cursor.

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - Survival: -0.01 per step.
    - Harmonic Hit: +0.1 for each enemy damaged by a harmonic field.
    - Enemy Destroyed: +1.0 per enemy destroyed.
    - Victory: +100 for surviving 1000 steps.
    - Defeat: -100 if an enemy reaches the defense line.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "A musical-themed tower defense game. Place and tune frequency towers to create harmonic fields that destroy incoming waves of dissonance."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to place or upgrade a tower. Press shift to cycle a tower's frequency."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_START = (10, 5, 20)
    COLOR_BG_END = (30, 10, 40)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_CURSOR = (255, 255, 255, 150)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_HARMONY = (255, 255, 255)

    # Game Parameters
    MAX_STEPS = 1000
    CURSOR_SPEED = 8
    INITIAL_RESOURCES = 150
    TOWER_COST = 50
    UPGRADE_COST_BASE = 75
    ENEMY_KILL_REWARD = 25
    DEFENSE_LINE_X = 50
    
    # Frequencies
    FREQUENCIES = [1, 2, 3, 4]
    FREQ_COLORS = {
        1: (255, 0, 128),  # Magenta
        2: (0, 255, 255),  # Cyan
        3: (255, 255, 0),  # Yellow
        4: (0, 255, 128),  # Spring Green
    }
    HARMONIC_PAIRS = frozenset([(1, 3), (2, 4)])

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.towers = []
        self.enemies = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.steps = 0
        self.score = 0
        self.resources = 0
        self.game_over = False
        self.victory = False
        self.last_enemy_spawn_time = 0
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False
        self.bg_surface = self._create_bg_surface()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.resources = self.INITIAL_RESOURCES
        self.game_over = False
        self.victory = False
        
        self.towers = []
        self.enemies = []
        self.particles = []
        
        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.last_enemy_spawn_time = 0
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Survival penalty to encourage efficiency
        self.steps += 1

        self._handle_input(action)
        self._spawn_enemies()
        
        damage_reward = self._update_damage_and_effects()
        kill_reward, terminal_penalty = self._update_enemies()
        reward += damage_reward + kill_reward
        
        terminated, terminal_reward = self._check_termination()
        reward += terminal_penalty + terminal_reward
        self.game_over = terminated
        
        self._update_particles()
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)
        
        # Check for rising edge on buttons
        space_pressed = space_held and not self.space_pressed_last_frame
        shift_pressed = shift_held and not self.shift_pressed_last_frame
        
        tower_at_cursor = self._get_tower_at_cursor()

        if space_pressed:
            if tower_at_cursor:
                self._upgrade_tower(tower_at_cursor)
            else:
                self._place_tower()
        
        if shift_pressed and tower_at_cursor:
            self._cycle_tower_frequency(tower_at_cursor)
            
        self.space_pressed_last_frame = space_held
        self.shift_pressed_last_frame = shift_held

    def _get_tower_at_cursor(self):
        for tower in self.towers:
            if math.hypot(tower['pos'][0] - self.cursor_pos[0], tower['pos'][1] - self.cursor_pos[1]) < tower['size']:
                return tower
        return None

    def _place_tower(self):
        if self.resources >= self.TOWER_COST and self.cursor_pos[0] > self.DEFENSE_LINE_X + 20:
            # Prevent placing towers too close to each other
            if not any(math.hypot(t['pos'][0] - self.cursor_pos[0], t['pos'][1] - self.cursor_pos[1]) < 30 for t in self.towers):
                self.resources -= self.TOWER_COST
                new_tower = {
                    'pos': list(self.cursor_pos),
                    'freq': 1,
                    'level': 1,
                    'size': 10,
                    'range': 70,
                    'damage': 0.5,
                    'pulse': 0.0
                }
                self.towers.append(new_tower)
                # sfx: tower_place.wav
                self._create_particles(self.cursor_pos, self.FREQ_COLORS[1], 20, 2.0)

    def _upgrade_tower(self, tower):
        upgrade_cost = self.UPGRADE_COST_BASE * tower['level']
        if self.resources >= upgrade_cost and tower['level'] < 5:
            self.resources -= upgrade_cost
            tower['level'] += 1
            tower['range'] += 15
            tower['damage'] += 0.25
            # sfx: upgrade_success.wav
            self._create_particles(tower['pos'], (200, 200, 255), 30, 3.0, 1.5)

    def _cycle_tower_frequency(self, tower):
        current_index = self.FREQUENCIES.index(tower['freq'])
        tower['freq'] = self.FREQUENCIES[(current_index + 1) % len(self.FREQUENCIES)]
        # sfx: freq_change.wav
        self._create_particles(tower['pos'], self.FREQ_COLORS[tower['freq']], 15, 1.5)

    def _spawn_enemies(self):
        spawn_interval = max(15, 60 - self.steps // 20)
        if self.steps - self.last_enemy_spawn_time > spawn_interval:
            self.last_enemy_spawn_time = self.steps
            
            base_health = 10 + self.steps // 10
            base_health *= (1 + 0.01 * (self.steps // 50)) # Health scaling
            
            speed = 0.8 + 0.02 * (self.steps // 100) # Speed scaling

            new_enemy = {
                'pos': [self.SCREEN_WIDTH + 20, self.np_random.uniform(20, self.SCREEN_HEIGHT - 20)],
                'health': base_health,
                'max_health': base_health,
                'speed': speed,
                'size': 8
            }
            self.enemies.append(new_enemy)
            
    def _update_damage_and_effects(self):
        reward = 0
        for tower in self.towers:
            tower['pulse'] = (tower['pulse'] + 0.05) % (2 * math.pi)

        for enemy in self.enemies:
            active_towers = []
            for tower in self.towers:
                if math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1]) < tower['range']:
                    active_towers.append(tower)
            
            # Apply base damage and effects
            for tower in active_towers:
                enemy['health'] -= tower['damage']
                # sfx: damage_tick.wav
                self._create_particles(enemy['pos'], self.FREQ_COLORS[tower['freq']], 1, 0.5, 0.5, life=10)

            # Apply harmonic damage
            if len(active_towers) >= 2:
                for t1, t2 in combinations(active_towers, 2):
                    freq_pair = tuple(sorted((t1['freq'], t2['freq'])))
                    if freq_pair in self.HARMONIC_PAIRS:
                        harmony_damage = (t1['damage'] + t2['damage']) * 1.5
                        enemy['health'] -= harmony_damage
                        reward += 0.1 # Reward for harmonic hit
                        # sfx: harmony_hit.wav
                        self._create_particles(enemy['pos'], self.COLOR_HARMONY, 3, 1.0, 1.0, life=15)
        return reward

    def _update_enemies(self):
        kill_reward = 0
        terminal_penalty = 0
        
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            enemy['pos'][0] -= enemy['speed']
            if enemy['health'] <= 0:
                # sfx: enemy_destroy.wav
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 40, 4.0, 2.0)
                kill_reward += 1.0
                self.resources += self.ENEMY_KILL_REWARD
                enemies_to_remove.append(i)
            elif enemy['pos'][0] < self.DEFENSE_LINE_X:
                # sfx: life_lost.wav
                terminal_penalty = -100 # Lose immediately if an enemy gets through
                self.game_over = True
                enemies_to_remove.append(i)
        
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]
            
        return kill_reward, terminal_penalty

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _check_termination(self):
        if self.game_over: # From enemy reaching the end
            self.victory = False
            return True, 0
        
        if self.steps >= self.MAX_STEPS:
            self.victory = True
            return True, 100.0
            
        return False, 0.0
    
    def _create_particles(self, pos, color, count, speed_mult=1.0, size_mult=1.0, life=30):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_mult
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(life // 2, life),
                'color': color,
                'size': self.np_random.uniform(1, 3) * size_mult
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resources": self.resources,
            "towers": len(self.towers),
            "enemies": len(self.enemies),
        }

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_bg_surface(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp,
                self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp,
                self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp,
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        
        # Add some static stars
        for _ in range(100):
            pos = (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT))
            size = random.randint(1, 2)
            brightness = random.randint(50, 100)
            bg.fill((brightness, brightness, brightness), (pos, (size, size)))
            
        return bg

    def _render_game_elements(self):
        # Defense Line
        pygame.draw.line(self.screen, (255, 0, 0, 50), (self.DEFENSE_LINE_X, 0), (self.DEFENSE_LINE_X, self.SCREEN_HEIGHT), 2)
        
        # Particles (rendered first, to be behind other elements)
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            pygame.draw.circle(self.screen, color, [int(p['pos'][0]), int(p['pos'][1])], int(p['size']))

        # Towers and their ranges/auras
        for tower in self.towers:
            pos_int = (int(tower['pos'][0]), int(tower['pos'][1]))
            color = self.FREQ_COLORS[tower['freq']]
            
            # Pulsating range indicator
            pulse_radius = tower['range'] * (0.95 + 0.05 * math.sin(tower['pulse']))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(pulse_radius), (*color, 50))
            
            # Tower Core
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(tower['size']), color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(tower['size']), (255, 255, 255))

            # Level indicator
            for i in range(tower['level']):
                angle = -math.pi/2 + (i - (tower['level']-1)/2) * 0.6
                lx = pos_int[0] + math.cos(angle) * (tower['size'] + 5)
                ly = pos_int[1] + math.sin(angle) * (tower['size'] + 5)
                pygame.draw.circle(self.screen, (255,255,255), (int(lx), int(ly)), 2)


        # Enemies
        for enemy in self.enemies:
            pos_int = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            size = int(enemy['size'])
            
            # Jagged polygon shape for 'dissonance'
            points = []
            num_points = 7
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points + self.steps * 0.1
                radius = size * (1 + 0.3 * math.sin(angle * 3 + self.steps * 0.05))
                points.append((pos_int[0] + math.cos(angle) * radius, pos_int[1] + math.sin(angle) * radius))
            
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

            # Health bar
            if enemy['health'] < enemy['max_health']:
                health_pct = enemy['health'] / enemy['max_health']
                bar_width = size * 2
                pygame.draw.rect(self.screen, (50, 0, 0), (pos_int[0] - size, pos_int[1] - size - 8, bar_width, 5))
                pygame.draw.rect(self.screen, (0, 255, 0), (pos_int[0] - size, pos_int[1] - size - 8, bar_width * health_pct, 5))

        # Cursor
        cursor_pos_int = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        tower_at_cursor = self._get_tower_at_cursor()
        cursor_color = self.COLOR_CURSOR
        if tower_at_cursor:
            cursor_color = (*self.FREQ_COLORS[tower_at_cursor['freq']], 180)
        pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 15, cursor_color)
        pygame.draw.line(self.screen, cursor_color, (cursor_pos_int[0] - 10, cursor_pos_int[1]), (cursor_pos_int[0] + 10, cursor_pos_int[1]), 1)
        pygame.draw.line(self.screen, cursor_color, (cursor_pos_int[0], cursor_pos_int[1] - 10), (cursor_pos_int[0], cursor_pos_int[1] + 10), 1)

    def _render_ui(self):
        # Resources
        res_text = self.font_ui.render(f"ENERGY: {self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_text, (10, 10))

        # Steps
        steps_text = self.font_ui.render(f"BAR: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 30))

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            msg = "VICTORY" if self.victory else "DISSONANCE OVERWHELMED"
            color = (0, 255, 128) if self.victory else (255, 50, 50)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # This method is for self-testing and can be removed in production
        try:
            # Test action space
            assert self.action_space.shape == (3,)
            assert self.action_space.nvec.tolist() == [5, 2, 2]
            
            # Test observation space
            test_obs = self._get_observation()
            assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert test_obs.dtype == np.uint8
            
            # Test reset
            obs, info = self.reset()
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(info, dict)
            
            # Test step
            test_action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(test_action)
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert trunc == False
            assert isinstance(info, dict)
            
            print("✓ Implementation validated successfully")
        except AssertionError as e:
            print(f"✗ Implementation validation failed: {e}")


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be run by the evaluation system
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    # We need a display for human playable mode
    pygame.display.init()
    display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Harmonic Defense")

    running = True
    while running:
        # Human control
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Pygame rendering for human playable mode
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        env.clock.tick(env.metadata["render_fps"])

    env.close()