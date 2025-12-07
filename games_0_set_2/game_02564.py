
# Generated: 2025-08-27T20:45:36.057323
# Source Brief: brief_02564.md
# Brief Index: 2564

        
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
        "Controls: Use arrow keys to move the placement cursor. "
        "Hold Shift to cycle tower types. Press Space to build or to start the next wave."
    )

    game_description = (
        "A top-down tower defense game. Place towers strategically to defend your base "
        "from 10 increasingly difficult waves of enemies. Don't let any enemies reach the end!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()

        # --- Game Configuration ---
        self._define_colors_and_fonts()
        self._define_game_parameters()
        self._define_level_layout()
        self._define_entity_properties()

        # --- Initialize State ---
        self.reset()
        
        # --- Self-Validation ---
        # self.validate_implementation()

    def _define_colors_and_fonts(self):
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_PATH = (45, 50, 62)
        self.COLOR_GRID = (60, 66, 82)
        self.COLOR_BASE = (68, 189, 50)
        self.COLOR_BASE_STROKE = (106, 216, 92)
        self.COLOR_ENEMY = (217, 30, 24)
        self.COLOR_ENEMY_STROKE = (231, 76, 60)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GOLD = (241, 196, 15)
        self.COLOR_CURSOR_VALID = (255, 255, 255, 100)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 100)
        self.FONT_UI = pygame.font.SysFont("Consolas", 20, bold=True)
        self.FONT_MSG = pygame.font.SysFont("Consolas", 32, bold=True)
        self.FONT_DESC = pygame.font.SysFont("Consolas", 14)

    def _define_game_parameters(self):
        self.FPS = 30
        self.MAX_STEPS = 15000 # Increased for a 10-wave game
        self.MAX_WAVES = 10
        self.STARTING_GOLD = 150
        self.GOLD_PER_KILL = 10

    def _define_level_layout(self):
        self.PATH_WAYPOINTS = [
            (-20, 150), (100, 150), (100, 250), (250, 250), (250, 100),
            (450, 100), (450, 300), (660, 300)
        ]
        self.BASE_POS = (640, 300)
        self.BASE_RECT = pygame.Rect(self.BASE_POS[0] - 10, self.BASE_POS[1] - 25, 10, 50)
        
        self.TOWER_SLOTS = [
            (100, 80), (180, 180), (180, 320), (350, 170), (350, 320), (530, 230)
        ]
        self.START_WAVE_BUTTON_POS = (560, 365)
        self.CURSOR_TARGETS = self.TOWER_SLOTS + [self.START_WAVE_BUTTON_POS]

    def _define_entity_properties(self):
        self.TOWER_DATA = {
            0: {"name": "Cannon", "cost": 50, "range": 80, "damage": 12, "fire_rate": 1.0, "color": (52, 152, 219)},
            1: {"name": "Gatling", "cost": 80, "range": 60, "damage": 5, "fire_rate": 0.3, "color": (26, 188, 156)},
            2: {"name": "Sniper", "cost": 120, "range": 150, "damage": 40, "fire_rate": 2.5, "color": (155, 89, 182)}
        }
        self.NUM_TOWER_TYPES = len(self.TOWER_DATA)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.gold = self.STARTING_GOLD
        self.wave_number = 0
        self.game_phase = "BUILD"  # "BUILD" or "WAVE"
        self.game_over = False
        self.game_won = False
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.wave_spawn_timer = 0
        self.enemies_to_spawn = []

        self.cursor_index = 0
        self.selected_tower_type = 0
        self.prev_shift_held = False
        self.prev_space_held = False

        self.message = "Prepare for the first wave!"
        self.message_timer = self.FPS * 3

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over and not self.game_won:
            self._handle_input(action)

            if self.game_phase == "WAVE":
                self._update_spawner()
                self._update_towers()
                reward += self._update_enemies()
                self._update_projectiles()
                
                if not self.enemies and not self.enemies_to_spawn:
                    reward += self._end_wave()
            
            self._update_particles()
        
        self.steps += 1
        if self.message_timer > 0:
            self.message_timer -= 1
        else:
            self.message = ""

        self.score += reward
        
        if self.game_over:
            reward = -100
            terminated = True
            self.message = "GAME OVER"
            self.message_timer = self.FPS * 5
        elif self.game_won:
            reward = 10
            terminated = True
            self.message = "VICTORY!"
            self.message_timer = self.FPS * 5
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_over: # Don't overwrite game over message
                 self.message = "Time Limit Reached"
                 self.message_timer = self.FPS * 5

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement != 0: # Only move on actual input
            # This logic is a bit naive, but good enough for discrete selection
            if movement == 1: self.cursor_index = (self.cursor_index - 1) % len(self.CURSOR_TARGETS)
            if movement == 2: self.cursor_index = (self.cursor_index + 1) % len(self.CURSOR_TARGETS)
            if movement == 3: self.cursor_index = (self.cursor_index - 1) % len(self.CURSOR_TARGETS)
            if movement == 4: self.cursor_index = (self.cursor_index + 1) % len(self.CURSOR_TARGETS)

        # --- Tower Type Selection (on press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % self.NUM_TOWER_TYPES
            # sfx: UI_cycle

        # --- Placement / Wave Start (on press) ---
        if space_held and not self.prev_space_held:
            cursor_pos = self.CURSOR_TARGETS[self.cursor_index]
            
            if cursor_pos == self.START_WAVE_BUTTON_POS: # Start wave button
                if self.game_phase == "BUILD":
                    self._start_wave()
                    # sfx: wave_start
            else: # Tower slot
                if self.game_phase == "BUILD":
                    self._place_tower(cursor_pos)

        self.prev_shift_held = shift_held
        self.prev_space_held = space_held

    def _start_wave(self):
        self.game_phase = "WAVE"
        self.wave_number += 1
        self.message = f"Wave {self.wave_number} Incoming!"
        self.message_timer = self.FPS * 2

        num_enemies = 5 + (self.wave_number - 1) * 2
        base_health = 10 * (1.1 ** (self.wave_number - 1))
        base_speed = 0.5 + self.wave_number * 0.1

        self.enemies_to_spawn = []
        for i in range(num_enemies):
            # Add slight variation to enemies
            health = base_health * self.rng.uniform(0.9, 1.1)
            speed = base_speed * self.rng.uniform(0.9, 1.1)
            self.enemies_to_spawn.append({"health": health, "speed": speed})
        
        self.wave_spawn_timer = 0

    def _end_wave(self):
        if self.wave_number >= self.MAX_WAVES:
            self.game_won = True
            return 1.0 # Return wave completion reward
        else:
            self.game_phase = "BUILD"
            self.message = "Wave Complete! Prepare for the next."
            self.message_timer = self.FPS * 3
            return 1.0 # Wave completion reward

    def _place_tower(self, pos):
        tower_data = self.TOWER_DATA[self.selected_tower_type]
        cost = tower_data["cost"]

        is_occupied = any(t['pos'] == pos for t in self.towers)
        if is_occupied:
            self.message = "Slot is already occupied!"
            self.message_timer = self.FPS * 2
            # sfx: error
            return

        if self.gold >= cost:
            self.gold -= cost
            self.towers.append({
                "pos": pos,
                "type": self.selected_tower_type,
                "cooldown": 0,
                "target": None
            })
            self.message = f"{tower_data['name']} placed!"
            self.message_timer = self.FPS * 2
            # sfx: place_tower
        else:
            self.message = "Not enough gold!"
            self.message_timer = self.FPS * 2
            # sfx: error

    def _update_spawner(self):
        if self.enemies_to_spawn:
            self.wave_spawn_timer -= 1 / self.FPS
            if self.wave_spawn_timer <= 0:
                enemy_data = self.enemies_to_spawn.pop(0)
                self.enemies.append({
                    "pos": list(self.PATH_WAYPOINTS[0]),
                    "max_health": enemy_data["health"],
                    "health": enemy_data["health"],
                    "speed": enemy_data["speed"],
                    "waypoint_index": 1
                })
                self.wave_spawn_timer = 0.5 # Time between spawns
                # sfx: enemy_spawn

    def _update_towers(self):
        for tower in self.towers:
            data = self.TOWER_DATA[tower['type']]
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1 / self.FPS
                continue

            # Find a target
            target = None
            min_dist = float('inf')
            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1])
                if dist <= data['range'] and dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower['cooldown'] = data['fire_rate']
                self.projectiles.append({
                    "start_pos": list(tower['pos']),
                    "pos": list(tower['pos']),
                    "target": target,
                    "speed": 15,
                    "damage": data['damage'],
                    "color": data['color']
                })
                # sfx: tower_fire

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            # --- Movement ---
            if enemy['waypoint_index'] < len(self.PATH_WAYPOINTS):
                target_pos = self.PATH_WAYPOINTS[enemy['waypoint_index']]
                dx = target_pos[0] - enemy['pos'][0]
                dy = target_pos[1] - enemy['pos'][1]
                dist = math.hypot(dx, dy)

                if dist < enemy['speed']:
                    enemy['pos'] = list(target_pos)
                    enemy['waypoint_index'] += 1
                else:
                    enemy['pos'][0] += (dx / dist) * enemy['speed']
                    enemy['pos'][1] += (dy / dist) * enemy['speed']
            else: # Reached the base
                self.game_over = True
                self.enemies.remove(enemy)
                # sfx: base_damage
                continue

            # --- Check Health ---
            if enemy['health'] <= 0:
                # sfx: enemy_death
                for _ in range(20): # Death particle effect
                    self.particles.append(self._create_particle(enemy['pos'], self.COLOR_ENEMY, 1.0))
                self.enemies.remove(enemy)
                self.gold += self.GOLD_PER_KILL
                reward += 0.1

        return reward

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            target = proj['target']
            if target not in self.enemies: # Target is dead
                self.projectiles.remove(proj)
                continue

            dx = target['pos'][0] - proj['pos'][0]
            dy = target['pos'][1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj['speed']:
                # Hit target
                target['health'] -= proj['damage']
                self.projectiles.remove(proj)
                # sfx: projectile_hit
                for _ in range(5): # Hit particle effect
                    self.particles.append(self._create_particle(target['pos'], proj['color'], 0.5))
            else:
                proj['pos'][0] += (dx / dist) * proj['speed']
                proj['pos'][1] += (dy / dist) * proj['speed']

    def _create_particle(self, pos, color, lifetime_mult):
        return {
            "pos": list(pos),
            "vel": [self.rng.uniform(-2, 2), self.rng.uniform(-2, 2)],
            "lifetime": self.rng.uniform(0.2, 0.5) * lifetime_mult * self.FPS,
            "color": color,
            "size": self.rng.uniform(2, 5)
        }

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_path()
        self._render_grid()
        self._render_base()
        for p in self.particles: self._render_particle(p)
        for t in self.towers: self._render_tower(t)
        for e in self.enemies: self._render_enemy(e)
        for p in self.projectiles: self._render_projectile(p)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_path(self):
        for i in range(len(self.PATH_WAYPOINTS) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.PATH_WAYPOINTS[i], self.PATH_WAYPOINTS[i+1], 40)

    def _render_grid(self):
        for slot in self.TOWER_SLOTS:
            pygame.gfxdraw.aacircle(self.screen, int(slot[0]), int(slot[1]), 20, self.COLOR_GRID)

    def _render_base(self):
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.BASE_RECT, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASE_STROKE, self.BASE_RECT, 2, border_radius=5)

    def _render_tower(self, tower):
        pos = (int(tower['pos'][0]), int(tower['pos'][1]))
        data = self.TOWER_DATA[tower['type']]
        color = data['color']
        
        # Draw range circle if cursor is over it
        cursor_pos = self.CURSOR_TARGETS[self.cursor_index]
        if cursor_pos == tower['pos']:
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], data['range'], (*color, 100))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], data['range'], (*color, 50))

        if data['name'] == "Cannon":
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, tuple(c*0.8 for c in color))
        elif data['name'] == "Gatling":
            points = [(pos[0], pos[1]-12), (pos[0]-10, pos[1]+8), (pos[0]+10, pos[1]+8)]
            pygame.gfxdraw.aapolygon(self.screen, points, tuple(c*0.8 for c in color))
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif data['name'] == "Sniper":
            rect = pygame.Rect(pos[0]-10, pos[1]-10, 20, 20)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), rect, 2)

    def _render_enemy(self, enemy):
        pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
        size = 8
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY_STROKE)

        # Health bar
        health_pct = max(0, enemy['health'] / enemy['max_health'])
        bar_width = 16
        bar_height = 3
        bar_x = pos[0] - bar_width // 2
        bar_y = pos[1] - size - 8
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, int(bar_width * health_pct), bar_height))

    def _render_projectile(self, proj):
        pos = (int(proj['pos'][0]), int(proj['pos'][1]))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, proj['color'])
        
    def _render_particle(self, particle):
        alpha = max(0, min(255, int(255 * (particle['lifetime'] / (0.5 * self.FPS)))))
        color = (*particle['color'], alpha)
        temp_surf = pygame.Surface((particle['size']*2, particle['size']*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (particle['size'], particle['size']), particle['size'])
        self.screen.blit(temp_surf, (int(particle['pos'][0] - particle['size']), int(particle['pos'][1] - particle['size'])))

    def _render_ui(self):
        # Top bar
        pygame.draw.rect(self.screen, (0,0,0,150), (0, 0, self.screen_width, 30))
        gold_text = self.FONT_UI.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (10, 5))
        wave_text = self.FONT_UI.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.screen_width - wave_text.get_width() - 10, 5))

        # Bottom bar (Tower selection)
        pygame.draw.rect(self.screen, (0,0,0,150), (0, self.screen_height - 40, self.screen_width, 40))
        for i, data in self.TOWER_DATA.items():
            is_selected = i == self.selected_tower_type
            color = data['color'] if self.gold >= data['cost'] else (100, 100, 100)
            if is_selected: color = (255, 255, 255)
            
            text = self.FONT_DESC.render(f"{i+1}:{data['name']} ${data['cost']}", True, color)
            self.screen.blit(text, (10 + i * 180, self.screen_height - 28))

        # Cursor
        cursor_pos = self.CURSOR_TARGETS[self.cursor_index]
        is_start_button = cursor_pos == self.START_WAVE_BUTTON_POS
        is_occupied = any(t['pos'] == cursor_pos for t in self.towers)
        
        if is_start_button:
            color = self.COLOR_CURSOR_VALID if self.game_phase == "BUILD" else self.COLOR_CURSOR_INVALID
            rect = pygame.Rect(0, 0, 120, 30)
            rect.center = cursor_pos
            pygame.draw.rect(self.screen, color, rect, 2, border_radius=5)
            text = self.FONT_UI.render("START", True, self.COLOR_TEXT)
            self.screen.blit(text, text.get_rect(center=rect.center))
        else: # Is a tower slot
            color = self.COLOR_CURSOR_INVALID if is_occupied else self.COLOR_CURSOR_VALID
            pygame.gfxdraw.aacircle(self.screen, cursor_pos[0], cursor_pos[1], 22, color)
            pygame.gfxdraw.aacircle(self.screen, cursor_pos[0], cursor_pos[1], 23, color)

        # Message
        if self.message and self.message_timer > 0:
            msg_surf = self.FONT_MSG.render(self.message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave_number,
            "phase": self.game_phase
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Validating implementation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Playable Demo ---
    # This part is for demonstration and will not be part of the final submission.
    # It allows playing the game with a keyboard.
    
    import sys
    
    env.reset()
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over. Score: {info['score']}, Wave: {info['wave']}")
            env.reset()

        # Pygame uses a different coordinate system for blitting numpy arrays
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()
    sys.exit()