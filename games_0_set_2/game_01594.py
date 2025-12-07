
# Generated: 2025-08-27T17:38:11.222407
# Source Brief: brief_01594.md
# Brief Index: 1594

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor, Space to place selected tower, Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Isometric tower defense. Place towers to defend your base from waves of enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 30000  # Approx 16 mins at 30fps
    WIN_WAVE = 20
    
    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_PATH = (50, 60, 70)
    COLOR_PATH_BORDER = (70, 80, 90)
    COLOR_GRID_VALID = (80, 150, 80, 100)
    COLOR_GRID_INVALID = (150, 80, 80, 100)
    COLOR_BASE = (60, 180, 220)
    COLOR_BASE_GLOW = (60, 180, 220, 50)
    
    COLOR_TEXT = (220, 220, 230)
    COLOR_GOLD = (255, 215, 0)
    COLOR_HEALTH = (220, 50, 50)
    
    # Isometric projection
    ISO_TILE_WIDTH_HALF = 28
    ISO_TILE_HEIGHT_HALF = 14
    ISO_ORIGIN_X = SCREEN_WIDTH // 2
    ISO_ORIGIN_Y = 80
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 14)
        self.font_m = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game Data
        self._define_game_data()

        # Initialize state variables
        self.cursor_pos_idx = 0
        self.selected_tower_type = 0
        self.last_shift_press = False
        self.last_space_press = False
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 0
        self.gold = 0
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.wave_timer = 0
        self.wave_in_progress = False
        self.pending_rewards = 0.0
        
        self.reset()
        
        self.validate_implementation()
        
    def _define_game_data(self):
        # Path defined in grid coordinates
        self.PATH = [(-1, 5), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (6, 1), (7, 1), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (7, 7), (6, 7), (5, 7), (4, 7), (3, 7), (3, 8), (3, 9), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10)]
        self.BASE_POS = self.PATH[-1]
        
        # Tower placement slots in grid coordinates
        self.PLACEMENT_SLOTS = [(1, 4), (3, 4), (4, 4), (6, 2), (7, 2), (7, 4), (7, 6), (6, 6), (4, 6), (2, 7), (2, 8), (2, 9), (4, 9), (5, 9), (6, 9), (8, 9)]
        
        # Tower Specifications
        self.TOWER_SPECS = {
            0: {"name": "Cannon", "cost": 50, "range": 3.0, "damage": 10, "fire_rate": 45, "color": (200, 200, 200), "proj_speed": 5, "proj_color": (255, 255, 100)},
            1: {"name": "Sniper", "cost": 120, "range": 5.0, "damage": 40, "fire_rate": 100, "color": (100, 100, 255), "proj_speed": 10, "proj_color": (150, 150, 255)},
        }

    def _to_screen_coords(self, grid_x, grid_y):
        screen_x = self.ISO_ORIGIN_X + (grid_x - grid_y) * self.ISO_TILE_WIDTH_HALF
        screen_y = self.ISO_ORIGIN_Y + (grid_x + grid_y) * self.ISO_TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.gold = 150
        self.wave_number = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos_idx = 0
        self.selected_tower_type = 0
        self.last_shift_press = False
        self.last_space_press = False
        self.wave_in_progress = False
        self.wave_timer = 150 # Time until first wave
        self.pending_rewards = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.pending_rewards = 0.0
        
        self._handle_player_input(action)
        self._update_game_logic()
        
        self.steps += 1
        
        terminated = self._check_termination()
        
        reward = self.pending_rewards
        # Add small penalty for existing towers to encourage efficiency
        reward -= len(self.towers) * 0.0001
        self.score += reward
        
        if terminated:
            if self.win:
                reward += 100
            else: # Loss
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement: Cycle through placement slots ---
        if movement != 0:
            # Simple debounce for movement
            if not hasattr(self, '_last_move_time') or self.steps > self._last_move_time + 5:
                if movement == 1: # Up
                    self.cursor_pos_idx = (self.cursor_pos_idx - 1) % len(self.PLACEMENT_SLOTS)
                elif movement == 2: # Down
                    self.cursor_pos_idx = (self.cursor_pos_idx + 1) % len(self.PLACEMENT_SLOTS)
                elif movement == 3: # Left
                    self.cursor_pos_idx = (self.cursor_pos_idx - 4) % len(self.PLACEMENT_SLOTS)
                elif movement == 4: # Right
                    self.cursor_pos_idx = (self.cursor_pos_idx + 4) % len(self.PLACEMENT_SLOTS)
                self._last_move_time = self.steps

        # --- Shift: Cycle tower type ---
        if shift_held and not self.last_shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self.last_shift_press = shift_held

        # --- Space: Place tower ---
        if space_held and not self.last_space_press:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.gold >= spec["cost"]:
                pos = self.PLACEMENT_SLOTS[self.cursor_pos_idx]
                is_occupied = any(t['pos'] == pos for t in self.towers)
                if not is_occupied:
                    self.gold -= spec["cost"]
                    self.towers.append({
                        "pos": pos,
                        "type": self.selected_tower_type,
                        "cooldown": 0,
                    })
                    # sfx: place_tower.wav
                    self._create_particles(self._to_screen_coords(*pos), 20, spec["color"])

        self.last_space_press = space_held

    def _update_game_logic(self):
        # --- Wave Management ---
        if not self.wave_in_progress and not self.game_over:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave_number += 1
                if self.wave_number > self.WIN_WAVE:
                    self.win = True
                    self.game_over = True
                else:
                    self._spawn_wave()
                    self.wave_in_progress = True

        # --- Update Towers ---
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            spec = self.TOWER_SPECS[tower['type']]
            target = None
            min_dist = spec['range'] ** 2 # Use squared distance
            
            for enemy in self.enemies:
                dist_sq = (tower['pos'][0] - enemy['pos'][0])**2 + (tower['pos'][1] - enemy['pos'][1])**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                tower['cooldown'] = spec['fire_rate']
                self.projectiles.append({
                    "start_pos": self._to_screen_coords(*tower['pos']),
                    "target_enemy": target,
                    "pos": list(self._to_screen_coords(*tower['pos'])),
                    "type": tower['type']
                })
                # sfx: tower_fire.wav
                self._create_particles(self._to_screen_coords(*tower['pos']), 5, (255, 150, 50), 0.5, 2, 5)

        # --- Update Projectiles ---
        for proj in self.projectiles[:]:
            spec = self.TOWER_SPECS[proj['type']]
            target_pos = self._to_screen_coords(*proj['target_enemy']['pos'])
            
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < spec['proj_speed']:
                # Hit
                proj['target_enemy']['health'] -= spec['damage']
                # sfx: enemy_hit.wav
                self._create_particles(target_pos, 15, spec['proj_color'])
                if proj in self.projectiles:
                    self.projectiles.remove(proj)
            else:
                proj['pos'][0] += (dx / dist) * spec['proj_speed']
                proj['pos'][1] += (dy / dist) * spec['proj_speed']

        # --- Update Enemies ---
        for enemy in self.enemies[:]:
            # Check for death
            if enemy['health'] <= 0:
                # sfx: enemy_die.wav
                self._create_particles(self._to_screen_coords(*enemy['pos']), 30, (255, 80, 80))
                self.gold += enemy['bounty']
                self.pending_rewards += 0.1
                self.enemies.remove(enemy)
                continue
            
            # Movement
            path_idx = enemy['path_index']
            target_node = self.PATH[path_idx]
            
            dx = target_node[0] - enemy['pos'][0]
            dy = target_node[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < enemy['speed']:
                enemy['path_index'] += 1
                if enemy['path_index'] >= len(self.PATH):
                    # Reached base
                    self.base_health -= enemy['damage']
                    self.base_health = max(0, self.base_health)
                    self._create_particles(self._to_screen_coords(*self.BASE_POS), 50, self.COLOR_HEALTH)
                    self.enemies.remove(enemy)
                    # sfx: base_damage.wav
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']
        
        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # --- Check for wave clear ---
        if self.wave_in_progress and not self.enemies:
            self.wave_in_progress = False
            self.wave_timer = 300 # Time until next wave
            self.pending_rewards += 1.0

    def _spawn_wave(self):
        num_enemies = 3 + self.wave_number * 2
        base_health = 20 + (self.wave_number - 1) * 10
        base_speed = 0.03 + (self.wave_number - 1) * 0.002
        
        for i in range(num_enemies):
            self.enemies.append({
                "pos": list(self.PATH[0]),
                "path_index": 1,
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed * self.np_random.uniform(0.9, 1.1),
                "damage": 10,
                "bounty": 5 + int(self.wave_number / 2),
                "spawn_delay": i * 30, # Stagger spawn
            })
    
    def _create_particles(self, pos, count, color, speed_mult=1.0, life_mult=1.0, size=3):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 30) * life_mult,
                "color": color,
                "size": self.np_random.integers(1, size+1)
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
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
        # --- Draw Path and Base ---
        path_points = [self._to_screen_coords(x, y) for x, y in self.PATH]
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, path_points, self.ISO_TILE_HEIGHT_HALF * 2)
        pygame.draw.lines(self.screen, self.COLOR_PATH_BORDER, False, path_points, self.ISO_TILE_HEIGHT_HALF * 2 + 2)

        base_screen_pos = self._to_screen_coords(*self.BASE_POS)
        pygame.gfxdraw.filled_circle(self.screen, base_screen_pos[0], base_screen_pos[1], 20, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, base_screen_pos[0], base_screen_pos[1], 20, self.COLOR_BASE)
        pygame.gfxdraw.filled_circle(self.screen, base_screen_pos[0], base_screen_pos[1], 30, self.COLOR_BASE_GLOW)

        # --- Draw Placement Slots and Cursor ---
        for i, pos in enumerate(self.PLACEMENT_SLOTS):
            is_occupied = any(t['pos'] == pos for t in self.towers)
            color = self.COLOR_GRID_INVALID if is_occupied else self.COLOR_GRID_VALID
            
            x, y = self._to_screen_coords(*pos)
            points = [
                (x, y - self.ISO_TILE_HEIGHT_HALF),
                (x + self.ISO_TILE_WIDTH_HALF, y),
                (x, y + self.ISO_TILE_HEIGHT_HALF),
                (x - self.ISO_TILE_WIDTH_HALF, y)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            
            if i == self.cursor_pos_idx:
                spec = self.TOWER_SPECS[self.selected_tower_type]
                cursor_color = (255, 255, 255) if self.gold >= spec['cost'] and not is_occupied else self.COLOR_HEALTH
                pygame.draw.polygon(self.screen, cursor_color, points, 2)

        # --- Draw Towers ---
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            x, y = self._to_screen_coords(*tower['pos'])
            pygame.draw.circle(self.screen, (30,30,30), (x, y - 5), 10)
            pygame.draw.circle(self.screen, spec['color'], (x, y - 8), 10)
            
        # --- Draw Enemies ---
        for enemy in sorted(self.enemies, key=lambda e: e['pos'][1]):
            x, y = self._to_screen_coords(enemy['pos'][0], enemy['pos'][1])
            size = 8
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (x - size, y - size - 15, size*2, size*2))
            
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_w = 20
            pygame.draw.rect(self.screen, (50,0,0), (x - bar_w/2, y - 25, bar_w, 4))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (x - bar_w/2, y - 25, bar_w * health_pct, 4))

        # --- Draw Projectiles ---
        for proj in self.projectiles:
            spec = self.TOWER_SPECS[proj['type']]
            x, y = int(proj['pos'][0]), int(proj['pos'][1])
            pygame.draw.circle(self.screen, spec['proj_color'], (x, y), 4)
            pygame.gfxdraw.aacircle(self.screen, x, y, 4, spec['proj_color'])

        # --- Draw Particles ---
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), p['size'])

    def _render_ui(self):
        # --- Top Bar ---
        ui_bar = pygame.Surface((self.SCREEN_WIDTH, 40))
        ui_bar.set_alpha(180)
        ui_bar.fill((10, 20, 30))
        self.screen.blit(ui_bar, (0, 0))

        # Gold
        gold_text = self.font_m.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (10, 10))
        
        # Base Health
        health_text = self.font_m.render(f"BASE HP: {self.base_health}", True, self.COLOR_BASE)
        self.screen.blit(health_text, (150, 10))
        
        # Wave
        wave_str = f"WAVE: {self.wave_number}/{self.WIN_WAVE}"
        if not self.wave_in_progress and not self.game_over:
            wave_str += f" (Next in {self.wave_timer // 30}s)"
        wave_text = self.font_m.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (320, 10))

        # --- Tower Selection UI ---
        for i, spec in self.TOWER_SPECS.items():
            is_selected = i == self.selected_tower_type
            
            box_rect = pygame.Rect(self.SCREEN_WIDTH - 160, 50 + i * 80, 150, 70)
            bg_color = (80, 90, 110) if is_selected else (40, 50, 60)
            pygame.draw.rect(self.screen, bg_color, box_rect, border_radius=5)
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_TEXT, box_rect, 2, 5)
            
            name_text = self.font_m.render(spec['name'], True, self.COLOR_TEXT)
            self.screen.blit(name_text, (box_rect.x + 10, box_rect.y + 5))
            
            cost_text = self.font_s.render(f"Cost: {spec['cost']}", True, self.COLOR_GOLD)
            self.screen.blit(cost_text, (box_rect.x + 10, box_rect.y + 28))
            
            dmg_text = self.font_s.render(f"Dmg: {spec['damage']} / Rng: {spec['range']:.1f}", True, self.COLOR_TEXT)
            self.screen.blit(dmg_text, (box_rect.x + 10, box_rect.y + 45))
        
        # --- Game Over/Win Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else self.COLOR_HEALTH
            end_text = self.font_l.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.wave_number,
            "enemies": len(self.enemies),
            "towers": len(self.towers)
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="human") # This will fail, but it's for local testing.
    # We need a different setup to render to screen.
    
    # --- Pygame setup for human play ---
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(GameEnv.user_guide)
    
    while not done:
        # --- Action mapping for human keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Rendering ---
        # The observation is already the rendered frame
        # We just need to convert it back to a Pygame surface and blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # 30 FPS
        
    print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
    env.close()