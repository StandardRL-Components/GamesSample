import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. Press SHIFT to cycle tower types. Press SPACE to build a tower. Survive the waves!"
    )

    game_description = (
        "Defend your base from waves of enemies in this minimalist isometric tower defense game. Place towers strategically to survive."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_PATH = (50, 50, 70)
    COLOR_GRID = (40, 40, 60)
    COLOR_BASE = (0, 200, 100)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_ENEMY_HEALTH = (50, 200, 50)
    COLOR_ENEMY_HEALTH_BG = (100, 30, 30)
    COLOR_TEXT = (230, 230, 230)
    COLOR_CURSOR_VALID = (255, 255, 0, 150)
    COLOR_CURSOR_INVALID = (255, 0, 0, 150)
    
    TOWER_COLORS = [(255, 180, 0), (0, 180, 255)] # Cannon, Laser

    # Game States
    STATE_PREP = 0
    STATE_WAVE = 1
    STATE_GAME_OVER = 2
    STATE_WIN = 3
    
    # Grid & Iso
    GRID_W, GRID_H = 22, 14
    TILE_W, TILE_H = 32, 16
    ISO_ORIGIN_X, ISO_ORIGIN_Y = 640 // 2, 60

    # Game Params
    MAX_WAVES = 20
    MAX_STEPS = 30 * 180 # 3 minutes at 30fps
    INITIAL_MONEY = 150
    INITIAL_BASE_HEALTH = 100
    PREP_PHASE_DURATION = 150 # 5 seconds at 30fps
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("sans-serif", 18)
        self.font_m = pygame.font.SysFont("sans-serif", 24)
        self.font_l = pygame.font.SysFont("sans-serif", 48)

        # Game path & tower spots
        self.path_nodes = [
            (-1, 5), (3, 5), (3, 9), (7, 9), (7, 2), (13, 2), 
            (13, 11), (18, 11), (18, 7), (22, 7)
        ]
        self.tower_spots = [
            (2, 7), (4, 7), (5, 9), (5, 4), (7, 4), (7, 7), (9, 2),
            (11, 2), (11, 4), (13, 4), (13, 9), (15, 11), (16, 9), (18, 9)
        ]
        self.base_pos = (21, 7)

        # Tower definitions: [cost, damage, range, fire_rate, proj_speed]
        self.tower_defs = {
            0: {"name": "Cannon", "cost": 50, "dmg": 25, "range": 3.5, "rate": 45, "proj_spd": 5},
            1: {"name": "Laser", "cost": 75, "dmg": 8, "range": 4.5, "rate": 10, "proj_spd": 20},
        }

        # The reset method will be called by the environment wrapper, so we don't need to call it here.
        # self.reset() 
        # self.validate_implementation() # This is also typically not called in __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.total_reward = 0
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.money = self.INITIAL_MONEY
        self.current_wave = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.game_state = self.STATE_PREP
        self.prep_timer = self.PREP_PHASE_DURATION
        
        self.cursor_idx = 0
        self.selected_tower_type = 0
        
        self.last_shift_press = False
        self.last_space_press = False
        self.last_move_press = False

        self._start_next_wave_prep()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        if self.game_state == self.STATE_PREP:
            # Cycle tower types (on press)
            if shift_held and not self.last_shift_press:
                self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_defs)
            
            # Move cursor (on press)
            if movement != 0 and not self.last_move_press:
                if movement == 1: # Up
                    self.cursor_idx = (self.cursor_idx - 1 + len(self.tower_spots)) % len(self.tower_spots)
                elif movement == 2: # Down
                    self.cursor_idx = (self.cursor_idx + 1) % len(self.tower_spots)
                elif movement == 3: # Left
                    self.cursor_idx = (self.cursor_idx - 1 + len(self.tower_spots)) % len(self.tower_spots)
                elif movement == 4: # Right
                    self.cursor_idx = (self.cursor_idx + 1) % len(self.tower_spots)
            
            # Place tower (on press)
            if space_held and not self.last_space_press:
                cursor_pos = self.tower_spots[self.cursor_idx]
                is_occupied = any(t['pos'] == cursor_pos for t in self.towers)
                cost = self.tower_defs[self.selected_tower_type]['cost']
                if not is_occupied and self.money >= cost:
                    self.money -= cost
                    new_tower = {
                        "pos": cursor_pos,
                        "type": self.selected_tower_type,
                        "cooldown": 0,
                        "fire_flash": 0
                    }
                    self.towers.append(new_tower)

        self.last_shift_press = shift_held
        self.last_space_press = space_held
        self.last_move_press = movement != 0

        # --- Update Game Logic ---
        if self.game_state == self.STATE_PREP:
            self.prep_timer -= 1
            if self.prep_timer <= 0:
                self._start_wave_active()

        elif self.game_state == self.STATE_WAVE:
            # Update Towers
            for tower in self.towers:
                tower['cooldown'] = max(0, tower['cooldown'] - 1)
                tower['fire_flash'] = max(0, tower['fire_flash'] - 1)
                if tower['cooldown'] == 0:
                    target = self._find_target(tower)
                    if target:
                        self._fire_projectile(tower, target)

            # Update Projectiles
            projectiles_to_remove = []
            for i, p in enumerate(self.projectiles):
                p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
                
                hit_enemy = None
                for enemy in self.enemies:
                    dist_sq = (p['pos'][0] - enemy['pixel_pos'][0])**2 + (p['pos'][1] - enemy['pixel_pos'][1])**2
                    if dist_sq < 10**2: # Hit radius
                        hit_enemy = enemy
                        break
                
                if hit_enemy:
                    hit_enemy['health'] -= p['dmg']
                    reward += 0.1 # Hit reward
                    self._create_particles(p['pos'], self.TOWER_COLORS[p['type']], 5)
                    projectiles_to_remove.append(i)
                elif not self.screen.get_rect().collidepoint(p['pos']):
                    projectiles_to_remove.append(i)

            for i in sorted(projectiles_to_remove, reverse=True):
                del self.projectiles[i]

            # Update Enemies
            enemies_to_remove = []
            for i, enemy in enumerate(self.enemies):
                if enemy['health'] <= 0:
                    enemies_to_remove.append(i)
                    self.money += enemy['value']
                    reward += 1.0 # Kill reward
                    continue

                # Movement
                if enemy['path_idx'] < len(self.path_nodes) - 1:
                    target_node = self.path_nodes[enemy['path_idx'] + 1]
                    target_pixel = self._iso_to_screen(*target_node)
                    
                    direction = (target_pixel[0] - enemy['pixel_pos'][0], target_pixel[1] - enemy['pixel_pos'][1])
                    dist = math.hypot(*direction)
                    
                    if dist < enemy['speed']:
                        enemy['pixel_pos'] = target_pixel
                        enemy['path_idx'] += 1
                    else:
                        norm_dir = (direction[0] / dist, direction[1] / dist)
                        enemy['pixel_pos'] = (enemy['pixel_pos'][0] + norm_dir[0] * enemy['speed'],
                                              enemy['pixel_pos'][1] + norm_dir[1] * enemy['speed'])
                else: # Reached base
                    self.base_health -= enemy['base_dmg']
                    reward -= 0.1 * enemy['base_dmg'] # Base damage penalty
                    enemies_to_remove.append(i)
                    self._create_particles(self._iso_to_screen(*self.base_pos), self.COLOR_BASE_DMG, 20)
            
            for i in sorted(enemies_to_remove, reverse=True):
                del self.enemies[i]

            # Check for wave end
            if not self.enemies and self.spawn_idx >= self.wave_enemy_count:
                reward += 5 # Wave survival reward
                self.money += 100 + self.current_wave * 10
                self._start_next_wave_prep()

            # Spawning
            self.spawn_timer -= 1
            if self.spawn_timer <= 0 and self.spawn_idx < self.wave_enemy_count:
                self._spawn_enemy()
                self.spawn_timer = self.spawn_interval

        # Update Particles
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                particles_to_remove.append(i)
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.base_health <= 0:
            terminated = True
            reward -= 100
            self.game_state = self.STATE_GAME_OVER
        elif self.current_wave > self.MAX_WAVES:
            terminated = True
            reward += 100
            self.game_state = self.STATE_WIN
        
        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.total_reward += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Helper Methods ---
    def _iso_to_screen(self, x, y):
        return (self.ISO_ORIGIN_X + (x - y) * self.TILE_W / 2,
                self.ISO_ORIGIN_Y + (x + y) * self.TILE_H / 2)

    def _start_next_wave_prep(self):
        if self.current_wave >= self.MAX_WAVES:
            self.game_state = self.STATE_WIN
            return
        self.current_wave += 1
        self.game_state = self.STATE_PREP
        self.prep_timer = self.PREP_PHASE_DURATION

    def _start_wave_active(self):
        self.game_state = self.STATE_WAVE
        self.wave_enemy_count = 2 + self.current_wave * 2
        self.spawn_idx = 0
        self.spawn_interval = max(10, 60 - self.current_wave * 2)
        self.spawn_timer = 0

    def _spawn_enemy(self):
        start_node = self.path_nodes[0]
        difficulty_mod = 1 + (self.current_wave - 1) * 0.05
        
        enemy = {
            'pos': start_node,
            'pixel_pos': self._iso_to_screen(*start_node),
            'health': 100 * difficulty_mod,
            'max_health': 100 * difficulty_mod,
            'speed': 1.0 * difficulty_mod,
            'path_idx': 0,
            'base_dmg': 10,
            'value': 10 + self.current_wave,
        }
        self.enemies.append(enemy)
        self.spawn_idx += 1

    def _find_target(self, tower):
        t_def = self.tower_defs[tower['type']]
        t_pos = self._iso_to_screen(*tower['pos'])
        
        in_range_enemies = []
        for enemy in self.enemies:
            if enemy['path_idx'] >= len(self.path_nodes) - 1:
                continue
            e_pos = enemy['pixel_pos']
            dist_sq = (t_pos[0] - e_pos[0])**2 + (t_pos[1] - e_pos[1])**2
            if dist_sq < (t_def['range'] * self.TILE_W)**2:
                in_range_enemies.append(enemy)
        
        if not in_range_enemies:
            return None
        
        return max(in_range_enemies, key=lambda e: (e['path_idx'], -math.hypot(e['pixel_pos'][0] - self._iso_to_screen(*self.path_nodes[e['path_idx']+1])[0], e['pixel_pos'][1] - self._iso_to_screen(*self.path_nodes[e['path_idx']+1])[1])))

    def _fire_projectile(self, tower, target):
        t_def = self.tower_defs[tower['type']]
        tower['cooldown'] = t_def['rate']
        tower['fire_flash'] = 5
        
        start_pos = self._iso_to_screen(*tower['pos'])
        start_pos = (start_pos[0], start_pos[1] - self.TILE_H)
        
        target_pos = target['pixel_pos']
        dist = math.hypot(target_pos[0] - start_pos[0], target_pos[1] - start_pos[1])
        time_to_hit = dist / t_def['proj_spd']
        
        future_pos = list(target['pixel_pos'])
        rem_dist = target['speed'] * time_to_hit
        path_idx = target['path_idx']
        while rem_dist > 0 and path_idx < len(self.path_nodes) - 1:
            next_node_pos = self._iso_to_screen(*self.path_nodes[path_idx + 1])
            seg_dist = math.hypot(next_node_pos[0] - future_pos[0], next_node_pos[1] - future_pos[1])
            if seg_dist > rem_dist:
                direction = (next_node_pos[0] - future_pos[0], next_node_pos[1] - future_pos[1])
                norm_dir = (direction[0] / seg_dist, direction[1] / seg_dist) if seg_dist > 0 else (0,0)
                future_pos[0] += norm_dir[0] * rem_dist
                future_pos[1] += norm_dir[1] * rem_dist
                rem_dist = 0
            else:
                rem_dist -= seg_dist
                future_pos = list(next_node_pos)
                path_idx += 1
        
        direction = (future_pos[0] - start_pos[0], future_pos[1] - start_pos[1])
        dist = math.hypot(*direction)
        norm_dir = (direction[0]/dist, direction[1]/dist) if dist > 0 else (0,0)

        projectile = {
            'pos': start_pos,
            'vel': (norm_dir[0] * t_def['proj_spd'], norm_dir[1] * t_def['proj_spd']),
            'dmg': t_def['dmg'],
            'type': tower['type']
        }
        self.projectiles.append(projectile)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': (math.cos(angle) * speed, math.sin(angle) * speed),
                'color': color,
                'lifespan': self.np_random.integers(10, 20)
            })

    def _get_info(self):
        return {
            "score": self.money,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_iso_poly(self, surface, color, x, y, w, h, z):
        points = [
            self._iso_to_screen(x, y + h),
            self._iso_to_screen(x + w, y + h),
            self._iso_to_screen(x + w, y),
            self._iso_to_screen(x, y),
        ]
        points = [(p[0], p[1] - z) for p in points]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_iso_cube(self, surface, color, x, y, size, height):
        top_color = tuple(min(255, c + 40) for c in color)
        side_color_right = color
        side_color_left = tuple(max(0, c - 40) for c in color)
        
        p = [self._iso_to_screen(x, y), self._iso_to_screen(x+size, y), self._iso_to_screen(x+size, y+size), self._iso_to_screen(x, y+size)]
        
        p_up = [(pi[0], pi[1]-height) for pi in p]

        pygame.gfxdraw.filled_polygon(surface, [p[3], p_up[3], p_up[0], p[0]], side_color_left)
        pygame.gfxdraw.filled_polygon(surface, [p[2], p_up[2], p_up[1], p[1]], side_color_right)
        pygame.gfxdraw.filled_polygon(surface, p_up, top_color)
        
    def _render_game(self):
        # Draw path
        for i in range(len(self.path_nodes) - 1):
            p1 = self._iso_to_screen(*self.path_nodes[i])
            p2 = self._iso_to_screen(*self.path_nodes[i+1])
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.TILE_H + 4)
        
        # Draw tower spots
        for spot in self.tower_spots:
            self._draw_iso_poly(self.screen, self.COLOR_GRID, spot[0], spot[1], 1, 1, 0)

        # Draw Base
        self._draw_iso_cube(self.screen, self.COLOR_BASE, self.base_pos[0], self.base_pos[1], 1, self.TILE_H)

        # Draw Towers
        for tower in self.towers:
            pos = self._iso_to_screen(*tower['pos'])
            t_def = self.tower_defs[tower['type']]
            color = self.TOWER_COLORS[tower['type']]
            self._draw_iso_cube(self.screen, (80,80,90), tower['pos'][0], tower['pos'][1], 1, self.TILE_H / 2)
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1] - self.TILE_H/2)), 8)
            if tower['fire_flash'] > 0:
                flash_color = (255, 255, 255, 200)
                radius = 12 * (1 - tower['fire_flash']/5)
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(s, int(radius), int(radius), int(radius), flash_color)
                self.screen.blit(s, (int(pos[0] - radius), int(pos[1] - self.TILE_H/2 - radius)))

        # Draw Enemies
        for enemy in self.enemies:
            pos = enemy['pixel_pos']
            rect = pygame.Rect(pos[0] - 8, pos[1] - 16, 16, 16)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            
            health_ratio = enemy['health'] / enemy['max_health']
            bar_w = 20
            bar_h = 4
            bar_x = pos[0] - bar_w / 2
            bar_y = pos[1] - 25
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, (bar_x, bar_y, max(0, bar_w * health_ratio), bar_h))

        # Draw Projectiles
        for p in self.projectiles:
            color = self.TOWER_COLORS[p['type']]
            if p['type'] == 0: # Cannon
                pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), 4)
            else: # Laser
                end_pos = (p['pos'][0] - p['vel'][0] * 0.5, p['pos'][1] - p['vel'][1] * 0.5)
                pygame.draw.line(self.screen, color, p['pos'], end_pos, 3)

        # Draw Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20))))
            color_with_alpha = (*p['color'], alpha)
            size = int(p['lifespan'] / 4)
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, size, size, size, color_with_alpha)
                self.screen.blit(temp_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

        # Draw Cursor
        if self.game_state == self.STATE_PREP:
            cursor_pos_grid = self.tower_spots[self.cursor_idx]
            cursor_pos_screen = self._iso_to_screen(*cursor_pos_grid)
            
            is_occupied = any(t['pos'] == cursor_pos_grid for t in self.towers)
            cost = self.tower_defs[self.selected_tower_type]['cost']
            can_afford = self.money >= cost
            
            color = self.COLOR_CURSOR_VALID if not is_occupied and can_afford else self.COLOR_CURSOR_INVALID
            
            s = pygame.Surface((self.TILE_W+2, self.TILE_H+2), pygame.SRCALPHA)
            pygame.draw.polygon(s, color, [(0, self.TILE_H/2), (self.TILE_W/2, 0), (self.TILE_W, self.TILE_H/2), (self.TILE_W/2, self.TILE_H)])
            self.screen.blit(s, (cursor_pos_screen[0] - self.TILE_W/2, cursor_pos_screen[1] - self.TILE_H/2))

    def _render_ui(self):
        # Top Left: Status
        health_text = self.font_m.render(f"Base Health: {int(self.base_health)}%", True, self.COLOR_TEXT)
        money_text = self.font_m.render(f"Money: ${self.money}", True, self.COLOR_TEXT)
        wave_text = self.font_m.render(f"Wave: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(money_text, (10, 35))
        self.screen.blit(wave_text, (10, 60))

        # Top Right: Tower Selection
        if self.game_state == self.STATE_PREP:
            t_def = self.tower_defs[self.selected_tower_type]
            tower_name = self.font_m.render(f"{t_def['name']}", True, self.TOWER_COLORS[self.selected_tower_type])
            tower_cost = self.font_s.render(f"Cost: ${t_def['cost']}", True, self.COLOR_TEXT)
            tower_dmg = self.font_s.render(f"Dmg: {t_def['dmg']}", True, self.COLOR_TEXT)
            tower_rate = self.font_s.render(f"Rate: {60/(t_def['rate']/30):.1f}/s", True, self.COLOR_TEXT)
            
            self.screen.blit(tower_name, (630 - tower_name.get_width(), 10))
            self.screen.blit(tower_cost, (630 - tower_cost.get_width(), 35))
            self.screen.blit(tower_dmg, (630 - tower_dmg.get_width(), 50))
            self.screen.blit(tower_rate, (630 - tower_rate.get_width(), 65))
            
            prep_time_text = self.font_m.render(f"Wave starting in: {math.ceil(self.prep_timer / 30)}s", True, self.COLOR_TEXT)
            self.screen.blit(prep_time_text, (self.screen.get_width()//2 - prep_time_text.get_width()//2, 10))

        # Game Over / Win message
        if self.game_state == self.STATE_GAME_OVER:
            text = self.font_l.render("GAME OVER", True, self.COLOR_ENEMY)
            self.screen.blit(text, (self.screen.get_width()//2 - text.get_width()//2, self.screen.get_height()//2 - text.get_height()//2))
        elif self.game_state == self.STATE_WIN:
            text = self.font_l.render("YOU WIN!", True, self.COLOR_BASE)
            self.screen.blit(text, (self.screen.get_width()//2 - text.get_width()//2, self.screen.get_height()//2 - text.get_height()//2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This is a helper for development and is not required by the API.
        # It's good practice to have such checks.
        print("Attempting to validate implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    
    # Run validation
    try:
        env.validate_implementation()
    except Exception as e:
        print(f"Validation failed: {e}")
        
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    # Key press state for manual play
    last_key_states = {
        pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False,
        pygame.K_SPACE: False, pygame.K_LSHIFT: False, pygame.K_RSHIFT: False
    }

    print(GameEnv.user_guide)

    while running:
        # Single-press logic for manual play
        movement = 0
        space_pressed = 0
        shift_pressed = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                # Map single key presses to actions
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_pressed = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    shift_pressed = 1
        
        # The environment's step function handles continuous holds,
        # but for manual play, we might want to simulate holds.
        # For simplicity with the current env logic, we'll use presses.
        # The environment logic is based on holds, so we'll map held keys.
        keys = pygame.key.get_pressed()
        
        mov_action = 0
        if keys[pygame.K_UP]: mov_action = 1
        elif keys[pygame.K_DOWN]: mov_action = 2
        elif keys[pygame.K_LEFT]: mov_action = 3
        elif keys[pygame.K_RIGHT]: mov_action = 4

        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Total Reward: {total_reward:.2f}, Final Info: {info}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Match the intended FPS

    env.close()