
# Generated: 2025-08-27T17:13:56.999530
# Source Brief: brief_01466.md
# Brief Index: 1466

        
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
        "Controls: Arrows to move cursor, Space to place selected tower, Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing musical towers in this rhythmic tower defense game. Survive all 5 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 30 * 120 # 120 seconds at 30fps
        self.MAX_WAVES = 5

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PATH = (60, 60, 75)
        self.COLOR_BASE = (0, 150, 255)
        self.COLOR_ENEMY_1 = (255, 80, 80)
        self.COLOR_TOWER_1 = (0, 255, 150)
        self.COLOR_TOWER_2 = (255, 200, 0)
        self.COLOR_PROJECTILE = (200, 220, 255)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (50, 50, 60, 180)
        
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
        self.font_s = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 24, bold=True)

        # Game assets (defined in code)
        self.path_waypoints = self._define_path()
        self.tower_types = self._define_towers()
        self.wave_definitions = self._define_waves()

        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.time_to_next_wave = 0
        self.time_to_next_spawn = 0
        self.spawn_index_in_wave = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 100
        self.resources = 100
        self.current_wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.time_to_next_wave = 150 # 5 seconds at 30fps
        self.time_to_next_spawn = 0
        self.spawn_index_in_wave = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.001 # Small penalty for time passing
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle player input
        self._handle_input(movement, space_held, shift_held)
        
        # Update game state
        reward += self._update_waves()
        self._update_towers()
        hit_reward = self._update_projectiles()
        kill_reward, resources_from_kills = self._check_enemy_deaths()
        reward += hit_reward + kill_reward
        self.resources += resources_from_kills
        self._update_enemies()
        self._update_particles()
        
        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.base_health <= 0 or self.steps >= self.MAX_STEPS:
                reward -= 100 # Loss penalty
            elif self.current_wave > self.MAX_WAVES:
                reward += 100 # Win bonus
            self.score += reward

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.GRID_ROWS - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.GRID_COLS - 1: self.cursor_pos[0] += 1
            
        # Place tower
        if space_held and not self.prev_space_held:
            tower_type = self.tower_types[self.selected_tower_type]
            if self.resources >= tower_type['cost']:
                is_on_path = any(self.cursor_pos == [c, r] for c, r in self.path_waypoints)
                is_occupied = any(self.cursor_pos == t['grid_pos'] for t in self.towers)
                if not is_on_path and not is_occupied:
                    self.resources -= tower_type['cost']
                    self.towers.append({
                        'grid_pos': list(self.cursor_pos),
                        'type_id': self.selected_tower_type,
                        'cooldown': 0
                    })
                    # sfx: tower_place.wav
                    self._create_particles(self.cursor_pos[0]*self.CELL_SIZE + self.CELL_SIZE/2, self.cursor_pos[1]*self.CELL_SIZE + self.CELL_SIZE/2, self.COLOR_TOWER_1, 10)

        # Cycle tower type
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_types)
            # sfx: ui_cycle.wav

    def _update_waves(self):
        reward = 0
        # If all enemies from current wave are spawned and defeated
        if self.spawn_index_in_wave >= len(self.wave_definitions[self.current_wave-1]) and not self.enemies and self.current_wave > 0:
            if self.current_wave <= self.MAX_WAVES:
                self.time_to_next_wave = 150 # 5s
                reward += 10 # Wave clear bonus
                if self.current_wave == self.MAX_WAVES:
                    self.current_wave += 1 # Win condition met
                    return reward
            self.spawn_index_in_wave = 0 # Reset for next wave

        if self.time_to_next_wave > 0:
            self.time_to_next_wave -= 1
            if self.time_to_next_wave == 0 and self.current_wave < self.MAX_WAVES:
                self.current_wave += 1
                self.spawn_index_in_wave = 0
        elif self.current_wave > 0 and self.current_wave <= self.MAX_WAVES:
            self.time_to_next_spawn -= 1
            if self.time_to_next_spawn <= 0 and self.spawn_index_in_wave < len(self.wave_definitions[self.current_wave-1]):
                enemy_type = self.wave_definitions[self.current_wave-1][self.spawn_index_in_wave]
                self._spawn_enemy(enemy_type)
                self.spawn_index_in_wave += 1
                self.time_to_next_spawn = 30 # 1s between spawns
        return reward

    def _spawn_enemy(self, enemy_type):
        start_pos = [p * self.CELL_SIZE + self.CELL_SIZE/2 for p in self.path_waypoints[0]]
        self.enemies.append({
            'pos': start_pos,
            'health': 50 + (self.current_wave - 1) * 10,
            'max_health': 50 + (self.current_wave - 1) * 10,
            'speed': 1.0 + (self.current_wave - 1) * 0.1,
            'waypoint_idx': 1,
            'value': 5
        })

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                tower_info = self.tower_types[tower['type_id']]
                tower_pos = [p * self.CELL_SIZE + self.CELL_SIZE/2 for p in tower['grid_pos']]
                
                target = None
                min_dist = float('inf')
                for enemy in self.enemies:
                    dist = math.hypot(enemy['pos'][0] - tower_pos[0], enemy['pos'][1] - tower_pos[1])
                    if dist <= tower_info['range'] and dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    self.projectiles.append({
                        'pos': list(tower_pos),
                        'target': target,
                        'speed': tower_info['proj_speed'],
                        'damage': tower_info['damage']
                    })
                    tower['cooldown'] = tower_info['fire_rate']
                    # sfx: laser_shoot.wav
                    self._create_particles(tower_pos[0], tower_pos[1], self.COLOR_PROJECTILE, 3, life=5, speed=2)

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj['target']['pos']
            dx, dy = target_pos[0] - proj['pos'][0], target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < proj['speed']:
                proj['target']['health'] -= proj['damage']
                reward += 0.1 # Hit reward
                self.projectiles.remove(proj)
                # sfx: enemy_hit.wav
                self._create_particles(target_pos[0], target_pos[1], (255, 255, 100), 5, life=8)
            else:
                proj['pos'][0] += (dx / dist) * proj['speed']
                proj['pos'][1] += (dy / dist) * proj['speed']
        return reward

    def _check_enemy_deaths(self):
        reward = 0
        resources = 0
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                reward += 1.0 # Kill reward
                resources += enemy['value']
                self.enemies.remove(enemy)
                # sfx: explosion.wav
                self._create_particles(enemy['pos'][0], enemy['pos'][1], self.COLOR_ENEMY_1, 20, life=20, speed=4)
        return reward, resources

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['waypoint_idx'] >= len(self.path_waypoints):
                self.base_health -= 10
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                continue
            
            target_wp = [p * self.CELL_SIZE + self.CELL_SIZE/2 for p in self.path_waypoints[enemy['waypoint_idx']]]
            dx, dy = target_wp[0] - enemy['pos'][0], target_wp[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < enemy['speed']:
                enemy['waypoint_idx'] += 1
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.current_wave > self.MAX_WAVES

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw path
        if len(self.path_waypoints) > 1:
            path_pixels = [[c*self.CELL_SIZE+self.CELL_SIZE/2, r*self.CELL_SIZE+self.CELL_SIZE/2] for c,r in self.path_waypoints]
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, path_pixels, self.CELL_SIZE)

        # Draw base (end of path)
        base_pos = self.path_waypoints[-1]
        base_rect = pygame.Rect(base_pos[0]*self.CELL_SIZE, base_pos[1]*self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.gfxdraw.rectangle(self.screen, base_rect, (*self.COLOR_BASE, 150))

        # Draw towers
        for tower in self.towers:
            tower_info = self.tower_types[tower['type_id']]
            pos_px = [int(p * self.CELL_SIZE + self.CELL_SIZE/2) for p in tower['grid_pos']]
            color = self.COLOR_TOWER_1 if tower['type_id'] == 0 else self.COLOR_TOWER_2
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], 12, color)
            pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], 12, color)

        # Draw projectiles
        for proj in self.projectiles:
            pos_px = [int(p) for p in proj['pos']]
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], 3, self.COLOR_PROJECTILE)
            
        # Draw enemies
        for enemy in self.enemies:
            pos_px = [int(p) for p in enemy['pos']]
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], 10, self.COLOR_ENEMY_1)
            pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], 10, self.COLOR_ENEMY_1)
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_w = int(20 * health_ratio)
            pygame.draw.rect(self.screen, (0,255,0), (pos_px[0]-10, pos_px[1]-15, bar_w, 3))
            pygame.draw.rect(self.screen, (255,0,0), (pos_px[0]-10+bar_w, pos_px[1]-15, 20-bar_w, 3))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

        # Draw cursor and range indicator
        cursor_px_x = self.cursor_pos[0] * self.CELL_SIZE
        cursor_px_y = self.cursor_pos[1] * self.CELL_SIZE
        cursor_rect = pygame.Rect(cursor_px_x, cursor_px_y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)
        
        tower_info = self.tower_types[self.selected_tower_type]
        range_color = (*self.COLOR_CURSOR, 50)
        pygame.gfxdraw.filled_circle(self.screen, cursor_px_x + self.CELL_SIZE//2, cursor_px_y + self.CELL_SIZE//2, tower_info['range'], range_color)

    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, font, color, x, y, align="left"):
            text_surf = font.render(text, True, (0,0,0))
            text_rect = text_surf.get_rect()
            if align == "right": text_rect.topright = (x, y)
            elif align == "center": text_rect.midtop = (x, y)
            else: text_rect.topleft = (x, y)
            self.screen.blit(text_surf, text_rect.move(1, 1))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, text_rect)

        # Top bar
        bar_rect = pygame.Rect(0, 0, self.WIDTH, 30)
        s = pygame.Surface((self.WIDTH, 30), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0,0))
        
        wave_text = f"Wave: {self.current_wave if self.current_wave <= self.MAX_WAVES else 'WIN!'}/{self.MAX_WAVES}"
        draw_text(wave_text, self.font_s, self.COLOR_UI_TEXT, 10, 5)
        
        draw_text(f"Resources: ${int(self.resources)}", self.font_s, self.COLOR_UI_TEXT, 200, 5)
        
        draw_text(f"Score: {int(self.score)}", self.font_s, self.COLOR_UI_TEXT, self.WIDTH - 10, 5, align="right")

        # Base health bar
        base_pos_px = [p*self.CELL_SIZE for p in self.path_waypoints[-1]]
        health_ratio = max(0, self.base_health / 100)
        bar_w = int(self.CELL_SIZE * health_ratio)
        pygame.draw.rect(self.screen, (0,255,0), (base_pos_px[0], base_pos_px[1]-8, bar_w, 5))
        pygame.draw.rect(self.screen, (255,0,0), (base_pos_px[0]+bar_w, base_pos_px[1]-8, self.CELL_SIZE-bar_w, 5))

        # Selected tower info
        tower_info = self.tower_types[self.selected_tower_type]
        tower_text = f"Selected: {tower_info['name']} (Cost: ${tower_info['cost']})"
        draw_text(tower_text, self.font_s, self.COLOR_UI_TEXT, 10, self.HEIGHT - 22)
        
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,150))
            self.screen.blit(s, (0,0))
            msg = "YOU WIN!" if self.current_wave > self.MAX_WAVES else "GAME OVER"
            draw_text(msg, self.font_m, (255,255,255), self.WIDTH/2, self.HEIGHT/2 - 20, align="center")

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "base_health": self.base_health, "wave": self.current_wave}

    def _define_path(self):
        path = []
        path.append((-1, 4))
        for i in range(5): path.append((i, 4))
        for i in range(4, 7): path.append((4, i))
        for i in range(4, -1, -1): path.append((i, 7))
        for i in range(7, 4, -1): path.append((0, i))
        for i in range(0, 11): path.append((i, 5))
        for i in range(5, 2, -1): path.append((10, i))
        for i in range(10, 16): path.append((i, 2))
        return path

    def _define_towers(self):
        return [
            {'name': 'Pulse Cannon', 'cost': 25, 'range': 80, 'damage': 10, 'fire_rate': 30, 'proj_speed': 8},
            {'name': 'Heavy Laser', 'cost': 60, 'range': 120, 'damage': 35, 'fire_rate': 75, 'proj_speed': 12},
        ]
    
    def _define_waves(self):
        waves = []
        for i in range(self.MAX_WAVES):
            num_enemies = 5 + i * 3
            waves.append([0] * num_enemies) # All enemies are type 0 for simplicity
        return waves
    
    def _create_particles(self, x, y, color, count, life=15, speed=3, size=3):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            s = self.np_random.uniform(0.5, 1.0) * speed
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * s, math.sin(angle) * s],
                'life': self.np_random.integers(life//2, life),
                'max_life': life,
                'color': color,
                'size': size
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Playable Demo ---
    # This part will not run in a headless environment but is useful for testing.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Tower Defense")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            movement = 0 # no-op
            space_held = False
            shift_held = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            if keys[pygame.K_DOWN]: movement = 2
            if keys[pygame.K_LEFT]: movement = 3
            if keys[pygame.K_RIGHT]: movement = 4
            if keys[pygame.K_SPACE]: space_held = True
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True
            
            action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Run at 30 FPS
            
    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("This is expected in a headless environment. The environment itself is functional.")
    finally:
        pygame.quit()