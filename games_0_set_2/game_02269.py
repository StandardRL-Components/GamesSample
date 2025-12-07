
# Generated: 2025-08-28T04:19:37.835583
# Source Brief: brief_02269.md
# Brief Index: 2269

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Hold Shift to cycle through tower types. Press Space to build the selected tower."
    )

    game_description = (
        "Defend your crystal from waves of enemies in this isometric tower defense game. "
        "Strategically place towers to survive all 15 waves."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 15, 15
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = 20, 10
        self.ORIGIN_X, self.ORIGIN_Y = self.WIDTH // 2, 80

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_GRID = (25, 35, 45)
        self.COLOR_CRYSTAL = (0, 200, 255)
        self.COLOR_PLAYER_CURSOR = (255, 255, 0)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_BG = (80, 20, 20)
        self.COLOR_HEALTH_FG = (50, 220, 50)
        self.TOWER_COLORS = [
            (100, 255, 100),  # Basic
            (255, 150, 50),   # Rapid
            (150, 100, 255),  # Splash
            (50, 150, 255),   # Slow
        ]

        # Game constants
        self.MAX_WAVES = 15
        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps
        self.INITIAL_CASTLE_HEALTH = 200
        self.INITIAL_GOLD = 150
        self.WAVE_BASE_INTERVAL = 30 * 20 # 20 seconds
        self.WAVE_INTERVAL_DECREMENT = 30 # 1 second

        # Define tower stats
        self.TOWER_STATS = {
            0: {"name": "Cannon", "cost": 50, "dmg": 10, "range": 3, "rate": 30, "proj_speed": 8, "type": "single"},
            1: {"name": "Gatling", "cost": 80, "dmg": 4, "range": 2.5, "rate": 8, "proj_speed": 10, "type": "single"},
            2: {"name": "Artillery", "cost": 120, "dmg": 25, "range": 4, "rate": 90, "proj_speed": 5, "type": "splash", "splash_radius": 1.5},
            3: {"name": "Frost", "cost": 70, "dmg": 2, "range": 2.5, "rate": 45, "proj_speed": 7, "type": "slow", "slow_factor": 0.5, "slow_duration": 60},
        }

        self.enemy_path = self._generate_path()
        self.buildable_tiles = self._get_buildable_tiles()

        # Initialize state variables
        self.reset()
        
        # Self-check
        self.validate_implementation()

    def _generate_path(self):
        path = []
        path.extend([(i, 0) for i in range(self.GRID_WIDTH - 1, 4, -1)])
        path.extend([(5, i) for i in range(1, self.GRID_HEIGHT - 5)])
        path.extend([(i, self.GRID_HEIGHT - 6) for i in range(5, self.GRID_WIDTH - 2)])
        path.extend([(self.GRID_WIDTH - 3, i) for i in range(self.GRID_HEIGHT - 6, self.GRID_HEIGHT - 1)])
        return path

    def _get_buildable_tiles(self):
        buildable = set()
        path_set = set(self.enemy_path)
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) not in path_set:
                    # Don't allow building too close to path
                    is_adjacent = False
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if (r + dr, c + dc) in path_set:
                                is_adjacent = True
                                break
                        if is_adjacent:
                            break
                    if not is_adjacent:
                        buildable.add((r, c))
        return buildable

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.castle_health = self.INITIAL_CASTLE_HEALTH
        self.gold = self.INITIAL_GOLD
        self.wave = 0
        self.wave_timer = 30 * 5 # 5s initial delay

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = (self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2)
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(self.metadata["render_fps"])
        self.steps += 1
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Game Logic Updates ---
        reward += self._update_waves()
        self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # Small passive reward for surviving
        reward += 0.001

        self.score += reward
        
        # --- Termination Check ---
        terminated = False
        if self.castle_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
        elif self.wave > self.MAX_WAVES:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            reward -= 50 # Penalty for timeout

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        new_cursor_pos = list(self.cursor_pos)
        if movement == 1: new_cursor_pos[0] -= 1  # Up
        elif movement == 2: new_cursor_pos[0] += 1  # Down
        elif movement == 3: new_cursor_pos[1] -= 1  # Left
        elif movement == 4: new_cursor_pos[1] += 1  # Right
        
        new_cursor_pos[0] = np.clip(new_cursor_pos[0], 0, self.GRID_HEIGHT - 1)
        new_cursor_pos[1] = np.clip(new_cursor_pos[1], 0, self.GRID_WIDTH - 1)
        self.cursor_pos = tuple(new_cursor_pos)

        # --- Tower Selection (on key press) ---
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_STATS)
            # sfx: UI_Bleep

        # --- Tower Placement (on key press) ---
        if space_held and not self.last_space_held:
            self._place_tower()
            # sfx: Build_Confirm or Build_Error

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _place_tower(self):
        stats = self.TOWER_STATS[self.selected_tower_type]
        is_occupied = any(t['grid_pos'] == self.cursor_pos for t in self.towers)
        
        if self.cursor_pos in self.buildable_tiles and not is_occupied and self.gold >= stats['cost']:
            self.gold -= stats['cost']
            screen_pos = self._iso_to_screen(*self.cursor_pos)
            new_tower = {
                "grid_pos": self.cursor_pos,
                "screen_pos": screen_pos,
                "type_id": self.selected_tower_type,
                "cooldown": 0,
                "target": None,
                "angle": -math.pi/2
            }
            self.towers.append(new_tower)
            # sfx: Build_Success

    def _update_waves(self):
        self.wave_timer -= 1
        if self.wave_timer <= 0 and self.wave <= self.MAX_WAVES:
            self.wave += 1
            if self.wave <= self.MAX_WAVES:
                self._spawn_wave()
                interval = self.WAVE_BASE_INTERVAL - (self.wave * self.WAVE_INTERVAL_DECREMENT)
                self.wave_timer = max(30 * 5, interval) # Min 5s interval
        return 0

    def _spawn_wave(self):
        num_enemies = 5 + self.wave * 2
        base_health = 20 + (self.wave - 1) * 8
        base_speed = 0.7 + (self.wave - 1) * 0.05
        
        for i in range(num_enemies):
            health = base_health * (1 + self.np_random.uniform(-0.1, 0.1))
            speed = base_speed * (1 + self.np_random.uniform(-0.1, 0.1))
            
            # Spawn enemies with a slight delay
            spawn_offset = -i * 15 # pixels
            start_pos = self._iso_to_screen(*self.enemy_path[0])
            start_pos = (start_pos[0] + spawn_offset, start_pos[1])

            self.enemies.append({
                "pos": list(start_pos),
                "health": health,
                "max_health": health,
                "speed": speed,
                "path_index": 0,
                "slow_timer": 0,
                "id": self.np_random.integers(1, 1_000_000)
            })
        # sfx: Wave_Start_Horn

    def _update_towers(self):
        for tower in self.towers:
            stats = self.TOWER_STATS[tower['type_id']]
            tower['cooldown'] = max(0, tower['cooldown'] - 1)

            # --- Target aquisition ---
            if tower['target'] is not None:
                # Check if target is still valid
                if tower['target'] not in self.enemies or \
                   np.linalg.norm(np.array(tower['screen_pos']) - np.array(tower['target']['pos'])) > stats['range'] * self.TILE_WIDTH_HALF * 2:
                    tower['target'] = None
            
            if tower['target'] is None:
                for enemy in self.enemies:
                    dist = np.linalg.norm(np.array(tower['screen_pos']) - np.array(enemy['pos']))
                    if dist <= stats['range'] * self.TILE_WIDTH_HALF * 2:
                        tower['target'] = enemy
                        break
            
            # --- Firing ---
            if tower['target'] is not None:
                # Turn to face target
                dx = tower['target']['pos'][0] - tower['screen_pos'][0]
                dy = tower['target']['pos'][1] - tower['screen_pos'][1]
                target_angle = math.atan2(dy, dx)
                
                # Smooth rotation
                angle_diff = (target_angle - tower['angle'] + math.pi) % (2 * math.pi) - math.pi
                tower['angle'] += np.clip(angle_diff, -0.2, 0.2)

                if tower['cooldown'] == 0 and abs(angle_diff) < 0.3:
                    tower['cooldown'] = stats['rate']
                    self.projectiles.append({
                        "pos": list(tower['screen_pos']),
                        "target": tower['target'],
                        "stats": stats
                    })
                    # sfx: Tower_Fire (different per type)
                    # Add muzzle flash particle
                    flash_end_pos = (tower['screen_pos'][0] + math.cos(tower['angle']) * 15,
                                     tower['screen_pos'][1] + math.sin(tower['angle']) * 15)
                    self.particles.append({"pos": flash_end_pos, "radius": 5, "life": 5, "color": (255, 255, 150)})


    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target_pos = proj['target']['pos']
            direction = np.array(target_pos) - np.array(proj['pos'])
            dist = np.linalg.norm(direction)
            
            if dist < proj['stats']['proj_speed']:
                # --- Hit ---
                self.projectiles.remove(proj)
                # sfx: Impact
                
                if proj['stats']['type'] == "single":
                    if proj['target'] in self.enemies:
                        proj['target']['health'] -= proj['stats']['dmg']
                
                elif proj['stats']['type'] == "splash":
                    for enemy in self.enemies:
                        if np.linalg.norm(np.array(enemy['pos']) - np.array(target_pos)) < proj['stats']['splash_radius'] * self.TILE_WIDTH_HALF * 2:
                            enemy['health'] -= proj['stats']['dmg']
                    # sfx: Explosion
                    self._create_explosion(target_pos, 30, 20)
                
                elif proj['stats']['type'] == "slow":
                    if proj['target'] in self.enemies:
                        proj['target']['health'] -= proj['stats']['dmg']
                        proj['target']['slow_timer'] = proj['stats']['slow_duration']
                    self._create_explosion(target_pos, 15, 10, self.TOWER_COLORS[3])
                
                # Check for kills (done in _update_enemies to handle all damage sources)
                
            else:
                # --- Move projectile ---
                direction = direction / dist
                proj['pos'][0] += direction[0] * proj['stats']['proj_speed']
                proj['pos'][1] += direction[1] * proj['stats']['proj_speed']
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            # --- Check for death ---
            if enemy['health'] <= 0:
                self.enemies.remove(enemy)
                self.gold += 5 + self.wave
                reward += 1
                # sfx: Enemy_Death
                self._create_explosion(enemy['pos'], 10, 15, self.COLOR_ENEMY)
                continue

            # --- Movement ---
            path_idx = enemy['path_index']
            if path_idx >= len(self.enemy_path):
                self.enemies.remove(enemy)
                self.castle_health -= 10
                reward -= 5
                # sfx: Crystal_Damage
                self._create_explosion(self._iso_to_screen(self.GRID_WIDTH - 1, self.GRID_HEIGHT - 1), 30, 15, self.COLOR_CRYSTAL)
                continue
            
            target_node = self.enemy_path[path_idx]
            target_pos = self._iso_to_screen(*target_node)
            
            direction = np.array(target_pos) - np.array(enemy['pos'])
            dist = np.linalg.norm(direction)

            current_speed = enemy['speed']
            if enemy['slow_timer'] > 0:
                enemy['slow_timer'] -= 1
                current_speed *= self.TOWER_STATS[3]['slow_factor']
            
            if dist < current_speed:
                enemy['path_index'] += 1
            else:
                direction = direction / dist
                enemy['pos'][0] += direction[0] * current_speed
                enemy['pos'][1] += direction[1] * current_speed
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            p['radius'] *= 0.95
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, num_particles, max_life, color=(255, 200, 100)):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": list(pos),
                "vel": velocity,
                "radius": self.np_random.uniform(2, 6),
                "life": self.np_random.integers(max_life // 2, max_life),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and path
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color = self.COLOR_GRID
                if (r, c) in self.enemy_path:
                    color = self.COLOR_PATH
                elif (r, c) in self.buildable_tiles:
                    color = (45, 60, 75)
                
                points = [
                    self._iso_to_screen(c, r),
                    self._iso_to_screen(c + 1, r),
                    self._iso_to_screen(c + 1, r + 1),
                    self._iso_to_screen(c, r + 1)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BG)

        # Draw crystal
        crystal_pos = self._iso_to_screen(self.GRID_WIDTH-1.5, self.GRID_HEIGHT-1.5)
        pygame.draw.circle(self.screen, self.COLOR_CRYSTAL, crystal_pos, 15)
        pygame.gfxdraw.aacircle(self.screen, int(crystal_pos[0]), int(crystal_pos[1]), 15, self.COLOR_CRYSTAL)
        
        # Draw towers
        for tower in self.towers:
            pos = tower['screen_pos']
            stats = self.TOWER_STATS[tower['type_id']]
            color = self.TOWER_COLORS[tower['type_id']]
            
            # Base
            pygame.draw.circle(self.screen, (30,30,40), pos, 8)
            pygame.draw.circle(self.screen, color, pos, 7)
            
            # Barrel
            end_x = pos[0] + math.cos(tower['angle']) * 12
            end_y = pos[1] + math.sin(tower['angle']) * 12
            pygame.draw.line(self.screen, (200,200,200), pos, (end_x, end_y), 3)

        # Draw projectiles
        for proj in self.projectiles:
            color = self.TOWER_COLORS[proj['stats']['cost'] > 100] # Simple color logic
            if proj['stats']['type'] == 'slow': color = self.TOWER_COLORS[3]
            pygame.draw.circle(self.screen, color, proj['pos'], 3)

        # Draw enemies
        for enemy in sorted(self.enemies, key=lambda e: e['pos'][1]):
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 6)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)
            
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_len = 12
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (pos[0] - bar_len/2, pos[1] - 12, bar_len, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (pos[0] - bar_len/2, pos[1] - 12, bar_len * health_pct, 3))
            
            if enemy['slow_timer'] > 0:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.TOWER_COLORS[3])

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                alpha_color = p['color'] + (int(255 * (p['life'] / 10)),)
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, alpha_color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw cursor
        cursor_points = [
            self._iso_to_screen(*self.cursor_pos),
            self._iso_to_screen(self.cursor_pos[1] + 1, self.cursor_pos[0]),
            self._iso_to_screen(self.cursor_pos[1] + 1, self.cursor_pos[0] + 1),
            self._iso_to_screen(self.cursor_pos[1], self.cursor_pos[0] + 1)
        ]
        pygame.draw.lines(self.screen, self.COLOR_PLAYER_CURSOR, True, cursor_points, 2)
        
    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, (10, 15, 25, 200), (0, 0, self.WIDTH, 40))
        
        # Health
        health_text = self.font_small.render(f"CRYSTAL: {int(self.castle_health)} / {self.INITIAL_CASTLE_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 12))

        # Gold
        gold_text = self.font_small.render(f"GOLD: {int(self.gold)}", True, (255, 215, 0))
        self.screen.blit(gold_text, (250, 12))

        # Wave
        if self.wave <= self.MAX_WAVES:
            wave_text = self.font_small.render(f"WAVE: {self.wave} / {self.MAX_WAVES}", True, self.COLOR_TEXT)
            self.screen.blit(wave_text, (self.WIDTH - 150, 12))

        # Selected Tower
        stats = self.TOWER_STATS[self.selected_tower_type]
        tower_text = self.font_small.render(f"BUILD: {stats['name']} ({stats['cost']}G)", True, self.TOWER_COLORS[self.selected_tower_type])
        self.screen.blit(tower_text, (400, 12))

        # Game Over / Win message
        if self.game_over:
            msg = "MISSION FAILED"
            color = self.COLOR_ENEMY
            if self.win:
                msg = "VICTORY!"
                color = self.COLOR_CRYSTAL
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,150))
            self.screen.blit(overlay, (0,0))
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _iso_to_screen(self, iso_x, iso_y):
        screen_x = self.ORIGIN_X + (iso_x - iso_y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (iso_x + iso_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "castle_health": self.castle_health,
            "gold": self.gold,
            "wave": self.wave
        }
    
    def close(self):
        pygame.quit()

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # To play manually, you need a pygame window.
    # The environment is designed to be headless, but we can display the frames.
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # Get keyboard input
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

    print(f"Game Over. Final Score: {total_reward}")
    print(f"Info: {info}")
    
    env.close()