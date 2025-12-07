
# Generated: 2025-08-28T04:07:50.746746
# Source Brief: brief_05154.md
# Brief Index: 5154

        
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
        "Controls: Arrow keys to move the cursor. Spacebar to place a tower on an empty grid cell."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist tower defense game. Place towers to defend your base from waves of enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.W, self.H = 640, 400
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.FPS = 30
        self.GRID_W, self.GRID_H = 8, 8
        self.CELL_SIZE = 50
        self.GRID_OFFSET_X = (self.W - self.GRID_W * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = 0

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PATH = (60, 60, 70)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_ENEMY = (210, 60, 60)
        self.COLOR_TOWER = (70, 150, 255)
        self.COLOR_PROJECTILE = (255, 220, 100)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_INVALID = (255, 100, 100)

        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game parameters
        self.INITIAL_GOLD = 40
        self.TOWER_COST = 20
        self.TOWER_RANGE_SQ = (self.CELL_SIZE * 1.5) ** 2 # Range of 1 grid square (center to center)
        self.TOWER_COOLDOWN = self.FPS // 2 # 0.5 seconds
        self.ENEMY_MAX_HEALTH = 3
        self.ENEMY_SPEED = self.CELL_SIZE / self.FPS # 1 grid square per second
        self.GOLD_PER_KILL = 5
        self.MAX_STEPS = 3000

        # Define waves
        self.wave_definitions = [5, 7, 10, 14, 18]

        # Path definition
        self.path_grid_coords = [(-1, 4), (0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (3, 2), (4, 2), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (6, 6), (7, 6), (8, 6)]
        self.path_pixel_coords = [self._grid_to_pixel_center(p[0], p[1]) for p in self.path_grid_coords]
        
        self.path_cells = set()
        for i in range(len(self.path_grid_coords) - 1):
            x1, y1 = self.path_grid_coords[i]
            x2, y2 = self.path_grid_coords[i+1]
            for x in range(min(x1, x2), max(x1, x2) + 1):
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    if 0 <= x < self.GRID_W and 0 <= y < self.GRID_H:
                        self.path_cells.add((x, y))
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        
        self.steps = 0
        self.total_reward = 0.0
        self.game_over = False
        self.game_won = False
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.gold = self.INITIAL_GOLD
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.space_pressed_last_frame = False
        self.move_cooldown = 0

        self.wave_number = 0
        self.wave_transition_timer = self.FPS * 3 # 3 second initial delay
        self.enemies_to_spawn_in_wave = 0
        self.spawn_timer = 0
        self.spawn_interval = self.FPS # 1 second between spawns

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.01 # Small reward for surviving
        self.game_over = self.game_over or self.steps >= self.MAX_STEPS

        if not self.game_over:
            self._handle_input(action)
            
            kill_reward = self._update_game_logic()
            reward += kill_reward

        if self.game_over:
            if self.game_won:
                reward += 100
            else:
                reward -= 100
        
        self.steps += 1
        self.total_reward += reward
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if self.move_cooldown == 0:
            moved = False
            if movement == 1 and self.cursor_pos[1] > 0: # Up
                self.cursor_pos[1] -= 1; moved = True
            elif movement == 2 and self.cursor_pos[1] < self.GRID_H - 1: # Down
                self.cursor_pos[1] += 1; moved = True
            elif movement == 3 and self.cursor_pos[0] > 0: # Left
                self.cursor_pos[0] -= 1; moved = True
            elif movement == 4 and self.cursor_pos[0] < self.GRID_W - 1: # Right
                self.cursor_pos[0] += 1; moved = True
            
            if moved:
                self.move_cooldown = 5 # 5-frame cooldown for smoother control

        if space_held and not self.space_pressed_last_frame:
            self._place_tower()
        self.space_pressed_last_frame = space_held

    def _place_tower(self):
        cx, cy = self.cursor_pos
        
        is_on_path = (cx, cy) in self.path_cells
        is_occupied = any(t['grid_pos'] == [cx, cy] for t in self.towers)

        if self.gold >= self.TOWER_COST and not is_on_path and not is_occupied:
            self.gold -= self.TOWER_COST
            px, py = self._grid_to_pixel_center(cx, cy)
            self.towers.append({
                'grid_pos': [cx, cy], 'pixel_pos': [px, py],
                'cooldown': 0, 'target': None, 'anim_scale': 0.0
            })
            # sfx: place_tower.wav
            for _ in range(20):
                self.particles.append(self._create_particle(px, py, self.COLOR_TOWER, 2, 4, 20))

    def _update_game_logic(self):
        self._update_waves()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        kill_count = self._reap_dead()
        if kill_count > 0:
            self.gold += kill_count * self.GOLD_PER_KILL
        return kill_count # Reward for kills

    def _update_waves(self):
        if self.enemies_to_spawn_in_wave == 0 and not self.enemies:
            if self.wave_transition_timer > 0:
                self.wave_transition_timer -= 1
            else:
                if self.wave_number >= len(self.wave_definitions):
                    self.game_over = True
                    self.game_won = True
                else:
                    self.enemies_to_spawn_in_wave = self.wave_definitions[self.wave_number]
                    self.wave_number += 1
                    self.spawn_timer = 0
        
        if self.enemies_to_spawn_in_wave > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_enemy()
                self.enemies_to_spawn_in_wave -= 1
                self.spawn_timer = self.spawn_interval

    def _spawn_enemy(self):
        start_pos = list(self.path_pixel_coords[0])
        self.enemies.append({
            'pos': start_pos, 'health': self.ENEMY_MAX_HEALTH, 'is_dead': False,
            'path_index': 1, 'anim_scale': 0.0
        })
        # sfx: spawn_enemy.wav

    def _update_towers(self):
        for tower in self.towers:
            tower['anim_scale'] = min(1.0, tower['anim_scale'] + 0.1)
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
            
            # Invalidate target if dead or out of range
            if tower['target']:
                if tower['target']['is_dead']:
                    tower['target'] = None
                else:
                    dist_sq = self._dist_sq(tower['pixel_pos'], tower['target']['pos'])
                    if dist_sq > self.TOWER_RANGE_SQ:
                        tower['target'] = None

            # Find new target if needed
            if not tower['target']:
                for enemy in self.enemies:
                    if not enemy['is_dead']:
                        dist_sq = self._dist_sq(tower['pixel_pos'], enemy['pos'])
                        if dist_sq <= self.TOWER_RANGE_SQ:
                            tower['target'] = enemy
                            break
            
            # Fire projectile
            if tower['target'] and tower['cooldown'] == 0:
                self.projectiles.append({
                    'pos': list(tower['pixel_pos']), 'target': tower['target'], 'is_dead': False
                })
                tower['cooldown'] = self.TOWER_COOLDOWN
                # sfx: tower_shoot.wav
                self.particles.append(self._create_particle(tower['pixel_pos'][0], tower['pixel_pos'][1], self.COLOR_PROJECTILE, 1, 2, 8))

    def _update_projectiles(self):
        PROJECTILE_SPEED = 8
        for proj in self.projectiles:
            if proj['target']['is_dead']:
                proj['is_dead'] = True
                continue
            
            target_pos = proj['target']['pos']
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < PROJECTILE_SPEED:
                proj['is_dead'] = True
                proj['target']['health'] -= 1
                # sfx: enemy_hit.wav
                for _ in range(10):
                    self.particles.append(self._create_particle(target_pos[0], target_pos[1], self.COLOR_ENEMY, 0.5, 1.5, 15))
            else:
                proj['pos'][0] += (dx / dist) * PROJECTILE_SPEED
                proj['pos'][1] += (dy / dist) * PROJECTILE_SPEED
    
    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['anim_scale'] = min(1.0, enemy['anim_scale'] + 0.1)
            target_pixel_pos = self.path_pixel_coords[enemy['path_index']]
            dx = target_pixel_pos[0] - enemy['pos'][0]
            dy = target_pixel_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < self.ENEMY_SPEED:
                enemy['pos'] = list(target_pixel_pos)
                enemy['path_index'] += 1
                if enemy['path_index'] >= len(self.path_pixel_coords):
                    self.game_over = True
                    self.game_won = False
                    enemy['is_dead'] = True # Remove from screen
                    # sfx: base_breached.wav
                    # Base explosion effect
                    base_pos = self.path_pixel_coords[-1]
                    for _ in range(50):
                        self.particles.append(self._create_particle(base_pos[0], base_pos[1], self.COLOR_BASE, 3, 8, 40))
            else:
                enemy['pos'][0] += (dx / dist) * self.ENEMY_SPEED
                enemy['pos'][1] += (dy / dist) * self.ENEMY_SPEED
            
            if enemy['health'] <= 0:
                enemy['is_dead'] = True
                # sfx: enemy_die.wav
                for _ in range(15):
                    self.particles.append(self._create_particle(enemy['pos'][0], enemy['pos'][1], self.COLOR_ENEMY, 1, 3, 25))
    
    def _update_particles(self):
        for p in self.particles:
            p['life'] -= 1
            p['radius'] -= p['decay']
            if p['life'] <= 0 or p['radius'] <= 0:
                p['is_dead'] = True

    def _reap_dead(self):
        kill_count = sum(1 for e in self.enemies if e['is_dead'])
        self.enemies = [e for e in self.enemies if not e['is_dead']]
        self.projectiles = [p for p in self.projectiles if not p['is_dead']]
        self.particles = [p for p in self.particles if not p['is_dead']]
        return kill_count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_W + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.H))
        for y in range(self.GRID_H + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_W * self.CELL_SIZE, py))

        # Draw path
        if len(self.path_pixel_coords) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_pixel_coords, width=self.CELL_SIZE)

        # Draw base
        base_pos = self._grid_to_pixel_center(7, 6)
        base_size = int(self.CELL_SIZE * 0.8)
        base_rect = pygame.Rect(base_pos[0] - base_size // 2, base_pos[1] - base_size // 2, base_size, base_size)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        
        # Draw towers
        for tower in self.towers:
            size = int(self.CELL_SIZE * 0.6 * tower['anim_scale'])
            pos = tower['pixel_pos']
            rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_TOWER, rect, border_radius=3)
        
        # Draw projectiles
        for proj in self.projectiles:
            px, py = int(proj['pos'][0]), int(proj['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, px, py, 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, px, py, 3, self.COLOR_PROJECTILE)

        # Draw enemies
        for enemy in self.enemies:
            px, py = int(enemy['pos'][0]), int(enemy['pos'][1])
            radius = int(self.CELL_SIZE * 0.3 * enemy['anim_scale'])
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_ENEMY)

            # Health bar
            if enemy['health'] < self.ENEMY_MAX_HEALTH:
                bar_w = self.CELL_SIZE * 0.6
                bar_h = 5
                health_pct = enemy['health'] / self.ENEMY_MAX_HEALTH
                health_w = bar_w * health_pct
                pygame.draw.rect(self.screen, (50, 50, 50), (px - bar_w // 2, py - radius - 10, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_BASE, (px - bar_w // 2, py - radius - 10, health_w, bar_h))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, int(p['radius']), int(p['radius']), int(p['radius']), color)
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

        # Draw cursor
        cx, cy = self.cursor_pos
        px, py = self._grid_to_pixel_center(cx, cy)
        is_on_path = (cx, cy) in self.path_cells
        is_occupied = any(t['grid_pos'] == [cx, cy] for t in self.towers)
        can_afford = self.gold >= self.TOWER_COST
        is_valid = not is_on_path and not is_occupied and can_afford
        cursor_color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID

        size = self.CELL_SIZE // 2
        pygame.draw.line(self.screen, cursor_color, (px - size, py), (px - 5, py), 2)
        pygame.draw.line(self.screen, cursor_color, (px + 5, py), (px + size, py), 2)
        pygame.draw.line(self.screen, cursor_color, (px, py - size), (px, py - 5), 2)
        pygame.draw.line(self.screen, cursor_color, (px, py + 5), (px, py + size), 2)
    
    def _render_ui(self):
        # Gold
        gold_text = self.font_main.render(f"GOLD: {self.gold}", True, self.COLOR_TEXT)
        self.screen.blit(gold_text, (10, 10))

        # Wave
        wave_str = f"WAVE: {self.wave_number}/{len(self.wave_definitions)}"
        if self.game_won: wave_str = "VICTORY!"
        wave_text = self.font_main.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.W - wave_text.get_width() - 10, 10))

        # Enemies remaining
        enemies_left = len(self.enemies) + self.enemies_to_spawn_in_wave
        if enemies_left > 0:
            enemy_text = self.font_main.render(f"ENEMIES: {enemies_left}", True, self.COLOR_TEXT)
            self.screen.blit(enemy_text, (self.W // 2 - enemy_text.get_width() // 2, self.H - 30))

        # Game Over / Win message
        if self.game_over:
            msg = "VICTORY" if self.game_won else "GAME OVER"
            color = self.COLOR_BASE if self.game_won else self.COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.W // 2 - end_text.get_width() // 2, self.H // 2 - end_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.total_reward,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave_number,
            "towers": len(self.towers),
            "enemies": len(self.enemies)
        }
    
    # Helper functions
    def _grid_to_pixel_center(self, gx, gy):
        px = self.GRID_OFFSET_X + gx * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + gy * self.CELL_SIZE + self.CELL_SIZE // 2
        return px, py
    
    def _dist_sq(self, pos1, pos2):
        return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2

    def _create_particle(self, x, y, color, min_r, max_r, life):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        radius = random.uniform(min_r, max_r)
        return {
            'pos': [x, y], 'radius': radius, 'color': color,
            'life': life, 'max_life': life, 'decay': radius / life, 'is_dead': False
        }

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Minimalist Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(env.FPS)
        
    pygame.quit()