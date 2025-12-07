
# Generated: 2025-08-27T22:31:50.553783
# Source Brief: brief_03156.md
# Brief Index: 3156

        
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
        "Controls: Use arrow keys to move the placement cursor. Press space to build the selected tower. Hold shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist tower defense game. Place towers to defend your base from waves of enemies. Earn gold for each kill to build more towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PATH = (40, 50, 70)
    COLOR_BASE = (60, 180, 75)
    COLOR_ENEMY = (210, 60, 60)
    COLOR_TEXT = (230, 230, 230)
    COLOR_GOLD = (255, 215, 0)
    COLOR_CURSOR_VALID = (255, 255, 255, 100)
    COLOR_CURSOR_INVALID = (255, 0, 0, 100)

    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 10
    GRID_ROWS = 6
    CELL_SIZE = 64  # 640/10, but 400/6 is 66.6. We'll use 64 and have a UI bar at bottom.
    
    # Game Parameters
    MAX_STEPS = 30 * 60 # 60 seconds at 30fps
    STARTING_GOLD = 150
    GOLD_PER_KILL = 15

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)

        # Tower definitions
        self.TOWER_TYPES = [
            {
                "name": "Cannon", "cost": 50, "range": 100, "fire_rate": 45, "damage": 12, 
                "color": (0, 150, 255), "proj_speed": 7, "shape": "square"
            },
            {
                "name": "Sniper", "cost": 120, "range": 200, "fire_rate": 90, "damage": 50, 
                "color": (255, 128, 0), "proj_speed": 15, "shape": "triangle"
            },
            {
                "name": "Machine Gun", "cost": 80, "range": 80, "fire_rate": 15, "damage": 5, 
                "color": (170, 0, 255), "proj_speed": 10, "shape": "circle"
            }
        ]
        
        # Define the enemy path (in grid coordinates)
        self.path_nodes = [(0, 2), (2, 2), (2, 4), (5, 4), (5, 1), (8, 1), (8, 3), (10, 3)]
        self.path_pixels = [self._grid_to_pixels(gx, gy) for gx, gy in self.path_nodes]
        self.base_pos = self.path_pixels[-1]
        
        # Initialize state variables
        self.np_random = None
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminal_reward_given = False

        self.gold = self.STARTING_GOLD
        self.wave_number = 1
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_type_idx = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for each time step

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1

        # 1. Handle Player Input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_input(movement, space_held, shift_held)

        # 2. Update Game Logic
        self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()

        # 3. Check for Wave Clear
        if not self.enemies and not self.game_over:
            if not self.terminal_reward_given:
                reward += 100 # Wave clear bonus
                self.score += 100
                self.terminal_reward_given = True
            self.game_over = True # Episode ends on win

        # 4. Check Termination Conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.game_over and not self.terminal_reward_given:
            reward -= 100 # Penalty for loss
            self.score -= 100
            self.terminal_reward_given = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # --- Cycle Tower Type (on key press) ---
        if shift_held and not self.last_shift_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.TOWER_TYPES)
        self.last_shift_held = shift_held

        # --- Place Tower (on key press) ---
        if space_held and not self.last_space_held:
            self._place_tower()
        self.last_space_held = space_held
        
        return 0 # Input itself gives no reward

    def _place_tower(self):
        selected_tower = self.TOWER_TYPES[self.selected_tower_type_idx]
        
        is_on_path = tuple(self.cursor_pos) in self.path_nodes[:-1] # Can't build on path
        is_occupied = any(t['grid_pos'] == self.cursor_pos for t in self.towers)
        can_afford = self.gold >= selected_tower['cost']

        if not is_on_path and not is_occupied and can_afford:
            self.gold -= selected_tower['cost']
            px, py = self._grid_to_pixels(self.cursor_pos[0], self.cursor_pos[1])
            new_tower = {
                "grid_pos": list(self.cursor_pos),
                "pixel_pos": (px, py),
                "type_idx": self.selected_tower_type_idx,
                "cooldown": 0,
                "target": None
            }
            self.towers.append(new_tower)
            # sfx: build_tower.wav
            self._create_particles(px, py, selected_tower['color'], 20, 3)

    def _spawn_wave(self):
        num_enemies = 5 + (self.wave_number - 1) * 2
        base_health = 10 * (1.1 ** (self.wave_number - 1))
        base_speed = 1.0 * (1.05 ** (self.wave_number - 1))
        
        for i in range(num_enemies):
            self.enemies.append({
                "pos": [self.path_pixels[0][0] - (i * 30), self.path_pixels[0][1]],
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed,
                "path_index": 0,
                "id": self.np_random.integers(1, 1_000_000)
            })

    def _update_towers(self):
        for tower in self.towers:
            stats = self.TOWER_TYPES[tower['type_idx']]
            tower['cooldown'] = max(0, tower['cooldown'] - 1)

            # Find new target if needed
            if tower.get('target') is None or not any(e['id'] == tower['target']['id'] for e in self.enemies):
                tower['target'] = None
                closest_enemy = None
                min_dist = float('inf')
                for enemy in self.enemies:
                    dist = math.hypot(enemy['pos'][0] - tower['pixel_pos'][0], enemy['pos'][1] - tower['pixel_pos'][1])
                    if dist <= stats['range'] and dist < min_dist:
                        min_dist = dist
                        closest_enemy = enemy
                if closest_enemy:
                    tower['target'] = closest_enemy
            
            # Fire if ready and has target
            if tower['cooldown'] == 0 and tower.get('target') is not None:
                tower['cooldown'] = stats['fire_rate']
                self.projectiles.append({
                    "pos": list(tower['pixel_pos']),
                    "target": tower['target'],
                    "type_idx": tower['type_idx']
                })
                # sfx: fire_weapon.wav

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            stats = self.TOWER_TYPES[proj['type_idx']]
            target_enemy = proj['target']
            
            # If target is gone, remove projectile
            if not any(e['id'] == target_enemy['id'] for e in self.enemies):
                self.projectiles.remove(proj)
                continue
                
            # Move projectile towards target
            dx = target_enemy['pos'][0] - proj['pos'][0]
            dy = target_enemy['pos'][1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < stats['proj_speed']:
                # Hit target
                target_enemy['health'] -= stats['damage']
                reward += 0.1 # Reward for hitting
                self.score += 0.1
                self.projectiles.remove(proj)
                self._create_particles(proj['pos'][0], proj['pos'][1], stats['color'], 10, 2)
                # sfx: hit_enemy.wav
            else:
                proj['pos'][0] += (dx / dist) * stats['proj_speed']
                proj['pos'][1] += (dy / dist) * stats['proj_speed']
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            # Check if defeated
            if enemy['health'] <= 0:
                reward += 1.0 # Reward for defeating
                self.score += 1.0
                self.gold += self.GOLD_PER_KILL
                self._create_particles(enemy['pos'][0], enemy['pos'][1], self.COLOR_ENEMY, 30, 4)
                self.enemies.remove(enemy)
                # sfx: enemy_explode.wav
                continue

            # Move along path
            if enemy['path_index'] < len(self.path_pixels) - 1:
                target_pos = self.path_pixels[enemy['path_index'] + 1]
                dx = target_pos[0] - enemy['pos'][0]
                dy = target_pos[1] - enemy['pos'][1]
                dist = math.hypot(dx, dy)
                
                if dist < enemy['speed']:
                    enemy['path_index'] += 1
                    enemy['pos'] = list(target_pos)
                else:
                    enemy['pos'][0] += (dx / dist) * enemy['speed']
                    enemy['pos'][1] += (dy / dist) * enemy['speed']
            else:
                # Reached base
                self.game_over = True
                self.enemies.remove(enemy)
                # sfx: base_breached.wav
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw path
        for i in range(len(self.path_pixels) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_pixels[i], self.path_pixels[i+1], self.CELL_SIZE)
        
        # Draw base
        pygame.draw.circle(self.screen, self.COLOR_BASE, self.base_pos, self.CELL_SIZE // 2)

        # Draw towers
        for tower in self.towers:
            stats = self.TOWER_TYPES[tower['type_idx']]
            pos = (int(tower['pixel_pos'][0]), int(tower['pixel_pos'][1]))
            if stats['shape'] == 'square':
                pygame.draw.rect(self.screen, stats['color'], (pos[0]-12, pos[1]-12, 24, 24))
            elif stats['shape'] == 'circle':
                pygame.draw.circle(self.screen, stats['color'], pos, 12)
            elif stats['shape'] == 'triangle':
                points = [(pos[0], pos[1]-12), (pos[0]-12, pos[1]+12), (pos[0]+12, pos[1]+12)]
                pygame.draw.polygon(self.screen, stats['color'], points)

        # Draw projectiles
        for proj in self.projectiles:
            stats = self.TOWER_TYPES[proj['type_idx']]
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.circle(self.screen, stats['color'], pos, 4)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, stats['color'])

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 8)
            # Health bar
            health_ratio = max(0, enemy['health'] / enemy['max_health'])
            bar_width = 20
            pygame.draw.rect(self.screen, (50,0,0), (pos[0]-bar_width/2, pos[1]-18, bar_width, 5))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (pos[0]-bar_width/2, pos[1]-18, bar_width * health_ratio, 5))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0]-p['size']), int(p['pos'][1]-p['size'])))
        
        # Draw cursor
        self._render_cursor()

    def _render_cursor(self):
        selected_tower = self.TOWER_TYPES[self.selected_tower_type_idx]
        cursor_px, cursor_py = self._grid_to_pixels(self.cursor_pos[0], self.cursor_pos[1])
        
        is_on_path = tuple(self.cursor_pos) in self.path_nodes[:-1]
        is_occupied = any(t['grid_pos'] == self.cursor_pos for t in self.towers)
        can_afford = self.gold >= selected_tower['cost']
        is_valid = not is_on_path and not is_occupied and can_afford
        
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        
        # Draw range indicator
        range_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(range_surf, cursor_color, (cursor_px, cursor_py), selected_tower['range'])
        self.screen.blit(range_surf, (0,0))

        # Draw tower preview
        if selected_tower['shape'] == 'square':
            pygame.draw.rect(self.screen, selected_tower['color'], (cursor_px-12, cursor_py-12, 24, 24), 2)
        elif selected_tower['shape'] == 'circle':
            pygame.draw.circle(self.screen, selected_tower['color'], (cursor_px, cursor_py), 12, 2)
        elif selected_tower['shape'] == 'triangle':
            points = [(cursor_px, cursor_py-12), (cursor_px-12, cursor_py+12), (cursor_px+12, cursor_py+12)]
            pygame.draw.polygon(self.screen, selected_tower['color'], points, 2)

    def _render_ui(self):
        ui_bar_y = self.GRID_ROWS * self.CELL_SIZE
        # Gold
        gold_text = self.font_small.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (10, ui_bar_y + 5))
        
        # Wave
        wave_text = self.font_small.render(f"Wave: {self.wave_number}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, ui_bar_y + 5))

        # Selected Tower
        tower = self.TOWER_TYPES[self.selected_tower_type_idx]
        tower_text = self.font_small.render(f"Selected: {tower['name']} (Cost: {tower['cost']})", True, tower['color'])
        self.screen.blit(tower_text, (self.SCREEN_WIDTH // 2 - tower_text.get_width() // 2, ui_bar_y + 5))

        # Game Over Text
        if self.game_over:
            is_win = not any(e for e in self.enemies)
            msg = "WAVE CLEARED" if is_win else "GAME OVER"
            color = self.COLOR_BASE if is_win else self.COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            pos = (self.SCREEN_WIDTH//2 - end_text.get_width()//2, self.SCREEN_HEIGHT//2 - end_text.get_height()//2)
            self.screen.blit(end_text, pos)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave_number,
            "enemies_left": len(self.enemies),
            "towers_placed": len(self.towers)
        }

    def _grid_to_pixels(self, gx, gy):
        return int((gx + 0.5) * self.CELL_SIZE), int((gy + 0.5) * self.CELL_SIZE)

    def _create_particles(self, x, y, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            self.particles.append({
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 20),
                "max_life": 20,
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a window to display the game
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()