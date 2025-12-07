
# Generated: 2025-08-28T04:42:26.730693
# Source Brief: brief_05338.md
# Brief Index: 5338

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to place the selected tower. Hold Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of geometric enemies by strategically placing towers on the grid."
    )

    # Frames auto-advance for real-time gameplay.
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
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Game Constants ---
        self.GRID_COLS, self.GRID_ROWS = 20, 10
        self.CELL_SIZE = 32
        self.GRID_OFFSET_X = 0
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_ROWS * self.CELL_SIZE) - 20 # 60
        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps
        self.MAX_WAVES = 10
        
        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_PATH = (45, 60, 90)
        self.COLOR_BASE = (0, 200, 100)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_GOLD = (255, 200, 0)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_INVALID = (255, 0, 0)
        
        # --- Fonts ---
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 64)
        self.font_small = pygame.font.Font(None, 18)

        # --- Game Entities & Path ---
        self.path_grid_coords = [
            (-1, 5), (2, 5), (2, 2), (6, 2), (6, 8), 
            (12, 8), (12, 1), (17, 1), (17, 5), (20, 5)
        ]
        self.path_pixel_coords = [self._grid_to_pixel(p, center=True) for p in self.path_grid_coords]
        self.total_path_length = self._calculate_path_length()
        
        self.TOWER_TYPES = [
            {
                "name": "Gun", "cost": 50, "range": 80, "damage": 12, 
                "fire_rate": 20, "color": (0, 255, 255), "proj_speed": 8
            },
            {
                "name": "Cannon", "cost": 100, "range": 120, "damage": 35, 
                "fire_rate": 60, "color": (255, 165, 0), "proj_speed": 6
            },
        ]
        
        # --- State Variables (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 0
        self.max_base_health = 100
        self.gold = 0
        self.wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.wave_cooldown = 0
        self.next_spawn_timer = 0
        self.enemies_to_spawn_in_wave = 0
        self.enemies_spawned_in_wave = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = self.max_base_health
        self.gold = 120
        self.wave = 0
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        self.wave_cooldown = 90  # Initial delay before first wave
        self.next_spawn_timer = 0
        self.enemies_to_spawn_in_wave = 0
        self.enemies_spawned_in_wave = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.001  # Small time penalty to encourage efficiency

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Player Input
        self._handle_input(action)
        
        # 2. Update Game Logic
        self._update_wave_logic()
        self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        self.score += reward
        self.steps += 1

        # 3. Check for Termination Conditions
        terminated = False
        if self.base_health <= 0:
            self.game_over = True
            terminated = True
            reward += -100  # Large penalty for losing
            self.score += -100
        elif self.game_won:
            self.game_over = True
            terminated = True
            reward += 100 # Large reward for winning
            self.score += 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Core Logic Sub-functions ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        if space_held and not self.last_space_held:
            self._place_tower()

        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
            # sfx: UI_Cycle.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_wave_logic(self):
        if self.wave_cooldown > 0:
            self.wave_cooldown -= 1
            if self.wave_cooldown == 0:
                self._start_next_wave()
            return 0

        if self.enemies_spawned_in_wave < self.enemies_to_spawn_in_wave:
            self.next_spawn_timer -= 1
            if self.next_spawn_timer <= 0:
                self._spawn_enemy()
                self.next_spawn_timer = 30 # Time between enemies in a wave

        # Check for wave completion
        if self.enemies_spawned_in_wave == self.enemies_to_spawn_in_wave and not self.enemies:
            if self.wave >= self.MAX_WAVES:
                self.game_won = True
            else:
                self.gold += 100 + self.wave * 10
                self.wave_cooldown = 240 # Time between waves
                # sfx: Wave_Complete.wav

    def _start_next_wave(self):
        self.wave += 1
        self.enemies_to_spawn_in_wave = 5 + self.wave * 2
        self.enemies_spawned_in_wave = 0
        self.next_spawn_timer = 0
        # sfx: New_Wave.wav

    def _spawn_enemy(self):
        health = 50 * (1.1 ** (self.wave - 1))
        speed = 1.0 * (1.05 ** (self.wave - 1))
        self.enemies.append({
            "pos": list(self.path_pixel_coords[0]),
            "health": health,
            "max_health": health,
            "speed": speed,
            "path_index": 0,
            "dist_on_path": 0,
            "id": self.np_random.integers(1, 1_000_000)
        })
        self.enemies_spawned_in_wave += 1

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["path_index"] >= len(self.path_pixel_coords) - 1:
                self.enemies.remove(enemy)
                self.base_health -= 10
                self.base_health = max(0, self.base_health)
                self._create_particles(self._grid_to_pixel((self.GRID_COLS, 5), center=True), self.COLOR_ENEMY, 20)
                # sfx: Base_Damage.wav
                continue

            target_node = self.path_pixel_coords[enemy["path_index"] + 1]
            current_pos = enemy["pos"]
            
            direction = np.array(target_node) - np.array(current_pos)
            dist_to_target = np.linalg.norm(direction)
            
            if dist_to_target < enemy["speed"]:
                enemy["path_index"] += 1
                enemy["dist_on_path"] += dist_to_target
            else:
                move_vec = (direction / dist_to_target) * enemy["speed"]
                enemy["pos"][0] += move_vec[0]
                enemy["pos"][1] += move_vec[1]
                enemy["dist_on_path"] += enemy["speed"]
        return reward

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] > 0:
                continue

            # Find best target (furthest along the path)
            best_target = None
            max_dist = -1
            for enemy in self.enemies:
                dist_sq = (tower['pixel_pos'][0] - enemy['pos'][0])**2 + (tower['pixel_pos'][1] - enemy['pos'][1])**2
                if dist_sq <= tower['range']**2 and enemy['dist_on_path'] > max_dist:
                    max_dist = enemy['dist_on_path']
                    best_target = enemy
            
            if best_target:
                tower['cooldown'] = tower['fire_rate']
                self.projectiles.append({
                    "pos": list(tower['pixel_pos']),
                    "target_id": best_target['id'],
                    "speed": tower['proj_speed'],
                    "damage": tower['damage'],
                    "color": tower['color']
                })
                # sfx: Tower_Fire.wav

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target_enemy = next((e for e in self.enemies if e['id'] == proj['target_id']), None)

            if not target_enemy:
                self.projectiles.remove(proj)
                continue

            target_pos = target_enemy['pos']
            proj_pos = np.array(proj['pos'])
            direction = np.array(target_pos) - proj_pos
            dist = np.linalg.norm(direction)

            if dist < proj['speed']:
                target_enemy['health'] -= proj['damage']
                self._create_particles(target_enemy['pos'], proj['color'], 5)
                # sfx: Enemy_Hit.wav
                if target_enemy['health'] <= 0:
                    self.gold += 10
                    reward += 0.1
                    self._create_particles(target_enemy['pos'], self.COLOR_ENEMY, 15, 2)
                    self.enemies.remove(target_enemy)
                    # sfx: Enemy_Destroyed.wav
                self.projectiles.remove(proj)
            else:
                move_vec = (direction / dist) * proj['speed']
                proj['pos'][0] += move_vec[0]
                proj['pos'][1] += move_vec[1]
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _place_tower(self):
        tower_spec = self.TOWER_TYPES[self.selected_tower_type]
        if self.gold < tower_spec['cost']:
            # sfx: Error.wav
            return

        grid_x, grid_y = self.cursor_pos
        if self._is_on_path((grid_x, grid_y)):
            # sfx: Error.wav
            return
        
        if any(t['grid_pos'] == [grid_x, grid_y] for t in self.towers):
            # sfx: Error.wav
            return

        self.gold -= tower_spec['cost']
        pixel_pos = self._grid_to_pixel(self.cursor_pos, center=True)
        self.towers.append({
            "grid_pos": list(self.cursor_pos),
            "pixel_pos": pixel_pos,
            "type": self.selected_tower_type,
            "cooldown": 0,
            **tower_spec
        })
        self._create_particles(pixel_pos, tower_spec['color'], 20, 1.5)
        # sfx: Place_Tower.wav

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw grid
        for x in range(self.GRID_COLS + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_ROWS * self.CELL_SIZE))
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_COLS * self.CELL_SIZE, py))

        # Draw path
        if len(self.path_pixel_coords) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_pixel_coords, width=self.CELL_SIZE)

        # Draw Base
        base_pos = self.path_pixel_coords[-1]
        base_rect = pygame.Rect(base_pos[0], base_pos[1] - self.CELL_SIZE // 2, 10, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

    def _render_towers(self):
        for tower in self.towers:
            pos = tower['pixel_pos']
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.CELL_SIZE // 3, tower['color'])
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.CELL_SIZE // 3, tower['color'])

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            size = 8
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos[0] - size, pos[1] - size, size*2, size*2))
            
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_width = 16
            bar_height = 3
            bar_pos = (pos[0] - bar_width // 2, pos[1] - size - 6)
            pygame.draw.rect(self.screen, (80, 0, 0), (*bar_pos, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (*bar_pos, int(bar_width * health_ratio), bar_height))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, proj['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, proj['color'])

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

    def _render_cursor(self):
        pixel_pos = self._grid_to_pixel(self.cursor_pos)
        rect = (pixel_pos[0], pixel_pos[1], self.CELL_SIZE, self.CELL_SIZE)
        
        tower_spec = self.TOWER_TYPES[self.selected_tower_type]
        is_valid = self.gold >= tower_spec['cost'] and not self._is_on_path(self.cursor_pos) and not any(t['grid_pos'] == self.cursor_pos for t in self.towers)
        color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID

        # Draw range indicator
        range_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        center_pos = (pixel_pos[0] + self.CELL_SIZE // 2, pixel_pos[1] + self.CELL_SIZE // 2)
        pygame.gfxdraw.aacircle(range_surf, center_pos[0], center_pos[1], tower_spec['range'], (*color, 50))
        pygame.gfxdraw.filled_circle(range_surf, center_pos[0], center_pos[1], tower_spec['range'], (*color, 50))
        self.screen.blit(range_surf, (0, 0))

        # Draw cursor box
        pygame.draw.rect(self.screen, color, rect, 2)

    def _render_ui(self):
        # Top Bar
        bar_surf = pygame.Surface((self.WIDTH, 40))
        bar_surf.fill((10, 15, 25))
        
        # Health
        health_text = self.font_ui.render(f"Base: {int(self.base_health)}/{self.max_base_health}", True, self.COLOR_TEXT)
        bar_surf.blit(health_text, (10, 10))

        # Gold
        gold_text = self.font_ui.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        bar_surf.blit(gold_text, (180, 10))

        # Wave
        wave_text = self.font_ui.render(f"Wave: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        bar_surf.blit(wave_text, (300, 10))
        
        if self.wave_cooldown > 0 and self.wave < self.MAX_WAVES:
            next_wave_in = f"Next wave in: {self.wave_cooldown / 30:.1f}s"
            wave_timer_text = self.font_small.render(next_wave_in, True, self.COLOR_TEXT)
            bar_surf.blit(wave_timer_text, (420, 12))

        self.screen.blit(bar_surf, (0, 0))

        # Bottom Bar (Tower Selection)
        bottom_bar_surf = pygame.Surface((self.WIDTH, self.HEIGHT - (self.GRID_OFFSET_Y + self.GRID_ROWS * self.CELL_SIZE)))
        bottom_bar_surf.fill((10, 15, 25))
        
        tower_spec = self.TOWER_TYPES[self.selected_tower_type]
        name_text = self.font_ui.render(f"Selected: {tower_spec['name']}", True, tower_spec['color'])
        cost_text = self.font_small.render(f"Cost: {tower_spec['cost']} | Dmg: {tower_spec['damage']} | Rng: {tower_spec['range']}", True, self.COLOR_TEXT)
        bottom_bar_surf.blit(name_text, (10, 5))
        bottom_bar_surf.blit(cost_text, (10, 25))
        
        self.screen.blit(bottom_bar_surf, (0, self.GRID_OFFSET_Y + self.GRID_ROWS * self.CELL_SIZE))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "YOU WIN!" if self.game_won else "GAME OVER"
        color = self.COLOR_BASE if self.game_won else self.COLOR_ENEMY
        text = self.font_title.render(message, True, color)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        overlay.blit(text, text_rect)
        self.screen.blit(overlay, (0, 0))

    # --- Helper Functions ---

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "gold": self.gold, "wave": self.wave, "base_health": self.base_health}

    def _grid_to_pixel(self, grid_pos, center=False):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        if center:
            x += self.CELL_SIZE // 2
            y += self.CELL_SIZE // 2
        return [x, y]

    def _is_on_path(self, grid_pos):
        gx, gy = grid_pos
        for i in range(len(self.path_grid_coords) - 1):
            p1 = self.path_grid_coords[i]
            p2 = self.path_grid_coords[i+1]
            if p1[0] == p2[0] == gx and min(p1[1], p2[1]) <= gy <= max(p1[1], p2[1]):
                return True
            if p1[1] == p2[1] == gy and min(p1[0], p2[0]) <= gx <= max(p1[0], p2[0]):
                return True
        return False

    def _calculate_path_length(self):
        length = 0
        for i in range(len(self.path_pixel_coords) - 1):
            p1 = np.array(self.path_pixel_coords[i])
            p2 = np.array(self.path_pixel_coords[i+1])
            length += np.linalg.norm(p2 - p1)
        return length

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "radius": self.np_random.integers(2, 5),
                "color": color,
                "lifespan": lifespan,
                "max_lifespan": lifespan
            })
            
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windows'

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Set to True to play manually with keyboard
    MANUAL_PLAY = True
    
    if MANUAL_PLAY:
        pygame.display.set_caption(env.game_description)
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # Action defaults
            movement = 0
            space_held = 0
            shift_held = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)

            # Render to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Run at 30 FPS
            
        pygame.quit()
    else:
        # --- Agent Training (example) ---
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0

        while not terminated:
            action = env.action_space.sample() # Replace with your agent's action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            if terminated or truncated:
                print(f"Episode finished after {step_count} steps. Total reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
                step_count = 0

    env.close()