
# Generated: 2025-08-27T14:33:59.999547
# Source Brief: brief_00720.md
# Brief Index: 720

        
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
        "Press Shift to cycle tower types. Press Space to build a tower."
    )

    game_description = (
        "Defend your base from waves of invading geometric enemies by strategically placing "
        "and managing your towers. Earn gold for each kill to build more defenses. Survive 10 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_GRID = (25, 30, 40)
    COLOR_BASE = (0, 255, 128)
    COLOR_BASE_GLOW = (0, 255, 128, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_GOLD = (255, 223, 0)
    COLOR_CURSOR_VALID = (200, 255, 200, 150)
    COLOR_CURSOR_INVALID = (255, 100, 100, 150)

    # Screen
    WIDTH, HEIGHT = 640, 400
    
    # Game parameters
    MAX_STEPS = 6000 # Approx 3.3 minutes at 30fps
    TOTAL_WAVES = 10
    BASE_START_HEALTH = 100
    STARTING_GOLD = 150
    GRID_SIZE = 40

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_small = pygame.font.SysFont("sans", 16, bold=True)
        self.font_large = pygame.font.SysFont("sans", 24, bold=True)
        
        # Tower definitions
        self.TOWER_TYPES = {
            0: {
                "name": "Gatling", "cost": 50, "range": 80, "fire_rate": 5, "damage": 5, 
                "color": (0, 150, 255), "proj_speed": 10, "proj_size": 2
            },
            1: {
                "name": "Cannon", "cost": 120, "range": 120, "fire_rate": 20, "damage": 25, 
                "color": (255, 150, 0), "proj_speed": 7, "proj_size": 4
            },
        }

        # Path definition
        self.path_points = [
            (-20, 100), (100, 100), (100, 300), (300, 300), 
            (300, 100), (500, 100), (500, 200), (self.WIDTH + 20, 200)
        ]
        self.base_pos = (self.WIDTH, 200)

        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.gold = 0
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.space_was_held = False
        self.shift_was_held = False
        self.wave_timer = 0
        self.enemies_to_spawn = []
        self.enemies_in_wave = 0
        self.reward_buffer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.BASE_START_HEALTH
        self.gold = self.STARTING_GOLD
        self.wave_number = 0
        self.wave_timer = 150 # 5 seconds to prepare for first wave
        self.enemies_to_spawn = []
        self.enemies_in_wave = 0
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.selected_tower_type = 0
        self.space_was_held = True # Prevent placement on first frame
        self.shift_was_held = True # Prevent cycle on first frame

        self._initialize_grid()
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_buffer = -0.01 # Time penalty
        self.steps += 1

        # --- Handle Input ---
        self._handle_input(action)

        # --- Game Logic Updates ---
        self._update_waves()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.base_health <= 0:
                self.reward_buffer -= 100
            elif self.wave_number > self.TOTAL_WAVES:
                self.reward_buffer += 100

        # --- Finalize Step ---
        reward = self.reward_buffer
        self.score = max(0, self.score)

        if self.auto_advance:
            self.clock.tick(30)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        cursor_speed = 10
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)
        
        # Cycle tower type on key press (rising edge)
        if shift_held and not self.shift_was_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
        self.shift_was_held = shift_held
        
        # Place tower on key press (rising edge)
        if space_held and not self.space_was_held:
            self._place_tower()
        self.space_was_held = space_held

    def _place_tower(self):
        grid_x = int(self.cursor_pos[0] // self.GRID_SIZE)
        grid_y = int(self.cursor_pos[1] // self.GRID_SIZE)
        
        if not (0 <= grid_y < len(self.grid) and 0 <= grid_x < len(self.grid[0])):
            return

        if self.grid[grid_y][grid_x]: # If cell is valid for placement
            tower_def = self.TOWER_TYPES[self.selected_tower_type]
            if self.gold >= tower_def["cost"]:
                self.gold -= tower_def["cost"]
                self.towers.append({
                    "x": grid_x * self.GRID_SIZE + self.GRID_SIZE // 2,
                    "y": grid_y * self.GRID_SIZE + self.GRID_SIZE // 2,
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    "target": None
                })
                self.grid[grid_y][grid_x] = False # Mark cell as occupied
                # sfx: place_tower.wav

    def _update_waves(self):
        if self.enemies_in_wave == 0 and len(self.enemies) == 0 and self.wave_number <= self.TOTAL_WAVES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()
        
        if self.enemies_to_spawn:
            if self.steps % self.enemies_to_spawn[0]['spawn_delay'] == 0:
                enemy_data = self.enemies_to_spawn.pop(0)
                self._spawn_enemy(enemy_data['health'], enemy_data['speed'], enemy_data['value'])

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES:
            return

        wave_difficulty = 1 + (self.wave_number - 1) * 0.15
        num_enemies = 3 + self.wave_number * 2
        
        self.enemies_to_spawn.clear()
        for i in range(num_enemies):
            self.enemies_to_spawn.append({
                'health': 50 * wave_difficulty,
                'speed': 1.0 * wave_difficulty,
                'value': 10 + self.wave_number,
                'spawn_delay': 15 # spawn every 0.5s
            })
        self.enemies_in_wave = num_enemies
        self.wave_timer = 240 # 8s between waves

    def _spawn_enemy(self, health, speed, value):
        self.enemies.append({
            "x": self.path_points[0][0], "y": self.path_points[0][1],
            "health": health, "max_health": health,
            "speed": speed, "value": value,
            "path_index": 1
        })

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            # Movement
            if enemy["path_index"] >= len(self.path_points):
                self.base_health -= 10
                self.reward_buffer -= 10
                self.enemies.remove(enemy)
                self.enemies_in_wave -= 1
                # sfx: base_damage.wav
                continue

            target_x, target_y = self.path_points[enemy["path_index"]]
            dx, dy = target_x - enemy["x"], target_y - enemy["y"]
            dist = math.hypot(dx, dy)
            
            if dist < enemy["speed"]:
                enemy["path_index"] += 1
            else:
                enemy["x"] += (dx / dist) * enemy["speed"]
                enemy["y"] += (dy / dist) * enemy["speed"]

    def _update_towers(self):
        for tower in self.towers:
            tower_def = self.TOWER_TYPES[tower["type"]]
            
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            # Find a target
            target = None
            min_dist = float('inf')
            for enemy in self.enemies:
                dist = math.hypot(tower["x"] - enemy["x"], tower["y"] - enemy["y"])
                if dist <= tower_def["range"] and dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            # Fire
            if target:
                tower["cooldown"] = tower_def["fire_rate"]
                self.projectiles.append({
                    "x": tower["x"], "y": tower["y"],
                    "target": target,
                    "damage": tower_def["damage"],
                    "speed": tower_def["proj_speed"],
                    "size": tower_def["proj_size"],
                    "color": tower_def["color"]
                })
                # sfx: tower_shoot.wav

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_x, target_y = proj["target"]["x"], proj["target"]["y"]
            dx, dy = target_x - proj["x"], target_y - proj["y"]
            dist = math.hypot(dx, dy)
            
            if dist < proj["speed"]:
                # Hit
                proj["target"]["health"] -= proj["damage"]
                self.score += proj["damage"]
                self.reward_buffer += 0.1
                
                self._create_particles(proj["x"], proj["y"], proj["color"], 5, 2)
                # sfx: projectile_hit.wav

                if proj["target"]["health"] <= 0:
                    self.gold += proj["target"]["value"]
                    self.reward_buffer += 1.0
                    self._create_particles(proj["target"]["x"], proj["target"]["y"], COLOR_ENEMY, 20, 4)
                    self.enemies.remove(proj["target"])
                    self.enemies_in_wave -= 1
                    # sfx: enemy_destroyed.wav
                
                self.projectiles.remove(proj)
            else:
                # Move
                proj["x"] += (dx / dist) * proj["speed"]
                proj["y"] += (dy / dist) * proj["speed"]

    def _update_particles(self):
        for p in reversed(self.particles):
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return (
            self.base_health <= 0 or
            self.steps >= self.MAX_STEPS or
            (self.wave_number > self.TOTAL_WAVES and len(self.enemies) == 0)
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.wave_number,
        }

    def _initialize_grid(self):
        self.grid = [[True for _ in range(self.WIDTH // self.GRID_SIZE)] for _ in range(self.HEIGHT // self.GRID_SIZE)]
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                px, py = x * self.GRID_SIZE + self.GRID_SIZE//2, y * self.GRID_SIZE + self.GRID_SIZE//2
                for i in range(len(self.path_points) - 1):
                    p1 = self.path_points[i]
                    p2 = self.path_points[i+1]
                    # Check distance from point to line segment
                    l2 = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                    if l2 == 0: continue
                    t = max(0, min(1, ((px-p1[0])*(p2[0]-p1[0]) + (py-p1[1])*(p2[1]-p1[1])) / l2))
                    proj_x = p1[0] + t * (p2[0] - p1[0])
                    proj_y = p1[1] + t * (p2[1] - p1[1])
                    if math.hypot(px-proj_x, py-proj_y) < self.GRID_SIZE:
                        self.grid[y][x] = False
                        break

    def _render_game(self):
        # Draw grid
        for r, row in enumerate(self.grid):
            for c, valid in enumerate(row):
                if valid:
                    rect = (c * self.GRID_SIZE, r * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_points, 30)
        
        # Draw base
        base_rect = pygame.Rect(self.WIDTH - 20, self.base_pos[1] - 20, 40, 40)
        pygame.gfxdraw.box(self.screen, base_rect, self.COLOR_BASE_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Draw towers
        for t in self.towers:
            tower_def = self.TOWER_TYPES[t["type"]]
            pygame.draw.circle(self.screen, tower_def["color"], (int(t['x']), int(t['y'])), 12)
            pygame.draw.circle(self.screen, self.COLOR_BG, (int(t['x']), int(t['y'])), 8)
            pygame.draw.circle(self.screen, tower_def["color"], (int(t['x']), int(t['y'])), 5)

        # Draw projectiles
        for p in self.projectiles:
            pygame.draw.circle(self.screen, p["color"], (int(p['x']), int(p['y'])), p['size'])

        # Draw enemies
        for e in self.enemies:
            pos = (int(e['x']), int(e['y']))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, self.COLOR_ENEMY)
            # Health bar
            health_ratio = e['health'] / e['max_health']
            bar_len = 14
            pygame.draw.rect(self.screen, (80,0,0), (pos[0] - bar_len//2, pos[1] - 15, bar_len, 3))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (pos[0] - bar_len//2, pos[1] - 15, bar_len * health_ratio, 3))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'][:3] + (alpha,)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['x']) - size, int(p['y']) - size))

        # Draw cursor
        self._render_cursor()

    def _render_cursor(self):
        grid_x = int(self.cursor_pos[0] // self.GRID_SIZE)
        grid_y = int(self.cursor_pos[1] // self.GRID_SIZE)
        
        is_valid = False
        if 0 <= grid_y < len(self.grid) and 0 <= grid_x < len(self.grid[0]):
            if self.grid[grid_y][grid_x]:
                is_valid = True
        
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        has_gold = self.gold >= tower_def["cost"]
        
        color = self.COLOR_CURSOR_VALID if is_valid and has_gold else self.COLOR_CURSOR_INVALID
        center = (grid_x * self.GRID_SIZE + self.GRID_SIZE//2, grid_y * self.GRID_SIZE + self.GRID_SIZE//2)

        # Draw range indicator
        range_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(range_surf, center[0], center[1], tower_def['range'], color)
        self.screen.blit(range_surf, (0,0))
        
        # Draw cursor box
        rect = (center[0] - self.GRID_SIZE//2, center[1] - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, color[:3], rect, 2)

    def _render_ui(self):
        # Gold
        gold_text = self.font_large.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (10, 10))
        
        # Wave
        wave_str = f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}"
        if self.wave_timer > 0 and self.wave_number <= self.TOTAL_WAVES and len(self.enemies) == 0:
            wave_str += f" (Next in {self.wave_timer/30:.1f}s)"
        elif self.wave_number > self.TOTAL_WAVES:
            wave_str = "VICTORY!"
        wave_text = self.font_large.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Base Health
        health_text = self.font_large.render(f"BASE HP: {max(0, self.base_health)}", True, self.COLOR_BASE)
        self.screen.blit(health_text, (10, self.HEIGHT - health_text.get_height() - 10))
        
        # Selected Tower
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        tower_str = f"Tower: {tower_def['name']} (Cost: {tower_def['cost']})"
        tower_text = self.font_small.render(tower_str, True, tower_def['color'])
        self.screen.blit(tower_text, (self.WIDTH - tower_text.get_width() - 10, self.HEIGHT - tower_text.get_height() - 10))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY" if self.base_health > 0 else "GAME OVER"
            color = self.COLOR_BASE if self.base_health > 0 else self.COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))

    def _create_particles(self, x, y, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'color': color,
                'size': self.np_random.uniform(1, 3)
            })

    def close(self):
        pygame.quit()

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
    # This block allows you to run the file directly to play the game
    # It's a simple example of how to use the environment
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # none
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to Screen ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            print("Press 'R' to reset.")
            # Wait for reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
                clock.tick(30)

        clock.tick(30) # Limit FPS
        
    env.close()