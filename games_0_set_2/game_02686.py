
# Generated: 2025-08-27T21:09:41.214222
# Source Brief: brief_02686.md
# Brief Index: 2686

        
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
        "Controls: ↑↓←→ to move the cursor. Press space to place a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of invading enemies by strategically placing towers in an isometric 2D world."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_PATH = (60, 60, 75)
    COLOR_GRID = (45, 45, 60)
    COLOR_BASE = (255, 200, 0)
    COLOR_TOWER = (0, 255, 128)
    COLOR_ENEMY = (255, 70, 70)
    COLOR_PROJECTILE = (100, 180, 255)
    COLOR_CURSOR = (0, 255, 255)
    COLOR_HEALTH_BG = (80, 20, 20)
    COLOR_HEALTH_FG = (120, 255, 120)
    COLOR_TEXT = (230, 230, 230)
    
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Grid & Isometric Projection
    GRID_WIDTH = 20
    GRID_HEIGHT = 14
    TILE_WIDTH_HALF = 16
    TILE_HEIGHT_HALF = 8
    ISO_ORIGIN_X = SCREEN_WIDTH // 2
    ISO_ORIGIN_Y = 100

    # Game Flow
    MAX_STEPS = 30 * 180 # 3 minutes at 30fps
    TOTAL_WAVES = 5
    WAVE_PREP_TIME = 150 # 5 seconds

    # Tower Stats
    TOWER_RANGE = 100
    TOWER_COOLDOWN = 20 # frames

    # Projectile Stats
    PROJECTILE_SPEED = 8
    PROJECTILE_DAMAGE = 10

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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self._define_path_and_grid()
        self.np_random = None
        
        # Initialize state variables
        self.reset()
    
    def _define_path_and_grid(self):
        self.path_waypoints = [
            (-1, 5), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5),
            (4, 6), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7),
            (8, 6), (8, 5), (8, 4), (8, 3), (9, 3), (10, 3),
            (11, 3), (12, 3), (13, 3), (14, 3), (15, 3), (16, 3),
            (16, 4), (16, 5), (16, 6), (16, 7), (16, 8), (16, 9),
            (15, 9), (14, 9), (13, 9), (12, 9), (11, 9),
            (11, 10), (11, 11), (11, 12), (12, 12), (13, 12),
            (14, 12), (15, 12), (16, 12), (17, 12), (18, 12),
            (19, 12), (20, 12)
        ]
        self.path_coords = set(self.path_waypoints)
        self.start_pos = self._grid_to_iso(self.path_waypoints[0][0], self.path_waypoints[0][1])
        self.base_pos_grid = self.path_waypoints[-2]
        
        self.buildable_tiles = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in self.path_coords:
                    self.buildable_tiles.add((x, y))

    def _grid_to_iso(self, x, y):
        iso_x = self.ISO_ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.ISO_ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return iso_x, iso_y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.wave_number = 0
        self.wave_state = "PREP_WAVE" # PREP_WAVE, SPAWNING
        self.wave_timer = self.WAVE_PREP_TIME
        self.enemies_to_spawn_in_wave = 0
        self.spawn_timer = 0
        
        self.prev_space_held = False
        
        self._start_next_wave()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES:
            self.game_over = True
            self.victory = True
            return

        self.wave_state = "SPAWNING"
        self.enemies_to_spawn_in_wave = 15 + self.wave_number * 5
        self.spawn_cooldown = max(5, 30 - self.wave_number * 4)
        self.spawn_timer = self.spawn_cooldown
        
        self.enemy_health = 100 * (1.1 ** (self.wave_number - 1))
        self.enemy_speed = 1.0 * (1.05 ** (self.wave_number - 1))
        
    def _spawn_enemy(self):
        enemy = {
            "pos": list(self.start_pos),
            "health": self.enemy_health,
            "max_health": self.enemy_health,
            "speed": self.enemy_speed,
            "path_index": 0,
            "id": self.np_random.integers(1, 1_000_000)
        }
        self.enemies.append(enemy)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0.0

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # --- 1. Handle Player Input ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        if space_held and not self.prev_space_held:
            cursor_tuple = tuple(self.cursor_pos)
            is_buildable = cursor_tuple in self.buildable_tiles
            is_occupied = any(t['grid_pos'] == cursor_tuple for t in self.towers)
            if is_buildable and not is_occupied:
                iso_pos = self._grid_to_iso(self.cursor_pos[0], self.cursor_pos[1])
                new_tower = {
                    "grid_pos": cursor_tuple,
                    "iso_pos": iso_pos,
                    "cooldown": 0,
                }
                self.towers.append(new_tower)
                # sfx: tower_place.wav
        
        self.prev_space_held = space_held

        # --- 2. Update Wave and Spawning ---
        if self.wave_state == "SPAWNING":
            if self.enemies_to_spawn_in_wave > 0:
                self.spawn_timer -= 1
                if self.spawn_timer <= 0:
                    self._spawn_enemy()
                    self.enemies_to_spawn_in_wave -= 1
                    self.spawn_timer = self.spawn_cooldown
            elif len(self.enemies) == 0:
                if self.wave_number == self.TOTAL_WAVES:
                    self.game_over = True
                    self.victory = True
                else:
                    step_reward += 50
                    self.wave_state = "PREP_WAVE"
                    self.wave_timer = self.WAVE_PREP_TIME
        
        elif self.wave_state == "PREP_WAVE":
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()

        # --- 3. Update Enemies ---
        for enemy in self.enemies[:]:
            path_idx = enemy["path_index"]
            if path_idx >= len(self.path_waypoints) - 1:
                self.enemies.remove(enemy)
                self.game_over = True # Loss condition
                continue

            target_grid_pos = self.path_waypoints[path_idx + 1]
            target_iso_pos = self._grid_to_iso(target_grid_pos[0], target_grid_pos[1])
            
            direction = np.array(target_iso_pos) - np.array(enemy["pos"])
            distance = np.linalg.norm(direction)
            
            if distance < enemy["speed"]:
                enemy["pos"] = list(target_iso_pos)
                enemy["path_index"] += 1
            else:
                move_vec = (direction / distance) * enemy["speed"]
                enemy["pos"][0] += move_vec[0]
                enemy["pos"][1] += move_vec[1]

        # --- 4. Update Towers and Fire Projectiles ---
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] > 0:
                continue

            target = None
            min_dist = self.TOWER_RANGE
            for enemy in self.enemies:
                dist = np.linalg.norm(np.array(tower["iso_pos"]) - np.array(enemy["pos"]))
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                projectile = {
                    "pos": list(tower["iso_pos"]),
                    "target_id": target["id"],
                    "damage": self.PROJECTILE_DAMAGE,
                }
                self.projectiles.append(projectile)
                tower["cooldown"] = self.TOWER_COOLDOWN
                # sfx: laser_shoot.wav

        # --- 5. Update Projectiles ---
        for proj in self.projectiles[:]:
            target_enemy = next((e for e in self.enemies if e["id"] == proj["target_id"]), None)
            
            if not target_enemy:
                self.projectiles.remove(proj)
                continue

            direction = np.array(target_enemy["pos"]) - np.array(proj["pos"])
            distance = np.linalg.norm(direction)
            
            if distance < self.PROJECTILE_SPEED:
                # Hit
                target_enemy["health"] -= proj["damage"]
                step_reward += 0.1
                # sfx: hit_damage.wav
                self._create_particles(proj["pos"], self.COLOR_PROJECTILE, 3)
                self.projectiles.remove(proj)
            else:
                move_vec = (direction / distance) * self.PROJECTILE_SPEED
                proj["pos"][0] += move_vec[0]
                proj["pos"][1] += move_vec[1]

        # --- 6. Process Dead Enemies ---
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                step_reward += 1.0
                self.score += 10
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 10)
                self.enemies.remove(enemy)
                # sfx: enemy_explode.wav

        # --- 7. Update Particles ---
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        # --- 8. Check Termination Conditions ---
        terminated = False
        reward = step_reward

        if self.game_over:
            terminated = True
            if self.victory:
                reward = 100
                self.score += 1000
            else: # Loss by leak
                reward = -100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -100 # Loss by timeout
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 20),
                "color": color
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "enemies_left": len(self.enemies) + self.enemies_to_spawn_in_wave,
            "towers": len(self.towers),
        }

    def _render_game(self):
        # --- Render Grid and Path ---
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                iso_x, iso_y = self._grid_to_iso(x, y)
                points = [
                    (iso_x, iso_y),
                    (iso_x + self.TILE_WIDTH_HALF, iso_y + self.TILE_HEIGHT_HALF),
                    (iso_x, iso_y + self.TILE_HEIGHT_HALF * 2),
                    (iso_x - self.TILE_WIDTH_HALF, iso_y + self.TILE_HEIGHT_HALF)
                ]
                color = self.COLOR_PATH if (x,y) in self.path_coords else self.COLOR_GRID
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, (color[0]+10, color[1]+10, color[2]+10))

        # --- Render Base ---
        base_iso_x, base_iso_y = self._grid_to_iso(self.base_pos_grid[0], self.base_pos_grid[1])
        base_points = [
            (base_iso_x, base_iso_y),
            (base_iso_x + self.TILE_WIDTH_HALF, base_iso_y + self.TILE_HEIGHT_HALF),
            (base_iso_x, base_iso_y + self.TILE_HEIGHT_HALF * 2),
            (base_iso_x - self.TILE_WIDTH_HALF, base_iso_y + self.TILE_HEIGHT_HALF)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, base_points, self.COLOR_BASE)
        pygame.gfxdraw.aapolygon(self.screen, base_points, (255,255,100))

        # --- Collect and sort all dynamic entities for correct Z-ordering ---
        render_list = []
        for t in self.towers:
            render_list.append({"type": "tower", "pos": t['iso_pos'], "y": t['iso_pos'][1]})
        for e in self.enemies:
            render_list.append({"type": "enemy", "entity": e, "y": e['pos'][1]})
        
        render_list.sort(key=lambda item: item['y'])

        # --- Render sorted entities ---
        for item in render_list:
            if item['type'] == 'tower':
                pos = item['pos']
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1] - self.TILE_HEIGHT_HALF), 8, self.COLOR_TOWER)
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1] - self.TILE_HEIGHT_HALF), 8, (200, 255, 200))
            elif item['type'] == 'enemy':
                e = item['entity']
                pos = e['pos']
                size = 8
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), size, self.COLOR_ENEMY)
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), size, (255,150,150))
                # Health bar
                health_perc = max(0, e['health'] / e['max_health'])
                bar_width = 20
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (pos[0] - bar_width/2, pos[1] - 18, bar_width, 4))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (pos[0] - bar_width/2, pos[1] - 18, bar_width * health_perc, 4))

        # --- Render Projectiles ---
        for proj in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 3, (200,220,255))

        # --- Render Particles ---
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20.0))
            color = (*p['color'], alpha)
            if alpha > 0:
                s = self.screen.copy()
                s.set_colorkey((0,0,0))
                pygame.gfxdraw.filled_circle(s, int(p['pos'][0]), int(p['pos'][1]), 2, color)
                s.set_alpha(alpha)
                self.screen.blit(s, (0,0))


        # --- Render Cursor ---
        cursor_iso_x, cursor_iso_y = self._grid_to_iso(self.cursor_pos[0], self.cursor_pos[1])
        cursor_points = [
            (cursor_iso_x, cursor_iso_y),
            (cursor_iso_x + self.TILE_WIDTH_HALF, cursor_iso_y + self.TILE_HEIGHT_HALF),
            (cursor_iso_x, cursor_iso_y + self.TILE_HEIGHT_HALF * 2),
            (cursor_iso_x - self.TILE_WIDTH_HALF, cursor_iso_y + self.TILE_HEIGHT_HALF)
        ]
        
        is_buildable = tuple(self.cursor_pos) in self.buildable_tiles
        is_occupied = any(t['grid_pos'] == tuple(self.cursor_pos) for t in self.towers)
        cursor_color = self.COLOR_CURSOR if is_buildable and not is_occupied else self.COLOR_ENEMY
        
        alpha = 128 + math.sin(self.steps * 0.2) * 64
        
        temp_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(temp_surface, cursor_points, (*cursor_color, int(alpha)))
        pygame.gfxdraw.aapolygon(temp_surface, cursor_points, cursor_color)
        self.screen.blit(temp_surface, (0,0))


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Wave info
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        enemies_left = len(self.enemies) + self.enemies_to_spawn_in_wave
        enemies_text = self.font_small.render(f"ENEMIES: {enemies_left}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_text, (10, 30))

        # Game Over / Victory Message
        if self.game_over:
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_TOWER if self.victory else self.COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)
        elif self.wave_state == "PREP_WAVE":
            prep_time_sec = math.ceil(self.wave_timer / 30)
            msg = f"WAVE {self.wave_number+1} IN {prep_time_sec}"
            prep_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = prep_text.get_rect(center=(self.SCREEN_WIDTH/2, 40))
            self.screen.blit(prep_text, text_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to see the game
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    
    # To store the state of the keys
    keys = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
    }

    # Re-initialize pygame with a display
    pygame.display.init()
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys:
                    keys[event.key] = True
                if event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
            if event.type == pygame.KEYUP:
                if event.key in keys:
                    keys[event.key] = False

        # --- Action Mapping ---
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Reward: {reward}")
            # In a real game, you might wait for a reset key press
            # For this demo, we'll let it sit on the game over screen
            # You can press 'R' to reset.

        # --- Rendering ---
        # The observation is already the rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()