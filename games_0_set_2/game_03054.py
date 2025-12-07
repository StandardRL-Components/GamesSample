
# Generated: 2025-08-27T22:13:48.258148
# Source Brief: brief_03054.md
# Brief Index: 3054

        
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
    """
    A minimalist tower defense game Gymnasium environment.
    The player places towers to defend a base against waves of enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. Press 'Space' to "
        "place the selected tower. Press 'Shift' to cycle through tower types."
    )
    game_description = (
        "Defend your base from descending waves of enemies by strategically "
        "placing defensive towers in this minimalist tower defense game."
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    # Game parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    CELL_SIZE = 40
    FPS = 30
    MAX_STEPS = 15000 # ~8 minutes at 30fps
    TOTAL_WAVES = 15

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_PATH = (50, 50, 80)
    COLOR_BASE = (0, 150, 50)
    COLOR_BASE_DMG = (200, 50, 50)
    COLOR_GOLD = (255, 200, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR_VALID = (200, 255, 200, 100)
    COLOR_CURSOR_INVALID = (255, 100, 100, 100)
    
    ENEMY_COLORS = [(220, 50, 50), (255, 120, 50), (255, 200, 50)]
    TOWER_COLORS = [(60, 120, 255), (60, 220, 255), (180, 60, 255)]
    PROJ_COLORS = [(150, 200, 255), (150, 255, 255), (220, 150, 255)]

    # Path definition (grid coordinates)
    PATH_WAYPOINTS = [
        (-1, 5), (2, 5), (2, 2), (6, 2), (6, 7), (10, 7), (10, 3), (13, 3), (13, 8), (16, 8)
    ]

    # Tower specifications: [cost, range_px, damage, fire_rate_ticks, unlock_wave]
    TOWER_SPECS = {
        0: [30, 80, 10, 30, 1],  # Basic
        1: [75, 60, 5, 10, 5],   # Rapid
        2: [100, 150, 35, 60, 10] # Sniper
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # Game state variables are initialized in reset()
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = 100
        self.gold = 80
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        # Wave management
        self.current_wave_idx = -1
        self.wave_spawn_timer = 0
        self.enemies_to_spawn = 0
        self.inter_wave_timer = 90 # 3 seconds before first wave

        # Player state
        self.cursor_pos = np.array([1, 1])
        self.selected_tower_type = 0
        self.available_tower_types = 1
        
        # Input handling flags
        self._space_pressed_last_frame = False
        self._shift_pressed_last_frame = False
        self._move_cooldown = 0

        # Pre-calculate path pixel coordinates
        self.path_pixels = [np.array(p) * self.CELL_SIZE + self.CELL_SIZE / 2 for p in self.PATH_WAYPOINTS]
        self.path_grid_coords = set()
        for i in range(len(self.PATH_WAYPOINTS) - 1):
            p1 = self.PATH_WAYPOINTS[i]
            p2 = self.PATH_WAYPOINTS[i+1]
            for x in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                self.path_grid_coords.add((x, p1[1]))
            for y in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                self.path_grid_coords.add((p2[0], y))
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = self.base_health <= 0 or self.steps >= self.MAX_STEPS
        if self.game_over:
            reward += -100 if not self.win else 0
            return self._get_observation(), reward, True, False, self._get_info()

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        reward += self._update_waves()
        self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # --- Finalization ---
        self.steps += 1
        self.score += reward
        self.clock.tick(self.FPS)
        
        terminated = self.game_over
        if self.win:
            reward += 100
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        if self._move_cooldown > 0:
            self._move_cooldown -= 1
        
        if movement != 0 and self._move_cooldown == 0:
            self._move_cooldown = 4 # Cooldown to prevent flying
            if movement == 1: self.cursor_pos[1] -= 1 # Up
            elif movement == 2: self.cursor_pos[1] += 1 # Down
            elif movement == 3: self.cursor_pos[0] -= 1 # Left
            elif movement == 4: self.cursor_pos[0] += 1 # Right
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # --- Place Tower (Space) ---
        space_just_pressed = space_held and not self._space_pressed_last_frame
        if space_just_pressed:
            cost = self.TOWER_SPECS[self.selected_tower_type][0]
            if self.gold >= cost and self._is_valid_placement(self.cursor_pos):
                self.gold -= cost
                new_tower = {
                    "pos": self.cursor_pos.copy(),
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    "target": None,
                }
                self.towers.append(new_tower)
                # SFX: place_tower.wav
        self._space_pressed_last_frame = space_held

        # --- Cycle Tower Type (Shift) ---
        shift_just_pressed = shift_held and not self._shift_pressed_last_frame
        if shift_just_pressed:
            self.selected_tower_type = (self.selected_tower_type + 1) % self.available_tower_types
            # SFX: cycle_weapon.wav
        self._shift_pressed_last_frame = shift_held

    def _is_valid_placement(self, grid_pos):
        if tuple(grid_pos) in self.path_grid_coords:
            return False
        for tower in self.towers:
            if np.array_equal(tower["pos"], grid_pos):
                return False
        return True

    def _update_waves(self):
        reward = 0
        
        # Check for win condition
        if self.current_wave_idx == self.TOTAL_WAVES - 1 and not self.enemies and self.enemies_to_spawn == 0:
            self.win = True
            return reward
        
        # Update available towers based on wave
        if self.current_wave_idx + 2 >= self.TOWER_SPECS[1][4]: self.available_tower_types = 2
        if self.current_wave_idx + 2 >= self.TOWER_SPECS[2][4]: self.available_tower_types = 3

        if self.inter_wave_timer > 0:
            self.inter_wave_timer -= 1
            return reward

        if self.enemies_to_spawn == 0 and not self.enemies:
            self.current_wave_idx += 1
            if self.current_wave_idx < self.TOTAL_WAVES:
                reward += 1.0 # Wave clear bonus
                self.inter_wave_timer = 150 # 5s between waves
                self._setup_wave(self.current_wave_idx)
            return reward
        
        self.wave_spawn_timer -= 1
        if self.wave_spawn_timer <= 0 and self.enemies_to_spawn > 0:
            self.enemies_to_spawn -= 1
            self.wave_spawn_timer = self.current_wave_config["spawn_delay"]
            self._spawn_enemy()
            
        return reward
        
    def _setup_wave(self, wave_idx):
        base_health = 10 + wave_idx * 5
        base_speed = 1.0 + wave_idx * 0.1
        num_enemies = 5 + wave_idx * 2
        
        # Apply difficulty scaling from brief
        scaled_health = base_health + (wave_idx // 2)
        scaled_speed = base_speed + (wave_idx // 3) * 0.05

        self.current_wave_config = {
            "num_enemies": num_enemies,
            "health": scaled_health,
            "speed": scaled_speed,
            "spawn_delay": max(5, 30 - wave_idx),
            "gold_reward": 2 + wave_idx // 2,
            "type": wave_idx % len(self.ENEMY_COLORS)
        }
        self.enemies_to_spawn = self.current_wave_config["num_enemies"]

    def _spawn_enemy(self):
        new_enemy = {
            "id": self.np_random.integers(1, 1_000_000),
            "pos": self.path_pixels[0].copy(),
            "health": self.current_wave_config["health"],
            "max_health": self.current_wave_config["health"],
            "speed": self.current_wave_config["speed"],
            "path_idx": 1,
            "type": self.current_wave_config["type"]
        }
        self.enemies.append(new_enemy)

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            tower_pos_px = tower["pos"] * self.CELL_SIZE + self.CELL_SIZE / 2
            
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            # Find a target
            target = None
            for enemy in self.enemies:
                dist = np.linalg.norm(enemy["pos"] - tower_pos_px)
                if dist <= spec[1]:
                    target = enemy
                    break
            
            if target:
                tower["cooldown"] = spec[3] # Set cooldown
                # SFX: shoot.wav
                self.projectiles.append({
                    "pos": tower_pos_px.copy(),
                    "target_id": target["id"],
                    "speed": 8,
                    "damage": spec[2],
                    "type": tower["type"]
                })

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            target_enemy = next((e for e in self.enemies if e["id"] == proj["target_id"]), None)
            
            if not target_enemy:
                continue # Target is gone, projectile fizzles

            direction = target_enemy["pos"] - proj["pos"]
            dist = np.linalg.norm(direction)
            
            if dist < proj["speed"]: # Hit
                target_enemy["health"] -= proj["damage"]
                # SFX: hit.wav
                self._create_hit_particles(proj["pos"], self.PROJ_COLORS[proj["type"]])
                if target_enemy["health"] <= 0:
                    reward += 0.1 # Kill reward
                    self.gold += self.current_wave_config["gold_reward"]
                    self.enemies = [e for e in self.enemies if e["id"] != target_enemy["id"]]
                    # SFX: enemy_die.wav
                    self._create_explosion_particles(target_enemy["pos"], self.ENEMY_COLORS[target_enemy["type"]])
            else: # Move projectile
                proj["pos"] += (direction / dist) * proj["speed"]
                projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep
        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_keep = []
        for enemy in self.enemies:
            target_waypoint = self.path_pixels[enemy["path_idx"]]
            direction = target_waypoint - enemy["pos"]
            dist = np.linalg.norm(direction)

            if dist < enemy["speed"]:
                enemy["path_idx"] += 1
                if enemy["path_idx"] >= len(self.path_pixels):
                    self.base_health -= 10 # Enemy reached base
                    # SFX: base_damage.wav
                    self._create_explosion_particles(enemy["pos"], self.COLOR_BASE_DMG)
                    continue # Enemy is removed
                enemy["pos"] = target_waypoint
            else:
                enemy["pos"] += (direction / dist) * enemy["speed"]
            
            enemies_to_keep.append(enemy)
        self.enemies = enemies_to_keep
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _create_hit_particles(self, pos, color):
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "life": self.np_random.integers(5, 10),
                "color": color,
                "size": self.np_random.integers(1, 3)
            })

    def _create_explosion_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "life": self.np_random.integers(10, 25),
                "color": color,
                "size": self.np_random.integers(2, 5)
            })
            
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
            "wave": self.current_wave_idx + 1,
        }

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_COLS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_SIZE, 0), (x * self.CELL_SIZE, self.SCREEN_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_SIZE), (self.SCREEN_WIDTH, y * self.CELL_SIZE))

        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_pixels, self.CELL_SIZE)
        
        # Draw base
        base_pos = self.path_pixels[-1] - self.CELL_SIZE/2
        pygame.draw.rect(self.screen, self.COLOR_BASE, (*base_pos, self.CELL_SIZE, self.CELL_SIZE))

        # Draw towers
        for tower in self.towers:
            pos_px = (tower["pos"] * self.CELL_SIZE) + (self.CELL_SIZE / 2)
            color = self.TOWER_COLORS[tower["type"]]
            points = [
                (pos_px[0], pos_px[1] - 12),
                (pos_px[0] - 10, pos_px[1] + 8),
                (pos_px[0] + 10, pos_px[1] + 8),
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)

        # Draw projectiles
        for proj in self.projectiles:
            color = self.PROJ_COLORS[proj["type"]]
            pygame.draw.circle(self.screen, color, proj["pos"].astype(int), 3)

        # Draw enemies
        for enemy in self.enemies:
            pos_int = enemy["pos"].astype(int)
            color = self.ENEMY_COLORS[enemy["type"]]
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 8, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, color)
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (255,0,0), (pos_int[0]-10, pos_int[1]-15, 20, 3))
            pygame.draw.rect(self.screen, (0,255,0), (pos_int[0]-10, pos_int[1]-15, int(20*health_ratio), 3))


        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 20.0))))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, p["pos"] - p["size"])
            
        # Draw cursor
        cursor_color = self.COLOR_CURSOR_VALID if self._is_valid_placement(self.cursor_pos) else self.COLOR_CURSOR_INVALID
        cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Draw transparent fill for cursor
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(cursor_color)
        self.screen.blit(s, cursor_rect.topleft)
        
        # Draw tower range preview
        spec = self.TOWER_SPECS[self.selected_tower_type]
        center_px = (cursor_rect.left + self.CELL_SIZE//2, cursor_rect.top + self.CELL_SIZE//2)
        pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], spec[1], (255,255,255,50))

    def _render_ui(self):
        # Health Bar
        pygame.draw.rect(self.screen, (50, 0, 0), (10, 10, 200, 20))
        health_w = max(0, int(200 * (self.base_health / 100)))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (10, 10, health_w, 20))
        health_text = self.font_s.render(f"Base: {self.base_health}/100", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Gold
        gold_text = self.font_m.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (self.SCREEN_WIDTH - gold_text.get_width() - 10, 10))

        # Wave Info
        wave_str = f"Wave {self.current_wave_idx + 1} / {self.TOTAL_WAVES}"
        if self.inter_wave_timer > 0 and self.current_wave_idx < self.TOTAL_WAVES -1:
            wave_str = f"Next wave in {self.inter_wave_timer / self.FPS:.1f}s"
        wave_text = self.font_m.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH // 2 - wave_text.get_width() // 2, self.SCREEN_HEIGHT - 35))

        # Selected Tower Info
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_info_str = f"Tower: {['Basic', 'Rapid', 'Sniper'][self.selected_tower_type]} (Cost: {spec[0]})"
        tower_info_text = self.font_s.render(tower_info_str, True, self.COLOR_TEXT)
        self.screen.blit(tower_info_text, (10, self.SCREEN_HEIGHT - 25))

        # Game Over/Win message
        if self.game_over or self.win:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.win else "GAME OVER"
            msg_text = self.font_l.render(msg, True, self.COLOR_GOLD if self.win else self.COLOR_BASE_DMG)
            self.screen.blit(msg_text, (self.SCREEN_WIDTH // 2 - msg_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_text.get_height() // 2))

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
    
    # Create a display window
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running and not terminated:
        # --- Player Input via Pygame ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        # Need to transpose it back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {total_reward}")
            print(f"Info: {info}")
            # Wait for a moment before closing
            pygame.time.wait(3000)
            running = False
            
    env.close()