
# Generated: 2025-08-28T04:15:35.963679
# Source Brief: brief_02257.md
# Brief Index: 2257

        
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
        "Controls: ↑↓←→ to move the placement cursor. Press space to place a tower."
    )

    game_description = (
        "Defend your base from waves of geometric enemies by strategically placing defensive towers."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PATH = (40, 50, 70)
    COLOR_GRID = (60, 70, 90)
    COLOR_BASE = (0, 200, 100)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TOWER = (0, 255, 150)
    COLOR_PROJECTILE = (100, 200, 255)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CURSOR_INVALID = (255, 100, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PARTICLE = (255, 200, 80)
    
    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Game Parameters
    MAX_STEPS = 3000 # Increased from 1000 to allow for 5 waves at 30fps
    NUM_WAVES = 5
    INITIAL_BASE_HEALTH = 100
    INITIAL_RESOURCES = 150
    TOWER_COST = 50
    REWARD_KILL = 1.0
    REWARD_HIT = 0.1
    REWARD_WAVE_CLEAR = 50.0
    REWARD_GAME_WIN = 100.0
    PENALTY_LEAK = -5.0
    
    # Grid
    GRID_OFFSET_X, GRID_OFFSET_Y = 40, 40
    CELL_SIZE = 40
    GRID_COLS, GRID_ROWS = 14, 8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game State (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.wave_countdown = 0
        self.spawn_timer = 0
        self.spawn_queue = []
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.grid = []
        self.cursor_pos = [0, 0]
        self.prev_space_held = False
        self.total_reward_this_step = 0
        
        # Path definition
        self.path_waypoints = [
            pygame.math.Vector2(-20, 200),
            pygame.math.Vector2(100, 200),
            pygame.math.Vector2(100, 100),
            pygame.math.Vector2(540, 100),
            pygame.math.Vector2(540, 300),
            pygame.math.Vector2(300, 300),
            pygame.math.Vector2(300, 220),
            pygame.math.Vector2(self.WIDTH + 20, 220)
        ]
        
        self.base_pos = pygame.math.Vector2(self.WIDTH - 30, 220)

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.current_wave = 0
        self.wave_countdown = 90  # 3 seconds at 30fps
        self.spawn_timer = 0
        self.spawn_queue = []
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.prev_space_held = True # Prevent placing tower on first frame
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.total_reward_this_step = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self._handle_input(action)
        
        self._update_waves()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        self.steps += 1
        
        terminated = self._check_termination()
        
        self.score += self.total_reward_this_step
        
        return (
            self._get_observation(),
            self.total_reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held_raw, _ = action
        space_held = space_held_raw == 1
        
        # --- Cursor Movement ---
        if movement == 1: # Up
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: # Down
            self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
        elif movement == 3: # Left
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: # Right
            self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)

        # --- Tower Placement ---
        if space_held and not self.prev_space_held:
            row, col = self.cursor_pos
            if self.resources >= self.TOWER_COST and self.grid[row][col] is None:
                self.resources -= self.TOWER_COST
                pos_x = self.GRID_OFFSET_X + col * self.CELL_SIZE + self.CELL_SIZE / 2
                pos_y = self.GRID_OFFSET_Y + row * self.CELL_SIZE + self.CELL_SIZE / 2
                new_tower = {
                    "pos": pygame.math.Vector2(pos_x, pos_y),
                    "range": 80,
                    "cooldown": 0,
                    "fire_rate": 25, # frames per shot
                    "damage": 10,
                    "angle": -90
                }
                self.towers.append(new_tower)
                self.grid[row][col] = new_tower
                # sfx: tower_place.wav

        self.prev_space_held = space_held

    def _update_waves(self):
        if not self.enemies and not self.spawn_queue and self.current_wave <= self.NUM_WAVES:
            if self.wave_countdown > 0:
                self.wave_countdown -= 1
            else:
                if self.current_wave > 0:
                    self.total_reward_this_step += self.REWARD_WAVE_CLEAR
                if self.current_wave < self.NUM_WAVES:
                    self._start_next_wave()
                else: # Game won
                    self.game_over = True
                    self.total_reward_this_step += self.REWARD_GAME_WIN
        
        if self.spawn_queue:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self.enemies.append(self.spawn_queue.pop(0))
                self.spawn_timer = 20 # frames between spawns

    def _start_next_wave(self):
        self.current_wave += 1
        self.wave_countdown = 150 # 5 seconds
        
        num_enemies = 5 + self.current_wave * 2
        enemy_health = 20 * (1 + (self.current_wave - 1) * 0.1)
        enemy_speed = 1.0 * (1 + (self.current_wave - 1) * 0.1)

        for _ in range(num_enemies):
            self.spawn_queue.append({
                "pos": self.path_waypoints[0].copy(),
                "health": enemy_health,
                "max_health": enemy_health,
                "speed": enemy_speed,
                "waypoint_index": 1,
                "radius": 7
            })

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            target = None
            max_dist = -1

            for enemy in self.enemies:
                dist_to_tower = tower["pos"].distance_to(enemy["pos"])
                if dist_to_tower <= tower["range"]:
                    # Target enemy closest to the base
                    dist_along_path = self._get_enemy_path_distance(enemy)
                    if dist_along_path > max_dist:
                        max_dist = dist_along_path
                        target = enemy
            
            if target:
                # sfx: laser_shoot.wav
                tower["cooldown"] = tower["fire_rate"]
                direction = (target["pos"] - tower["pos"]).normalize()
                tower["angle"] = direction.angle_to(pygame.math.Vector2(1, 0))
                
                self.projectiles.append({
                    "pos": tower["pos"].copy(),
                    "vel": direction * 8,
                    "damage": tower["damage"],
                    "lifespan": 40
                })

    def _get_enemy_path_distance(self, enemy):
        dist = 0
        # Distance of completed segments
        for i in range(enemy["waypoint_index"] - 1):
            dist += self.path_waypoints[i].distance_to(self.path_waypoints[i+1])
        # Distance along current segment
        dist += enemy["pos"].distance_to(self.path_waypoints[enemy["waypoint_index"] - 1])
        return dist

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj["pos"] += proj["vel"]
            proj["lifespan"] -= 1

            if proj["lifespan"] <= 0:
                self.projectiles.remove(proj)
                continue

            for enemy in self.enemies:
                if proj["pos"].distance_to(enemy["pos"]) < enemy["radius"]:
                    enemy["health"] -= proj["damage"]
                    self.total_reward_this_step += self.REWARD_HIT
                    self._create_particles(proj["pos"], 5, self.COLOR_PROJECTILE)
                    # sfx: hit_enemy.wav
                    if proj in self.projectiles:
                        self.projectiles.remove(proj)
                    break

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy["waypoint_index"] >= len(self.path_waypoints):
                self.base_health -= 10
                self.total_reward_this_step += self.PENALTY_LEAK
                self._create_particles(enemy["pos"], 20, self.COLOR_BASE_DMG)
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                continue
            
            target_pos = self.path_waypoints[enemy["waypoint_index"]]
            direction = (target_pos - enemy["pos"])
            
            if direction.length() < enemy["speed"]:
                enemy["pos"] = target_pos.copy()
                enemy["waypoint_index"] += 1
            else:
                enemy["pos"] += direction.normalize() * enemy["speed"]
            
            if enemy["health"] <= 0:
                self.resources += 20
                self.total_reward_this_step += self.REWARD_KILL
                self._create_particles(enemy["pos"], 30, self.COLOR_PARTICLE)
                self.enemies.remove(enemy)
                # sfx: enemy_explode.wav

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        # Win condition is handled in _update_waves
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_path()
        self._render_grid()
        self._render_base()
        for tower in self.towers: self._render_tower(tower)
        for proj in self.projectiles: self._render_projectile(proj)
        for enemy in self.enemies: self._render_enemy(enemy)
        for p in self.particles: self._render_particle(p)
        self._render_cursor()

    def _render_path(self):
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, [(int(p.x), int(p.y)) for p in self.path_waypoints], 20)

    def _render_grid(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                x = self.GRID_OFFSET_X + c * self.CELL_SIZE
                y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
    
    def _render_base(self):
        size = 30
        rect = pygame.Rect(self.base_pos.x - size/2, self.base_pos.y - size/2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_BASE, rect)
        pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_BASE), rect, 3)

    def _render_tower(self, tower):
        pos = (int(tower["pos"].x), int(tower["pos"].y))
        size = 12
        
        # Body
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_TOWER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_TOWER)
        
        # Barrel
        angle_rad = math.radians(-tower["angle"])
        end_x = pos[0] + (size + 5) * math.cos(angle_rad)
        end_y = pos[1] + (size + 5) * math.sin(angle_rad)
        pygame.draw.line(self.screen, self.COLOR_TOWER, pos, (int(end_x), int(end_y)), 4)
        
    def _render_enemy(self, enemy):
        pos = (int(enemy["pos"].x), int(enemy["pos"].y))
        radius = int(enemy["radius"])
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)

        # Health bar
        if enemy["health"] < enemy["max_health"]:
            bar_w = 20
            bar_h = 4
            bar_x = pos[0] - bar_w / 2
            bar_y = pos[1] - radius - bar_h - 2
            health_pct = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, bar_w * health_pct, bar_h))

    def _render_projectile(self, proj):
        pos = (int(proj["pos"].x), int(proj["pos"].y))
        pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, pos, 3)

    def _render_particle(self, p):
        alpha = max(0, min(255, int(255 * (p["lifespan"] / p["max_lifespan"]))))
        color = p["color"] + (alpha,)
        temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
        self.screen.blit(temp_surf, (int(p["pos"].x - p["size"]), int(p["pos"].y - p["size"])))

    def _render_cursor(self):
        row, col = self.cursor_pos
        x = self.GRID_OFFSET_X + col * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + row * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        can_afford = self.resources >= self.TOWER_COST
        is_empty = self.grid[row][col] is None
        color = self.COLOR_CURSOR if (can_afford and is_empty) else self.COLOR_CURSOR_INVALID
        
        pygame.draw.rect(self.screen, color, rect, 2)

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.WIDTH, 35), pygame.SRCALPHA)
        ui_panel.fill((10, 15, 30, 180))
        self.screen.blit(ui_panel, (0, self.HEIGHT - 35))

        # Base Health
        health_text = self.font_small.render(f"Base HP: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, self.HEIGHT - 28))
        
        # Resources
        resource_text = self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(resource_text, (160, self.HEIGHT - 28))
        
        # Wave
        wave_str = f"Wave: {self.current_wave}/{self.NUM_WAVES}"
        if not self.enemies and not self.spawn_queue and self.current_wave < self.NUM_WAVES:
            wave_str += f" (Next in {self.wave_countdown / 30:.1f}s)"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (320, self.HEIGHT - 28))
        
        # Score
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (520, self.HEIGHT - 28))

        # Game Over / Win Text
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            
            msg = "YOU WIN!" if self.base_health > 0 and self.current_wave > self.NUM_WAVES else "GAME OVER"
            color = self.COLOR_BASE if msg == "YOU WIN!" else self.COLOR_ENEMY
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            lifespan = random.randint(15, 30)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "size": random.randint(1, 4),
                "color": color
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "resources": self.resources,
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
        assert self.base_health == self.INITIAL_BASE_HEALTH
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # Test reward range
        assert -10 <= reward <= 10 or abs(reward) >= 50 # Allow for large terminal/wave rewards
        
        # Test termination
        self.steps = self.MAX_STEPS + 1
        assert self._check_termination() is True
        self.reset()
        self.base_health = 0
        assert self._check_termination() is True
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    
    # Game loop
    running = True
    terminated = False
    
    # To store the action
    action = env.action_space.sample()
    action[0] = 0 # no movement
    action[1] = 0 # space not held
    action[2] = 0 # shift not held

    # Create a window to display the game
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    print(GameEnv.user_guide)

    while running:
        if terminated:
            obs, info = env.reset()
            terminated = False

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 0
        
        # --- Continuous Key Presses for Movement ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Tick the Clock ---
        clock.tick(30) # Run at 30 FPS

    env.close()