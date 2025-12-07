
# Generated: 2025-08-28T00:41:49.754896
# Source Brief: brief_03872.md
# Brief Index: 3872

        
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
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Space to build the selected tower. Press Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of zombies by strategically placing defensive towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    CELL_SIZE = 20
    UI_HEIGHT = 60
    GAME_AREA_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_PATH = (25, 30, 45)
    COLOR_BASE = (0, 150, 200)
    COLOR_BASE_GLOW = (0, 150, 200, 50)
    COLOR_ZOMBIE = (220, 50, 50)
    COLOR_ZOMBIE_GLOW = (220, 50, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (40, 50, 70, 200)
    
    TOWER_SPECS = {
        0: {"name": "Gatling", "cost": 50, "range": 4, "damage": 5, "fire_rate": 5, "color": (0, 200, 255), "proj_speed": 8},
        1: {"name": "Cannon", "cost": 120, "range": 6, "damage": 25, "fire_rate": 30, "color": (255, 180, 0), "proj_speed": 6},
    }

    # Game parameters
    MAX_STEPS = 5400 # 30fps * 180s (3 minutes)
    INITIAL_MONEY = 150
    BASE_MAX_HEALTH = 100
    INTERMISSION_TIME = 150 # 5 seconds
    
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
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Path definition (grid coordinates)
        self.path_points = [(-1, 10), (10, 10), (10, 4), (22, 4), (22, 13), (32, 13)]
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.money = self.INITIAL_MONEY
        self.base_health = self.BASE_MAX_HEALTH
        self.game_over = False
        self.game_won = False
        
        self.towers = []
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.wave_number = 0
        self.wave_state = "intermission" # intermission, spawning, active
        self.wave_timer = self.INTERMISSION_TIME // 2

        self.cursor_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        self.selected_tower_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reward_this_step = 0

        self.occupied_cells = set()
        self._precompute_path_cells()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_this_step = 0

        self._handle_input(action)
        self._update_game_state()

        self.steps += 1
        
        reward = self.reward_this_step
        terminated = self._check_termination()

        if terminated:
            if self.game_won:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        if space_held and not self.prev_space_held:
            self._place_tower()
        
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_state(self):
        self._update_waves()
        self._update_towers()
        self._update_zombies()
        self._update_projectiles()
        self._update_particles()
    
    def _update_waves(self):
        if self.wave_state == "intermission":
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_new_wave()
        elif self.wave_state == "spawning":
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.zombies_to_spawn > 0:
                self._spawn_zombie()
                self.zombies_to_spawn -= 1
                self.wave_timer = 15 # Time between spawns
            elif self.zombies_to_spawn == 0:
                self.wave_state = "active"
        elif self.wave_state == "active":
            if not self.zombies:
                self.reward_this_step += 5 # Wave survived bonus
                if self.wave_number >= 5:
                    self.game_won = True
                    self.game_over = True
                else:
                    self.wave_state = "intermission"
                    self.wave_timer = self.INTERMISSION_TIME
    
    def _start_new_wave(self):
        self.wave_number += 1
        self.wave_state = "spawning"
        self.zombies_to_spawn = 4 + math.ceil(self.wave_number * self.wave_number * 0.8)
        self.wave_timer = 0

    def _spawn_zombie(self):
        health = 50 + self.wave_number * 10
        speed = 0.03 + self.wave_number * 0.002
        self.zombies.append({
            "pos": np.array([-0.5, 10.5]),
            "health": health,
            "max_health": health,
            "speed": speed,
            "path_index": 1,
        })
    
    def _update_towers(self):
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] > 0:
                continue

            target = None
            min_dist = tower["spec"]["range"]
            for zombie in self.zombies:
                dist = np.linalg.norm(tower["pos"] - zombie["pos"])
                if dist < min_dist:
                    min_dist = dist
                    target = zombie
            
            if target:
                # sfx: tower_shoot
                tower["cooldown"] = tower["spec"]["fire_rate"]
                self.projectiles.append({
                    "pos": tower["pos"].copy(),
                    "target": target,
                    "spec": tower["spec"],
                })

    def _update_zombies(self):
        for z in self.zombies[:]:
            if z["path_index"] >= len(self.path_points):
                self.base_health -= 10
                self.zombies.remove(z)
                # sfx: base_hit
                self._create_particles(np.array([self.GRID_WIDTH-1, 13.5]), 20, (255, 0, 0))
                continue

            target_pos = np.array(self.path_points[z["path_index"]]) + 0.5
            direction = target_pos - z["pos"]
            dist = np.linalg.norm(direction)

            if dist < z["speed"]:
                z["pos"] = target_pos
                z["path_index"] += 1
            else:
                z["pos"] += (direction / dist) * z["speed"]

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            if p["target"] not in self.zombies:
                self.projectiles.remove(p)
                continue

            direction = p["target"]["pos"] - p["pos"]
            dist = np.linalg.norm(direction)
            
            if dist < 0.5:
                # sfx: zombie_hit
                p["target"]["health"] -= p["spec"]["damage"]
                self.reward_this_step += 0.1
                self._create_particles(p["pos"], 5, p["spec"]["color"])
                if p["target"]["health"] <= 0:
                    # sfx: zombie_die
                    self._create_particles(p["target"]["pos"], 15, self.COLOR_ZOMBIE)
                    self.zombies.remove(p["target"])
                    self.score += 10
                    self.money += 5
                    self.reward_this_step += 1
                self.projectiles.remove(p)
            else:
                p["pos"] += (direction / dist) * p["spec"]["proj_speed"] * 0.1

    def _update_particles(self):
        for particle in self.particles[:]:
            particle["pos"] += particle["vel"]
            particle["lifetime"] -= 1
            if particle["lifetime"] <= 0:
                self.particles.remove(particle)

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.money < spec["cost"]:
            return # sfx: error_sound

        pos_tuple = tuple(self.cursor_pos)
        if pos_tuple in self.occupied_cells:
            return # sfx: error_sound

        # sfx: place_tower
        self.money -= spec["cost"]
        self.towers.append({
            "pos": self.cursor_pos.astype(float) + 0.5,
            "cooldown": 0,
            "type": self.selected_tower_type,
            "spec": spec
        })
        self.occupied_cells.add(pos_tuple)

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
        
        for tower in self.towers:
            self._render_tower(tower)
        
        for zombie in self.zombies:
            self._render_zombie(zombie)
            
        for proj in self.projectiles:
            self._render_projectile(proj)
            
        for particle in self.particles:
            self._render_particle(particle)

        self._render_cursor()

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GAME_AREA_HEIGHT))
        for y in range(0, self.GAME_AREA_HEIGHT + 1, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_path(self):
        for x, y in self.path_cells:
            pygame.draw.rect(self.screen, self.COLOR_PATH, (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

    def _render_base(self):
        px, py = self.GRID_WIDTH * self.CELL_SIZE, 13 * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_BASE, (px - 10, py, 10, self.CELL_SIZE))
        for i in range(5):
            alpha = self.COLOR_BASE_GLOW[3] - i * 10
            if alpha > 0:
                pygame.gfxdraw.box(self.screen, (px - 10 - i, py - i, 10 + 2*i, self.CELL_SIZE + 2*i), (*self.COLOR_BASE[:3], alpha))

    def _render_tower(self, tower):
        px, py = int(tower["pos"][0] * self.CELL_SIZE), int(tower["pos"][1] * self.CELL_SIZE)
        spec = tower["spec"]
        radius = self.CELL_SIZE // 3
        
        # Glow
        for i in range(radius, radius + 5):
            alpha = 50 - (i - radius) * 10
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, px, py, i, (*spec["color"], alpha))
        
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, spec["color"])
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, (255, 255, 255))
    
    def _render_zombie(self, zombie):
        px, py = int(zombie["pos"][0] * self.CELL_SIZE), int(zombie["pos"][1] * self.CELL_SIZE)
        size = self.CELL_SIZE // 2
        
        # Glow
        for i in range(size // 2, size // 2 + 4):
            alpha = self.COLOR_ZOMBIE_GLOW[3] - (i - size // 2) * 12
            if alpha > 0:
                pygame.gfxdraw.box(self.screen, (px - i, py - i, i*2, i*2), (*self.COLOR_ZOMBIE[:3], alpha))

        rect = pygame.Rect(px - size//2, py - size//2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, rect)
        
        # Health bar
        health_ratio = zombie["health"] / zombie["max_health"]
        bar_width = int(self.CELL_SIZE * 0.8)
        health_width = int(bar_width * health_ratio)
        pygame.draw.rect(self.screen, (50, 50, 50), (px - bar_width//2, py - size, bar_width, 3))
        pygame.draw.rect(self.screen, (0, 255, 0), (px - bar_width//2, py - size, health_width, 3))

    def _render_projectile(self, proj):
        px, py = int(proj["pos"][0] * self.CELL_SIZE), int(proj["pos"][1] * self.CELL_SIZE)
        color = proj["spec"]["color"]
        pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 3)
        pygame.draw.circle(self.screen, color, (px, py), 2)

    def _render_particle(self, p):
        size = int(p["lifetime"] / p["max_lifetime"] * 4)
        if size > 0:
            px, py = int(p["pos"][0] * self.CELL_SIZE), int(p["pos"][1] * self.CELL_SIZE)
            pygame.draw.circle(self.screen, p["color"], (px, py), size)

    def _render_cursor(self):
        px, py = self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE
        
        can_place = self.money >= self.TOWER_SPECS[self.selected_tower_type]["cost"] and tuple(self.cursor_pos) not in self.occupied_cells
        color = (0, 255, 0, 100) if can_place else (255, 0, 0, 100)
        
        cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        cursor_surface.fill(color)
        self.screen.blit(cursor_surface, (px, py))
        
        # Range indicator
        range_px = self.TOWER_SPECS[self.selected_tower_type]["range"] * self.CELL_SIZE
        pygame.gfxdraw.aacircle(self.screen, px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2, range_px, (*color[:3], 150))

    def _render_ui(self):
        ui_surface = pygame.Surface((self.SCREEN_WIDTH, self.UI_HEIGHT), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surface, (0, self.SCREEN_HEIGHT - self.UI_HEIGHT))

        y_pos = self.SCREEN_HEIGHT - self.UI_HEIGHT + 10

        # Score, Money, Base Health
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, y_pos))
        
        money_text = self.font_medium.render(f"MONEY: ${self.money}", True, self.COLOR_TEXT)
        self.screen.blit(money_text, (15, y_pos + 25))
        
        health_text = self.font_medium.render(f"BASE HP: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (200, y_pos))

        # Wave info
        if self.wave_state == "intermission":
            wave_info = f"WAVE {self.wave_number + 1} IN {self.wave_timer / 30:.1f}s"
        else:
            wave_info = f"WAVE {self.wave_number} / 5"
        wave_text = self.font_medium.render(wave_info, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (200, y_pos + 25))

        # Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_title_text = self.font_medium.render(f"BUILD: {spec['name']}", True, spec["color"])
        self.screen.blit(tower_title_text, (400, y_pos))
        
        cost_text = self.font_small.render(f"Cost: ${spec['cost']} | Dmg: {spec['damage']} | Rng: {spec['range']}", True, self.COLOR_TEXT)
        self.screen.blit(cost_text, (400, y_pos + 25))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "money": self.money,
            "base_health": self.base_health,
            "wave": self.wave_number,
        }

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.05, 0.2)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifetime = random.randint(10, 25)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": lifetime,
                "max_lifetime": lifetime,
                "color": color
            })

    def _precompute_path_cells(self):
        self.path_cells = set()
        for i in range(len(self.path_points) - 1):
            p1 = np.array(self.path_points[i])
            p2 = np.array(self.path_points[i+1])
            
            if p1[0] == p2[0]: # Vertical segment
                for y in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    self.path_cells.add((p1[0], y))
            else: # Horizontal segment
                for x in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    self.path_cells.add((x, p1[1]))
        
        for cell in self.path_cells:
            self.occupied_cells.add(cell)

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # To control the game with keyboard
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    while running:
        # --- Human Input ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render ---
        # The observation is already the rendered image
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    pygame.quit()