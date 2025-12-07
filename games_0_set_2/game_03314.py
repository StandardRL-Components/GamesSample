
# Generated: 2025-08-27T22:59:22.860860
# Source Brief: brief_03314.md
# Brief Index: 3314

        
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
    A top-down tower defense game where the player places towers to defend a base
    from waves of enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrows to move placement cursor. Space to build selected tower. Shift to cycle tower types."
    )

    # User-facing game description
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers. Earn gold for each kill to build more defenses. Survive all 15 waves to win!"
    )

    # Auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_WAVES = 15
        self.MAX_STEPS = self.FPS * 5 * 60 # 5 minutes max

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_GRID = (30, 40, 50)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_GOLD = (255, 215, 0)
        
        # --- Tower Definitions ---
        self.TOWER_TYPES = [
            {
                "name": "Gun Turret",
                "cost": 100,
                "range": 100,
                "damage": 5,
                "fire_rate": 0.3, # seconds
                "color": (0, 150, 255),
                "projectile_speed": 8,
                "shape": "triangle"
            },
            {
                "name": "Cannon",
                "cost": 250,
                "range": 150,
                "damage": 25,
                "fire_rate": 1.5,
                "color": (255, 150, 0),
                "projectile_speed": 6,
                "shape": "square"
            }
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game State (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.base_health = 0
        self.max_base_health = 0
        self.gold = 0
        self.wave_number = 0
        self.wave_timer = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.path = []
        self.grid_size = 40
        self.grid_w = self.WIDTH // self.grid_size
        self.grid_h = self.HEIGHT // self.grid_size
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.cursor_move_cooldown = 0
        self.grid = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.max_base_health = 100
        self.base_health = self.max_base_health
        self.gold = 250
        self.wave_number = 0
        self.wave_timer = self.FPS * 5  # 5 seconds until first wave

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self._create_path_and_grid()
        
        self.cursor_pos = [self.grid_w // 2, self.grid_h // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.cursor_move_cooldown = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        step_reward = 0

        if not self.game_over:
            self._handle_input(movement, space_held, shift_held)
            
            # --- Update Game Logic ---
            step_reward += self._update_towers()
            step_reward += self._update_projectiles()
            step_reward += self._update_enemies()
            self._update_particles()
            self._update_wave_spawner()

            # Continuous survival penalty to encourage speed
            step_reward -= 0.001 

        self.score += step_reward
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.victory:
                self.score += 100
            else:
                self.score -= 100
            self.game_over = True
            
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_path_and_grid(self):
        self.path = [
            (-20, 180), (100, 180), (100, 80), (260, 80), (260, 280),
            (420, 280), (420, 120), (580, 120), (580, 220)
        ]
        self.base_pos = (580, 220)
        
        self.grid = [[{"valid": True} for _ in range(self.grid_h)] for _ in range(self.grid_w)]
        
        # Invalidate grid cells that are on the path
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            for t in np.linspace(0, 1, int(dist / (self.grid_size * 0.5))):
                px = p1[0] * (1-t) + p2[0] * t
                py = p1[1] * (1-t) + p2[1] * t
                gx, gy = int(px / self.grid_size), int(py / self.grid_size)
                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    self.grid[gx][gy]["valid"] = False

    def _handle_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        if movement != 0 and self.cursor_move_cooldown <= 0:
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_w - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_h - 1)
            self.cursor_move_cooldown = 5 # frames
        if self.cursor_move_cooldown > 0:
            self.cursor_move_cooldown -= 1
            
        # --- Cycle Tower ---
        if shift_held and not self.last_shift_held:
            # sfx: menu_cycle.wav
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
        
        # --- Place Tower ---
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        gx, gy = self.cursor_pos
        can_place = self.grid[gx][gy]["valid"] and self.gold >= tower_def["cost"]
        if space_held and not self.last_space_held and can_place:
            # sfx: build_tower.wav
            self.gold -= tower_def["cost"]
            self.towers.append({
                "type": self.selected_tower_type,
                "pos": (gx * self.grid_size + self.grid_size/2, gy * self.grid_size + self.grid_size/2),
                "cooldown": 0
            })
            self.grid[gx][gy]["valid"] = False

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_wave_spawner(self):
        if not self.enemies and self.wave_number < self.MAX_WAVES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave_number += 1
                self._spawn_wave()
                self.wave_timer = self.FPS * 10 # 10s between waves
    
    def _spawn_wave(self):
        # sfx: new_wave_alert.wav
        num_enemies = 3 + (self.wave_number - 1)
        base_health = 10
        health_multiplier = 1.1 ** (self.wave_number - 1)
        enemy_health = int(base_health * health_multiplier)
        
        for i in range(num_enemies):
            self.enemies.append({
                "pos": [self.path[0][0] - i * 20, self.path[0][1]],
                "health": enemy_health,
                "max_health": enemy_health,
                "speed": 1.5 + self.np_random.uniform(-0.2, 0.2),
                "path_index": 1,
                "value": 10 # gold reward
            })

    def _update_enemies(self):
        reward = 0
        for enemy in list(self.enemies):
            target_pos = self.path[enemy["path_index"]]
            dx = target_pos[0] - enemy["pos"][0]
            dy = target_pos[1] - enemy["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < enemy["speed"]:
                enemy["path_index"] += 1
                if enemy["path_index"] >= len(self.path):
                    # sfx: base_damage.wav
                    self.base_health -= 10
                    self.enemies.remove(enemy)
                    continue
            else:
                enemy["pos"][0] += (dx / dist) * enemy["speed"]
                enemy["pos"][1] += (dy / dist) * enemy["speed"]
        return reward

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue
            
            tower_def = self.TOWER_TYPES[tower["type"]]
            target = None
            min_dist = tower_def["range"]
            
            for enemy in self.enemies:
                dist = math.hypot(enemy["pos"][0] - tower["pos"][0], enemy["pos"][1] - tower["pos"][1])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                # sfx: tower_shoot.wav
                self.projectiles.append({
                    "pos": list(tower["pos"]),
                    "target": target,
                    "speed": tower_def["projectile_speed"],
                    "damage": tower_def["damage"],
                    "color": tower_def["color"]
                })
                tower["cooldown"] = tower_def["fire_rate"] * self.FPS
        return 0

    def _update_projectiles(self):
        reward = 0
        for proj in list(self.projectiles):
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj["target"]["pos"]
            dx = target_pos[0] - proj["pos"][0]
            dy = target_pos[1] - proj["pos"][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj["speed"]:
                # sfx: enemy_hit.wav
                proj["target"]["health"] -= proj["damage"]
                self._create_explosion(proj["pos"], proj["color"])
                if proj["target"]["health"] <= 0:
                    # sfx: enemy_die.wav
                    self.gold += proj["target"]["value"]
                    reward += 0.1 # Kill reward
                    self._create_explosion(proj["target"]["pos"], self.COLOR_ENEMY, 15, 20)
                    self.enemies.remove(proj["target"])
                    
                    if not self.enemies and self.wave_number > 0:
                        reward += 1.0 # Wave clear bonus
                
                self.projectiles.remove(proj)
            else:
                proj["pos"][0] += (dx / dist) * proj["speed"]
                proj["pos"][1] += (dy / dist) * proj["speed"]
        return reward
        
    def _update_particles(self):
        for p in list(self.particles):
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, count=5, max_life=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(max_life // 2, max_life),
                "max_life": max_life,
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def _check_termination(self):
        if self.base_health <= 0:
            self.victory = False
            return True
        if self.wave_number >= self.MAX_WAVES and not self.enemies:
            self.victory = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.victory = False
            return True
        return False
        
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
            "wave": self.wave_number,
            "base_health": self.base_health,
            "enemies_left": len(self.enemies)
        }

    def _render_text(self, text, pos, font, color, center=False):
        surface = font.render(text, True, color)
        rect = surface.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.screen.blit(surface, rect)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 40)
        
        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.base_pos[0]-15, self.base_pos[1]-15, 30, 30))
        
        # Draw towers
        for tower in self.towers:
            tower_def = self.TOWER_TYPES[tower["type"]]
            pos = (int(tower["pos"][0]), int(tower["pos"][1]))
            if tower_def["shape"] == "triangle":
                points = [(pos[0], pos[1]-10), (pos[0]-10, pos[1]+7), (pos[0]+10, pos[1]+7)]
                pygame.gfxdraw.aapolygon(self.screen, points, tower_def["color"])
                pygame.gfxdraw.filled_polygon(self.screen, points, tower_def["color"])
            else: # square
                pygame.draw.rect(self.screen, tower_def["color"], (pos[0]-8, pos[1]-8, 16, 16))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.draw.rect(self.screen, proj["color"], (pos[0]-2, pos[1]-2, 4, 4))
            
        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            # Health bar
            if enemy["health"] < enemy["max_health"]:
                health_pct = enemy["health"] / enemy["max_health"]
                pygame.draw.rect(self.screen, (50,0,0), (pos[0]-10, pos[1]-15, 20, 4))
                pygame.draw.rect(self.screen, (0,200,0), (pos[0]-10, pos[1]-15, int(20 * health_pct), 4))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, p["size"], p["size"]))
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]/2), int(p["pos"][1] - p["size"]/2)))

        # Draw cursor
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        gx, gy = self.cursor_pos
        cursor_world_pos = (gx * self.grid_size, gy * self.grid_size)
        is_valid = self.grid[gx][gy]["valid"] and self.gold >= tower_def["cost"]
        cursor_color = (0, 255, 0, 100) if is_valid else (255, 0, 0, 100)
        
        # Range indicator
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        center = (cursor_world_pos[0] + self.grid_size//2, cursor_world_pos[1] + self.grid_size//2)
        pygame.gfxdraw.filled_circle(s, center[0], center[1], int(tower_def["range"]), (*cursor_color[:3], 50))
        pygame.gfxdraw.aacircle(s, center[0], center[1], int(tower_def["range"]), (*cursor_color[:3], 150))
        self.screen.blit(s, (0,0))
        
        # Placement box
        pygame.draw.rect(self.screen, cursor_color[:3], cursor_world_pos + (self.grid_size, self.grid_size), 2)


    def _render_ui(self):
        # Top-left info
        self._render_text(f"Gold: {self.gold}", (10, 10), self.font_medium, self.COLOR_GOLD)
        self._render_text(f"Base Health: {max(0, self.base_health)} / {self.max_base_health}", (10, 35), self.font_small, self.COLOR_TEXT)
        
        # Top-right info
        wave_str = f"Wave: {self.wave_number}/{self.MAX_WAVES}"
        self._render_text(wave_str, (self.WIDTH - 10 - self.font_medium.size(wave_str)[0], 10), self.font_medium, self.COLOR_TEXT)
        
        # Intermission timer
        if not self.enemies and self.wave_number < self.MAX_WAVES and not self.game_over:
            timer_sec = self.wave_timer // self.FPS
            timer_str = f"Next wave in: {timer_sec}"
            self._render_text(timer_str, (self.WIDTH - 10 - self.font_small.size(timer_str)[0], 35), self.font_small, self.COLOR_TEXT)

        # Bottom-right tower selector
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        self._render_text("Selected Tower:", (self.WIDTH - 150, self.HEIGHT - 55), self.font_small, self.COLOR_TEXT)
        self._render_text(f"{tower_def['name']}", (self.WIDTH - 150, self.HEIGHT - 35), self.font_small, tower_def['color'])
        self._render_text(f"Cost: {tower_def['cost']}", (self.WIDTH - 150, self.HEIGHT - 20), self.font_small, self.COLOR_GOLD)

        # Game Over / Victory Screen
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            if self.victory:
                self._render_text("VICTORY", (self.WIDTH/2, self.HEIGHT/2 - 20), self.font_large, (0, 255, 0), center=True)
            else:
                self._render_text("GAME OVER", (self.WIDTH/2, self.HEIGHT/2 - 20), self.font_large, (255, 0, 0), center=True)
            self._render_text(f"Final Score: {self.score:.0f}", (self.WIDTH/2, self.HEIGHT/2 + 20), self.font_medium, self.COLOR_TEXT, center=True)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # To run headlessly, Pygame needs a display. For manual play, we create one.
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Display the observation from the environment ---
        # The observation is (H, W, C), but pygame wants (W, H) for a surface
        # and its array format is (W, H, C). So we need to transpose back.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()