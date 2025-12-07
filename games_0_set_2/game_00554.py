
# Generated: 2025-08-27T13:59:47.492511
# Source Brief: brief_00554.md
# Brief Index: 554

        
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

    user_guide = (
        "Controls: Arrow keys to move the placement cursor. "
        "Space to build a short-range 'Cannon' tower. "
        "Shift to build a long-range 'Archer' tower."
    )

    game_description = (
        "A classic tower defense game. Strategically place towers on the grid to "
        "defend your base from waves of incoming enemies. Survive all 10 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.FPS = 30
        self.MAX_STEPS = 15000 # Approx 8 minutes at 30fps

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PATH = (60, 70, 90)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_ENEMY = (210, 60, 60)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 100)
        self.COLOR_TOWER_1 = (70, 150, 255) # Cannon
        self.COLOR_TOWER_2 = (255, 215, 0) # Archer

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.max_base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = (0, 0)
        self.path = []
        self.grid = [[]]
        self.wave_timer = 0
        self.victory = False
        self.terminal_reward_given = False

        self.reset()
        # self.validate_implementation() # Uncomment for self-testing

    def _define_path(self):
        self.path = [
            (-1, 5), (1, 5), (1, 2), (4, 2), (4, 7), (7, 7),
            (7, 1), (11, 1), (11, 8), (14, 8), (14, 4), (17, 4)
        ]
        self.base_pos = (15, 4)

    def _get_pixel_pos(self, grid_pos):
        return (
            int(grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2),
            int(grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2)
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._define_path()
        self.grid = [[0 for _ in range(self.GRID_H)] for _ in range(self.GRID_W)]
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            for x in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                if 0 <= x < self.GRID_W: self.grid[x][p1[1]] = 1 # Mark path
            for y in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                if 0 <= p1[0] < self.GRID_W: self.grid[p1[0]][y] = 1 # Mark path

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.terminal_reward_given = False
        self.max_base_health = 20
        self.base_health = self.max_base_health
        self.resources = 100
        self.current_wave = 0
        self.wave_timer = 150 # 5 seconds before first wave
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_pos = (self.GRID_W // 2, self.GRID_H // 2)

        return self._get_observation(), self._get_info()

    def _spawn_wave(self):
        self.current_wave += 1
        if self.current_wave > 10: return

        num_enemies = 5 + self.current_wave * 2
        difficulty_mod = 1 + (self.current_wave - 1) * 0.05
        
        for i in range(num_enemies):
            enemy = {
                "pos": self._get_pixel_pos(self.path[0]),
                "path_index": 0,
                "health": int(50 * difficulty_mod),
                "max_health": int(50 * difficulty_mod),
                "speed": 0.8 * difficulty_mod,
                "spawn_delay": i * 20, # Stagger spawn
                "value": 5 + self.current_wave,
            }
            self.enemies.append(enemy)

    def step(self, action):
        reward = -0.001 # Small penalty for existing
        self.game_over = self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.victory
        if self.game_over:
            if not self.terminal_reward_given:
                if self.victory:
                    reward += 100
                else:
                    reward += -100
                self.terminal_reward_given = True
            return self._get_observation(), reward, True, False, self._get_info()

        # 1. Handle Player Input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos = (self.cursor_pos[0], max(0, self.cursor_pos[1] - 1))
        elif movement == 2: self.cursor_pos = (self.cursor_pos[0], min(self.GRID_H - 1, self.cursor_pos[1] + 1))
        elif movement == 3: self.cursor_pos = (max(0, self.cursor_pos[0] - 1), self.cursor_pos[1])
        elif movement == 4: self.cursor_pos = (min(self.GRID_W - 1, self.cursor_pos[0] + 1), self.cursor_pos[1])

        if space_held:
            reward += self._place_tower(self.cursor_pos, 1)
        if shift_held:
            reward += self._place_tower(self.cursor_pos, 2)

        # 2. Update Wave Logic
        if not self.enemies and self.current_wave <= 10:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._spawn_wave()
                self.wave_timer = 300 # 10s between waves
        
        if not self.enemies and self.current_wave > 10:
            self.victory = True

        # 3. Update Enemies
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            if enemy["spawn_delay"] > 0:
                enemy["spawn_delay"] -= 1
                continue
            
            target_waypoint_pos = self._get_pixel_pos(self.path[enemy["path_index"] + 1])
            dx = target_waypoint_pos[0] - enemy["pos"][0]
            dy = target_waypoint_pos[1] - enemy["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < enemy["speed"]:
                enemy["path_index"] += 1
                if enemy["path_index"] >= len(self.path) - 1:
                    self.base_health -= 1
                    reward -= 10
                    self._create_particles(self._get_pixel_pos(self.base_pos), self.COLOR_ENEMY, 20)
                    enemies_to_remove.append(i)
                    continue
            else:
                enemy["pos"] = (
                    enemy["pos"][0] + (dx / dist) * enemy["speed"],
                    enemy["pos"][1] + (dy / dist) * enemy["speed"],
                )
        
        self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]

        # 4. Update Towers and Projectiles
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] == 0:
                target = self._find_target(tower)
                if target:
                    # sfx: fire_weapon
                    self.projectiles.append({
                        "pos": tower["pixel_pos"],
                        "target": target,
                        "speed": 10,
                        "damage": tower["damage"],
                        "color": tower["color"]
                    })
                    tower["cooldown"] = tower["fire_rate"]

        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            if p["target"] not in self.enemies:
                projectiles_to_remove.append(i)
                continue

            dx = p["target"]["pos"][0] - p["pos"][0]
            dy = p["target"]["pos"][1] - p["pos"][1]
            dist = math.hypot(dx, dy)
            
            if dist < p["speed"]:
                # sfx: hit_enemy
                p["target"]["health"] -= p["damage"]
                reward += 0.1 # Reward for hit
                self._create_particles(p["target"]["pos"], p["color"], 5)
                projectiles_to_remove.append(i)
                if p["target"]["health"] <= 0:
                    reward += 1 # Reward for kill
                    self.score += p["target"]["value"]
                    self.resources += p["target"]["value"]
                    self._create_particles(p["target"]["pos"], self.COLOR_ENEMY, 15)
                    self.enemies.remove(p["target"])
            else:
                p["pos"] = (
                    p["pos"][0] + (dx / dist) * p["speed"],
                    p["pos"][1] + (dy / dist) * p["speed"]
                )
        
        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]

        # 5. Update Particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] = (p["pos"][0] + p["vel"][0], p["pos"][1] + p["vel"][1])
            p["life"] -= 1

        # 6. Finalize Step
        self.steps += 1
        terminated = self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.victory
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_tower(self, pos, tower_type):
        if self.grid[pos[0]][pos[1]] != 0: return 0 # Can't place on path or existing tower

        tower_stats = {
            1: {"cost": 40, "range": 80, "damage": 25, "fire_rate": 45, "color": self.COLOR_TOWER_1}, # Cannon
            2: {"cost": 60, "range": 160, "damage": 15, "fire_rate": 30, "color": self.COLOR_TOWER_2}  # Archer
        }
        stats = tower_stats[tower_type]

        if self.resources >= stats["cost"]:
            # sfx: place_tower
            self.resources -= stats["cost"]
            self.grid[pos[0]][pos[1]] = 2 # Mark as tower
            self.towers.append({
                "pos": pos,
                "pixel_pos": self._get_pixel_pos(pos),
                "type": tower_type,
                "cooldown": 0,
                **stats
            })
            return 0.5 # Small reward for placing a tower
        return -0.1 # Penalty for failed placement attempt

    def _find_target(self, tower):
        best_target = None
        max_dist_on_path = -1

        for enemy in self.enemies:
            if enemy["spawn_delay"] > 0: continue
            
            dist = math.hypot(enemy["pos"][0] - tower["pixel_pos"][0], enemy["pos"][1] - tower["pixel_pos"][1])
            if dist <= tower["range"]:
                # Target enemy furthest along the path
                enemy_path_dist = enemy["path_index"] + (1 - (math.hypot(enemy["pos"][0] - self._get_pixel_pos(self.path[enemy["path_index"]+1])[0], enemy["pos"][1] - self._get_pixel_pos(self.path[enemy["path_index"]+1])[1]) / self.GRID_SIZE))
                if enemy_path_dist > max_dist_on_path:
                    max_dist_on_path = enemy_path_dist
                    best_target = enemy
        return best_target
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(10, 20),
                "color": color
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
            "base_health": self.base_health,
            "resources": self.resources,
            "current_wave": self.current_wave,
            "enemies_remaining": len([e for e in self.enemies if e["spawn_delay"] <= 0]),
            "victory": self.victory
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw path
        for i in range(len(self.path) - 1):
            p1 = self._get_pixel_pos(self.path[i])
            p2 = self._get_pixel_pos(self.path[i+1])
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.GRID_SIZE)
        
        # Draw Base
        base_px = self._get_pixel_pos(self.base_pos)
        base_rect = pygame.Rect(base_px[0] - self.GRID_SIZE//2, base_px[1] - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.gfxdraw.rectangle(self.screen, base_rect, (*self.COLOR_BASE, 150))

        # Draw Towers
        for tower in self.towers:
            pygame.gfxdraw.filled_circle(self.screen, tower["pixel_pos"][0], tower["pixel_pos"][1], self.GRID_SIZE//2 - 4, tower["color"])
            pygame.gfxdraw.aacircle(self.screen, tower["pixel_pos"][0], tower["pixel_pos"][1], self.GRID_SIZE//2 - 4, tower["color"])

        # Draw Projectiles
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 3, p["color"])
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 3, p["color"])

        # Draw Enemies
        for enemy in self.enemies:
            if enemy["spawn_delay"] > 0: continue
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            # Health bar
            if enemy["health"] < enemy["max_health"]:
                health_pct = enemy["health"] / enemy["max_health"]
                pygame.draw.rect(self.screen, (60, 60, 60), (pos[0] - 10, pos[1] - 15, 20, 3))
                pygame.draw.rect(self.screen, self.COLOR_BASE, (pos[0] - 10, pos[1] - 15, 20 * health_pct, 3))

        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = (*p["color"], alpha)
            pygame.draw.circle(self.screen, color, (int(p["pos"][0]), int(p["pos"][1])), int(p["life"] / 5))

        # Draw Cursor
        cursor_px_pos = self._get_pixel_pos(self.cursor_pos)
        is_valid_pos = self.grid[self.cursor_pos[0]][self.cursor_pos[1]] == 0
        cursor_color = self.COLOR_CURSOR if is_valid_pos else self.COLOR_CURSOR_INVALID
        cursor_rect = pygame.Rect(self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        
        # Blinking effect for the cursor fill
        if (self.steps // 8) % 2 == 0:
            s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            s.fill(cursor_color)
            self.screen.blit(s, cursor_rect.topleft)
        pygame.gfxdraw.rectangle(self.screen, cursor_rect, (*cursor_color[:3], 200))

    def _render_ui(self):
        # Top-left info panel
        info_panel = pygame.Surface((200, 85), pygame.SRCALPHA)
        info_panel.fill((20, 25, 40, 180))
        
        wave_text = f"Wave  : {self.current_wave}/10"
        res_text = f"Gold  : {self.resources}"
        score_text = f"Score : {self.score}"

        info_panel.blit(self.font_small.render(wave_text, True, self.COLOR_TEXT), (10, 10))
        info_panel.blit(self.font_small.render(res_text, True, self.COLOR_TEXT), (10, 35))
        info_panel.blit(self.font_small.render(score_text, True, self.COLOR_TEXT), (10, 60))
        self.screen.blit(info_panel, (5, 5))

        # Base health bar
        health_pct = max(0, self.base_health / self.max_base_health)
        pygame.draw.rect(self.screen, (60, 60, 60), (self.WIDTH - 160, 15, 150, 20))
        if health_pct > 0:
            pygame.draw.rect(self.screen, self.COLOR_BASE, (self.WIDTH - 160, 15, 150 * health_pct, 20))
        health_text = self.font_small.render(f"Base: {self.base_health}/{self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH - 155, 17))
        
        # Game Over / Victory Text
        if self.game_over:
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_BASE if self.victory else self.COLOR_ENEMY
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.font.quit()
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Set a dummy video driver to run headless
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Test with the validation method ---
    env.validate_implementation()

    # --- Manual play example ---
    # To run this, you need a display. Comment out the os.environ line above.
    # And change the GameEnv init to: env = GameEnv(render_mode="human")
    # You'll also need to add a "human" render mode that blits to a real screen.
    # This example below just tests the logic with random actions.
    
    print("\n--- Running random agent for 1000 steps ---")
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    for i in range(1000):
        if terminated:
            print(f"Episode finished after {i+1} steps. Final Info: {info}")
            break
            
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if (i+1) % 100 == 0:
            print(f"Step {i+1}: Reward={reward:.2f}, Total Reward={total_reward:.2f}, Info={info}")

    print("--- Run complete ---")
    env.close()