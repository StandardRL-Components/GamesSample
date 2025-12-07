
# Generated: 2025-08-28T02:48:09.619122
# Source Brief: brief_04570.md
# Brief Index: 4570

        
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
        "Controls: Arrow keys to move cursor. Space to place selected tower. Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers in an isometric world."
    )

    auto_advance = True

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (45, 50, 64)
    COLOR_PATH = (60, 66, 82)
    COLOR_BASE = (0, 150, 255)
    COLOR_BASE_DAMAGED = (255, 100, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GOLD = (255, 215, 0)
    COLOR_CURSOR = (255, 255, 255)
    
    # --- Game Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 20, 10
    MAX_STEPS = 30 * 120 # 2 minutes at 30fps
    MAX_WAVES = 20
    INITIAL_GOLD = 150
    INITIAL_BASE_HEALTH = 100
    WAVE_COOLDOWN_FRAMES = 30 * 5 # 5 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = 120

        self._define_path()
        self._define_towers()
        
        self.reset()
        
        # This check is for development, can be removed in production
        # self.validate_implementation()

    def _define_path(self):
        self.path_grid_coords = [
            (-1, 4), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (4, 3), (4, 2),
            (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (10, 2),
            (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (9, 7), (8, 7),
            (7, 7), (6, 7), (6, 8), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9),
            (11, 9), (12, 9), (13, 9), (14, 9), (15, 9), (16, 9)
        ]
        self.path_world_coords = [self._grid_to_world(p[0], p[1]) for p in self.path_grid_coords]
        self.base_pos = self.path_world_coords[-1]

    def _define_towers(self):
        self.tower_types = [
            {"name": "Gatling", "cost": 50, "range": 100, "damage": 5, "fire_rate": 5, "color": (0, 255, 0), "proj_speed": 8},
            {"name": "Cannon", "cost": 120, "range": 150, "damage": 25, "fire_rate": 30, "color": (255, 165, 0), "proj_speed": 6},
            {"name": "Sniper", "cost": 200, "range": 250, "damage": 80, "fire_rate": 60, "color": (138, 43, 226), "proj_speed": 15},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.gold = self.INITIAL_GOLD
        self.wave_number = 0
        self.wave_timer = self.WAVE_COOLDOWN_FRAMES // 2
        self.enemies_in_wave = 0
        self.enemies_spawned_this_wave = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.damage_flash_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        self._handle_input(movement, space_held, shift_held)
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.damage_flash_timer = max(0, self.damage_flash_timer - 1)

        wave_reward = self._update_waves()
        reward += wave_reward

        enemy_reward, base_damage_penalty = self._update_enemies()
        reward += enemy_reward
        reward += base_damage_penalty

        self._update_towers()
        self._update_projectiles()
        self._update_particles()
        
        terminated = self._check_termination()
        
        if terminated:
            if self.game_won:
                reward += 100
            else:
                reward -= 100

        self.score += reward
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if self.auto_advance:
            self.clock.tick(30)
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Cycle Tower ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_types)
            # sfx: UI_switch.wav
        
        # --- Place Tower ---
        if space_held and not self.prev_space_held:
            self._place_tower()

    def _place_tower(self):
        tower_spec = self.tower_types[self.selected_tower_type]
        if self.gold < tower_spec["cost"]:
            # sfx: UI_error.wav
            return

        grid_x, grid_y = self.cursor_pos
        is_on_path = any(p == (grid_x, grid_y) for p in self.path_grid_coords)
        is_occupied = any(t["grid_pos"] == [grid_x, grid_y] for t in self.towers)

        if not is_on_path and not is_occupied:
            self.gold -= tower_spec["cost"]
            world_pos = self._grid_to_world(grid_x, grid_y)
            new_tower = {
                "spec": tower_spec,
                "grid_pos": [grid_x, grid_y],
                "world_pos": world_pos,
                "cooldown": 0,
            }
            self.towers.append(new_tower)
            # sfx: place_tower.wav

    def _update_waves(self):
        reward = 0
        if not self.enemies and self.enemies_spawned_this_wave == self.enemies_in_wave and self.wave_number <= self.MAX_WAVES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                if self.wave_number > 0:
                    reward += 1.0 # Wave complete reward
                    self.gold += 100 + self.wave_number * 10
                self.wave_number += 1
                if self.wave_number > self.MAX_WAVES:
                    return reward
                self.wave_timer = self.WAVE_COOLDOWN_FRAMES
                self.enemies_in_wave = 3 + self.wave_number * 2
                self.enemies_spawned_this_wave = 0
        
        if self.wave_number > 0 and self.enemies_spawned_this_wave < self.enemies_in_wave:
            if self.steps % 15 == 0: # Spawn delay
                self._spawn_enemy()
                self.enemies_spawned_this_wave += 1
        return reward

    def _spawn_enemy(self):
        health = 20 * (1.05 ** (self.wave_number -1))
        speed = 1.0 * (1.02 ** (self.wave_number - 1))
        start_pos = list(self.path_world_coords[0])
        self.enemies.append({
            "pos": start_pos,
            "max_health": health,
            "health": health,
            "speed": speed,
            "path_index": 1,
        })

    def _update_enemies(self):
        reward = 0
        penalty = 0
        for enemy in reversed(self.enemies):
            if enemy["path_index"] >= len(self.path_world_coords):
                self.base_health -= 10
                penalty -= 1.0 # -0.1 per health point
                self.enemies.remove(enemy)
                self.damage_flash_timer = 10
                # sfx: base_damage.wav
                continue

            target_pos = self.path_world_coords[enemy["path_index"]]
            direction = np.array(target_pos) - np.array(enemy["pos"])
            distance = np.linalg.norm(direction)

            if distance < enemy["speed"]:
                enemy["pos"] = list(target_pos)
                enemy["path_index"] += 1
            else:
                move = direction / distance * enemy["speed"]
                enemy["pos"][0] += move[0]
                enemy["pos"][1] += move[1]

        return reward, penalty

    def _update_towers(self):
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] > 0:
                continue

            target = None
            min_dist = tower["spec"]["range"] ** 2

            for enemy in self.enemies:
                dist_sq = (tower["world_pos"][0] - enemy["pos"][0]) ** 2 + \
                          (tower["world_pos"][1] - enemy["pos"][1]) ** 2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                self.projectiles.append({
                    "pos": list(tower["world_pos"]),
                    "target": target,
                    "spec": tower["spec"],
                })
                tower["cooldown"] = tower["spec"]["fire_rate"]
                # sfx: tower_fire.wav

    def _update_projectiles(self):
        reward = 0
        for proj in reversed(self.projectiles):
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj["target"]["pos"]
            direction = np.array(target_pos) - np.array(proj["pos"])
            distance = np.linalg.norm(direction)
            
            speed = proj["spec"]["proj_speed"]
            if distance < speed:
                reward += self._damage_enemy(proj["target"], proj["spec"]["damage"])
                self._create_particles(proj["pos"], proj["spec"]["color"], 5)
                self.projectiles.remove(proj)
                # sfx: enemy_hit.wav
            else:
                move = direction / distance * speed
                proj["pos"][0] += move[0]
                proj["pos"][1] += move[1]
        return reward

    def _damage_enemy(self, enemy, damage):
        enemy["health"] -= damage
        if enemy["health"] <= 0:
            self._create_particles(enemy["pos"], self.COLOR_ENEMY, 20)
            self.enemies.remove(enemy)
            self.gold += 5
            # sfx: enemy_destroy.wav
            return 0.1 # Kill reward
        return 0

    def _update_particles(self):
        for p in reversed(self.particles):
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(10, 20),
                "color": color,
            })

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.wave_number > self.MAX_WAVES and not self.enemies:
            self.game_over = True
            self.game_won = True
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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }

    def _grid_to_world(self, grid_x, grid_y):
        screen_x = self.origin_x + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.origin_y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return [screen_x, screen_y]

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._grid_to_world(x, y)
                p2 = self._grid_to_world(x + 1, y)
                p3 = self._grid_to_world(x + 1, y + 1)
                p4 = self._grid_to_world(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

        # Draw path
        if len(self.path_world_coords) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_world_coords, 20)

        # Draw base
        base_color = self.COLOR_BASE if self.damage_flash_timer == 0 else self.COLOR_BASE_DAMAGED
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 12, base_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 12, base_color)

        # Draw towers
        for tower in self.towers:
            pos = tower["world_pos"]
            color = tower["spec"]["color"]
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1] - 5), 6, color)
            pygame.gfxdraw.box(self.screen, (int(pos[0]-4), int(pos[1]-5), 8, 5), color)

        # Draw cursor
        cursor_world_pos = self._grid_to_world(self.cursor_pos[0], self.cursor_pos[1])
        p1 = self._grid_to_world(self.cursor_pos[0], self.cursor_pos[1])
        p2 = self._grid_to_world(self.cursor_pos[0] + 1, self.cursor_pos[1])
        p3 = self._grid_to_world(self.cursor_pos[0] + 1, self.cursor_pos[1] + 1)
        p4 = self._grid_to_world(self.cursor_pos[0], self.cursor_pos[1] + 1)
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, [p1, p2, p3, p4], 2)

        # Draw tower range for selected tower
        tower_spec = self.tower_types[self.selected_tower_type]
        if tower_spec["cost"] <= self.gold:
            pygame.gfxdraw.aacircle(self.screen, int(cursor_world_pos[0]), int(cursor_world_pos[1]), tower_spec["range"], (255,255,255,50))

        # Draw enemies
        for enemy in self.enemies:
            pos = enemy["pos"]
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 5, self.COLOR_ENEMY)
            health_percent = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (50,50,50), (int(pos[0]-10), int(pos[1]-15), 20, 3))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (int(pos[0]-10), int(pos[1]-15), 20 * health_percent, 3))

        # Draw projectiles
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, proj["spec"]["color"], (int(proj["pos"][0]), int(proj["pos"][1])), 3)
            
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20.0))
            color = (*p["color"], alpha)
            s = pygame.Surface((2, 2), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p["pos"][0]), int(p["pos"][1])))

    def _render_ui(self):
        # Top bar background
        pygame.draw.rect(self.screen, (15, 18, 26), (0, 0, self.SCREEN_WIDTH, 40))

        # Health
        health_text = self.font_ui.render(f"Base HP: {max(0, self.base_health)}/{self.INITIAL_BASE_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 12))
        
        # Gold
        gold_text = self.font_ui.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (200, 12))

        # Wave
        wave_str = f"Wave: {self.wave_number}/{self.MAX_WAVES}" if self.wave_number <= self.MAX_WAVES else "All Waves Done"
        wave_text = self.font_ui.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (320, 12))

        # Selected Tower UI
        tower_spec = self.tower_types[self.selected_tower_type]
        tower_name = self.font_ui.render(f"Tower: {tower_spec['name']}", True, tower_spec['color'])
        tower_cost = self.font_ui.render(f"Cost: {tower_spec['cost']}", True, self.COLOR_GOLD if self.gold >= tower_spec['cost'] else self.COLOR_ENEMY)
        self.screen.blit(tower_name, (450, 5))
        self.screen.blit(tower_cost, (450, 20))

        # Game Over / Win Message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (0, 255, 0) if self.game_won else (255, 0, 0)
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To run with a human player ---
    import pygame
    
    # Re-initialize pygame for display
    pygame.init()
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # Map pygame keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not done:
        # --- Human Input ---
        movement_action = 0  # no-op
        space_action = 0
        shift_action = 0
        
        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Only one movement at a time
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # Pygame uses (width, height), numpy uses (height, width)
        # So we need to transpose from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'r' key
        
        clock.tick(30) # Limit to 30 FPS

    env.close()
    pygame.quit()