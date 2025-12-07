
# Generated: 2025-08-27T13:24:03.874622
# Source Brief: brief_00352.md
# Brief Index: 352

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place selected block. Shift to cycle block types."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing walls and turrets on a grid."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.CELL_SIZE = self.WIDTH // self.GRID_W

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_BASE = (46, 204, 113)
        self.COLOR_BASE_DMG = (231, 76, 60)
        self.COLOR_ENEMY = (231, 76, 60)
        self.COLOR_PROJECTILE = (241, 196, 15)
        self.COLOR_CURSOR = (236, 240, 241)
        self.COLOR_TEXT = (236, 240, 241)
        self.BLOCK_TYPES = [
            {"name": "Turret", "color": (52, 152, 219), "type": "turret", "range_sq": 5**2, "fire_rate": 25, "projectile_speed": 6, "damage": 25},
            {"name": "Wall", "color": (127, 140, 141), "type": "wall"},
        ]

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.FONT_UI = pygame.font.SysFont("Consolas", 16, bold=True)
        self.FONT_MSG = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game Parameters
        self.MAX_STEPS = 3000
        self.MAX_WAVES = 10
        self.INITIAL_BASE_HEALTH = 100
        self.WAVE_PREP_TIME = 150 # frames

        # Initialize state variables
        self.rng = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 0
        self.wave_number = 0
        self.time_until_next_wave = 0
        self.enemies = []
        self.blocks = {}
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_block_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.base_pos = (self.GRID_W // 2, self.GRID_H - 1)
        self.grid_for_pathfinding = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.wave_number = 0
        self.time_until_next_wave = self.WAVE_PREP_TIME

        self.enemies.clear()
        self.blocks.clear()
        self.projectiles.clear()
        self.particles.clear()

        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_block_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        self._rebuild_pathfinding_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1

        if not self.game_over:
            self._handle_input(action)
            
            wave_survived = self._update_waves()
            if wave_survived:
                reward += 1.0
                self.score += 100

            self._update_turrets()
            self._update_projectiles()
            
            kills = self._update_enemies()
            if kills > 0:
                reward += 0.1 * kills
                self.score += 10 * kills
        
        self._update_particles()

        terminated = self._check_termination()
        if terminated and not self.win:
            reward = -100.0
        elif terminated and self.win:
            reward = 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_W
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_H

        # --- Place Block (on key press) ---
        if space_held and not self.last_space_held:
            pos_tuple = tuple(self.cursor_pos)
            if pos_tuple not in self.blocks and pos_tuple != self.base_pos:
                block_info = self.BLOCK_TYPES[self.selected_block_type_idx]
                self.blocks[pos_tuple] = {
                    "type": block_info["type"],
                    "color": block_info["color"],
                    "cooldown": 0
                }
                self._rebuild_pathfinding_grid()
                # Recalculate paths for all enemies
                for enemy in self.enemies:
                    enemy["path"] = self._find_path(enemy["grid_pos"], self.base_pos)
                # sfx: place_block.wav
                self._create_particles(self._grid_to_pixel(pos_tuple), block_info["color"], 10, 2)


        # --- Cycle Block Type (on key press) ---
        if shift_held and not self.last_shift_held:
            self.selected_block_type_idx = (self.selected_block_type_idx + 1) % len(self.BLOCK_TYPES)
            # sfx: cycle_type.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_waves(self):
        if self.win: return False

        if not self.enemies and self.wave_number <= self.MAX_WAVES:
            self.time_until_next_wave -= 1
            if self.time_until_next_wave <= 0:
                self.wave_number += 1
                if self.wave_number > self.MAX_WAVES:
                    self.win = True
                    return False
                self._spawn_wave()
                self.time_until_next_wave = self.WAVE_PREP_TIME
                return True # Wave survived
        return False

    def _spawn_wave(self):
        num_enemies = 5 + self.wave_number * 2
        enemy_speed = 0.02 + self.wave_number * 0.005
        enemy_health = 50 + self.wave_number * 15
        
        spawn_x = self.rng.integers(0, self.GRID_W)
        spawn_pos = (spawn_x, 0)
        
        for i in range(num_enemies):
            offset_x = self.rng.uniform(-0.4, 0.4)
            offset_y = self.rng.uniform(-0.4, 0.4) - i * 0.8
            
            start_grid_pos = (spawn_x, -i)
            pixel_pos = self._grid_to_pixel((spawn_x + offset_x, offset_y))

            enemy = {
                "pixel_pos": list(pixel_pos),
                "grid_pos": start_grid_pos,
                "health": enemy_health,
                "max_health": enemy_health,
                "speed": enemy_speed,
                "path": self._find_path(start_grid_pos, self.base_pos),
                "path_progress": 0.0
            }
            self.enemies.append(enemy)

    def _update_turrets(self):
        turret_info = self.BLOCK_TYPES[0]
        for pos, block in self.blocks.items():
            if block["type"] == "turret":
                block["cooldown"] = max(0, block["cooldown"] - 1)
                if block["cooldown"] == 0:
                    target = self._find_nearest_enemy(pos, turret_info["range_sq"])
                    if target:
                        block["cooldown"] = turret_info["fire_rate"]
                        start_pixel = self._grid_to_pixel(pos)
                        
                        proj = {
                            "pos": list(start_pixel),
                            "target": target,
                            "speed": turret_info["projectile_speed"],
                            "damage": turret_info["damage"]
                        }
                        self.projectiles.append(proj)
                        # sfx: turret_fire.wav

    def _find_nearest_enemy(self, turret_pos, range_sq):
        nearest_enemy = None
        min_dist_sq = float('inf')
        
        turret_pixel_pos = self._grid_to_pixel(turret_pos)
        
        for enemy in self.enemies:
            dist_sq = (enemy["pixel_pos"][0] - turret_pixel_pos[0])**2 + (enemy["pixel_pos"][1] - turret_pixel_pos[1])**2
            if dist_sq < min_dist_sq and dist_sq <= range_sq:
                min_dist_sq = dist_sq
                nearest_enemy = enemy
        return nearest_enemy

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            target = proj["target"]
            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue

            direction_vec = np.array(target["pixel_pos"]) - np.array(proj["pos"])
            dist = np.linalg.norm(direction_vec)
            
            if dist < proj["speed"]:
                target["health"] -= proj["damage"]
                self._create_particles(target["pixel_pos"], self.COLOR_PROJECTILE, 15, 3)
                # sfx: enemy_hit.wav
                self.projectiles.remove(proj)
            else:
                direction_vec /= dist
                proj["pos"][0] += direction_vec[0] * proj["speed"]
                proj["pos"][1] += direction_vec[1] * proj["speed"]

    def _update_enemies(self):
        kills = 0
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                self.enemies.remove(enemy)
                self._create_particles(enemy["pixel_pos"], self.COLOR_ENEMY, 30, 4)
                # sfx: enemy_die.wav
                kills += 1
                continue

            if not enemy["path"]:
                continue

            enemy["path_progress"] += enemy["speed"]
            
            if enemy["path_progress"] >= 1.0:
                enemy["path_progress"] = 0.0
                enemy["path"].pop(0)
                if not enemy["path"]: # Reached destination
                    if enemy["grid_pos"] == self.base_pos:
                        self.base_health -= 10
                        self.base_health = max(0, self.base_health)
                        self.enemies.remove(enemy)
                        # sfx: base_damage.wav
                        self._create_particles(self._grid_to_pixel(self.base_pos), self.COLOR_BASE_DMG, 50, 5)
                    continue

            current_node_idx = 0
            start_node = enemy["path"][current_node_idx]
            end_node = enemy["path"][current_node_idx + 1]
            
            start_pixel = self._grid_to_pixel(start_node)
            end_pixel = self._grid_to_pixel(end_node)

            enemy["pixel_pos"][0] = start_pixel[0] + (end_pixel[0] - start_pixel[0]) * enemy["path_progress"]
            enemy["pixel_pos"][1] = start_pixel[1] + (end_pixel[1] - start_pixel[1]) * enemy["path_progress"]
            enemy["grid_pos"] = end_node
        return kills

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _rebuild_pathfinding_grid(self):
        self.grid_for_pathfinding = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        for pos in self.blocks:
            self.grid_for_pathfinding[pos] = 1 # Obstacle
        self.grid_for_pathfinding[self.base_pos] = 0 # Destination is walkable

    def _find_path(self, start_pos, end_pos):
        start_pos = (int(start_pos[0]), int(start_pos[1]))
        end_pos = (int(end_pos[0]), int(end_pos[1]))

        if not (0 <= start_pos[0] < self.GRID_W and 0 <= start_pos[1] < self.GRID_H):
            start_pos = (max(0, min(self.GRID_W - 1, start_pos[0])),
                         max(0, min(self.GRID_H - 1, start_pos[1])))

        q = deque([(start_pos, [start_pos])])
        visited = {start_pos}

        while q:
            (vx, vy), path = q.popleft()
            if (vx, vy) == end_pos:
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = vx + dx, vy + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and (nx, ny) not in visited:
                    if self.grid_for_pathfinding[nx, ny] == 0 or (nx, ny) == end_pos:
                        visited.add((nx, ny))
                        new_path = list(path)
                        new_path.append((nx, ny))
                        q.append(((nx, ny), new_path))
        return [] # No path found

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.win and not self.enemies:
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
        # Draw grid
        for x in range(self.GRID_W):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_SIZE, 0), (x * self.CELL_SIZE, self.HEIGHT))
        for y in range(self.GRID_H):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_SIZE), (self.WIDTH, y * self.CELL_SIZE))

        # Draw base
        base_px, base_py = self._grid_to_pixel(self.base_pos)
        base_rect = pygame.Rect(base_px - self.CELL_SIZE//2, base_py - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=3)
        
        # Draw blocks
        for pos, block in self.blocks.items():
            px, py = self._grid_to_pixel(pos)
            rect = pygame.Rect(px - self.CELL_SIZE//2, py - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, block["color"], rect, border_radius=3)
            if block["type"] == "turret":
                pygame.gfxdraw.aacircle(self.screen, int(px), int(py), self.CELL_SIZE//4, (255,255,255))

        # Draw enemies
        for enemy in self.enemies:
            px, py = enemy["pixel_pos"]
            radius = self.CELL_SIZE // 3
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, self.COLOR_ENEMY)
            # Health bar
            if enemy["health"] < enemy["max_health"]:
                bar_w = radius * 2
                bar_h = 4
                health_pct = enemy["health"] / enemy["max_health"]
                pygame.draw.rect(self.screen, (80,0,0), (px - radius, py - radius - 8, bar_w, bar_h))
                pygame.draw.rect(self.screen, (0,200,0), (px - radius, py - radius - 8, bar_w * health_pct, bar_h))


        # Draw projectiles
        for proj in self.projectiles:
            px, py = proj["pos"]
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), 3, self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0,0, p['size'], p['size']))
            self.screen.blit(temp_surf, (p['pos'][0] - p['size']//2, p['pos'][1] - p['size']//2))

        # Draw cursor
        cx_px, cy_px = self._grid_to_pixel(self.cursor_pos)
        cursor_rect = pygame.Rect(cx_px - self.CELL_SIZE//2, cy_px - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=3)
        # Glow effect
        glow_surf = pygame.Surface((self.CELL_SIZE * 2, self.CELL_SIZE * 2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_CURSOR, 50), glow_surf.get_rect(), 10, border_radius=8)
        self.screen.blit(glow_surf, (cursor_rect.x - self.CELL_SIZE//2, cursor_rect.y - self.CELL_SIZE//2), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill((30, 40, 50, 200))
        self.screen.blit(ui_panel, (0, self.HEIGHT - 40))

        # Health
        health_text = self.FONT_UI.render(f"Base Health: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, self.HEIGHT - 30))
        
        # Score
        score_text = self.FONT_UI.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (200, self.HEIGHT - 30))

        # Wave
        wave_str = f"Wave: {self.wave_number}/{self.MAX_WAVES}"
        if not self.enemies and self.wave_number < self.MAX_WAVES and not self.win:
            wave_str += f" (Next in {self.time_until_next_wave // 30}s)"
        wave_text = self.FONT_UI.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (350, self.HEIGHT - 30))

        # Selected Block
        sel_block = self.BLOCK_TYPES[self.selected_block_type_idx]
        sel_text = self.FONT_UI.render(f"Selected: {sel_block['name']}", True, self.COLOR_TEXT)
        self.screen.blit(sel_text, (self.WIDTH - 150, self.HEIGHT - 30))
        pygame.draw.rect(self.screen, sel_block['color'], (self.WIDTH - 165, self.HEIGHT - 32, 10, 24), border_radius=2)
        
        # Game Over / Win Message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BASE if self.win else self.COLOR_ENEMY
            msg_text = self.FONT_MSG.render(msg, True, color)
            msg_rect = msg_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "enemies_remaining": len(self.enemies),
        }

    def _grid_to_pixel(self, grid_pos):
        x = (grid_pos[0] + 0.5) * self.CELL_SIZE
        y = (grid_pos[1] + 0.5) * self.CELL_SIZE
        return x, y

    def _create_particles(self, pos, color, count, speed_factor):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 3) * speed_factor
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.rng.integers(15, 30),
                'max_life': 30,
                'color': color,
                'size': self.rng.integers(2, 5)
            })

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Grid Defense")
    clock = pygame.time.Clock()

    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    pygame.quit()