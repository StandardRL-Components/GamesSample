
# Generated: 2025-08-28T05:41:08.044971
# Source Brief: brief_05658.md
# Brief Index: 5658

        
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
        "Controls: ↑↓←→ to move the cursor. Press Space to place the selected tower. Shift does nothing."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    CELL_SIZE = 20
    UI_HEIGHT = 40
    GAME_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PATH = (40, 40, 60)
    COLOR_GRID = (30, 30, 45)
    COLOR_BASE = (0, 150, 50)
    COLOR_BASE_DMG = (200, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CURSOR_INVALID = (255, 50, 50)

    # Tower Types
    TOWER_SPECS = {
        0: {"name": "Gun", "cost": 50, "range": 80, "damage": 5, "fire_rate": 15, "color": (50, 150, 255)},
        1: {"name": "Sniper", "cost": 120, "range": 200, "damage": 25, "fire_rate": 60, "color": (255, 200, 50)},
    }
    
    # Enemy Types
    ENEMY_SPECS = {
        0: {"name": "Grunt", "health": 20, "speed": 1.0, "reward": 5, "color": (255, 50, 50)},
        1: {"name": "Tank", "health": 100, "speed": 0.6, "reward": 20, "color": (200, 40, 120)},
    }
    
    # Game Parameters
    MAX_STEPS = 18000 # 10 minutes at 30fps
    TOTAL_WAVES = 10
    INITIAL_RESOURCES = 150
    INITIAL_BASE_HEALTH = 100

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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_huge = pygame.font.SysFont("monospace", 48, bold=True)

        self._define_path()
        self.reset()
        
        self.validate_implementation()

    def _define_path(self):
        self.path_grid_coords = [
            (-1, 9), (4, 9), (4, 4), (12, 4), (12, 14), 
            (22, 14), (22, 7), (28, 7), (28, 11), (32, 11)
        ]
        self.path_pixel_coords = [
            (c[0] * self.CELL_SIZE + self.CELL_SIZE // 2, c[1] * self.CELL_SIZE + self.CELL_SIZE // 2) 
            for c in self.path_grid_coords
        ]
        self.path_rects = []
        for i in range(len(self.path_pixel_coords) - 1):
            p1 = self.path_pixel_coords[i]
            p2 = self.path_pixel_coords[i+1]
            rect = pygame.Rect(
                min(p1[0], p2[0]) - self.CELL_SIZE // 2,
                min(p1[1], p2[1]) - self.CELL_SIZE // 2,
                abs(p1[0] - p2[0]) + self.CELL_SIZE,
                abs(p1[1] - p2[1]) + self.CELL_SIZE
            )
            self.path_rects.append(rect)

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

        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_type = 0
        self.space_was_held = False

        self.current_wave = 0
        self.wave_timer = 150 # Time before first wave
        self.wave_spawning = False
        self.enemies_to_spawn = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = -0.001 # Small penalty for time passing
        self.steps += 1

        if not self.game_over:
            # --- Handle Input ---
            self._handle_actions(action)

            # --- Update Game Logic ---
            self._update_wave_manager()
            self._update_towers()
            reward += self._update_projectiles()
            reward += self._update_enemies()
            self._update_particles()

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.victory:
            reward -= 100 # Penalty for base destruction
        elif self.victory:
            reward += 100 # Bonus for winning

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # Place tower
        if space_held and not self.space_was_held:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.resources >= spec["cost"] and self._is_valid_placement(self.cursor_pos):
                # sfx: place_tower
                self.resources -= spec["cost"]
                center_x = self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                self.towers.append({
                    "pos": (center_x, center_y),
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    "target": None
                })
                self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            else:
                # sfx: placement_fail
                pass
        self.space_was_held = space_held

    def _is_valid_placement(self, grid_pos):
        # Check if on path
        cursor_rect = pygame.Rect(grid_pos[0] * self.CELL_SIZE, grid_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        if any(cursor_rect.colliderect(path_r) for path_r in self.path_rects):
            return False
        # Check if another tower is there
        for tower in self.towers:
            tower_grid_x = (tower["pos"][0] - self.CELL_SIZE // 2) // self.CELL_SIZE
            tower_grid_y = (tower["pos"][1] - self.CELL_SIZE // 2) // self.CELL_SIZE
            if grid_pos[0] == tower_grid_x and grid_pos[1] == tower_grid_y:
                return False
        return True

    def _update_wave_manager(self):
        if self.victory: return

        if not self.wave_spawning and not self.enemies:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                if self.current_wave > self.TOTAL_WAVES:
                    self.victory = True
                    return
                self.wave_spawning = True
                self._generate_wave()

        if self.wave_spawning:
            if self.wave_timer > 0:
                self.wave_timer -= 1
            elif self.enemies_to_spawn:
                enemy_type = self.enemies_to_spawn.pop(0)
                self._spawn_enemy(enemy_type)
                self.wave_timer = 20 # Spawn delay
            else:
                self.wave_spawning = False
                self.wave_timer = 300 # Time until next wave

    def _generate_wave(self):
        num_grunts = 3 + self.current_wave * 2
        num_tanks = self.current_wave // 2
        
        wave_list = [0] * num_grunts + [1] * num_tanks
        self.np_random.shuffle(wave_list)
        self.enemies_to_spawn = wave_list

    def _spawn_enemy(self, enemy_type):
        spec = self.ENEMY_SPECS[enemy_type]
        speed_multiplier = 1 + (self.current_wave - 1) * 0.05
        self.enemies.append({
            "pos": list(self.path_pixel_coords[0]),
            "type": enemy_type,
            "health": spec["health"],
            "max_health": spec["health"],
            "speed": spec["speed"] * speed_multiplier,
            "path_index": 0,
            "dist_on_path": 0,
        })
        
    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            # Invalidate target if out of range or dead
            if tower["target"] is not None:
                if tower["target"] not in self.enemies or \
                   math.dist(tower["pos"], tower["target"]["pos"]) > spec["range"]:
                    tower["target"] = None
            
            # Find new target if needed
            if tower["target"] is None:
                best_target = None
                max_dist = -1
                for enemy in self.enemies:
                    d = math.dist(tower["pos"], enemy["pos"])
                    if d <= spec["range"] and enemy["dist_on_path"] > max_dist:
                        max_dist = enemy["dist_on_path"]
                        best_target = enemy
                tower["target"] = best_target

            # Fire
            if tower["target"] is not None and tower["cooldown"] <= 0:
                # sfx: shoot
                self.projectiles.append({
                    "pos": list(tower["pos"]),
                    "target": tower["target"],
                    "damage": spec["damage"],
                    "speed": 8,
                })
                tower["cooldown"] = spec["fire_rate"]

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target = proj["target"]
            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue

            direction = (target["pos"][0] - proj["pos"][0], target["pos"][1] - proj["pos"][1])
            dist = math.hypot(*direction)
            
            if dist < proj["speed"]:
                # Hit
                # sfx: enemy_hit
                reward += 0.1
                target["health"] -= proj["damage"]
                self._create_particles(proj["pos"], self.TOWER_SPECS[0]["color"], 5, 2)
                if target["health"] <= 0:
                    # sfx: enemy_die
                    reward += 1.0
                    self.resources += self.ENEMY_SPECS[target["type"]]["reward"]
                    self._create_particles(target["pos"], target["color"], 20, 4)
                    self.enemies.remove(target)
                self.projectiles.remove(proj)
            else:
                # Move
                norm_dir = (direction[0] / dist, direction[1] / dist)
                proj["pos"][0] += norm_dir[0] * proj["speed"]
                proj["pos"][1] += norm_dir[1] * proj["speed"]
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["path_index"] >= len(self.path_pixel_coords) - 1:
                # Reached base
                # sfx: base_damage
                self.base_health -= enemy["health"] // 4 + 1
                self.enemies.remove(enemy)
                self._create_particles((self.SCREEN_WIDTH - 10, enemy["pos"][1]), self.COLOR_BASE_DMG, 30, 5)
                continue

            target_pos = self.path_pixel_coords[enemy["path_index"] + 1]
            direction = (target_pos[0] - enemy["pos"][0], target_pos[1] - enemy["pos"][1])
            dist = math.hypot(*direction)

            move_dist = min(dist, enemy["speed"])
            
            if dist > 0:
                enemy["pos"][0] += (direction[0] / dist) * move_dist
                enemy["pos"][1] += (direction[1] / dist) * move_dist
                enemy["dist_on_path"] += move_dist

            if dist < 1:
                enemy["path_index"] += 1
        return reward

    def _create_particles(self, pos, color, count, max_life):
        for _ in range(count):
            self.particles.append({
                "pos": list(pos),
                "vel": [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)],
                "life": self.np_random.integers(max_life // 2, max_life),
                "max_life": max_life,
                "color": color
            })
            
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            p["life"] -= 0.1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.victory:
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
        
        if self.game_over:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GAME_HEIGHT))
        for y in range(0, self.GAME_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw path
        for rect in self.path_rects:
            pygame.draw.rect(self.screen, self.COLOR_PATH, rect)
        
        # Draw base
        base_rect = pygame.Rect(self.SCREEN_WIDTH - 20, 0, 20, self.GAME_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        
        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            pos = (int(tower["pos"][0]), int(tower["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CELL_SIZE // 2 - 2, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CELL_SIZE // 2 - 2, spec["color"])
            
        # Draw enemies
        for enemy in self.enemies:
            spec = self.ENEMY_SPECS[enemy["type"]]
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, spec["color"])
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            bar_width = 14
            pygame.draw.rect(self.screen, (50,50,50), (pos[0] - bar_width//2, pos[1] - 12, bar_width, 3))
            pygame.draw.rect(self.screen, (50,255,50), (pos[0] - bar_width//2, pos[1] - 12, int(bar_width * health_pct), 3))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, (255, 255, 100))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, (255, 255, 100))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            s = pygame.Surface((2, 2), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p["pos"][0]), int(p["pos"][1])))

        # Draw cursor
        if not self.game_over:
            cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            is_valid = self._is_valid_placement(self.cursor_pos)
            tower_spec = self.TOWER_SPECS[self.selected_tower_type]
            can_afford = self.resources >= tower_spec["cost"]
            
            color = self.COLOR_CURSOR if (is_valid and can_afford) else self.COLOR_CURSOR_INVALID
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, 60), (0, 0, self.CELL_SIZE, self.CELL_SIZE))
            pygame.draw.rect(s, (*color, 180), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 1)
            self.screen.blit(s, cursor_rect.topleft)
            
            # Draw tower range preview
            if is_valid:
                center_x = cursor_rect.centerx
                center_y = cursor_rect.centery
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, tower_spec["range"], (*color, 80))

    def _render_ui(self):
        ui_rect = pygame.Rect(0, self.GAME_HEIGHT, self.SCREEN_WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, (25, 25, 40), ui_rect)
        pygame.draw.line(self.screen, (60, 60, 80), (0, self.GAME_HEIGHT), (self.SCREEN_WIDTH, self.GAME_HEIGHT), 2)

        # Base Health
        health_text = self.font_small.render(f"Base: {max(0, self.base_health)}%", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, self.GAME_HEIGHT + 12))
        
        # Resources
        resource_text = self.font_small.render(f"Resources: ${self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(resource_text, (160, self.GAME_HEIGHT + 12))

        # Wave
        wave_str = f"Wave: {self.current_wave}/{self.TOTAL_WAVES}" if self.current_wave > 0 else "Waiting..."
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (330, self.GAME_HEIGHT + 12))

        # Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_text = self.font_small.render(f"Next: {spec['name']} (${spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (480, self.GAME_HEIGHT + 12))
        pygame.draw.circle(self.screen, spec['color'], (465, self.GAME_HEIGHT + 20), 7)

    def _render_end_screen(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        
        end_text_str = "VICTORY" if self.victory else "GAME OVER"
        color = (100, 255, 100) if self.victory else (255, 100, 100)
        end_text = self.font_huge.render(end_text_str, True, color)
        text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
        s.blit(end_text, text_rect)
        
        score_text = self.font_large.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 40))
        s.blit(score_text, score_rect)
        
        self.screen.blit(s, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a way to display the output.
    # The environment is headless by design, but we can use Pygame to show the frames.
    
    # Set up display for manual play
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Action state
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = 0
    shift_held = 0

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            # Handle key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0

        # Handle continuous movement keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        else: movement = 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over. Final Info: {info}")
    
    # Keep the window open for a few seconds to see the end screen
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 3000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
    
    env.close()