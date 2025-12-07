
# Generated: 2025-08-27T20:09:33.719867
# Source Brief: brief_02368.md
# Brief Index: 2368

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ↑↓←→ to move the cursor. Press space to place a tower. Hold shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    CELL_SIZE = SCREEN_WIDTH // GRID_WIDTH

    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 52, 100)
    COLOR_PATH = (60, 66, 78)
    COLOR_BASE = (66, 165, 245)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_INVALID_CURSOR = (255, 82, 82)
    
    TOWER_SPECS = {
        0: {"name": "Cannon", "cost": 100, "range": 80, "fire_rate": 45, "damage": 25, "color": (255, 202, 40)}, # Yellow
        1: {"name": "Frost", "cost": 75, "range": 60, "fire_rate": 60, "damage": 5, "slow_potency": 0.5, "slow_duration": 60, "color": (33, 150, 243)} # Blue
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        self.render_mode = render_mode
        self.game_over_message = ""

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Create a default RNG if no seed is provided
            if not hasattr(self, 'rng'):
                self.rng = np.random.default_rng()

        self.steps = 0
        self.max_steps = 5000
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = 100
        self.resources = 150
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        self.last_shift_state = 0
        self.placement_feedback = 0 # Timer for showing placement success/failure
        self.placement_feedback_color = (0,0,0)

        self._define_path()
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.wave_number = 0
        self.total_waves = 5
        self.wave_active = False
        self.enemies_in_wave = []
        self.spawn_timer = 0
        self.inter_wave_timer = 90 # Time before first wave starts

        return self._get_observation(), self._get_info()

    def _define_path(self):
        self.path = []
        points = [
            (-1, 5), (5, 5), (5, 15), (15, 15), (15, 2), (25, 2), (25, 10), (self.GRID_WIDTH + 1, 10)
        ]
        for i in range(len(points) - 1):
            p1 = np.array(points[i]) * self.CELL_SIZE
            p2 = np.array(points[i+1]) * self.CELL_SIZE
            self.path.append((p1, p2))
        self.base_pos_grid = points[-2]
        self.base_pos_px = np.array(self.base_pos_grid) * self.CELL_SIZE + self.CELL_SIZE / 2

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01 # Small penalty per step to encourage action
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        step_reward = self._handle_input(movement, space_held, shift_held)
        reward += step_reward

        if not self.wave_active and not self.game_won:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0:
                self._start_next_wave()
        
        if self.wave_active:
            self._update_wave_spawner()
        
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        reward += self._check_game_state()

        terminated = self.game_over or self.steps >= self.max_steps
        if self.steps >= self.max_steps and not self.game_over:
             self.game_over = True
             self.game_over_message = "TIME LIMIT REACHED"
             reward -= 50

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # --- Move Cursor ---
        if movement == 1: self.cursor_pos[1] -= 1
        if movement == 2: self.cursor_pos[1] += 1
        if movement == 3: self.cursor_pos[0] -= 1
        if movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = self.cursor_pos[0] % self.GRID_WIDTH
        self.cursor_pos[1] = self.cursor_pos[1] % self.GRID_HEIGHT

        # --- Cycle Tower ---
        if shift_held and not self.last_shift_state:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self.last_shift_state = shift_held

        # --- Place Tower ---
        if space_held:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.resources >= spec["cost"] and self._is_valid_placement(self.cursor_pos):
                self.resources -= spec["cost"]
                self.towers.append({
                    "pos": list(self.cursor_pos),
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    "target": None
                })
                self.placement_feedback = 30
                self.placement_feedback_color = spec["color"]
                # sfx: tower_placed
                return -spec["cost"] * 0.01 # Small penalty for spending
            else:
                self.placement_feedback = 30
                self.placement_feedback_color = self.COLOR_INVALID_CURSOR
                # sfx: placement_failed
        return 0

    def _is_valid_placement(self, grid_pos):
        # Check if on another tower
        for tower in self.towers:
            if tower["pos"] == grid_pos:
                return False
        # Check if on path
        cursor_center = (np.array(grid_pos) + 0.5) * self.CELL_SIZE
        for p1, p2 in self.path:
            # Simple bounding box check
            min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
            min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
            if (min_x - self.CELL_SIZE < cursor_center[0] < max_x + self.CELL_SIZE and
                min_y - self.CELL_SIZE < cursor_center[1] < max_y + self.CELL_SIZE):
                 return False
        return True

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.total_waves:
            return

        self.wave_active = True
        base_health = 50 + (self.wave_number - 1) * 20
        base_speed = 1.0 + (self.wave_number - 1) * 0.1
        enemy_count = 5 + self.wave_number * 2
        spawn_delay = max(15, 60 - self.wave_number * 5)
        
        self.enemies_in_wave = []
        for _ in range(enemy_count):
            self.enemies_in_wave.append({
                "health": base_health * (self.rng.random() * 0.4 + 0.8), # 80-120%
                "speed": base_speed * (self.rng.random() * 0.4 + 0.8),
                "spawn_delay": spawn_delay
            })

    def _update_wave_spawner(self):
        if not self.enemies_in_wave:
            return
        
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            enemy_spec = self.enemies_in_wave.pop(0)
            self.spawn_timer = enemy_spec["spawn_delay"]
            self.enemies.append({
                "pos": np.array(self.path[0][0], dtype=float),
                "path_index": 0,
                "health": enemy_spec["health"],
                "max_health": enemy_spec["health"],
                "speed": enemy_spec["speed"],
                "effects": {"slow": 0},
                "id": self.rng.integers(1, 1_000_000)
            })

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            
            if tower["cooldown"] > 0:
                continue

            # Find a target
            target = None
            min_dist = spec["range"] ** 2
            tower_pos_px = (np.array(tower["pos"]) + 0.5) * self.CELL_SIZE
            
            valid_enemies = [e for e in self.enemies if e is not None]

            for enemy in valid_enemies:
                dist_sq = np.sum((enemy["pos"] - tower_pos_px) ** 2)
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                tower["cooldown"] = spec["fire_rate"]
                self.projectiles.append({
                    "start_pos": tower_pos_px.copy(),
                    "pos": tower_pos_px.copy(),
                    "target_id": target["id"],
                    "type": tower["type"],
                    "speed": 8.0,
                })
                # sfx: tower_fire
        return reward

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            target = next((e for e in self.enemies if e and e["id"] == proj["target_id"]), None)
            
            if not target:
                continue # Target is gone, projectile fizzles

            spec = self.TOWER_SPECS[proj["type"]]
            dir_vec = target["pos"] - proj["pos"]
            dist = np.linalg.norm(dir_vec)

            if dist < proj["speed"]:
                # Hit!
                target["health"] -= spec["damage"]
                reward += 0.1 # Reward for hit
                if "slow_potency" in spec:
                    target["effects"]["slow"] = max(target["effects"]["slow"], spec["slow_duration"])
                
                self._create_explosion(target["pos"], 5, spec["color"])
                # sfx: projectile_hit
            else:
                proj["pos"] += (dir_vec / dist) * proj["speed"]
                projectiles_to_keep.append(proj)
        self.projectiles = projectiles_to_keep
        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_keep = []
        for enemy in self.enemies:
            if enemy["health"] <= 0:
                self.score += 10
                self.resources += 25
                reward += 1.0 # Reward for kill
                self._create_explosion(enemy["pos"], 15, (255, 82, 82))
                # sfx: enemy_destroyed
                continue

            speed = enemy["speed"]
            if enemy["effects"]["slow"] > 0:
                enemy["effects"]["slow"] -= 1
                slow_potency = self.TOWER_SPECS[1]["slow_potency"]
                speed *= (1 - slow_potency)

            path_start, path_end = self.path[enemy["path_index"]]
            direction = path_end - path_start
            dist_to_end = np.linalg.norm(path_end - enemy["pos"])

            if dist_to_end < speed:
                enemy["path_index"] += 1
                if enemy["path_index"] >= len(self.path):
                    # Reached base
                    self.base_health = max(0, self.base_health - 10)
                    self._create_explosion(self.base_pos_px, 20, self.COLOR_BASE)
                    # sfx: base_hit
                    continue
                else:
                    enemy["pos"] = path_end.copy()
            else:
                move_vec = direction / np.linalg.norm(direction)
                enemy["pos"] += move_vec * speed
            
            enemies_to_keep.append(enemy)
        self.enemies = enemies_to_keep
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _check_game_state(self):
        reward = 0
        # Check for loss
        if self.base_health <= 0 and not self.game_over:
            self.game_over = True
            self.game_over_message = "BASE DESTROYED"
            reward -= 100
            return reward

        # Check for wave completion
        if self.wave_active and not self.enemies and not self.enemies_in_wave:
            self.wave_active = False
            self.score += 100
            self.resources += 50 + self.wave_number * 10
            reward += 10
            
            if self.wave_number >= self.total_waves:
                self.game_over = True
                self.game_won = True
                self.game_over_message = "VICTORY!"
                reward += 100
            else:
                self.inter_wave_timer = 300 # Time before next wave
        
        return reward

    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 + 1
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle), math.sin(angle)]) * speed,
                "life": self.rng.integers(10, 20),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_path()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_path(self):
        for p1, p2 in self.path:
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.CELL_SIZE)
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos_px[0]), int(self.base_pos_px[1]), self.CELL_SIZE // 2, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, int(self.base_pos_px[0]), int(self.base_pos_px[1]), self.CELL_SIZE // 2, self.COLOR_BASE)

    def _render_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            pos_px = (np.array(tower["pos"]) + 0.5) * self.CELL_SIZE
            x, y = int(pos_px[0]), int(pos_px[1])
            
            # Draw tower triangle
            points = [
                (x, y - 8),
                (x - 7, y + 5),
                (x + 7, y + 5)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, spec["color"])
            pygame.gfxdraw.filled_polygon(self.screen, points, spec["color"])

    def _render_enemies(self):
        for enemy in self.enemies:
            x, y = int(enemy["pos"][0]), int(enemy["pos"][1])
            radius = self.CELL_SIZE // 3
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, (211, 47, 47)) # Red
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, (211, 47, 47))
            
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            bar_width = radius * 2
            pygame.draw.rect(self.screen, (50, 50, 50), (x - radius, y - radius - 8, bar_width, 5))
            pygame.draw.rect(self.screen, (118, 255, 3), (x - radius, y - radius - 8, bar_width * health_ratio, 5))

    def _render_projectiles(self):
        for proj in self.projectiles:
            spec = self.TOWER_SPECS[proj["type"]]
            pygame.draw.rect(self.screen, spec["color"], (proj["pos"][0] - 2, proj["pos"][1] - 2, 4, 4))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 20.0))))
            color = p["color"] + (alpha,)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p["pos"][0]), int(p["pos"][1])))

    def _render_cursor(self):
        pos_px = np.array(self.cursor_pos) * self.CELL_SIZE
        spec = self.TOWER_SPECS[self.selected_tower_type]
        is_valid = self._is_valid_placement(self.cursor_pos) and self.resources >= spec["cost"]
        
        # Draw placement rect
        cursor_color = self.COLOR_CURSOR if is_valid else self.COLOR_INVALID_CURSOR
        rect = pygame.Rect(pos_px[0], pos_px[1], self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, cursor_color, rect, 2)
        
        # Draw range indicator
        range_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(range_surface, pos_px[0] + self.CELL_SIZE // 2, pos_px[1] + self.CELL_SIZE // 2, spec["range"], (*cursor_color, 30))
        pygame.gfxdraw.aacircle(range_surface, pos_px[0] + self.CELL_SIZE // 2, pos_px[1] + self.CELL_SIZE // 2, spec["range"], (*cursor_color, 100))
        self.screen.blit(range_surface, (0, 0))

        if self.placement_feedback > 0:
            self.placement_feedback -= 1
            alpha = int(255 * (self.placement_feedback / 30))
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((*self.placement_feedback_color, alpha))
            self.screen.blit(s, pos_px)

    def _render_ui(self):
        # Top Left Info
        health_text = self.font_small.render(f"BASE HP: {self.base_health}", True, self.COLOR_UI_TEXT)
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(score_text, (10, 30))
        
        # Top Right Info
        wave_text = self.font_small.render(f"WAVE: {min(self.wave_number, self.total_waves)}/{self.total_waves}", True, self.COLOR_UI_TEXT)
        resource_text = self.font_small.render(f"$: {self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        self.screen.blit(resource_text, (self.SCREEN_WIDTH - resource_text.get_width() - 10, 30))

        # Bottom Center: Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_name = self.font_large.render(spec["name"], True, spec["color"])
        tower_cost = self.font_small.render(f"Cost: ${spec['cost']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(tower_name, (self.SCREEN_WIDTH/2 - tower_name.get_width()/2, self.SCREEN_HEIGHT - 45))
        self.screen.blit(tower_cost, (self.SCREEN_WIDTH/2 - tower_cost.get_width()/2, self.SCREEN_HEIGHT - 25))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        color = (100, 255, 100) if self.game_won else (255, 100, 100)
        message_surf = self.font_large.render(self.game_over_message, True, color)
        pos = (self.SCREEN_WIDTH/2 - message_surf.get_width()/2, self.SCREEN_HEIGHT/2 - message_surf.get_height()/2)
        self.screen.blit(message_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
            "game_over": self.game_over,
        }

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # --- Keyboard to Action Mapping ---
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    print(GameEnv.user_guide)
    print(GameEnv.game_description)

    while not terminated:
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human playability
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()