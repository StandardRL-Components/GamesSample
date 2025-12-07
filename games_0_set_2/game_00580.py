
# Generated: 2025-08-27T14:04:58.084459
# Source Brief: brief_00580.md
# Brief Index: 580

        
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
        "Controls: Arrow keys to move cursor. Space to place Basic Tower. "
        "Shift to place Fast Tower. Space+Shift for Long-Range Tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers. "
        "Survive 5 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_PATH = (50, 60, 70)
    COLOR_PATH_BORDER = (70, 80, 90)
    COLOR_BASE = (0, 150, 50)
    COLOR_BASE_DAMAGED = (255, 0, 0)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 255)
    
    TOWER_COLORS = {
        "basic": (60, 120, 220),
        "fast": (220, 180, 60),
        "long_range": (180, 60, 220)
    }
    PROJECTILE_COLORS = {
        "basic": (100, 180, 255),
        "fast": (255, 220, 100),
        "long_range": (220, 100, 255)
    }

    # Game parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 15000 # ~8 minutes at 30fps
    FPS = 30
    TOTAL_WAVES = 5
    TIME_BETWEEN_WAVES = 20 * FPS # 20 seconds
    
    BASE_STARTING_HEALTH = 100
    STARTING_RESOURCES = 150
    RESOURCES_PER_KILL = 15
    
    CURSOR_SPEED = 8
    PATH_WIDTH = 40
    
    TOWER_SPECS = {
        "basic": {"cost": 50, "range": 80, "fire_rate": 45, "damage": 10, "proj_speed": 6},
        "fast": {"cost": 75, "range": 60, "fire_rate": 20, "damage": 6, "proj_speed": 8},
        "long_range": {"cost": 100, "range": 120, "fire_rate": 90, "damage": 25, "proj_speed": 5}
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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.render_mode = render_mode
        self.np_random = None
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.BASE_STARTING_HEALTH
        self.resources = self.STARTING_RESOURCES
        self.current_wave = 0
        self.wave_timer = self.TIME_BETWEEN_WAVES // 2

        self.path_waypoints = self._generate_path()
        self.base_pos = self.path_waypoints[-1]
        self.base_rect = pygame.Rect(self.base_pos[0] - 20, self.base_pos[1] - 20, 40, 40)
        self.base_damage_flash = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.placement_cursor = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.last_placement_attempt = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.001 # Small time penalty
        
        self._handle_player_action(movement, space_held, shift_held)
        
        self._update_wave_system()
        step_reward = self._update_game_logic()
        reward += step_reward

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.win:
                reward += 100
            else: # Loss
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_player_action(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.placement_cursor[1] -= self.CURSOR_SPEED
        elif movement == 2: self.placement_cursor[1] += self.CURSOR_SPEED
        elif movement == 3: self.placement_cursor[0] -= self.CURSOR_SPEED
        elif movement == 4: self.placement_cursor[0] += self.CURSOR_SPEED
        self.placement_cursor[0] = np.clip(self.placement_cursor[0], 0, self.SCREEN_WIDTH)
        self.placement_cursor[1] = np.clip(self.placement_cursor[1], 0, self.SCREEN_HEIGHT)

        # Cooldown for placing towers to prevent spamming
        if self.last_placement_attempt > 0:
            self.last_placement_attempt -= 1
            return

        # Place tower
        tower_type = None
        if space_held and shift_held: tower_type = "long_range"
        elif space_held: tower_type = "basic"
        elif shift_held: tower_type = "fast"
        
        if tower_type:
            spec = self.TOWER_SPECS[tower_type]
            if self.resources >= spec["cost"] and self._is_valid_placement(self.placement_cursor):
                self.resources -= spec["cost"]
                self.towers.append({
                    "pos": list(self.placement_cursor), "type": tower_type, 
                    "cooldown": 0, "fire_flash": 0, **spec
                })
                # sfx: place_tower.wav
                self.last_placement_attempt = 10 # 1/3 second cooldown

    def _is_valid_placement(self, pos):
        # Check distance to path
        for i in range(len(self.path_waypoints) - 1):
            p1 = np.array(self.path_waypoints[i])
            p2 = np.array(self.path_waypoints[i+1])
            p_pos = np.array(pos)
            
            d = np.linalg.norm(np.cross(p2-p1, p1-p_pos))/np.linalg.norm(p2-p1) if np.linalg.norm(p2-p1) > 0 else np.linalg.norm(p1-p_pos)
            if d < self.PATH_WIDTH:
                return False
        # Check distance to other towers
        for tower in self.towers:
            if np.linalg.norm(np.array(pos) - np.array(tower["pos"])) < 20:
                return False
        # Check distance to base
        if np.linalg.norm(np.array(pos) - np.array(self.base_pos)) < 40:
            return False
        return True

    def _update_wave_system(self):
        if self.current_wave < self.TOTAL_WAVES and not self.enemies:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                self._spawn_wave()
                self.wave_timer = self.TIME_BETWEEN_WAVES
        elif not self.enemies and self.current_wave >= self.TOTAL_WAVES:
            self.win = True

    def _spawn_wave(self):
        # sfx: wave_start.wav
        num_enemies = 3 + self.current_wave * 2
        base_health = 20 * (1.1 ** self.current_wave)
        base_speed = 1.0 * (1.05 ** self.current_wave)
        
        for i in range(num_enemies):
            self.enemies.append({
                "pos": list(self.path_waypoints[0]),
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed * self.np_random.uniform(0.9, 1.1),
                "path_index": 1,
                "spawn_cooldown": i * 20 # Stagger spawns
            })
            
    def _update_game_logic(self):
        reward = 0
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        return reward

    def _update_towers(self):
        for tower in self.towers:
            if tower["fire_flash"] > 0: tower["fire_flash"] -= 1
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            
            if tower["cooldown"] == 0:
                target = None
                # Find first enemy in range
                for enemy in self.enemies:
                    if enemy["spawn_cooldown"] > 0: continue
                    dist = np.linalg.norm(np.array(tower["pos"]) - np.array(enemy["pos"]))
                    if dist <= tower["range"]:
                        target = enemy
                        break
                
                if target:
                    # sfx: shoot.wav
                    tower["cooldown"] = tower["fire_rate"]
                    tower["fire_flash"] = 5
                    self.projectiles.append({
                        "pos": list(tower["pos"]), "target": target, "type": tower["type"],
                        "damage": tower["damage"], "speed": tower["proj_speed"]
                    })
        return 0

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target_pos = np.array(proj["target"]["pos"])
            proj_pos = np.array(proj["pos"])
            
            direction = target_pos - proj_pos
            dist = np.linalg.norm(direction)

            if dist < proj["speed"]:
                # Hit
                proj["target"]["health"] -= proj["damage"]
                reward += 0.1 # Reward for hitting
                # sfx: hit.wav
                self.projectiles.remove(proj)
                self._create_hit_particles(proj_pos, self.PROJECTILE_COLORS[proj["type"]])
            else:
                # Move
                direction = direction / dist
                proj["pos"] += direction * proj["speed"]
        return reward
        
    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["spawn_cooldown"] > 0:
                enemy["spawn_cooldown"] -= 1
                continue

            if enemy["health"] <= 0:
                # sfx: enemy_die.wav
                self._create_explosion(enemy["pos"], self.COLOR_ENEMY)
                self.enemies.remove(enemy)
                self.resources += self.RESOURCES_PER_KILL
                self.score += 10
                reward += 1 # Reward for kill
                continue

            target_waypoint = np.array(self.path_waypoints[enemy["path_index"]])
            enemy_pos = np.array(enemy["pos"])
            
            direction = target_waypoint - enemy_pos
            dist = np.linalg.norm(direction)
            
            if dist < enemy["speed"]:
                enemy["path_index"] += 1
                if enemy["path_index"] >= len(self.path_waypoints):
                    # Reached base
                    self.base_health -= 10
                    self.base_damage_flash = 10
                    self.enemies.remove(enemy)
                    reward -= 5 # Penalty for base damage
                    # sfx: base_damage.wav
            else:
                direction = direction / dist
                enemy["pos"] += direction * enemy["speed"]
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.game_over: return True
        if self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.win:
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
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave,
            "win": self.win
        }
    
    # --- Rendering ---

    def _render_game(self):
        self._render_path()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()

    def _render_path(self):
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, p1, p2, self.PATH_WIDTH + 4)
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.PATH_WIDTH)

    def _render_base(self):
        color = self.COLOR_BASE_DAMAGED if self.base_damage_flash > 0 else self.COLOR_BASE
        if self.base_damage_flash > 0: self.base_damage_flash -= 1
        pygame.draw.rect(self.screen, color, self.base_rect)
        pygame.draw.rect(self.screen, self.COLOR_PATH_BORDER, self.base_rect, 2)
        
    def _render_towers(self):
        for tower in self.towers:
            pos = (int(tower["pos"][0]), int(tower["pos"][1]))
            color = (255, 255, 255) if tower["fire_flash"] > 0 else self.TOWER_COLORS[tower["type"]]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_TEXT)

    def _render_enemies(self):
        for enemy in self.enemies:
            if enemy["spawn_cooldown"] > 0: continue
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            size = 8
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (255,0,0), (rect.left, rect.top - 6, rect.width, 3))
            pygame.draw.rect(self.screen, (0,255,0), (rect.left, rect.top - 6, rect.width * health_pct, 3))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            color = self.PROJECTILE_COLORS[proj["type"]]
            pygame.draw.circle(self.screen, color, pos, 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / p["max_life"]))
            p_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.rect(p_surf, (*p["color"], alpha), (0,0,p["size"]*2,p["size"]*2))
            self.screen.blit(p_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])))

    def _render_cursor(self):
        pos = (int(self.placement_cursor[0]), int(self.placement_cursor[1]))
        is_valid = self._is_valid_placement(pos)
        
        # Determine tower type based on potential action
        tower_type = "basic" # Default for range display
        if pygame.key.get_pressed()[pygame.K_SPACE] and pygame.key.get_pressed()[pygame.K_LSHIFT]:
            tower_type = "long_range"
        elif pygame.key.get_pressed()[pygame.K_LSHIFT]:
            tower_type = "fast"
        
        cost = self.TOWER_SPECS[tower_type]["cost"]
        can_afford = self.resources >= cost
        
        cursor_color = self.COLOR_CURSOR
        if not is_valid or not can_afford:
            cursor_color = self.COLOR_ENEMY

        # Draw range indicator
        range_val = self.TOWER_SPECS[tower_type]["range"]
        range_surf = pygame.Surface((range_val * 2, range_val * 2), pygame.SRCALPHA)
        pygame.draw.circle(range_surf, (*cursor_color, 50), (range_val, range_val), range_val)
        pygame.draw.circle(range_surf, (*cursor_color, 100), (range_val, range_val), range_val, 1)
        self.screen.blit(range_surf, (pos[0] - range_val, pos[1] - range_val))
        
        # Draw cursor crosshair
        pygame.draw.line(self.screen, cursor_color, (pos[0] - 10, pos[1]), (pos[0] + 10, pos[1]), 1)
        pygame.draw.line(self.screen, cursor_color, (pos[0], pos[1] - 10), (pos[0], pos[1] + 10), 1)

    def _render_ui(self):
        # Top-left: Wave info
        if self.current_wave > 0 and self.current_wave <= self.TOTAL_WAVES:
            wave_text = f"WAVE: {self.current_wave} / {self.TOTAL_WAVES}"
        else:
            wave_text = "PREPARING..."
        if self.win: wave_text = "VICTORY!"
        if self.base_health <= 0: wave_text = "DEFEAT!"

        text_surf = self.font_small.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        if self.wave_timer > 0 and self.current_wave < self.TOTAL_WAVES and not self.enemies:
            timer_text = f"Next wave in: {self.wave_timer / self.FPS:.1f}s"
            timer_surf = self.font_small.render(timer_text, True, self.COLOR_TEXT)
            self.screen.blit(timer_surf, (10, 30))

        # Top-right: Resources
        res_text = f"RESOURCES: ${self.resources}"
        text_surf = self.font_small.render(res_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))
        
        # Bottom-center: Base Health
        hp_text = f"BASE HP: {max(0, self.base_health)} / {self.BASE_STARTING_HEALTH}"
        text_surf = self.font_small.render(hp_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH/2 - text_surf.get_width()/2, self.SCREEN_HEIGHT - 30))
        
        if self.win or self.base_health <= 0:
            end_text = "VICTORY" if self.win else "GAME OVER"
            end_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            self.screen.blit(end_surf, (self.SCREEN_WIDTH/2 - end_surf.get_width()/2, self.SCREEN_HEIGHT/2 - end_surf.get_height()/2))
    
    # --- Helper methods for effects and generation ---

    def _generate_path(self):
        points = []
        y_start = self.np_random.integers(100, self.SCREEN_HEIGHT - 100)
        points.append((50, y_start))
        
        num_segments = 4
        x_spacing = (self.SCREEN_WIDTH - 150) / num_segments
        
        for i in range(1, num_segments + 1):
            x = 50 + i * x_spacing
            y = self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
            points.append((x, y))
        
        points.append((self.SCREEN_WIDTH - 50, self.SCREEN_HEIGHT // 2))
        return points

    def _create_explosion(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 25), "max_life": 25,
                "color": color, "size": self.np_random.integers(2, 4)
            })

    def _create_hit_particles(self, pos, color):
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(5, 15), "max_life": 15,
                "color": color, "size": self.np_random.integers(1, 3)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {obs.shape}"
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import time
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    # This part is for human interaction and visualization, not part of the env itself.
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    running = True
    total_reward = 0
    
    while running:
        # Map pygame keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the display window
        # The observation is (H, W, C), but pygame blit needs a Surface.
        # So we'll get the env's internal screen surface.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            time.sleep(3) # Pause on the end screen
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.FPS)
        
    env.close()