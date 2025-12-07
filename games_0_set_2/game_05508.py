
# Generated: 2025-08-28T05:14:06.746506
# Source Brief: brief_05508.md
# Brief Index: 5508

        
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
        "Controls: Arrow keys to move the cursor. "
        "Space to place a tower. Shift to upgrade a tower."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing and "
        "upgrading towers in this top-down tower defense game."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000 * 3  # ~1.5 mins at 30fps
    TOTAL_WAVES = 10

    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_PATH = (40, 50, 65)
    COLOR_BASE = (0, 150, 136)
    COLOR_ENEMY = (244, 67, 54)
    COLOR_TOWER = (33, 150, 243)
    COLOR_PROJECTILE = (255, 235, 59)
    COLOR_PARTICLE = (224, 224, 224)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_UI_TEXT = (200, 200, 220)
    COLOR_UI_RED = (255, 82, 82)

    # --- Game Config ---
    INITIAL_RESOURCES = 150
    TOWER_COST = 50
    UPGRADE_COST_MULTIPLIER = 1.5
    REWARD_PER_KILL = 5
    CURSOR_SPEED = 15
    WAVE_DELAY_FRAMES = 90  # 3 seconds

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
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 48)

        self.np_random = None
        self.game_over_message = ""
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.game_over_message = ""

        self.resources = self.INITIAL_RESOURCES
        self.wave_number = 0
        self.wave_timer = 0
        self.enemies_to_spawn = []
        self.spawn_timer = 0

        self.path_waypoints = self._generate_path()
        self.base_pos = self.path_waypoints[-1]

        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.last_space_held = False
        self.last_shift_held = False
        
        self.insufficient_funds_flash = 0

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = -0.001  # Small penalty for time passing
        terminated = self.game_over

        if not terminated:
            self._handle_input(action)
            self._update_game_logic()
            reward += self._process_collisions_and_rewards()
            
            self.steps += 1
            terminated, win_reward = self._check_termination()
            reward += win_reward
            self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # Place tower on key press (rising edge)
        if space_held and not self.last_space_held:
            self._place_tower()

        # Upgrade tower on key press (rising edge)
        if shift_held and not self.last_shift_held:
            self._upgrade_tower()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        if self.insufficient_funds_flash > 0:
            self.insufficient_funds_flash -= 1

    def _update_game_logic(self):
        self._update_wave_spawner()
        self._update_towers()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
    def _process_collisions_and_rewards(self):
        reward = 0
        
        # Projectile-Enemy collisions
        for proj in self.projectiles[:]:
            for enemy in self.enemies[:]:
                if not proj["active"] or not enemy["active"]: continue
                dist = np.linalg.norm(proj["pos"] - enemy["pos"])
                if dist < enemy["radius"]:
                    # sfx: enemy_hit
                    enemy["health"] -= proj["damage"]
                    proj["active"] = False
                    reward += 0.1
                    if enemy["health"] <= 0:
                        enemy["active"] = False
                        reward += self.REWARD_PER_KILL
                        self.resources += enemy["value"]
                        self._create_particles(enemy["pos"], 20, self.COLOR_ENEMY)
                        # sfx: enemy_death
                    else:
                        self._create_particles(proj["pos"], 5, self.COLOR_PROJECTILE)
                    break
        
        # Remove inactive entities
        self.projectiles = [p for p in self.projectiles if p["active"]]
        self.enemies = [e for e in self.enemies if e["active"]]

        # Check for enemies reaching the base
        for enemy in self.enemies[:]:
            dist_to_base = np.linalg.norm(enemy["pos"] - self.base_pos)
            if dist_to_base < 10:
                self.game_over = True
                self.game_over_message = "GAME OVER"
                reward -= 10
                # sfx: game_over
                break

        return reward

    def _check_termination(self):
        if self.game_over:
            return True, 0

        if self.wave_number > self.TOTAL_WAVES and not self.enemies and not self.enemies_to_spawn:
            self.game_won = True
            self.game_over = True
            self.game_over_message = "YOU WIN!"
            # sfx: game_win
            return True, 100

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.game_over_message = "TIME UP"
            return True, 0
        
        return False, 0
    
    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES:
            return

        num_enemies = 5 + self.wave_number * 2
        health = 10 + (self.wave_number - 1) * 10
        speed = 0.8 + (self.wave_number - 1) * 0.1
        value = 2 + (self.wave_number - 1)
        
        self.enemies_to_spawn = []
        for i in range(num_enemies):
            self.enemies_to_spawn.append({
                "delay": i * 30, # Spawn one enemy per second
                "health": health,
                "speed": speed,
                "value": value
            })

    def _update_wave_spawner(self):
        if not self.enemies and not self.enemies_to_spawn and self.wave_number <= self.TOTAL_WAVES:
            self.wave_timer += 1
            if self.wave_timer >= self.WAVE_DELAY_FRAMES:
                self.wave_timer = 0
                self._start_next_wave()
        
        if self.enemies_to_spawn:
            self.spawn_timer += 1
            if self.spawn_timer >= self.enemies_to_spawn[0]["delay"]:
                enemy_data = self.enemies_to_spawn.pop(0)
                self.enemies.append({
                    "pos": self.path_waypoints[0].copy(),
                    "health": enemy_data["health"],
                    "max_health": enemy_data["health"],
                    "speed": enemy_data["speed"],
                    "value": enemy_data["value"],
                    "path_index": 1,
                    "radius": 8,
                    "active": True
                })
                # Reset spawn timer for the next enemy in the list
                if self.enemies_to_spawn:
                    self.spawn_timer = 0
    
    def _place_tower(self):
        if self.resources >= self.TOWER_COST:
            # Check if too close to another tower or the path
            cursor_pos_tuple = (self.cursor_pos[0], self.cursor_pos[1])
            if any(np.linalg.norm(self.cursor_pos - t['pos']) < 30 for t in self.towers):
                return
            
            min_dist_to_path = float('inf')
            for i in range(len(self.path_waypoints) - 1):
                p1 = self.path_waypoints[i]
                p2 = self.path_waypoints[i+1]
                dist = self._point_line_segment_distance(cursor_pos_tuple, (p1[0], p1[1]), (p2[0], p2[1]))
                if dist < min_dist_to_path:
                    min_dist_to_path = dist
            
            if min_dist_to_path < 25:
                return

            self.resources -= self.TOWER_COST
            self.towers.append({
                "pos": self.cursor_pos.copy(),
                "level": 1,
                "range": 100,
                "damage": 5,
                "fire_rate": 30, # frames per shot
                "cooldown": 0,
            })
            # sfx: place_tower
        else:
            self.insufficient_funds_flash = 10
            # sfx: error

    def _upgrade_tower(self):
        for tower in self.towers:
            if np.linalg.norm(self.cursor_pos - tower["pos"]) < 15:
                cost = int(self.TOWER_COST * self.UPGRADE_COST_MULTIPLIER ** tower["level"])
                if self.resources >= cost:
                    self.resources -= cost
                    tower["level"] += 1
                    tower["range"] *= 1.1
                    tower["damage"] *= 1.25
                    tower["fire_rate"] = max(10, tower["fire_rate"] * 0.9)
                    # sfx: upgrade_tower
                    break
                else:
                    self.insufficient_funds_flash = 10
                    # sfx: error
                    break
    
    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            target = None
            min_dist = tower["range"]
            for enemy in self.enemies:
                dist = np.linalg.norm(tower["pos"] - enemy["pos"])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                self.projectiles.append({
                    "pos": tower["pos"].copy(),
                    "target_pos": target["pos"].copy(),
                    "speed": 8,
                    "damage": tower["damage"],
                    "active": True
                })
                tower["cooldown"] = tower["fire_rate"]
                # sfx: tower_shoot

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy["path_index"] >= len(self.path_waypoints):
                enemy["active"] = False
                continue
            
            target_waypoint = self.path_waypoints[enemy["path_index"]]
            direction = target_waypoint - enemy["pos"]
            dist = np.linalg.norm(direction)

            if dist < enemy["speed"]:
                enemy["path_index"] += 1
            else:
                enemy["pos"] += (direction / dist) * enemy["speed"]

    def _update_projectiles(self):
        for proj in self.projectiles:
            direction = proj["target_pos"] - proj["pos"]
            dist = np.linalg.norm(direction)
            if dist < proj["speed"]:
                proj["active"] = False # Reached target's last known location
            else:
                proj["pos"] += (direction / dist) * proj["speed"]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        if len(self.path_waypoints) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 10)
        
        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.base_pos[0]-12, self.base_pos[1]-12, 24, 24))
        pygame.gfxdraw.rectangle(self.screen, (self.base_pos[0]-12, self.base_pos[1]-12, 24, 24), self.COLOR_UI_TEXT)

        # Draw towers
        for tower in self.towers:
            pos_int = (int(tower["pos"][0]), int(tower["pos"][1]))
            size = 10 + tower["level"]
            pygame.draw.rect(self.screen, self.COLOR_TOWER, (pos_int[0]-size, pos_int[1]-size, size*2, size*2))
            for i in range(tower["level"]):
                pygame.gfxdraw.rectangle(self.screen, (pos_int[0]-size+i*2, pos_int[1]-size+i*2, (size-i*2)*2, (size-i*2)*2), self.COLOR_UI_TEXT)

        # Draw enemies
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            radius = int(enemy["radius"])
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            if health_ratio < 1.0:
                bar_width = 16
                pygame.draw.rect(self.screen, (80,0,0), (pos_int[0]-bar_width/2, pos_int[1]-15, bar_width, 4))
                pygame.draw.rect(self.screen, (0,200,0), (pos_int[0]-bar_width/2, pos_int[1]-15, bar_width * health_ratio, 4))

        # Draw projectiles
        for proj in self.projectiles:
            pos_int = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 3, self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            size = int(p["size"] * (p["life"] / p["max_life"]))
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, size, size, size, color)
                self.screen.blit(temp_surf, (p["pos"][0]-size, p["pos"][1]-size))

        # Draw cursor and tower range
        cursor_pos_int = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 10, (*self.COLOR_CURSOR, 50))
        pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 10, self.COLOR_CURSOR)
        
        # Show tower range if hovering
        hovered_tower = None
        for tower in self.towers:
            if np.linalg.norm(self.cursor_pos - tower['pos']) < 15:
                hovered_tower = tower
                break
        
        if hovered_tower:
            pos = (int(hovered_tower['pos'][0]), int(hovered_tower['pos'][1]))
            radius = int(hovered_tower['range'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.COLOR_TOWER, 100))
        else: # Show placement range
            pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 100, (*self.COLOR_CURSOR, 50))

    def _render_ui(self):
        # Wave text
        wave_text = f"Wave: {min(self.wave_number, self.TOTAL_WAVES)} / {self.TOTAL_WAVES}"
        text_surf = self.font_small.render(wave_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Resources text
        res_color = self.COLOR_UI_RED if self.insufficient_funds_flash > 0 else self.COLOR_UI_TEXT
        resource_text = f"Resources: {self.resources}"
        text_surf = self.font_small.render(resource_text, True, res_color)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)
        
        # Game Over/Win message
        if self.game_over:
            color = self.COLOR_BASE if self.game_won else self.COLOR_ENEMY
            text_surf = self.font_large.render(self.game_over_message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "towers": len(self.towers),
            "enemies": len(self.enemies)
        }
        
    def _generate_path(self):
        waypoints = []
        x, y = 50, self.np_random.integers(100, self.SCREEN_HEIGHT - 100)
        waypoints.append(np.array([x, y], dtype=float))
        
        while x < self.SCREEN_WIDTH - 50:
            x += self.np_random.integers(50, 100)
            x = min(x, self.SCREEN_WIDTH - 50)
            y_offset = self.np_random.integers(-80, 81)
            y = np.clip(y + y_offset, 50, self.SCREEN_HEIGHT - 50)
            waypoints.append(np.array([x, y], dtype=float))
        return waypoints

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "life": self.np_random.integers(10, 20),
                "max_life": 20,
                "color": color,
                "size": self.np_random.integers(2, 5)
            })
    
    @staticmethod
    def _point_line_segment_distance(p, a, b):
        p, a, b = np.array(p), np.array(a), np.array(b)
        if np.array_equal(a, b):
            return np.linalg.norm(p - a)
        
        l2 = np.sum((a - b)**2)
        t = max(0, min(1, np.dot(p - a, b - a) / l2))
        projection = a + t * (b - a)
        return np.linalg.norm(p - projection)

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
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Human Play Controls ---
    # pip install keyboard
    try:
        import keyboard
        print("\n" + "="*30)
        print("Human Controls Enabled:")
        print(GameEnv.user_guide)
        print("Press 'q' to quit.")
        print("="*30 + "\n")

        while not terminated:
            movement = 0 # none
            if keyboard.is_pressed('up arrow'): movement = 1
            elif keyboard.is_pressed('down arrow'): movement = 2
            elif keyboard.is_pressed('left arrow'): movement = 3
            elif keyboard.is_pressed('right arrow'): movement = 4
            
            space = 1 if keyboard.is_pressed('space') else 0
            shift = 1 if keyboard.is_pressed('shift') else 0

            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Convert to Pygame surface for display
            frame = np.transpose(obs, (1, 0, 2))
            game_surface = pygame.surfarray.make_surface(frame)
            
            # Create a display if one doesn't exist
            try:
                display_surface = pygame.display.get_surface()
                if display_surface is None:
                    raise AttributeError
            except (pygame.error, AttributeError):
                display_surface = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
                pygame.display.set_caption("Tower Defense")

            display_surface.blit(game_surface, (0, 0))
            pygame.display.flip()

            if keyboard.is_pressed('q'):
                break

    except ImportError:
        print("\n'keyboard' library not found. Running random agent.")
        print("Install it with 'pip install keyboard' to play manually.")
        
        for _ in range(5000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode finished. Info: {info}")
                obs, info = env.reset()

    env.close()
    pygame.quit()