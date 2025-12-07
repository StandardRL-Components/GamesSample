import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Space to place the selected tower and cycle to the next type. "
        "Press Shift to start the next wave."
    )

    game_description = (
        "A minimalist tower defense game. Strategically place towers to defend your base "
        "from waves of enemies. Survive 20 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_PATH = (45, 48, 58)
        self.COLOR_BASE = (0, 200, 100)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_PLACEMENT_VALID = (0, 255, 0, 50)
        self.COLOR_PLACEMENT_INVALID = (255, 0, 0, 50)

        # --- Game Constants ---
        self.MAX_STEPS = 20000 # Increased to allow for longer games
        self.MAX_WAVES = 20
        self.INITIAL_RESOURCES = 150
        self.INITIAL_BASE_HEALTH = 20
        self.CURSOR_SPEED = 8.0

        # --- Tower Specifications ---
        self.TOWER_SPECS = {
            0: {"name": "Gun", "cost": 50, "range": 100, "damage": 10, "fire_rate": 30, "color": (50, 150, 255)}, # Blue
            1: {"name": "Sniper", "cost": 100, "range": 180, "damage": 40, "fire_rate": 90, "color": (255, 200, 50)}, # Yellow
            2: {"name": "Minigun", "cost": 75, "range": 70, "damage": 5, "fire_rate": 8, "color": (255, 120, 30)}, # Orange
        }
        
        # Initialize state variables
        self.reset()
        
        # Run self-check
        # try:
        #     self.validate_implementation()
        # except AssertionError as e:
        #     print(f"Validation failed: {e}")


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.game_phase = "BETWEEN_WAVES" # "WAVE_ACTIVE"
        self.wave_number = 0
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.enemies_to_spawn = []
        self.spawn_cooldown = 0

        self.cursor_pos = np.array([self.width / 2, self.height / 2], dtype=np.float32)
        self.selected_tower_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._define_world()

        return self._get_observation(), self._get_info()

    def _define_world(self):
        self.path_points = [
            np.array([-20, 100], dtype=np.float32), np.array([150, 100], dtype=np.float32), np.array([150, 300], dtype=np.float32),
            np.array([450, 300], dtype=np.float32), np.array([450, 100], dtype=np.float32), np.array([self.width + 20, 100], dtype=np.float32)
        ]
        self.base_pos = self.path_points[-2] # Base is on the path before exit
        self.base_radius = 20
        
        self.placement_spots = [
            {"pos": np.array([80, 200], dtype=np.float32), "radius": 40},
            {"pos": np.array([250, 200], dtype=np.float32), "radius": 50},
            {"pos": np.array([380, 200], dtype=np.float32), "radius": 50},
            {"pos": np.array([520, 200], dtype=np.float32), "radius": 40},
            {"pos": np.array([300, 50], dtype=np.float32), "radius": 40},
            {"pos": np.array([300, 350], dtype=np.float32), "radius": 40},
        ]

    def step(self, action):
        reward = 0
        self.game_over = self._check_termination()
        
        if self.game_over:
            if self.base_health <= 0:
                reward = -100.0
            elif self.wave_number > self.MAX_WAVES:
                reward = 100.0
            
            return self._get_observation(), reward, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._handle_input(movement, space_held, shift_held)
        
        if self.game_phase == "WAVE_ACTIVE":
            self._spawn_enemies()
            self._update_towers()
            
            projectile_reward = self._update_projectiles()
            enemy_reward = self._update_enemies()
            reward += projectile_reward + enemy_reward
            
            if not self.enemies and not self.enemies_to_spawn:
                self.game_phase = "BETWEEN_WAVES"
                reward += 5.0 # Wave clear bonus
                self.resources += 75 + self.wave_number * 10
                self.score += 100 * self.wave_number
                if self.wave_number >= self.MAX_WAVES:
                    self.game_over = True
        
        self._update_particles()
        
        self.steps += 1
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        terminated = self._check_termination()
        truncated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True
            truncated = True

        if terminated and not self.game_over: # Apply terminal reward only on the final step
            if self.base_health <= 0:
                reward = -100.0
            elif self.wave_number > self.MAX_WAVES:
                reward = 100.0
        
        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        if movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        if movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        if movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.width)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.height)

        is_space_press = space_held and not self.prev_space_held
        if is_space_press and self.game_phase == "BETWEEN_WAVES":
            self._place_tower()
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)

        is_shift_press = shift_held and not self.prev_shift_held
        if is_shift_press and self.game_phase == "BETWEEN_WAVES":
            self._start_next_wave()

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources < spec["cost"]:
            # sfx: error_buzz
            return

        valid_spot = None
        for spot in self.placement_spots:
            if np.linalg.norm(self.cursor_pos - spot["pos"]) < spot["radius"]:
                valid_spot = spot
                break
        
        if not valid_spot:
            # sfx: error_buzz
            return

        for tower in self.towers:
            if np.linalg.norm(self.cursor_pos - tower["pos"]) < 20: # Prevent stacking
                # sfx: error_buzz
                return
        
        self.resources -= spec["cost"]
        self.towers.append({
            "pos": self.cursor_pos.copy(),
            "type": self.selected_tower_type,
            "spec": spec,
            "cooldown": 0,
            "angle": 0.0
        })
        # sfx: place_tower

    def _start_next_wave(self):
        if self.wave_number >= self.MAX_WAVES: return

        self.wave_number += 1
        self.game_phase = "WAVE_ACTIVE"
        
        num_enemies = 5 + (self.wave_number - 1) * 2
        enemy_health = 20 + self.wave_number * 10 + (self.wave_number // 5) * 20
        enemy_speed = 1.0 + math.floor((self.wave_number - 1) / 4) * 0.2
        
        for _ in range(num_enemies):
            self.enemies_to_spawn.append({
                "pos": self.path_points[0].copy(),
                "health": enemy_health,
                "max_health": enemy_health,
                "speed": enemy_speed + self.np_random.uniform(-0.1, 0.1),
                "path_index": 0,
                "value": 1.0,
                "radius": 7
            })
        # sfx: wave_start_horn

    def _spawn_enemies(self):
        self.spawn_cooldown -= 1
        if self.spawn_cooldown <= 0 and self.enemies_to_spawn:
            self.enemies.append(self.enemies_to_spawn.pop(0))
            self.spawn_cooldown = max(10, 45 - self.wave_number) # Spawn faster on later waves

    def _update_enemies(self):
        reward = 0
        for enemy in reversed(self.enemies):
            if enemy["path_index"] >= len(self.path_points) - 1:
                self.base_health -= 1
                self.enemies.remove(enemy)
                self.score -= 50
                reward -= 1.0
                # sfx: base_damage
                continue

            target_pos = self.path_points[enemy["path_index"] + 1]
            direction = target_pos - enemy["pos"]
            distance = np.linalg.norm(direction)

            if distance < enemy["speed"]:
                enemy["pos"] = target_pos.copy()
                enemy["path_index"] += 1
            else:
                enemy["pos"] += (direction / distance) * enemy["speed"]
        return reward

    def _update_towers(self):
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] > 0:
                continue

            target = None
            min_dist = tower["spec"]["range"]
            for enemy in self.enemies:
                dist = np.linalg.norm(tower["pos"] - enemy["pos"])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                # sfx: shoot
                tower["cooldown"] = tower["spec"]["fire_rate"]
                direction = target["pos"] - tower["pos"]
                tower["angle"] = math.atan2(direction[1], direction[0])
                
                self.projectiles.append({
                    "pos": tower["pos"].copy(),
                    "target_pos": target["pos"].copy(), # Fire at current position
                    "speed": 12.0,
                    "damage": tower["spec"]["damage"],
                    "color": tower["spec"]["color"],
                    "owner_tower": tower
                })

    def _update_projectiles(self):
        reward = 0
        for proj in reversed(self.projectiles):
            direction = proj["target_pos"] - proj["pos"]
            distance = np.linalg.norm(direction)
            
            if distance < proj["speed"]:
                if proj in self.projectiles: self.projectiles.remove(proj)
                continue
            
            proj["pos"] += (direction / distance) * proj["speed"]
            
            for enemy in reversed(self.enemies):
                if np.linalg.norm(proj["pos"] - enemy["pos"]) < enemy["radius"]:
                    # sfx: enemy_hit
                    enemy["health"] -= proj["damage"]
                    if proj in self.projectiles: self.projectiles.remove(proj)

                    if enemy["health"] <= 0:
                        # sfx: enemy_die
                        reward += 0.5 # Kill reward
                        self.score += 10
                        self._create_particles(enemy["pos"], self.COLOR_ENEMY, 10)
                        self.enemies.remove(enemy)
                    else:
                        self._create_particles(proj["pos"], proj["color"], 3)
                    break
        return reward

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "velocity": velocity,
                "lifetime": self.np_random.integers(10, 20),
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in reversed(self.particles):
            p["pos"] += p["velocity"]
            p["lifetime"] -= 1
            p["velocity"] *= 0.95 # Damping
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.base_health <= 0 or self.wave_number > self.MAX_WAVES

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_path()
        self._render_placement_spots()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        if self.game_phase == "BETWEEN_WAVES":
            self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "resources": self.resources,
            "phase": self.game_phase,
        }

    # --- Rendering Methods ---
    def _render_path(self):
        for i in range(len(self.path_points) - 1):
            p1 = self.path_points[i].astype(int)
            p2 = self.path_points[i+1].astype(int)
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, 40)

    def _render_base(self):
        pos = self.base_pos.astype(int)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.base_radius, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.base_radius, self.COLOR_BASE)

    def _render_placement_spots(self):
        if self.game_phase == "WAVE_ACTIVE": return
        for spot in self.placement_spots:
            pos = spot["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], spot["radius"], (100, 100, 100, 20))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spot["radius"], (100, 100, 100, 40))

    def _render_towers(self):
        for tower in self.towers:
            pos = tower["pos"].astype(int)
            color = tower["spec"]["color"]
            
            # Base
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_PATH)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, color)
            
            # Turret
            end_x = pos[0] + 15 * math.cos(tower["angle"])
            end_y = pos[1] + 15 * math.sin(tower["angle"])
            pygame.draw.line(self.screen, color, pos, (end_x, end_y), 5)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = enemy["pos"].astype(int)
            radius = int(enemy["radius"])
            
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (255, 150, 150))
            
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            bar_width = 16
            bar_height = 3
            bar_x = pos[0] - bar_width // 2
            bar_y = pos[1] - radius - 8
            pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, int(bar_width * health_pct), bar_height))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = proj["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, proj["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, (255, 255, 255))
    
    def _render_particles(self):
        for p in self.particles:
            pos = p["pos"].astype(int)
            radius = int(p["radius"])
            alpha = int(255 * (p["lifetime"] / 20))
            color = (*p["color"], alpha)
            try:
                # Use a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))
            except:
                pass # Ignore errors from invalid alpha or size

    def _render_cursor(self):
        pos = self.cursor_pos.astype(int)
        spec = self.TOWER_SPECS[self.selected_tower_type]
        
        # Check if placement is valid
        is_valid = False
        if self.resources >= spec["cost"]:
            for spot in self.placement_spots:
                if np.linalg.norm(self.cursor_pos - spot["pos"]) < spot["radius"]:
                    is_valid = True
                    break
        
        color = self.COLOR_PLACEMENT_VALID if is_valid else self.COLOR_PLACEMENT_INVALID
        
        # Draw range indicator
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], spec["range"], color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spec["range"], (*color[:3], 150))
        
        # Draw ghost tower
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, (*spec["color"], 150))
        pygame.draw.line(self.screen, (*spec["color"], 150), (pos[0]-15, pos[1]), (pos[0]+15, pos[1]), 2)
        pygame.draw.line(self.screen, (*spec["color"], 150), (pos[0], pos[1]-15), (pos[0], pos[1]+15), 2)
        
    def _render_ui(self):
        # Health
        health_text = self.font_small.render(f"Base Health: {self.base_health}/{self.INITIAL_BASE_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Resources
        resource_text = self.font_small.render(f"Resources: ${self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(resource_text, (10, 30))
        
        # Wave
        wave_text = self.font_large.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.width - wave_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.width - score_text.get_width() - 10, 40))

        # Phase indicator
        if self.game_phase == "BETWEEN_WAVES" and self.wave_number < self.MAX_WAVES:
            phase_text = self.font_large.render("Press SHIFT to start next wave", True, self.COLOR_TEXT)
            text_rect = phase_text.get_rect(center=(self.width/2, self.height - 30))
            self.screen.blit(phase_text, text_rect)
            
            # Selected tower info
            spec = self.TOWER_SPECS[self.selected_tower_type]
            tower_info = f"Selected: {spec['name']} (Cost: {spec['cost']})"
            tower_text = self.font_small.render(tower_info, True, self.COLOR_TEXT)
            self.screen.blit(tower_text, (10, self.height - 25))

        if self.game_over:
            result_text_str = "VICTORY!" if self.wave_number > self.MAX_WAVES else "GAME OVER"
            result_text = self.font_large.render(result_text_str, True, self.COLOR_BASE if self.wave_number > self.MAX_WAVES else self.COLOR_ENEMY)
            text_rect = result_text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(result_text, text_rect)


    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To see the game, comment out the `os.environ` line at the top of the file
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up a window to see the game
    pygame.display.set_caption("Tower Defense Gym Environment")
    screen = pygame.display.set_mode((env.width, env.height))

    total_reward = 0
    
    while not done:
        # --- Manual Control ---
        movement = 0 # none
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
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Render to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(60) # Run at 60 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    pygame.quit()