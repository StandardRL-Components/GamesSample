
# Generated: 2025-08-27T19:38:03.568219
# Source Brief: brief_02214.md
# Brief Index: 2214

        
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

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle tower types. Press Space to place the selected tower."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 15000 # Approx 8 minutes at 30fps
        self.MAX_WAVES = 10

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_PATH = (45, 50, 64)
        self.COLOR_GRID = (35, 40, 52)
        self.COLOR_BASE = (60, 179, 113)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_HEALTH = (220, 20, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_INVALID = (255, 0, 0, 100)
        self.COLOR_TOWER_TYPES = [
            (50, 150, 255), (255, 100, 50), (200, 50, 200), (0, 255, 150)
        ]
        self.ENEMY_COLORS = [(255, 80, 80), (255, 150, 80), (200, 80, 255)]

        # Tower definitions
        self.TOWER_SPECS = [
            {"name": "Gun Turret", "cost": 50, "range": 100, "damage": 5, "fire_rate": 20, "splash": 0},
            {"name": "Cannon", "cost": 120, "range": 80, "damage": 15, "fire_rate": 60, "splash": 25},
            {"name": "Sniper", "cost": 150, "range": 200, "damage": 40, "fire_rate": 100, "splash": 0},
            {"name": "Slow Tower", "cost": 80, "range": 90, "damage": 1, "fire_rate": 30, "splash": 30, "slow": 0.5},
        ]

        # Enemy definitions
        self.ENEMY_SPECS = [
            {"name": "Scout", "health": 50, "speed": 1.2, "gold": 5, "damage": 5},
            {"name": "Bruiser", "health": 120, "speed": 0.8, "gold": 10, "damage": 10},
            {"name": "Swarm", "health": 25, "speed": 1.5, "gold": 3, "damage": 3},
        ]

        # Path waypoints
        self.PATH_WAYPOINTS = [
            (-20, 100), (100, 100), (100, 300), (300, 300),
            (300, 50), (500, 50), (500, 250), (self.WIDTH + 20, 250)
        ]

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 64)

        # Initialize state variables
        self.reset()
        
        # This check is for development; comment out for production
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        # Player state
        self.base_health = 100
        self.gold = 150
        
        # Game entities
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        # Wave management
        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_timer = self.FPS * 5  # 5 seconds until first wave
        self.enemies_to_spawn = []

        # Player controls state
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        self.selected_tower_type = 0
        self.prev_space_held = 0
        self.prev_shift_held = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        # Time penalty
        reward -= 0.001

        # Handle input and update game state
        reward += self._handle_input(action)
        self._update_waves()
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()

        # Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.victory:
                reward += 100
            elif self.base_health <= 0:
                reward -= 100
        
        self.score += reward

        # Tick clock for auto-advance
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action
        cursor_speed = 8

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # Cycle tower type on shift press
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_cycle_sound
        self.prev_shift_held = shift_held

        # Place tower on space press
        if space_held and not self.prev_space_held:
            self._place_tower()
        self.prev_space_held = space_held
        
        return 0

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.gold >= spec["cost"] and self._is_valid_placement(self.cursor_pos):
            self.gold -= spec["cost"]
            self.towers.append({
                "pos": self.cursor_pos.copy(),
                "type": self.selected_tower_type,
                "cooldown": 0,
                "target": None
            })
            # sfx: place_tower_sound
            self._create_particles(self.cursor_pos, self.COLOR_TOWER_TYPES[self.selected_tower_type], 20, 3)

    def _is_valid_placement(self, pos):
        # Check proximity to path
        for i in range(len(self.PATH_WAYPOINTS) - 1):
            p1 = np.array(self.PATH_WAYPOINTS[i])
            p2 = np.array(self.PATH_WAYPOINTS[i+1])
            d = np.linalg.norm(np.cross(p2-p1, p1-pos))/np.linalg.norm(p2-p1) if np.linalg.norm(p2-p1) != 0 else np.linalg.norm(p1-pos)
            if d < 40: # 20 path radius + 20 buffer
                return False
        # Check proximity to other towers
        for tower in self.towers:
            if np.linalg.norm(pos - tower["pos"]) < 20:
                return False
        return True

    def _update_waves(self):
        if self.game_over: return

        if not self.wave_in_progress and self.wave_number < self.MAX_WAVES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave_number += 1
                self.wave_in_progress = True
                self.enemies_to_spawn = self._generate_wave()
                # sfx: wave_start_horn
        
        if self.wave_in_progress and self.enemies_to_spawn:
            if self.steps % 15 == 0: # Spawn interval
                enemy_type = self.enemies_to_spawn.pop(0)
                self._spawn_enemy(enemy_type)

        if self.wave_in_progress and not self.enemies and not self.enemies_to_spawn:
            self.wave_in_progress = False
            self.wave_timer = self.FPS * 10 # 10 seconds between waves
            self.gold += 100 + self.wave_number * 10 # End of wave bonus

    def _generate_wave(self):
        wave_composition = []
        num_enemies = 5 + self.wave_number * 3
        for _ in range(num_enemies):
            # Introduce enemy types gradually
            max_type = min(len(self.ENEMY_SPECS) - 1, self.wave_number // 3)
            enemy_type = self.np_random.integers(0, max_type + 1)
            wave_composition.append(enemy_type)
        return wave_composition

    def _spawn_enemy(self, enemy_type):
        spec = self.ENEMY_SPECS[enemy_type].copy()
        difficulty_mod = 1 + (self.wave_number - 1) * 0.1
        self.enemies.append({
            "pos": np.array(self.PATH_WAYPOINTS[0], dtype=float),
            "type": enemy_type,
            "health": spec["health"] * difficulty_mod,
            "max_health": spec["health"] * difficulty_mod,
            "speed": spec["speed"],
            "gold": spec["gold"],
            "damage": spec["damage"],
            "waypoint_idx": 1,
            "slow_timer": 0
        })

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            
            if tower["cooldown"] == 0:
                target = None
                min_dist = spec["range"]
                for enemy in self.enemies:
                    dist = np.linalg.norm(tower["pos"] - enemy["pos"])
                    if dist <= min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    tower["cooldown"] = spec["fire_rate"]
                    self._create_projectile(tower, target)
                    self._create_particles(tower["pos"], (255,255,200), 3, 1) # Muzzle flash
                    # sfx: tower_fire_sound
        return reward

    def _create_projectile(self, tower, target):
        spec = self.TOWER_SPECS[tower["type"]]
        self.projectiles.append({
            "pos": tower["pos"].copy(),
            "type": tower["type"],
            "target": target,
            "speed": 10
        })

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            if p["target"] not in self.enemies:
                self.projectiles.remove(p)
                continue

            direction = p["target"]["pos"] - p["pos"]
            dist = np.linalg.norm(direction)
            if dist < p["speed"]:
                p["pos"] = p["target"]["pos"]
            else:
                p["pos"] += (direction / dist) * p["speed"]

            if np.linalg.norm(p["pos"] - p["target"]["pos"]) < 10:
                reward += self._hit_enemy(p["target"], p)
                self.projectiles.remove(p)
                self._create_particles(p["pos"], self.COLOR_TOWER_TYPES[p["type"]], 10, 2)
                # sfx: projectile_hit_sound
        return reward

    def _hit_enemy(self, enemy, projectile):
        reward = 0
        spec = self.TOWER_SPECS[projectile["type"]]
        enemy["health"] -= spec["damage"]
        reward += 0.01 # Small reward for a hit

        if "slow" in spec:
            enemy["slow_timer"] = self.FPS * 2 # Slow for 2 seconds

        # Splash damage
        if spec["splash"] > 0:
            for other_enemy in self.enemies:
                if other_enemy is not enemy and np.linalg.norm(enemy["pos"] - other_enemy["pos"]) <= spec["splash"]:
                    other_enemy["health"] -= spec["damage"] * 0.5 # Splash does 50% damage
                    if "slow" in spec:
                        other_enemy["slow_timer"] = self.FPS * 2
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                self.gold += enemy["gold"]
                reward += 1
                self.enemies.remove(enemy)
                self._create_particles(enemy["pos"], self.ENEMY_COLORS[enemy["type"]], 30, 4)
                # sfx: enemy_death_explosion
                continue

            if enemy["waypoint_idx"] >= len(self.PATH_WAYPOINTS):
                self.base_health -= enemy["damage"]
                reward -= 10
                self.enemies.remove(enemy)
                self._create_particles(self.PATH_WAYPOINTS[-1], self.COLOR_HEALTH, 50, 5)
                # sfx: base_damage_sound
                continue
            
            target_pos = np.array(self.PATH_WAYPOINTS[enemy["waypoint_idx"]])
            direction = target_pos - enemy["pos"]
            dist = np.linalg.norm(direction)
            
            speed = enemy["speed"]
            if enemy["slow_timer"] > 0:
                enemy["slow_timer"] -= 1
                slow_spec = next((s for s in self.TOWER_SPECS if "slow" in s), None)
                if slow_spec:
                    speed *= slow_spec["slow"]

            if dist < speed:
                enemy["pos"] = target_pos
                enemy["waypoint_idx"] += 1
            else:
                enemy["pos"] += (direction / dist) * speed
        return reward

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * max_speed
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.game_over: return True
        
        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
            return True
        
        if self.wave_number >= self.MAX_WAVES and not self.wave_in_progress and not self.enemies:
            self.game_over = True
            self.victory = True
            return True

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.PATH_WAYPOINTS, 40)
        for point in self.PATH_WAYPOINTS:
            pygame.draw.circle(self.screen, self.COLOR_PATH, point, 20)

        # Draw base
        base_pos = self.PATH_WAYPOINTS[-1]
        pygame.draw.circle(self.screen, self.COLOR_BASE, (int(base_pos[0]), int(base_pos[1])), 25)

    def _render_game_elements(self):
        # Draw towers and their ranges if cursor is nearby
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            color = self.COLOR_TOWER_TYPES[tower["type"]]
            pos = (int(tower["pos"][0]), int(tower["pos"][1]))
            pygame.draw.circle(self.screen, color, pos, 10)
            pygame.draw.circle(self.screen, (255,255,255), pos, 10, 1)

        # Draw projectiles
        for p in self.projectiles:
            color = self.COLOR_TOWER_TYPES[p["type"]]
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.draw.circle(self.screen, color, pos, 3)

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            color = self.ENEMY_COLORS[enemy["type"]]
            size = 8
            pygame.draw.circle(self.screen, color, pos, size)
            
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            bar_width = 20
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (pos[0] - bar_width/2, pos[1] - 15, bar_width, 4))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (pos[0] - bar_width/2, pos[1] - 15, bar_width * health_pct, 4))
            
            if enemy["slow_timer"] > 0:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size + 2, self.TOWER_SPECS[3]["color"])

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(p["lifetime"] * 255 / 30)))
            color = p["color"] + (alpha,)
            temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2,2), 2)
            self.screen.blit(temp_surf, (int(p["pos"][0]-2), int(p["pos"][1]-2)))

    def _render_ui(self):
        # Top bar background
        pygame.draw.rect(self.screen, (0,0,0,150), (0, 0, self.WIDTH, 40))

        # Health
        health_text = self.font_medium.render(f"❤️ {int(self.base_health)}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Gold
        gold_text = self.font_medium.render(f"💰 {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (130, 10))

        # Wave
        if self.wave_in_progress:
            wave_str = f"Wave: {self.wave_number}/{self.MAX_WAVES}"
        elif self.wave_number < self.MAX_WAVES:
            wave_str = f"Next wave in: {int(self.wave_timer / self.FPS)}s"
        else:
            wave_str = "All waves cleared!"
        wave_text = self.font_medium.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH / 2 - wave_text.get_width() / 2, 10))

        # Cursor and tower placement preview
        if not self.game_over:
            self._render_cursor()

    def _render_cursor(self):
        cursor_pos_int = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        spec = self.TOWER_SPECS[self.selected_tower_type]
        valid_placement = self._is_valid_placement(self.cursor_pos)
        can_afford = self.gold >= spec["cost"]

        # Draw range
        range_color = (255, 255, 255, 50) if valid_placement and can_afford else self.COLOR_INVALID
        temp_surf = pygame.Surface((spec["range"] * 2, spec["range"] * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, range_color, (spec["range"], spec["range"]), spec["range"])
        self.screen.blit(temp_surf, (cursor_pos_int[0] - spec["range"], cursor_pos_int[1] - spec["range"]))

        # Draw cursor
        cursor_color = self.COLOR_CURSOR if valid_placement and can_afford else (255, 0, 0)
        pygame.draw.circle(self.screen, cursor_color, cursor_pos_int, 10, 2)
        pygame.draw.line(self.screen, cursor_color, (cursor_pos_int[0] - 15, cursor_pos_int[1]), (cursor_pos_int[0] + 15, cursor_pos_int[1]), 2)
        pygame.draw.line(self.screen, cursor_color, (cursor_pos_int[0], cursor_pos_int[1] - 15), (cursor_pos_int[0], cursor_pos_int[1] + 15), 2)
        
        # Draw selected tower info
        tower_info_y = self.HEIGHT - 70
        pygame.draw.rect(self.screen, (0,0,0,150), (0, tower_info_y, self.WIDTH, 70))
        
        name_text = self.font_medium.render(f"[{self.selected_tower_type+1}] {spec['name']}", True, self.COLOR_TOWER_TYPES[self.selected_tower_type])
        self.screen.blit(name_text, (10, tower_info_y + 10))

        cost_color = self.COLOR_GOLD if can_afford else self.COLOR_HEALTH
        cost_text = self.font_small.render(f"Cost: {spec['cost']}", True, cost_color)
        self.screen.blit(cost_text, (10, tower_info_y + 40))

        stats_text = self.font_small.render(f"Dmg: {spec['damage']} | Range: {spec['range']} | Rate: {spec['fire_rate']}", True, self.COLOR_TEXT)
        self.screen.blit(stats_text, (150, tower_info_y + 40))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        msg = "VICTORY!" if self.victory else "DEFEAT"
        color = self.COLOR_BASE if self.victory else self.COLOR_HEALTH
        text = self.font_large.render(msg, True, color)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
        self.screen.blit(text, text_rect)

        score_text = self.font_medium.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 30))
        self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.wave_number
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("🔬 Validating implementation...")
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

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    terminated = False
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        # --- End Human Controls ---

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)

    env.close()