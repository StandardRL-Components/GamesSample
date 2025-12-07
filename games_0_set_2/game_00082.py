
# Generated: 2025-08-27T12:33:02.681948
# Source Brief: brief_00082.md
# Brief Index: 82

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to select a build location. SHIFT to cycle turret type. SPACE to build."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing turrets."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500  # 50 seconds at 30fps

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_PATH = (40, 40, 55)
    COLOR_BASE = (0, 200, 100)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_ZONE_EMPTY = (100, 100, 120, 100)
    COLOR_ZONE_SELECT = (255, 255, 0)
    
    # Game parameters
    BASE_MAX_HEALTH = 100
    INITIAL_RESOURCES = 250
    WAVE_COUNT = 5
    ZONES_PER_ROW = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.font_reward = pygame.font.Font(None, 20)

        # Game data structures
        self._define_turret_specs()
        self._define_enemy_path()
        self._define_placement_zones()
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def _define_turret_specs(self):
        self.TURRET_SPECS = {
            0: {"name": "Gatling", "cost": 50, "damage": 4, "fire_rate": 8, "range": 90, "color": (0, 150, 255)},
            1: {"name": "Cannon", "cost": 125, "damage": 30, "fire_rate": 45, "range": 130, "color": (255, 150, 0)},
            2: {"name": "Frost", "cost": 80, "damage": 1, "fire_rate": 20, "range": 70, "color": (100, 200, 255), "slow": 0.5}
        }

    def _define_enemy_path(self):
        self.path_points = [
            pygame.math.Vector2(-20, 50),
            pygame.math.Vector2(120, 50),
            pygame.math.Vector2(120, 200),
            pygame.math.Vector2(520, 200),
            pygame.math.Vector2(520, 350),
            pygame.math.Vector2(self.SCREEN_WIDTH + 20, 350)
        ]
        self.base_pos = pygame.math.Vector2(self.SCREEN_WIDTH - 40, 350)

    def _define_placement_zones(self):
        self.placement_zones = []
        for y_offset in [120, 280]:
            for i in range(self.ZONES_PER_ROW):
                x = 100 + i * 90
                self.placement_zones.append(pygame.math.Vector2(x, y_offset))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = self.BASE_MAX_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.current_wave = 0
        self.wave_timer = self.FPS * 3 # 3 seconds until first wave
        self.wave_complete = True
        
        self.enemies = []
        self.turrets = [None] * len(self.placement_zones)
        self.projectiles = []
        self.particles = []
        
        self.selected_zone_idx = 0
        self.selected_turret_type_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reward_popups = deque(maxlen=10)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.001 # Small penalty for existing
        
        self._handle_input(action)
        
        if not self.game_over:
            self._update_waves()
            reward += self._update_turrets()
            reward += self._update_projectiles()
            reward += self._update_enemies()
            self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and self.win:
            reward += 100
        elif terminated and self.base_health <= 0:
            reward -= 100

        self.score += reward
        if abs(reward) > 0.01:
             # Add a popup for significant rewards
             popup_pos = self.base_pos + self.np_random.uniform(-30, 30, 2)
             self.reward_popups.append([f"{reward:+.1f}", popup_pos, 45])
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement: Select placement zone ---
        if movement != 0:
            row = self.selected_zone_idx // self.ZONES_PER_ROW
            col = self.selected_zone_idx % self.ZONES_PER_ROW
            if movement == 1: # Up
                row = max(0, row - 1)
            elif movement == 2: # Down
                row = min(1, row + 1)
            elif movement == 3: # Left
                col = max(0, col - 1)
            elif movement == 4: # Right
                col = min(self.ZONES_PER_ROW - 1, col + 1)
            self.selected_zone_idx = row * self.ZONES_PER_ROW + col

        # --- Shift: Cycle turret type (on press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_turret_type_idx = (self.selected_turret_type_idx + 1) % len(self.TURRET_SPECS)
            # SFX: UI_BEEP

        # --- Space: Place turret (on press) ---
        if space_held and not self.prev_space_held:
            self._place_turret()

        self.prev_space_held, self.prev_shift_held = space_held, shift_held

    def _place_turret(self):
        spec = self.TURRET_SPECS[self.selected_turret_type_idx]
        if self.turrets[self.selected_zone_idx] is None and self.resources >= spec["cost"]:
            self.resources -= spec["cost"]
            pos = self.placement_zones[self.selected_zone_idx]
            self.turrets[self.selected_zone_idx] = {
                "spec": spec,
                "pos": pos,
                "cooldown": 0,
                "target": None,
                "angle": 0
            }
            # SFX: BUILD_TURRET
            self._create_particles(pos, spec["color"], 20, 2, 4)

    def _update_waves(self):
        if self.wave_complete and self.current_wave < self.WAVE_COUNT:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                self.wave_complete = False
                self._spawn_wave()
    
    def _spawn_wave(self):
        num_enemies = 5 + (self.current_wave - 1) * 2
        health = 10 * (1.1 ** (self.current_wave - 1))
        speed = 1.0 * (1.05 ** (self.current_wave - 1))
        spawn_delay = int(self.FPS * 0.5)

        for i in range(num_enemies):
            self.enemies.append({
                "pos": self.path_points[0].copy(),
                "health": health,
                "max_health": health,
                "speed": speed,
                "path_idx": 1,
                "spawn_timer": i * spawn_delay,
                "slow_timer": 0
            })
        # SFX: WAVE_START

    def _update_turrets(self):
        reward = 0
        for turret in self.turrets:
            if turret is None:
                continue

            turret["cooldown"] = max(0, turret["cooldown"] - 1)
            
            # Find a new target if needed
            if turret["target"] is None or turret["target"] not in self.enemies or turret["pos"].distance_to(turret["target"]["pos"]) > turret["spec"]["range"]:
                turret["target"] = None
                in_range_enemies = [e for e in self.enemies if e["spawn_timer"] <= 0 and turret["pos"].distance_to(e["pos"]) <= turret["spec"]["range"]]
                if in_range_enemies:
                    turret["target"] = min(in_range_enemies, key=lambda e: turret["pos"].distance_to(e["pos"]))

            if turret["target"] and turret["cooldown"] == 0:
                # SFX: TURRET_FIRE
                spec = turret["spec"]
                turret["cooldown"] = spec["fire_rate"]
                target_pos = turret["target"]["pos"]
                
                # Aim at target
                dx = target_pos.x - turret["pos"].x
                dy = target_pos.y - turret["pos"].y
                turret["angle"] = math.degrees(math.atan2(-dy, dx))

                self._create_particles(turret["pos"], spec["color"], 5, 1, 2)
                
                self.projectiles.append({
                    "pos": turret["pos"].copy(),
                    "target": turret["target"],
                    "speed": 10,
                    "spec": spec
                })
        return reward

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target = proj["target"]
            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue

            direction = (target["pos"] - proj["pos"]).normalize()
            proj["pos"] += direction * proj["speed"]

            if proj["pos"].distance_to(target["pos"]) < 10:
                # SFX: PROJECTILE_HIT
                reward += self._damage_enemy(target, proj["spec"])
                self._create_particles(proj["pos"], proj["spec"]["color"], 10, 1, 3)
                self.projectiles.remove(proj)
        return reward
    
    def _damage_enemy(self, enemy, spec):
        reward = 0.1 # Reward for hitting
        enemy["health"] -= spec["damage"]
        if "slow" in spec:
            enemy["slow_timer"] = self.FPS # Slow for 1 second

        if enemy["health"] <= 0:
            # SFX: ENEMY_DESTROYED
            reward += 1.0 # Reward for kill
            self.resources += int(enemy["max_health"] / 5)
            self._create_particles(enemy["pos"], self.COLOR_ENEMY, 30, 3, 6)
            self.enemies.remove(enemy)
        return reward

    def _update_enemies(self):
        reward = 0
        if not self.enemies and self.current_wave > 0:
            self.wave_complete = True
            if self.current_wave == self.WAVE_COUNT:
                self.win = True

        for enemy in self.enemies[:]:
            if enemy["spawn_timer"] > 0:
                enemy["spawn_timer"] -= 1
                continue
            
            enemy["slow_timer"] = max(0, enemy["slow_timer"] - 1)
            current_speed = enemy["speed"] * (0.5 if enemy["slow_timer"] > 0 else 1.0)
            
            target_point = self.path_points[enemy["path_idx"]]
            direction = (target_point - enemy["pos"])
            
            if direction.length() < current_speed:
                enemy["pos"] = target_point.copy()
                enemy["path_idx"] += 1
                if enemy["path_idx"] >= len(self.path_points):
                    # SFX: BASE_DAMAGE
                    self.base_health -= 10
                    reward -= 10
                    self.enemies.remove(enemy)
                    self._create_particles(self.base_pos, self.COLOR_BASE_DMG, 40, 5, 8)
                    continue
            else:
                enemy["pos"] += direction.normalize() * current_speed
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, min_speed, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": self.np_random.integers(10, 20),
                "color": color
            })

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.win:
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
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, [tuple(p) for p in self.path_points], 30)

        # Placement Zones
        for i, zone_pos in enumerate(self.placement_zones):
            color = self.COLOR_ZONE_SELECT if i == self.selected_zone_idx else self.COLOR_ZONE_EMPTY
            is_filled = self.turrets[i] is not None
            pygame.gfxdraw.filled_circle(self.screen, int(zone_pos.x), int(zone_pos.y), 20, self.COLOR_PATH)
            pygame.gfxdraw.aacircle(self.screen, int(zone_pos.x), int(zone_pos.y), 20, color)
            if is_filled:
                 pygame.gfxdraw.filled_circle(self.screen, int(zone_pos.x), int(zone_pos.y), 5, color)


        # Turrets
        for turret in self.turrets:
            if turret:
                self._render_turret(turret)

        # Base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.base_pos.x - 15, self.base_pos.y - 15, 30, 30))

        # Enemies
        for enemy in self.enemies:
            if enemy["spawn_timer"] <= 0:
                self._render_enemy(enemy)

        # Projectiles
        for proj in self.projectiles:
            p1 = proj["pos"]
            p2 = proj["pos"] - (proj["target"]["pos"] - proj["pos"]).normalize() * 5
            pygame.draw.aaline(self.screen, proj["spec"]["color"], tuple(p1), tuple(p2), 2)

        # Particles
        for p in self.particles:
            size = max(1, int(p["lifespan"] / 5))
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y)), size)

    def _render_turret(self, turret):
        pos = turret["pos"]
        spec = turret["spec"]
        
        # Base
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 15, spec["color"])
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 15, (255, 255, 255))
        
        # Barrel
        angle_rad = math.radians(turret["angle"])
        end_x = pos.x + 20 * math.cos(angle_rad)
        end_y = pos.y - 20 * math.sin(angle_rad)
        pygame.draw.line(self.screen, (255, 255, 255), (pos.x, pos.y), (end_x, end_y), 4)

    def _render_enemy(self, enemy):
        pos = enemy["pos"]
        size = 10
        points = [
            (pos.x, pos.y - size),
            (pos.x - size / 2, pos.y + size / 2),
            (pos.x + size / 2, pos.y + size / 2)
        ]
        
        # Determine rotation from movement direction
        if enemy["path_idx"] < len(self.path_points):
            direction = self.path_points[enemy["path_idx"]] - pos
            if direction.length() > 0:
                angle = math.atan2(-direction.y, direction.x)
                # Rotate points
                center = pygame.math.Vector2(pos.x, pos.y)
                rotated_points = []
                for p_x, p_y in points:
                    v = pygame.math.Vector2(p_x, p_y) - center
                    v = v.rotate(math.degrees(-angle) - 90) # Adjust for triangle orientation
                    rotated_points.append(tuple(v + center))
                points = rotated_points

        color = (180, 220, 255) if enemy["slow_timer"] > 0 else self.COLOR_ENEMY
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Health bar
        if enemy["health"] < enemy["max_health"]:
            health_pct = enemy["health"] / enemy["max_health"]
            bar_width = 20
            pygame.draw.rect(self.screen, (50, 0, 0), (pos.x - bar_width/2, pos.y - 20, bar_width, 4))
            pygame.draw.rect(self.screen, (0, 255, 0), (pos.x - bar_width/2, pos.y - 20, bar_width * health_pct, 4))

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 1, pos[1] + 1))
            self.screen.blit(content, pos)

        # Top-left: Wave info
        wave_text = f"Wave: {self.current_wave}/{self.WAVE_COUNT}"
        if self.wave_complete and self.current_wave < self.WAVE_COUNT:
            wave_text += f" (Next in {self.wave_timer / self.FPS:.1f}s)"
        draw_text(wave_text, self.font_small, self.COLOR_TEXT, (10, 10))

        # Top-right: Resources
        draw_text(f"Resources: ${self.resources}", self.font_small, (255, 223, 0), (self.SCREEN_WIDTH - 150, 10))

        # Bottom-left: Turret selection
        spec = self.TURRET_SPECS[self.selected_turret_type_idx]
        draw_text("Selected Turret:", self.font_small, self.COLOR_TEXT, (10, self.SCREEN_HEIGHT - 60))
        draw_text(f"{spec['name']}", self.font_small, spec['color'], (10, self.SCREEN_HEIGHT - 40))
        draw_text(f"Cost: ${spec['cost']}", self.font_small, (255, 223, 0), (10, self.SCREEN_HEIGHT - 20))

        # Base Health Bar
        health_pct = self.base_health / self.BASE_MAX_HEALTH
        bar_width, bar_height = 100, 10
        bar_pos = (self.base_pos.x - bar_width / 2, self.base_pos.y - 35)
        pygame.draw.rect(self.screen, (50, 0, 0), (*bar_pos, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (*bar_pos, bar_width * health_pct, bar_height))
        
        # Reward Popups
        for popup in self.reward_popups:
            popup[2] -= 1
            if popup[2] > 0:
                alpha = min(255, popup[2] * 10)
                text_surf = self.font_reward.render(popup[0], True, self.COLOR_TEXT)
                text_surf.set_alpha(alpha)
                pos = popup[1] - pygame.math.Vector2(0, 45 - popup[2])
                self.screen.blit(text_surf, (int(pos.x), int(pos.y)))

        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            text = self.font_large.render(msg, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "current_wave": self.current_wave,
            "enemies_left": len(self.enemies),
            "game_over": self.game_over,
            "win": self.win
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Player Controls ---
    # Map keyboard keys to the MultiDiscrete action space
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    total_reward = 0
    total_steps = 0

    print("--- Playing Game ---")
    print(env.user_guide)

    while not done:
        # Construct the action based on keyboard state
        movement_action = 0
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        for key, move in key_to_action.items():
            if keys[key]:
                movement_action = move
                break # only one movement at a time
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_steps += 1

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    print("\n--- Game Over ---")
    print(f"Final Score: {info['score']:.2f}")
    print(f"Total Steps: {info['steps']}")
    print(f"Result: {'VICTORY!' if info['win'] else 'DEFEAT'}")

    env.close()