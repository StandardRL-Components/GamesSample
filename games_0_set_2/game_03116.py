
# Generated: 2025-08-27T22:25:16.864496
# Source Brief: brief_03116.md
# Brief Index: 3116

        
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
    """
    A tower defense game where the player places turrets to defend a base from waves of zombies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press space to place the selected turret. Hold shift to cycle turret types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base against waves of increasingly difficult zombies by strategically placing turrets."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 6000  # 200 seconds at 30fps
    MAX_WAVES = 20

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_BASE = (60, 179, 113)
    COLOR_BASE_HP = (90, 220, 140)
    COLOR_ZONE = (50, 60, 70)
    COLOR_ZONE_INVALID = (100, 50, 50)
    COLOR_ZONE_VALID = (50, 100, 50)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_ZOMBIE = (220, 60, 60)
    COLOR_ZOMBIE_HP = (255, 90, 90)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BG = (40, 45, 55, 180)

    TURRET_TYPES = [
        {
            "name": "Gun", "cost": 100, "range": 100, "damage": 5, "fire_rate": 10,
            "color": (0, 150, 255), "proj_speed": 8, "proj_size": 2
        },
        {
            "name": "Cannon", "cost": 250, "range": 150, "damage": 25, "fire_rate": 60,
            "color": (0, 80, 200), "proj_speed": 5, "proj_size": 4, "splash": 25
        },
        {
            "name": "Slow", "cost": 150, "range": 80, "damage": 0, "fire_rate": 30,
            "color": (100, 200, 255), "slow_factor": 0.5, "slow_duration": 60
        }
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 20, bold=True)
        self.font_wave = pygame.font.SysFont("sans-serif", 48, bold=True)

        # Game state is initialized in reset()
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.money = 0
        self.game_over = False
        self.base_pos = pygame.math.Vector2(self.SCREEN_WIDTH - 40, self.SCREEN_HEIGHT / 2)
        self.base_health = 0
        self.max_base_health = 0
        self.wave_number = 0
        self.wave_in_progress = False
        self.inter_wave_timer = 0
        self.zombie_spawn_queue = []
        self.zombie_spawn_timer = 0
        self.zombies = []
        self.turrets = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = pygame.math.Vector2(0, 0)
        self.selected_turret_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.placement_zones = []

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
        self.money = 400
        self.game_over = False

        self.base_health = 1000
        self.max_base_health = 1000

        self.wave_number = 0
        self.wave_in_progress = False
        self.inter_wave_timer = self.FPS * 5  # 5 seconds before first wave

        self.zombie_spawn_queue = []
        self.zombie_spawn_timer = 0

        self.zombies = []
        self.turrets = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.selected_turret_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.placement_zones = [
            {"pos": pygame.math.Vector2(150, 100), "radius": 40, "turret": None},
            {"pos": pygame.math.Vector2(150, 300), "radius": 40, "turret": None},
            {"pos": pygame.math.Vector2(320, 200), "radius": 40, "turret": None},
            {"pos": pygame.math.Vector2(450, 100), "radius": 40, "turret": None},
            {"pos": pygame.math.Vector2(450, 300), "radius": 40, "turret": None},
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # --- Handle Actions ---
        self._handle_actions(action)

        # --- Update Game Logic ---
        self._update_wave_system()
        zombie_kills, base_damage = self._update_zombies()
        self._update_turrets()
        projectile_hits = self._update_projectiles()
        self._update_particles()

        # --- Calculate Reward ---
        reward -= 0.001  # Small penalty for time to encourage efficiency
        reward += zombie_kills * 0.1
        self.money += zombie_kills * 25
        self.score += zombie_kills * 10
        self.base_health -= base_damage * 50
        
        if self.wave_cleared_this_frame:
            reward += 1.0
            self.score += 100 * self.wave_number
            self.money += 100 + 50 * self.wave_number

        # --- Check Termination ---
        self.steps += 1
        if self.base_health <= 0:
            terminated = True
            reward -= 100
            self.score -= 1000
        elif self.wave_number > self.MAX_WAVES:
            terminated = True
            reward += 100
            self.score += 5000
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Timeout
        
        self.game_over = terminated

        return (
            self._get_observation(),
            np.clip(reward, -100, 100),
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_val, shift_val = action
        space_held = space_val == 1
        shift_held = shift_val == 1

        # Cursor movement
        cursor_speed = 10
        if movement == 1: self.cursor_pos.y -= cursor_speed
        elif movement == 2: self.cursor_pos.y += cursor_speed
        elif movement == 3: self.cursor_pos.x -= cursor_speed
        elif movement == 4: self.cursor_pos.x += cursor_speed
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

        # Cycle turret type (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_turret_type = (self.selected_turret_type + 1) % len(self.TURRET_TYPES)
            # sfx: menu_cycle

        # Place turret (on press)
        if space_held and not self.prev_space_held:
            turret_info = self.TURRET_TYPES[self.selected_turret_type]
            if self.money >= turret_info["cost"]:
                for zone in self.placement_zones:
                    if zone["turret"] is None and self.cursor_pos.distance_to(zone["pos"]) < zone["radius"]:
                        new_turret = {
                            "pos": zone["pos"].copy(),
                            "type_idx": self.selected_turret_type,
                            "cooldown": 0,
                            "target": None
                        }
                        self.turrets.append(new_turret)
                        zone["turret"] = new_turret
                        self.money -= turret_info["cost"]
                        # sfx: place_turret
                        break

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
    def _update_wave_system(self):
        self.wave_cleared_this_frame = False
        if self.wave_in_progress:
            # Spawn zombies from the queue
            if self.zombie_spawn_queue and self.zombie_spawn_timer <= 0:
                self.zombies.append(self.zombie_spawn_queue.pop(0))
                self.zombie_spawn_timer = self.FPS // 2 # 0.5s between spawns
            self.zombie_spawn_timer -= 1

            # Check for wave clear
            if not self.zombies and not self.zombie_spawn_queue:
                self.wave_in_progress = False
                self.inter_wave_timer = self.FPS * 8 # 8s between waves
                self.wave_cleared_this_frame = True
                # sfx: wave_cleared
        else: # Inter-wave period
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0 and self.wave_number <= self.MAX_WAVES:
                self.wave_number += 1
                self._spawn_wave()

    def _spawn_wave(self):
        self.wave_in_progress = True
        num_zombies = 5 + self.wave_number * 2
        base_health = 20 + self.wave_number * 10
        base_speed = 0.5 + self.wave_number * 0.1

        for _ in range(num_zombies):
            spawn_y = self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
            health = base_health + self.np_random.integers(-5, 5)
            speed = base_speed * self.np_random.uniform(0.8, 1.2)
            self.zombie_spawn_queue.append({
                "pos": pygame.math.Vector2(-20, spawn_y),
                "health": health, "max_health": health,
                "speed": speed, "slow_factor": 1.0, "slow_timer": 0
            })
        
    def _update_zombies(self):
        kills = 0
        base_damage = 0
        for z in self.zombies[:]:
            # Update slow effect
            if z["slow_timer"] > 0:
                z["slow_timer"] -= 1
            else:
                z["slow_factor"] = 1.0
            
            # Move towards base
            direction = (self.base_pos - z["pos"]).normalize()
            z["pos"] += direction * z["speed"] * z["slow_factor"]

            # Check collision with base
            if z["pos"].distance_to(self.base_pos) < 20:
                base_damage += 1
                self.zombies.remove(z)
                # sfx: base_hit
                self._create_particles(self.base_pos, self.COLOR_ZOMBIE, 20)
            
            # Check for death
            if z["health"] <= 0:
                kills += 1
                self.zombies.remove(z)
                # sfx: zombie_death
                self._create_particles(z["pos"], self.COLOR_ZOMBIE, 15)
        return kills, base_damage

    def _update_turrets(self):
        for t in self.turrets:
            t["cooldown"] -= 1
            turret_info = self.TURRET_TYPES[t["type_idx"]]
            
            # Find target
            target = None
            min_dist = turret_info["range"]
            for z in self.zombies:
                dist = t["pos"].distance_to(z["pos"])
                if dist < min_dist:
                    min_dist = dist
                    target = z
            
            # Fire
            if t["cooldown"] <= 0 and target:
                t["cooldown"] = turret_info["fire_rate"]
                # sfx: turret_fire
                self._create_projectile(t, target)

    def _create_projectile(self, turret, target):
        turret_info = self.TURRET_TYPES[turret["type_idx"]]
        start_pos = turret["pos"].copy()
        
        if turret_info["damage"] > 0: # Gun or Cannon
            direction = (target["pos"] - start_pos).normalize()
            self.projectiles.append({
                "pos": start_pos, "vel": direction * turret_info["proj_speed"],
                "type_idx": turret["type_idx"], "life": 120
            })
        else: # Slow tower pulse effect
            self._create_particles(start_pos, turret_info["color"], 20, count=1, p_type='pulse',
                                   data={"range": turret_info["range"], "duration": 10})
            for z in self.zombies:
                if z["pos"].distance_to(start_pos) <= turret_info["range"]:
                    z["slow_factor"] = turret_info["slow_factor"]
                    z["slow_timer"] = turret_info["slow_duration"]

    def _update_projectiles(self):
        hits = 0
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            
            if p["life"] <= 0:
                self.projectiles.remove(p)
                continue

            proj_info = self.TURRET_TYPES[p["type_idx"]]
            
            for z in self.zombies:
                if p["pos"].distance_to(z["pos"]) < 10: # Hitbox size
                    # sfx: projectile_hit
                    if "splash" in proj_info:
                        self._create_particles(p["pos"], proj_info["color"], 10, count=1, p_type='pulse',
                                               data={"range": proj_info["splash"], "duration": 5})
                        for splash_z in self.zombies:
                            if splash_z["pos"].distance_to(p["pos"]) < proj_info["splash"]:
                                splash_z["health"] -= proj_info["damage"]
                    else:
                        z["health"] -= proj_info["damage"]
                        self._create_particles(p["pos"], proj_info["color"], 5)

                    hits += 1
                    self.projectiles.remove(p)
                    break
        return hits

    def _create_particles(self, pos, color, size, count=10, p_type='burst', data=None):
        for _ in range(count):
            if p_type == 'burst':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                life = self.np_random.integers(10, 20)
                self.particles.append({"pos": pos.copy(), "vel": vel, "life": life, "max_life": life, "color": color, "type": "spark"})
            elif p_type == 'pulse':
                self.particles.append({"pos": pos.copy(), "life": data["duration"], "max_life": data["duration"], "color": color, "type": "pulse", "max_radius": data["range"]})

    def _update_particles(self):
        for part in self.particles[:]:
            part["life"] -= 1
            if part["life"] <= 0:
                self.particles.remove(part)
                continue
            if part["type"] == "spark":
                part["pos"] += part["vel"]
                part["vel"] *= 0.9 # friction
    
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
            "money": self.money,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "zombies": len(self.zombies),
        }

    def _render_game(self):
        # Draw placement zones
        for zone in self.placement_zones:
            pygame.gfxdraw.aacircle(self.screen, int(zone["pos"].x), int(zone["pos"].y), zone["radius"], self.COLOR_ZONE)

        # Draw Base
        base_rect = pygame.Rect(self.base_pos.x - 20, self.base_pos.y - 20, 40, 40)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        
        # Draw Zombies
        for z in self.zombies:
            pos_x, pos_y = int(z["pos"].x), int(z["pos"].y)
            color = (255, 255, 100) if z["slow_timer"] > 0 else self.COLOR_ZOMBIE
            pygame.draw.rect(self.screen, color, (pos_x - 5, pos_y - 8, 10, 16), border_radius=2)
            # Health bar
            hp_ratio = z["health"] / z["max_health"]
            pygame.draw.rect(self.screen, (80,0,0), (pos_x - 8, pos_y - 15, 16, 4))
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_HP, (pos_x - 8, pos_y - 15, max(0, 16 * hp_ratio), 4))
            
        # Draw Turrets
        for t in self.turrets:
            info = self.TURRET_TYPES[t["type_idx"]]
            pos_x, pos_y = int(t["pos"].x), int(t["pos"].y)
            if info["name"] == "Gun":
                pygame.draw.circle(self.screen, info["color"], (pos_x, pos_y), 10)
            elif info["name"] == "Cannon":
                pygame.draw.rect(self.screen, info["color"], (pos_x - 8, pos_y - 8, 16, 16), border_radius=3)
            elif info["name"] == "Slow":
                 pygame.gfxdraw.filled_trigon(self.screen, pos_x, pos_y - 9, pos_x-8, pos_y+6, pos_x+8, pos_y+6, info["color"])

        # Draw Projectiles
        for p in self.projectiles:
            info = self.TURRET_TYPES[p["type_idx"]]
            pygame.draw.circle(self.screen, info["color"], (int(p["pos"].x), int(p["pos"].y)), info["proj_size"])

        # Draw Particles
        for part in self.particles:
            alpha = int(255 * (part["life"] / part["max_life"]))
            color = (*part["color"], alpha)
            if part["type"] == "spark":
                pygame.draw.circle(self.screen, part["color"], (int(part["pos"].x), int(part["pos"].y)), int(part["life"] * 0.2))
            elif part["type"] == "pulse":
                radius = int(part["max_radius"] * (1 - part["life"] / part["max_life"]))
                pygame.gfxdraw.aacircle(self.screen, int(part["pos"].x), int(part["pos"].y), radius, color)

        # Draw Cursor and turret preview
        self._render_cursor()

        # Draw inter-wave text
        if not self.wave_in_progress and self.wave_number <= self.MAX_WAVES and not self.game_over:
            text = f"Wave {self.wave_number + 1} starting soon..."
            if self.wave_number == 0:
                text = "First wave starting soon..."
            
            wave_surf = self.font_wave.render(text, True, self.COLOR_UI_TEXT)
            wave_rect = wave_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            alpha = min(255, int(255 * (self.inter_wave_timer / (self.FPS * 2))))
            wave_surf.set_alpha(alpha)
            self.screen.blit(wave_surf, wave_rect)


    def _render_cursor(self):
        cursor_x, cursor_y = int(self.cursor_pos.x), int(self.cursor_pos.y)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x - 8, cursor_y), (cursor_x + 8, cursor_y), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y - 8), (cursor_x, cursor_y + 8), 1)

        # Preview placement
        turret_info = self.TURRET_TYPES[self.selected_turret_type]
        can_place = False
        is_on_zone = False
        for zone in self.placement_zones:
            if self.cursor_pos.distance_to(zone["pos"]) < zone["radius"]:
                is_on_zone = True
                if zone["turret"] is None and self.money >= turret_info["cost"]:
                    can_place = True
                break
        
        preview_color = (*self.COLOR_ZONE_VALID, 100) if can_place else (*self.COLOR_ZONE_INVALID, 100)
        if is_on_zone:
            pygame.gfxdraw.filled_circle(self.screen, cursor_x, cursor_y, 15, preview_color)
        
        pygame.gfxdraw.aacircle(self.screen, cursor_x, cursor_y, turret_info["range"], (*turret_info["color"], 50))

    def _render_ui(self):
        # UI Panel
        panel_surf = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        panel_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(panel_surf, (0, 0))

        # Base Health
        health_text = self.font_ui.render(f"Base HP: {int(self.base_health)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 15))
        pygame.draw.rect(self.screen, (80,0,0), (110, 18, 150, 15))
        hp_ratio = self.base_health / self.max_base_health
        pygame.draw.rect(self.screen, self.COLOR_BASE_HP, (110, 18, max(0, 150 * hp_ratio), 15))

        # Money and Score
        money_text = self.font_ui.render(f"$ {self.money}", True, (255, 223, 0))
        self.screen.blit(money_text, (280, 15))
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (380, 15))

        # Wave
        wave_text = self.font_ui.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (520, 15))
        
        # Selected Turret UI
        turret_info = self.TURRET_TYPES[self.selected_turret_type]
        cost_color = (255, 223, 0) if self.money >= turret_info["cost"] else (255, 80, 80)
        turret_name = self.font_ui.render(f"Build: {turret_info['name']}", True, self.COLOR_UI_TEXT)
        turret_cost = self.font_ui.render(f"Cost: {turret_info['cost']}", True, cost_color)
        self.screen.blit(turret_name, (self.cursor_pos.x + 15, self.cursor_pos.y - 25))
        self.screen.blit(turret_cost, (self.cursor_pos.x + 15, self.cursor_pos.y - 5))

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Player Input ---
        movement = 0 # none
        space = 0    # released
        shift = 0    # released

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            obs, info = env.reset()

        # --- Render ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()