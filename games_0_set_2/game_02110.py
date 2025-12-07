
# Generated: 2025-08-27T19:18:17.299156
# Source Brief: brief_02110.md
# Brief Index: 2110

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. SHIFT to cycle tower type. "
        "SPACE to place tower. Press SPACE away from a build site to start the wave."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from enemy waves by placing towers. "
        "Manage resources and choose tower types strategically to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 20000
    STARTING_BASE_HEALTH = 100
    STARTING_MONEY = 250
    WAVE_BONUS_MONEY = 100
    MAX_WAVES = 10

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PATH = (40, 50, 80)
    COLOR_BASE = (0, 200, 100)
    COLOR_ZONE = (60, 70, 100)
    COLOR_ZONE_HIGHLIGHT = (100, 120, 180)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    TOWER_SPECS = {
        0: {"name": "Gatling", "cost": 75, "range": 80, "damage": 2, "fire_rate": 5, "color": (0, 150, 255)},
        1: {"name": "Cannon", "cost": 125, "range": 120, "damage": 20, "fire_rate": 1, "color": (255, 150, 0)},
        2: {"name": "Sniper", "cost": 200, "range": 250, "damage": 50, "fire_rate": 0.5, "color": (200, 200, 200)},
        3: {"name": "Slower", "cost": 100, "range": 100, "damage": 0, "fire_rate": 2, "color": (150, 50, 255), "slow": 0.5}
    }

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
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.np_random = None
        
        # State variables are initialized in reset()
        self.reset()
        
        # self.validate_implementation() # Optional: Call for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.game_phase = "PREP" # "PREP" or "WAVE"
        self.wave_number = 0
        self.base_health = self.STARTING_BASE_HEALTH
        self.money = self.STARTING_MONEY

        self.path = self._generate_path()
        self.tower_zones = self._generate_tower_zones()
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.selected_tower_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._start_new_prep_phase()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = -0.01 # Small penalty for existing
        
        self._process_inputs(action)

        if self.game_phase == "WAVE":
            self._update_wave_logic()

        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _process_inputs(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos.y -= 5 # Up
        elif movement == 2: self.cursor_pos.y += 5 # Down
        elif movement == 3: self.cursor_pos.x -= 5 # Left
        elif movement == 4: self.cursor_pos.x += 5 # Right
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

        # --- Edge-triggered actions ---
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        if self.game_phase == "PREP":
            if shift_press:
                # sfx: ui_cycle.wav
                self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            
            if space_press:
                zone_idx = self._get_zone_at_cursor()
                if zone_idx is not None:
                    self._try_place_tower(zone_idx)
                else:
                    # sfx: wave_start.wav
                    self._start_wave()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_wave_logic(self):
        # --- Towers act ---
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] == 0:
                target = self._find_target(tower)
                if target:
                    self._fire_projectile(tower, target)
                    tower["cooldown"] = 60 / tower["spec"]["fire_rate"]

        # --- Projectiles move and hit ---
        for proj in self.projectiles[:]:
            proj["pos"] += proj["vel"]
            if proj["pos"].distance_to(proj["target_pos"]) < proj["speed"] or not self._is_on_screen(proj["pos"]):
                self._handle_projectile_impact(proj)
                self.projectiles.remove(proj)
            
        # --- Enemies move ---
        for enemy in self.enemies[:]:
            enemy["slow_timer"] = max(0, enemy["slow_timer"] - 1)
            current_speed = enemy["base_speed"] * (enemy["spec"]["slow"] if enemy["slow_timer"] > 0 else 1)

            if enemy["path_idx"] < len(self.path) - 1:
                target_pos = self.path[enemy["path_idx"] + 1]
                direction = (target_pos - enemy["pos"]).normalize()
                enemy["pos"] += direction * current_speed
                if enemy["pos"].distance_to(target_pos) < current_speed:
                    enemy["path_idx"] += 1
                    enemy["pos"] = pygame.Vector2(target_pos)
            else: # Reached base
                # sfx: base_damage.wav
                self.base_health -= enemy["health"]
                self.reward_this_step -= 10 * enemy["health"]
                self._create_explosion(enemy["pos"], self.COLOR_BASE, 20)
                self.enemies.remove(enemy)

        # --- Check for wave end ---
        if not self.enemies and self.game_phase == "WAVE":
            self._start_new_prep_phase()

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
            "wave": self.wave_number,
            "money": self.money,
            "base_health": self.base_health,
            "game_phase": self.game_phase
        }

    # --- Game Logic Helpers ---

    def _generate_path(self):
        path = []
        y = self.np_random.integers(100, self.SCREEN_HEIGHT - 100)
        path.append(pygame.Vector2(0, y))
        for i in range(1, 5):
            x = int(i * self.SCREEN_WIDTH / 5)
            y = self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
            path.append(pygame.Vector2(x, y))
        path.append(pygame.Vector2(self.SCREEN_WIDTH, y))
        return path

    def _generate_tower_zones(self):
        zones = []
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            midpoint = p1.lerp(p2, 0.5)
            perp = (p2 - p1).rotate(90).normalize()
            dist = self.np_random.uniform(40, 60) * (1 if self.np_random.random() > 0.5 else -1)
            zone_pos = midpoint + perp * dist
            zones.append({"pos": zone_pos, "tower": None})
        return zones

    def _get_zone_at_cursor(self):
        for i, zone in enumerate(self.tower_zones):
            if self.cursor_pos.distance_to(zone["pos"]) < 20 and zone["tower"] is None:
                return i
        return None

    def _try_place_tower(self, zone_idx):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.money >= spec["cost"]:
            # sfx: place_tower.wav
            self.money -= spec["cost"]
            tower = {
                "pos": self.tower_zones[zone_idx]["pos"],
                "type": self.selected_tower_type,
                "spec": spec,
                "cooldown": 0,
            }
            self.towers.append(tower)
            self.tower_zones[zone_idx]["tower"] = tower
        else:
            # sfx: error.wav
            pass # Not enough money

    def _start_new_prep_phase(self):
        if self.wave_number > 0:
            self.money += self.WAVE_BONUS_MONEY
        
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            self.game_over = True # Win condition
            self.reward_this_step += 100
            return

        self.game_phase = "PREP"

    def _start_wave(self):
        if self.game_phase == "PREP":
            self.game_phase = "WAVE"
            self._spawn_wave()

    def _spawn_wave(self):
        num_enemies = 3 + self.wave_number * 2
        base_health = 10 + self.wave_number * 5
        base_speed = 0.8 + self.wave_number * 0.05
        
        for i in range(num_enemies):
            offset = self.np_random.uniform(-10, 10)
            spawn_pos = pygame.Vector2(self.path[0].x - (i * 25), self.path[0].y + offset)
            self.enemies.append({
                "pos": spawn_pos,
                "health": base_health,
                "max_health": base_health,
                "base_speed": base_speed,
                "path_idx": 0,
                "slow_timer": 0,
                "spec": self.TOWER_SPECS[3] # for slow effect reference
            })

    def _find_target(self, tower):
        for enemy in self.enemies:
            if tower["pos"].distance_to(enemy["pos"]) <= tower["spec"]["range"]:
                return enemy
        return None

    def _fire_projectile(self, tower, target):
        # sfx: fire_gatling.wav or fire_cannon.wav
        proj_speed = 10
        direction = (target["pos"] - tower["pos"]).normalize()
        self.projectiles.append({
            "pos": pygame.Vector2(tower["pos"]),
            "vel": direction * proj_speed,
            "speed": proj_speed,
            "target_pos": pygame.Vector2(target["pos"]),
            "damage": tower["spec"]["damage"],
            "color": tower["spec"]["color"],
            "tower_type": tower["type"]
        })
        # Muzzle flash
        self.particles.append({"pos": tower["pos"] + direction * 15, "radius": 8, "color": (255, 255, 100), "life": 3})

    def _handle_projectile_impact(self, proj):
        # Find nearest enemy to impact point
        impact_radius = 10
        if proj["tower_type"] == 1: # Cannon has splash
            impact_radius = 40
        
        hit_enemies = [e for e in self.enemies if e["pos"].distance_to(proj["pos"]) < impact_radius]

        if not hit_enemies: return

        # sfx: hit_enemy.wav
        self._create_explosion(proj["pos"], proj["color"], 10)

        for enemy in hit_enemies:
            damage = proj["damage"]
            enemy["health"] -= damage
            self.reward_this_step += 0.1 # Reward for hit
            
            if proj["tower_type"] == 3: # Slower tower
                enemy["slow_timer"] = 120 # 2 seconds at 60fps

            if enemy["health"] <= 0:
                # sfx: enemy_die.wav
                self.score += 10
                self.money += 5
                self.reward_this_step += 1
                self._create_explosion(enemy["pos"], self.COLOR_ENEMY, 30)
                if enemy in self.enemies:
                    self.enemies.remove(enemy)

    def _is_on_screen(self, pos):
        return 0 <= pos.x <= self.SCREEN_WIDTH and 0 <= pos.y <= self.SCREEN_HEIGHT

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            self.reward_this_step -= 100
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
        
    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = self.np_random.uniform(2, 5)
            life = self.np_random.integers(15, 30)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "radius": radius, "color": color, "life": life})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] *= 0.95
            if p["life"] <= 0 or p["radius"] < 0.5:
                self.particles.remove(p)

    # --- Rendering ---

    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, [(int(p.x), int(p.y)) for p in self.path], 30)
        
        # Base
        base_rect = pygame.Rect(self.SCREEN_WIDTH - 20, self.path[-1].y - 20, 20, 40)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Tower Zones
        cursor_on_zone_idx = self._get_zone_at_cursor()
        for i, zone in enumerate(self.tower_zones):
            color = self.COLOR_ZONE_HIGHLIGHT if i == cursor_on_zone_idx else self.COLOR_ZONE
            if zone["tower"] is None:
                pygame.gfxdraw.aacircle(self.screen, int(zone["pos"].x), int(zone["pos"].y), 20, color)

        # Towers
        for tower in self.towers:
            pos = (int(tower["pos"].x), int(tower["pos"].y))
            spec = tower["spec"]
            pygame.draw.circle(self.screen, spec["color"], pos, 12)
            pygame.draw.circle(self.screen, self.COLOR_BG, pos, 8)
        
        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            pygame.gfxdraw.filled_trigon(self.screen, pos[0], pos[1]-8, pos[0]-7, pos[1]+6, pos[0]+7, pos[1]+6, self.COLOR_ENEMY)
            # Health bar
            if enemy["health"] < enemy["max_health"]:
                bar_w = 14
                ratio = enemy["health"] / enemy["max_health"]
                pygame.draw.rect(self.screen, (255,0,0), (pos[0]-bar_w/2, pos[1]-15, bar_w, 3))
                pygame.draw.rect(self.screen, (0,255,0), (pos[0]-bar_w/2, pos[1]-15, bar_w * ratio, 3))

        # Projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"].x), int(proj["pos"].y))
            pygame.draw.circle(self.screen, proj["color"], pos, 4)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, proj["color"])

        # Particles
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            radius = int(p["radius"])
            if radius > 0:
                pygame.draw.circle(self.screen, p["color"], pos, radius)

        # Cursor
        if self.game_phase == "PREP":
            pos = (int(self.cursor_pos.x), int(self.cursor_pos.y))
            color = self.TOWER_SPECS[self.selected_tower_type]["color"]
            pygame.draw.circle(self.screen, color, pos, 5)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, color)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color, center=None):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            text_rect = text_surf.get_rect()
            if center == "x": text_rect.centerx = pos[0]
            elif center == "y": text_rect.centery = pos[1]
            else: text_rect.topleft = pos
            
            self.screen.blit(shadow_surf, (text_rect.x+2, text_rect.y+2))
            self.screen.blit(text_surf, text_rect)

        # Top-left: Wave info
        draw_text(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", self.font_medium, self.COLOR_TEXT, (15, 10), self.COLOR_TEXT_SHADOW)
        
        # Top-right: Health and Money
        draw_text(f"HP: {max(0, self.base_health)}/{self.STARTING_BASE_HEALTH}", self.font_medium, self.COLOR_BASE, (self.SCREEN_WIDTH - 150, 10), self.COLOR_TEXT_SHADOW)
        draw_text(f"$ {self.money}", self.font_medium, self.COLOR_CURSOR, (self.SCREEN_WIDTH - 150, 35), self.COLOR_TEXT_SHADOW)

        # Bottom-left: Selected tower info (in PREP phase)
        if self.game_phase == "PREP":
            spec = self.TOWER_SPECS[self.selected_tower_type]
            draw_text("TOWER:", self.font_small, self.COLOR_TEXT, (15, self.SCREEN_HEIGHT - 70), self.COLOR_TEXT_SHADOW)
            draw_text(f"{spec['name']}", self.font_medium, spec['color'], (15, self.SCREEN_HEIGHT - 55), self.COLOR_TEXT_SHADOW)
            draw_text(f"Cost: ${spec['cost']}", self.font_small, self.COLOR_TEXT, (15, self.SCREEN_HEIGHT - 30), self.COLOR_TEXT_SHADOW)
            
            if self.game_over:
                msg = "YOU WIN!" if self.wave_number > self.MAX_WAVES else "GAME OVER"
                draw_text(msg, self.font_large, self.COLOR_CURSOR, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 30), self.COLOR_TEXT_SHADOW, center="x")
                draw_text(f"Final Score: {self.score}", self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20), self.COLOR_TEXT_SHADOW, center="x")
            else:
                 draw_text("Place towers or press SPACE in empty area to start wave", self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT-20), self.COLOR_TEXT_SHADOW, center="x")


    def validate_implementation(self):
        print("Running implementation validation...")
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
    
    # --- To play manually ---
    # To run, you need pygame installed and a windowing system.
    # If running on a headless server, this will fail.
    try:
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Tower Defense")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        print("\n" + "="*30)
        print(env.game_description)
        print(env.user_guide)
        print("="*30 + "\n")

        while not done:
            # --- Action mapping for human play ---
            keys = pygame.key.get_pressed()
            mov = 0
            if keys[pygame.K_UP]: mov = 1
            elif keys[pygame.K_DOWN]: mov = 2
            elif keys[pygame.K_LEFT]: mov = 3
            elif keys[pygame.K_RIGHT]: mov = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [mov, space, shift]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            clock.tick(30) # Limit to 30 FPS

        print(f"Game Over! Final Info: {info}")
        pygame.quit()

    except pygame.error as e:
        print(f"Pygame display could not be initialized: {e}")
        print("This is expected on a headless server. The environment is still valid.")
        # You can still test the environment logic without rendering
        env.validate_implementation()
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished after {_} steps. Info: {info}")
                obs, info = env.reset()