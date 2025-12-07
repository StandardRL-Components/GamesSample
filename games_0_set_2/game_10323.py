import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:18:32.265398
# Source Brief: brief_00323.md
# Brief Index: 323
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Maneuver your ship in a vector-based arena, shooting down incoming enemies. "
        "Deploy clones to distract foes and switch between weapons to survive the onslaught."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to move. Tap space to fire your weapon, "
        "or hold space to deploy a decoy clone. Press shift to cycle between weapons."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    PLAYER_SPEED = 5
    PLAYER_SIZE = 15
    PLAYER_HEALTH_MAX = 100
    
    ENEMY_SPEED = 2
    ENEMY_SIZE = 12
    ENEMY_HEALTH_MAX = 1
    
    CLONE_MAX_ACTIVE = 3
    CLONE_LIFETIME = 150 # 5 seconds at 30 FPS
    CLONE_RADIUS = 100
    CLONE_ATTRACTION_FORCE = 0.3
    
    SPACE_HOLD_THRESHOLD = 5 # frames to distinguish tap from hold

    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_CLONE = (100, 150, 255)
    COLOR_CLONE_FIELD = (100, 150, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BAR = (40, 40, 60)
    COLOR_HEALTH_BAR = (0, 200, 100)
    COLOR_CHARGE_BAR = (255, 180, 0)
    
    WEAPON_SPECS = {
        0: {"name": "Blaster", "cooldown": 8, "speed": 10, "color": (255, 255, 0), "size": 3, "damage": 1},
        1: {"name": "Pulse", "cooldown": 20, "speed": 6, "color": (0, 200, 255), "size": 8, "damage": 3},
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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_health = None
        self.enemies = None
        self.clones = None
        self.player_projectiles = None
        self.enemy_projectiles = None
        self.particles = None
        self.stars = None
        
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.current_weapon = None
        self.weapon_cooldown = None
        self.clone_cooldown = None
        
        self.last_space_held = None
        self.last_shift_held = None
        self.space_press_duration = None
        
        self.screen_flash_timer = None
        self.screen_flash_color = None

        self.initial_enemy_spawn_steps = 100
        self.enemy_spawn_timer = None
        self.enemy_projectile_speed_bonus = None

        # The reset call is not strictly necessary in __init__ for Gymnasium API,
        # but it's kept here to ensure all state variables are initialized
        # before any potential internal method calls.
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_health = self.PLAYER_HEALTH_MAX
        
        self.enemies = []
        self.clones = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.current_weapon = 0
        self.weapon_cooldown = 0
        self.clone_cooldown = 0
        
        self.last_space_held = 0
        self.last_shift_held = 0
        self.space_press_duration = 0
        
        self.screen_flash_timer = 0
        
        self.enemy_spawn_timer = self.initial_enemy_spawn_steps
        self.enemy_projectile_speed_bonus = 0

        if not hasattr(self, 'stars') or self.stars is None:
            self.stars = [
                {
                    "pos": pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                    "speed": self.np_random.uniform(0.1, 0.5),
                    "size": self.np_random.uniform(1, 2.5),
                    "color": self.np_random.integers(50, 101)
                } for _ in range(150)
            ]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        self._handle_input(action)
        self._update_player()
        self._update_enemies()
        self._update_clones()
        self._update_projectiles()
        self._update_particles()
        
        collision_rewards = self._handle_collisions()
        reward += collision_rewards
        
        self.steps += 1
        self.weapon_cooldown = max(0, self.weapon_cooldown - 1)
        self.clone_cooldown = max(0, self.clone_cooldown - 1)
        self.screen_flash_timer = max(0, self.screen_flash_timer - 1)

        # Difficulty scaling
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self._spawn_enemy()
            spawn_period = self.initial_enemy_spawn_steps * max(0.1, 1 - self.steps * 0.0001)
            self.enemy_spawn_timer = int(spawn_period)
        
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_projectile_speed_bonus += 0.1

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and self.player_health <= 0:
            reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        if move_vec.length() > 0:
            self.player_pos += move_vec.normalize() * self.PLAYER_SPEED

        # --- Weapon Cycle (Shift) ---
        if shift_held and not self.last_shift_held:
            self.current_weapon = (self.current_weapon + 1) % len(self.WEAPON_SPECS)
            # SFX: Weapon cycle
        self.last_shift_held = shift_held

        # --- Fire / Deploy Clone (Space) ---
        if space_held:
            self.space_press_duration += 1
        
        if not space_held and self.last_space_held: # On release
            if self.space_press_duration <= self.SPACE_HOLD_THRESHOLD: # Tap
                self._fire_weapon()
            else: # Hold
                self._deploy_clone()
            self.space_press_duration = 0
            
        self.last_space_held = space_held

    def _fire_weapon(self):
        if self.weapon_cooldown > 0: return
        
        spec = self.WEAPON_SPECS[self.current_weapon]
        self.weapon_cooldown = spec["cooldown"]
        
        proj = {
            "pos": self.player_pos.copy(),
            "vel": pygame.Vector2(0, -spec["speed"]), # Always fires "up" screen
            "spec": spec,
        }
        self.player_projectiles.append(proj)
        # SFX: Player shoot
        
        # Muzzle flash
        for _ in range(5):
            self._create_particle(self.player_pos + pygame.Vector2(0, -self.PLAYER_SIZE), spec["color"], count=1, speed_range=(1,3), size_range=(2,4))

    def _deploy_clone(self):
        if self.clone_cooldown > 0 or len(self.clones) >= self.CLONE_MAX_ACTIVE: return
        
        self.clone_cooldown = 30 # 1 sec cooldown
        self.clones.append({
            "pos": self.player_pos.copy(),
            "lifetime": self.CLONE_LIFETIME,
            "pulse": 0
        })
        # SFX: Clone deploy
        self._create_particle(self.player_pos, self.COLOR_CLONE, count=20, speed_range=(2, 5))

    def _update_player(self):
        self.player_pos.x = self.player_pos.x % self.SCREEN_WIDTH
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.SCREEN_HEIGHT)

    def _spawn_enemy(self):
        side = self.np_random.integers(0, 4)
        if side == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_SIZE)
        elif side == 1: # Right
            pos = pygame.Vector2(self.SCREEN_WIDTH + self.ENEMY_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        elif side == 2: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ENEMY_SIZE)
        else: # Left
            pos = pygame.Vector2(-self.ENEMY_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
        self.enemies.append({
            "pos": pos,
            "health": self.ENEMY_HEALTH_MAX,
            "fire_cooldown": self.np_random.integers(60, 120)
        })

    def _update_enemies(self):
        for enemy in self.enemies:
            # --- Movement ---
            total_force = pygame.Vector2(0, 0)
            
            # Attraction to player
            dir_to_player = self.player_pos - enemy["pos"]
            if dir_to_player.length() > 0:
                total_force += dir_to_player.normalize()
            
            # Attraction to clones
            for clone in self.clones:
                dir_to_clone = clone["pos"] - enemy["pos"]
                dist_to_clone = dir_to_clone.length()
                if 0 < dist_to_clone < self.CLONE_RADIUS:
                    # Force is stronger closer to the clone
                    strength = (1 - (dist_to_clone / self.CLONE_RADIUS)) * self.CLONE_ATTRACTION_FORCE
                    total_force += dir_to_clone.normalize() * strength * 5 # Clones are very attractive
            
            if total_force.length() > 0:
                enemy["pos"] += total_force.normalize() * self.ENEMY_SPEED
            
            # --- Firing ---
            enemy["fire_cooldown"] -= 1
            if enemy["fire_cooldown"] <= 0:
                dir_to_player = self.player_pos - enemy["pos"]
                if dir_to_player.length() > 0:
                    self.enemy_projectiles.append({
                        "pos": enemy["pos"].copy(),
                        "vel": dir_to_player.normalize() * (3 + self.enemy_projectile_speed_bonus)
                    })
                    enemy["fire_cooldown"] = self.np_random.integers(90, 150)
                    # SFX: Enemy shoot

    def _update_clones(self):
        for clone in self.clones:
            clone["lifetime"] -= 1
            clone["pulse"] = (clone["pulse"] + 0.1) % (2 * math.pi)
        self.clones = [c for c in self.clones if c["lifetime"] > 0]

    def _update_projectiles(self):
        for proj in self.player_projectiles:
            proj["pos"] += proj["vel"]
        for proj in self.enemy_projectiles:
            proj["pos"] += proj["vel"]
        
        # Remove off-screen projectiles
        self.player_projectiles = [p for p in self.player_projectiles if self.screen.get_rect().collidepoint(p["pos"])]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if self.screen.get_rect().collidepoint(p["pos"])]

    def _create_particle(self, pos, color, count, lifetime_range=(10, 20), speed_range=(1, 4), size_range=(1, 3)):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            lifetime = self.np_random.integers(*lifetime_range)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "lifetime": lifetime,
                "max_lifetime": lifetime,
                "color": color,
                "size": self.np_random.uniform(*size_range)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if proj["pos"].distance_to(enemy["pos"]) < self.ENEMY_SIZE + proj["spec"]["size"]:
                    enemy["health"] -= proj["spec"]["damage"]
                    reward += 0.1
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    
                    if enemy["health"] <= 0:
                        reward += 1
                        self.score += 100
                        self._create_particle(enemy["pos"], (255, 150, 0), count=30, speed_range=(2,6))
                        # SFX: Explosion
                        if enemy in self.enemies: self.enemies.remove(enemy)
                    else:
                        self._create_particle(proj["pos"], (255, 255, 255), count=5, speed_range=(1,2))
                    break
        
        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles[:]:
            if proj["pos"].distance_to(self.player_pos) < self.PLAYER_SIZE:
                self.player_health -= 10
                reward -= 0.1
                self.screen_flash_timer = 5
                self.screen_flash_color = (*self.COLOR_ENEMY, 100)
                if proj in self.enemy_projectiles: self.enemy_projectiles.remove(proj)
                self._create_particle(self.player_pos, self.COLOR_ENEMY, count=10)
                # SFX: Player hit
                if self.player_health <= 0:
                    self.game_over = True
                    self._create_particle(self.player_pos, (255, 150, 0), count=100, speed_range=(2,8))
                    # SFX: Player Explosion
                break

        # Enemies vs Clones
        for enemy in self.enemies[:]:
            for clone in self.clones:
                if enemy["pos"].distance_to(clone["pos"]) < self.ENEMY_SIZE + 10:
                    reward += 1
                    self.score += 100
                    self._create_particle(enemy["pos"], self.COLOR_CLONE_FIELD, count=30, speed_range=(2,6))
                    # SFX: Zapped
                    if enemy in self.enemies: self.enemies.remove(enemy)
                    break
        
        return reward

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(self.screen_flash_timer / 5 * self.screen_flash_color[3])
            flash_surface.fill((*self.screen_flash_color[:3], alpha))
            self.screen.blit(flash_surface, (0,0))
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for star in self.stars:
            star["pos"].x = (star["pos"].x - star["speed"]) % self.SCREEN_WIDTH
            c = star["color"]
            pygame.draw.circle(self.screen, (c,c,c), star["pos"], star["size"])

        # Clones
        for clone in self.clones:
            pulse_alpha = 50 + 40 * math.sin(clone["pulse"])
            pygame.gfxdraw.aacircle(self.screen, int(clone["pos"].x), int(clone["pos"].y), self.CLONE_RADIUS, (*self.COLOR_CLONE_FIELD, int(pulse_alpha)))
            pygame.gfxdraw.filled_circle(self.screen, int(clone["pos"].x), int(clone["pos"].y), self.CLONE_RADIUS, (*self.COLOR_CLONE_FIELD, int(pulse_alpha/2)))
            
            core_radius = 10 + 3 * math.sin(clone["pulse"] * 2)
            pygame.gfxdraw.aacircle(self.screen, int(clone["pos"].x), int(clone["pos"].y), int(core_radius), self.COLOR_CLONE)
            pygame.gfxdraw.filled_circle(self.screen, int(clone["pos"].x), int(clone["pos"].y), int(core_radius), self.COLOR_CLONE)

        # Projectiles
        for proj in self.enemy_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, proj["pos"], 5)
        for proj in self.player_projectiles:
            spec = proj["spec"]
            pygame.draw.circle(self.screen, spec["color"], proj["pos"], spec["size"])
            pygame.gfxdraw.aacircle(self.screen, int(proj["pos"].x), int(proj["pos"].y), spec["size"], spec["color"])

        # Enemies
        for enemy in self.enemies:
            p1 = enemy["pos"] + pygame.Vector2(0, -self.ENEMY_SIZE).rotate(180)
            p2 = enemy["pos"] + pygame.Vector2(self.ENEMY_SIZE / 1.5, self.ENEMY_SIZE / 2).rotate(180)
            p3 = enemy["pos"] + pygame.Vector2(-self.ENEMY_SIZE / 1.5, self.ENEMY_SIZE / 2).rotate(180)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_ENEMY)

        # Player
        if self.player_health > 0:
            # Glow
            for i in range(4):
                alpha = 100 - i * 25
                pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE + i * 3, (*self.COLOR_PLAYER_GLOW, alpha))
            
            # Ship
            p1 = self.player_pos + pygame.Vector2(0, -self.PLAYER_SIZE)
            p2 = self.player_pos + pygame.Vector2(self.PLAYER_SIZE / 1.5, self.PLAYER_SIZE / 2)
            p3 = self.player_pos + pygame.Vector2(-self.PLAYER_SIZE / 1.5, self.PLAYER_SIZE / 2)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, p["pos"] - pygame.Vector2(p["size"], p["size"]), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Health bar
        health_ratio = np.clip(self.player_health / self.PLAYER_HEALTH_MAX, 0, 1)
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_small.render(f"HP", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Enemy Count
        enemy_text = self.font_large.render(f"ENEMIES: {len(self.enemies)}", True, self.COLOR_TEXT)
        self.screen.blit(enemy_text, (self.SCREEN_WIDTH - enemy_text.get_width() - 10, 10))

        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, self.SCREEN_HEIGHT - score_text.get_height() - 10))
        
        # Weapon display
        weapon_spec = self.WEAPON_SPECS[self.current_weapon]
        weapon_text = self.font_large.render(weapon_spec["name"].upper(), True, self.COLOR_TEXT)
        text_rect = weapon_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40))
        self.screen.blit(weapon_text, text_rect)
        
        # Clone charge indicator
        if self.space_press_duration > 0:
            charge_ratio = min(1, self.space_press_duration / (self.SPACE_HOLD_THRESHOLD + 1))
            bar_width = 100
            bar_x = self.SCREEN_WIDTH / 2 - bar_width / 2
            bar_y = self.SCREEN_HEIGHT - 25
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_width, 10))
            if charge_ratio < 1: # Tap charge (fire)
                pygame.draw.rect(self.screen, weapon_spec["color"], (bar_x, bar_y, int(bar_width * charge_ratio), 10))
            else: # Hold charge (clone)
                pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR, (bar_x, bar_y, bar_width, 10))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "enemies": len(self.enemies),
            "clones": len(self.clones),
            "current_weapon": self.current_weapon
        }

    def close(self):
        pygame.quit()
        

if __name__ == '__main__':
    # This block is for manual play and debugging.
    # It will not be executed by the evaluation setup.
    # To run, you need to have pygame installed: `pip install pygame`
    # You also need to unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    
    terminated = False
    total_reward = 0
    
    pygame.display.set_caption("Vector Combat")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # obs, info = env.reset() # Uncomment to auto-restart

    env.close()