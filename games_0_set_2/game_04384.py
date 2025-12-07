
# Generated: 2025-08-28T02:15:41.906906
# Source Brief: brief_04384.md
# Brief Index: 4384

        
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
        "Controls: Arrow keys to move. Hold space to fire your weapon. Press shift to activate your shield (if available)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive for 60 seconds against waves of asteroids and enemy ships in this top-down arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 60 * FPS

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_EXHAUST = (255, 100, 0)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ASTEROID = (150, 150, 150)
    COLOR_PROJECTILE_PLAYER = (255, 255, 255)
    COLOR_PROJECTILE_ENEMY = (255, 150, 150)
    COLOR_POWERUP = (100, 100, 255)
    COLOR_SHIELD = (100, 100, 255, 100) # RGBA
    COLOR_UI_TEXT = (220, 220, 220)
    
    # Game Parameters
    PLAYER_SIZE = 12
    PLAYER_SPEED = 5
    PLAYER_FRICTION = 0.9
    PLAYER_HEALTH_MAX = 100
    PLAYER_FIRE_COOLDOWN = 5 # frames
    
    ENEMY_SIZE = 10
    ENEMY_HEALTH_MAX = 20
    ENEMY_SHOOT_COOLDOWN = 20
    ENEMY_PROJECTILE_SPEED = 6
    ENEMY_SPAWN_RATE_INITIAL = 90 # frames
    
    ASTEROID_HEALTH_MAX = 10
    ASTEROID_SPEED_INITIAL = 1.0
    ASTEROID_COLLISION_DAMAGE = 20
    
    PROJECTILE_SPEED = 10
    PROJECTILE_DAMAGE = 10
    
    POWERUP_SPAWN_RATE = 150 # frames
    SHIELD_DURATION = 120 # frames

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        self.rng = np.random.default_rng()
        
        # This will be initialized in reset()
        self.player = {}
        self.enemies = []
        self.asteroids = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.powerup = None
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.player = {
            "pos": np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50], dtype=float),
            "vel": np.array([0.0, 0.0], dtype=float),
            "angle": -math.pi / 2,
            "health": self.PLAYER_HEALTH_MAX,
            "fire_cooldown": 0,
            "has_shield_powerup": False,
            "shield_active": False,
            "shield_timer": 0
        }

        self.enemies = []
        self.asteroids = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.powerup = None

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False

        self.enemy_spawn_timer = self.ENEMY_SPAWN_RATE_INITIAL
        self.enemy_spawn_rate = self.ENEMY_SPAWN_RATE_INITIAL
        self.powerup_spawn_timer = self.POWERUP_SPAWN_RATE
        self.asteroid_speed_multiplier = 1.0

        if not self.stars:
            self._create_stars()
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1 # Survival reward
        
        self._handle_input(action)
        self._update_game_state()
        
        collision_rewards = self._handle_collisions()
        reward += collision_rewards
        
        self._update_difficulty()
        self._handle_spawning()

        self.steps += 1
        
        terminated = self.player["health"] <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.player["health"] > 0: # Victory
                reward += 100
            else: # Loss
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.player["vel"][1] -= 1
        if movement == 2: self.player["vel"][1] += 1
        if movement == 3: self.player["vel"][0] -= 1
        if movement == 4: self.player["vel"][0] += 1

        # Update angle based on movement
        if movement in [1, 2, 3, 4]:
            self.player["angle"] = math.atan2(self.player["vel"][1], self.player["vel"][0])

        # --- Firing ---
        if space_held and not self.last_space_held and self.player["fire_cooldown"] <= 0:
            self._fire_player_projectile()
        self.last_space_held = space_held
        
        # --- Shield ---
        if shift_held and not self.last_shift_held and self.player["has_shield_powerup"] and not self.player["shield_active"]:
            self.player["has_shield_powerup"] = False
            self.player["shield_active"] = True
            self.player["shield_timer"] = self.SHIELD_DURATION
            # SFX: Shield activate
        self.last_shift_held = shift_held

    def _fire_player_projectile(self):
        self.player["fire_cooldown"] = self.PLAYER_FIRE_COOLDOWN
        vel = np.array([math.cos(self.player["angle"]), math.sin(self.player["angle"])]) * self.PROJECTILE_SPEED
        pos = self.player["pos"] + vel * 1.5
        self.player_projectiles.append({"pos": pos, "vel": vel})
        # SFX: Player shoot

    def _update_game_state(self):
        # Update player
        self.player["vel"] *= self.PLAYER_FRICTION
        if np.linalg.norm(self.player["vel"]) > self.PLAYER_SPEED:
            self.player["vel"] = self.player["vel"] / np.linalg.norm(self.player["vel"]) * self.PLAYER_SPEED
        self.player["pos"] += self.player["vel"]
        self.player["pos"][0] = np.clip(self.player["pos"][0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player["pos"][1] = np.clip(self.player["pos"][1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)
        if self.player["fire_cooldown"] > 0: self.player["fire_cooldown"] -= 1
        if self.player["shield_active"]:
            self.player["shield_timer"] -= 1
            if self.player["shield_timer"] <= 0:
                self.player["shield_active"] = False
                # SFX: Shield deactivate

        # Update entities
        for p in self.player_projectiles: p["pos"] += p["vel"]
        for p in self.enemy_projectiles: p["pos"] += p["vel"]
        for a in self.asteroids:
            a["pos"] += a["vel"]
            a["angle"] += a["rot_speed"]
        for e in self.enemies:
            e["pos"] += e["vel"]
            e["path_t"] += 0.05
            e["pos"][0] = e["start_x"] + math.sin(e["path_t"]) * 50
            if e["shoot_cooldown"] > 0: e["shoot_cooldown"] -= 1
            else:
                e["shoot_cooldown"] = self.ENEMY_SHOOT_COOLDOWN
                direction = self.player["pos"] - e["pos"]
                if np.linalg.norm(direction) > 0:
                    vel = direction / np.linalg.norm(direction) * self.ENEMY_PROJECTILE_SPEED
                    self.enemy_projectiles.append({"pos": e["pos"].copy(), "vel": vel})
                    # SFX: Enemy shoot
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] - 0.2)

        # Cleanup off-screen and dead entities
        self.player_projectiles = [p for p in self.player_projectiles if 0 < p["pos"][0] < self.SCREEN_WIDTH and 0 < p["pos"][1] < self.SCREEN_HEIGHT]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if 0 < p["pos"][0] < self.SCREEN_WIDTH and 0 < p["pos"][1] < self.SCREEN_HEIGHT]
        self.asteroids = [a for a in self.asteroids if 0 < a["pos"][0] < self.SCREEN_WIDTH and 0 < a["pos"][1] < self.SCREEN_HEIGHT]
        self.enemies = [e for e in self.enemies if 0 < e["pos"][1] < self.SCREEN_HEIGHT + 20]
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Asteroids/Enemies
        for p in self.player_projectiles[:]:
            for a in self.asteroids[:]:
                if np.linalg.norm(p["pos"] - a["pos"]) < a["size"]:
                    a["health"] -= self.PROJECTILE_DAMAGE
                    if a["health"] <= 0:
                        self.asteroids.remove(a)
                        self.score += 50
                        reward += 0.5
                        self._create_explosion(a["pos"], a["size"])
                    if p in self.player_projectiles: self.player_projectiles.remove(p)
                    break
            else: # continue if the inner loop wasn't broken
                for e in self.enemies[:]:
                    if np.linalg.norm(p["pos"] - e["pos"]) < self.ENEMY_SIZE:
                        e["health"] -= self.PROJECTILE_DAMAGE
                        if e["health"] <= 0:
                            self.enemies.remove(e)
                            self.score += 100
                            reward += 1.0
                            self._create_explosion(e["pos"], self.ENEMY_SIZE)
                        if p in self.player_projectiles: self.player_projectiles.remove(p)
                        break

        if self.player["shield_active"]:
            return reward

        # Enemy projectiles vs Player
        for p in self.enemy_projectiles[:]:
            if np.linalg.norm(p["pos"] - self.player["pos"]) < self.PLAYER_SIZE:
                self.player["health"] -= self.PROJECTILE_DAMAGE
                self.enemy_projectiles.remove(p)
                self._create_explosion(self.player["pos"], self.PLAYER_SIZE / 2, self.COLOR_PLAYER)
                # SFX: Player hit
        
        # Asteroids vs Player
        for a in self.asteroids[:]:
            if np.linalg.norm(a["pos"] - self.player["pos"]) < self.PLAYER_SIZE + a["size"]:
                self.player["health"] -= self.ASTEROID_COLLISION_DAMAGE
                self.asteroids.remove(a)
                self.score += 50 # Score for destroying it
                reward += 0.5
                self._create_explosion(a["pos"], a["size"])
                self._create_explosion(self.player["pos"], self.PLAYER_SIZE, self.COLOR_PLAYER)
                # SFX: Player hit hard
        
        # Power-up vs Player
        if self.powerup and np.linalg.norm(self.powerup["pos"] - self.player["pos"]) < self.PLAYER_SIZE + 10:
            self.player["has_shield_powerup"] = True
            self.powerup = None
            reward += 2.0
            # SFX: Powerup collect
            
        self.player["health"] = max(0, self.player["health"])
        return reward

    def _update_difficulty(self):
        # Increase enemy spawn rate every 10 seconds
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.enemy_spawn_rate = max(20, self.enemy_spawn_rate - 10)
        
        # Increase asteroid speed every 5 seconds
        if self.steps > 0 and self.steps % (5 * self.FPS) == 0:
            self.asteroid_speed_multiplier += 0.05

    def _handle_spawning(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self._spawn_enemy()
            self.enemy_spawn_timer = self.enemy_spawn_rate
            
        self.powerup_spawn_timer -= 1
        if self.powerup_spawn_timer <= 0 and self.powerup is None:
            self._spawn_powerup()
            self.powerup_spawn_timer = self.POWERUP_SPAWN_RATE

        # Keep a minimum number of asteroids
        if len(self.asteroids) < 5:
            self._spawn_asteroid()

    def _spawn_enemy(self):
        pos = np.array([self.rng.uniform(50, self.SCREEN_WIDTH - 50), -self.ENEMY_SIZE], dtype=float)
        self.enemies.append({
            "pos": pos,
            "vel": np.array([0, 1.5]),
            "health": self.ENEMY_HEALTH_MAX,
            "shoot_cooldown": self.rng.integers(0, self.ENEMY_SHOOT_COOLDOWN),
            "start_x": pos[0],
            "path_t": self.rng.uniform(0, 2 * math.pi)
        })

    def _spawn_asteroid(self):
        edge = self.rng.integers(0, 4)
        if edge == 0: # top
            pos = np.array([self.rng.uniform(0, self.SCREEN_WIDTH), -20], dtype=float)
        elif edge == 1: # right
            pos = np.array([self.SCREEN_WIDTH + 20, self.rng.uniform(0, self.SCREEN_HEIGHT)], dtype=float)
        elif edge == 2: # bottom
            pos = np.array([self.rng.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20], dtype=float)
        else: # left
            pos = np.array([-20, self.rng.uniform(0, self.SCREEN_HEIGHT)], dtype=float)

        target = np.array([self.rng.uniform(self.SCREEN_WIDTH*0.25, self.SCREEN_WIDTH*0.75), 
                           self.rng.uniform(self.SCREEN_HEIGHT*0.25, self.SCREEN_HEIGHT*0.75)])
        direction = target - pos
        vel = direction / np.linalg.norm(direction) * self.ASTEROID_SPEED_INITIAL * self.asteroid_speed_multiplier
        
        size = self.rng.uniform(8, 20)
        num_points = self.rng.integers(5, 9)
        shape = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = self.rng.uniform(0.7, 1.1) * size
            shape.append((math.cos(angle) * dist, math.sin(angle) * dist))

        self.asteroids.append({
            "pos": pos,
            "vel": vel,
            "health": self.ASTEROID_HEALTH_MAX,
            "size": size,
            "shape": shape,
            "angle": self.rng.uniform(0, 2 * math.pi),
            "rot_speed": self.rng.uniform(-0.05, 0.05)
        })

    def _spawn_powerup(self):
        self.powerup = {
            "pos": np.array([self.rng.uniform(50, self.SCREEN_WIDTH - 50), self.rng.uniform(50, self.SCREEN_HEIGHT - 50)], dtype=float)
        }

    def _create_explosion(self, pos, size, color=None):
        num_particles = int(size * 2)
        base_color = color if color is not None else (255, self.rng.integers(100, 200), 0)
        for _ in range(num_particles):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": self.rng.uniform(1, 4),
                "life": self.rng.integers(10, 20),
                "color": (
                    max(0, min(255, base_color[0] + self.rng.integers(-20, 20))),
                    max(0, min(255, base_color[1] + self.rng.integers(-20, 20))),
                    max(0, min(255, base_color[2] + self.rng.integers(-20, 20)))
                )
            })
        # SFX: Explosion

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_stars(self):
        self.stars = []
        for i in range(200):
            size = self.rng.choice([1, 2, 3])
            self.stars.append({
                "pos": (self.rng.integers(0, self.SCREEN_WIDTH), self.rng.integers(0, self.SCREEN_HEIGHT)),
                "size": size,
                "speed": 0.2 * size
            })

    def _render_game(self):
        # Stars
        for star in self.stars:
            star["pos"] = (star["pos"][0], (star["pos"][1] + star["speed"]) % self.SCREEN_HEIGHT)
            color_val = 50 * star["size"]
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), star["pos"], star["size"] / 2)

        # Power-up
        if self.powerup:
            pos = self.powerup["pos"].astype(int)
            t = self.steps % 30 / 30.0
            radius = 10 + 3 * math.sin(t * 2 * math.pi)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius), self.COLOR_POWERUP)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius), self.COLOR_POWERUP)

        # Asteroids
        for a in self.asteroids:
            points = []
            for p in a["shape"]:
                x = p[0] * math.cos(a["angle"]) - p[1] * math.sin(a["angle"])
                y = p[0] * math.sin(a["angle"]) + p[1] * math.cos(a["angle"])
                points.append((a["pos"][0] + x, a["pos"][1] + y))
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
        
        # Enemies
        for e in self.enemies:
            pos = e["pos"].astype(int)
            p1 = (pos[0], pos[1] - self.ENEMY_SIZE)
            p2 = (pos[0] - self.ENEMY_SIZE // 2, pos[1] + self.ENEMY_SIZE // 2)
            p3 = (pos[0] + self.ENEMY_SIZE // 2, pos[1] + self.ENEMY_SIZE // 2)
            pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_ENEMY)
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_ENEMY)

        # Player projectiles
        for p in self.player_projectiles:
            pos = p["pos"].astype(int)
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE_PLAYER, pos, 3)

        # Enemy projectiles
        for p in self.enemy_projectiles:
            pos = p["pos"].astype(int)
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE_ENEMY, pos, 3)

        # Player
        if self.player["health"] > 0:
            pos = self.player["pos"].astype(int)
            angle = self.player["angle"]
            size = self.PLAYER_SIZE
            
            # Exhaust trail
            if np.linalg.norm(self.player["vel"]) > 0.5:
                num_exhaust = self.rng.integers(1, 4)
                for _ in range(num_exhaust):
                    offset = self.rng.uniform(size * 0.8, size * 1.2)
                    p_pos = self.player["pos"] - np.array([math.cos(angle), math.sin(angle)]) * offset
                    self.particles.append({
                        "pos": p_pos, "vel": self.player["vel"] * -0.1,
                        "radius": self.rng.uniform(1, 3), "life": 8, "color": self.COLOR_PLAYER_EXHAUST
                    })

            # Ship body
            p1 = (pos[0] + math.cos(angle) * size, pos[1] + math.sin(angle) * size)
            p2 = (pos[0] + math.cos(angle + 2.5) * size * 0.8, pos[1] + math.sin(angle + 2.5) * size * 0.8)
            p3 = (pos[0] + math.cos(angle - 2.5) * size * 0.8, pos[1] + math.sin(angle - 2.5) * size * 0.8)
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)

            # Shield
            if self.player["shield_active"]:
                alpha = 100 + 50 * math.sin(self.steps * 0.5)
                temp_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surface, pos[0], pos[1], int(size * 1.8), (*self.COLOR_SHIELD[:3], alpha))
                pygame.gfxdraw.aacircle(temp_surface, pos[0], pos[1], int(size * 1.8), (*self.COLOR_SHIELD[:3], alpha))
                self.screen.blit(temp_surface, (0, 0))

        # Particles
        for p in self.particles:
            pos = p["pos"].astype(int)
            pygame.draw.circle(self.screen, p["color"], pos, int(p["radius"]))
            
    def _render_ui(self):
        # Health
        health_text = self.font_ui.render(f"HEALTH: {int(self.player['health'])}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) // self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 10))
        
        # Shield indicator
        if self.player["has_shield_powerup"]:
            shield_text = self.font_ui.render("SHIELD READY", True, self.COLOR_POWERUP)
            self.screen.blit(shield_text, (10, 35))

        # Game Over message
        if self.game_over:
            if self.player["health"] <= 0:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            else:
                msg = "VICTORY!"
                color = self.COLOR_PLAYER
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "time_left": (self.MAX_STEPS - self.steps) // self.FPS
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Create a display for human playing
    pygame.display.set_caption("Arcade Space Shooter")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        move_action = 0 # none
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.FPS)
        
    env.close()