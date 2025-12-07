
# Generated: 2025-08-27T13:53:00.048283
# Source Brief: brief_00512.md
# Brief Index: 512

        
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
        "Controls: Arrow keys to apply thrust. Hold space to fire your weapon. "
        "Your ship aims in the last direction you moved."
    )

    game_description = (
        "Pilot a spaceship, blast asteroids, and collect fuel to survive in a "
        "procedurally generated asteroid field. Collect 50 fuel to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.WIN_FUEL = 50.0
        self.STARTING_FUEL = 25.0

        # Player
        self.PLAYER_RADIUS = 12
        self.PLAYER_ACCELERATION = 0.6
        self.PLAYER_MAX_SPEED = 6.0
        self.PLAYER_FRICTION = 0.97
        self.SHOOT_COOLDOWN_FRAMES = 8

        # Bullets
        self.BULLET_SPEED = 10.0
        self.BULLET_LIFESPAN = 40 # in frames

        # Asteroids
        self.ASTEROID_MIN_RADIUS = 15
        self.ASTEROID_MAX_RADIUS = 40
        self.ASTEROID_MIN_VERTICES = 7
        self.ASTEROID_MAX_VERTICES = 12
        self.FUEL_DROP_CHANCE = 0.5

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_BULLET = (0, 255, 255)
        self.COLOR_ASTEROID = (160, 160, 160)
        self.COLOR_FUEL = (255, 223, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_UI_BAR = (40, 60, 80)
        self.PARTICLE_COLORS = [(255, 64, 0), (255, 165, 0), (255, 215, 0)]

        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # --- State Variables ---
        self.np_random = None
        self.player = {}
        self.asteroids = []
        self.bullets = []
        self.fuel_canisters = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.fuel = 0.0
        self.game_over = False
        self.win = False
        self.shoot_cooldown = 0
        self.last_move_action = 1
        self.asteroid_spawn_rate = 0.0
        self.asteroid_base_speed = 0.0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.player = {
            "pos": np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64),
            "vel": np.array([0.0, 0.0], dtype=np.float64),
            "angle": -90.0, # Pointing up
        }
        self.last_move_action = 1

        self.steps = 0
        self.score = 0
        self.fuel = self.STARTING_FUEL
        self.game_over = False
        self.win = False
        self.shoot_cooldown = 0

        self.asteroids.clear()
        self.bullets.clear()
        self.fuel_canisters.clear()
        self.particles.clear()

        self.asteroid_spawn_rate = 0.02
        self.asteroid_base_speed = 1.0
        for _ in range(5):
            self._spawn_asteroid()

        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
            for _ in range(150)
        ]
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0.0

        self._handle_input(action)
        self._update_game_state()
        reward += self._handle_collisions()
        self._spawn_new_asteroids()
        self._update_difficulty()

        self.fuel = max(0, self.fuel - 0.01)
        reward -= 0.01

        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        move_force = np.array([0.0, 0.0])
        if movement != 0: self.last_move_action = movement
        if movement == 1: move_force[1] = -self.PLAYER_ACCELERATION
        elif movement == 2: move_force[1] = self.PLAYER_ACCELERATION
        elif movement == 3: move_force[0] = -self.PLAYER_ACCELERATION
        elif movement == 4: move_force[0] = self.PLAYER_ACCELERATION
        self.player["vel"] += move_force

        if self.last_move_action == 1: self.player['angle'] = -90
        elif self.last_move_action == 2: self.player['angle'] = 90
        elif self.last_move_action == 3: self.player['angle'] = 180
        elif self.last_move_action == 4: self.player['angle'] = 0

        if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
        if space_held and self.shoot_cooldown <= 0 and self.fuel > 0:
            self._spawn_bullet()
            # sfx: player_shoot.wav
            self.shoot_cooldown = self.SHOOT_COOLDOWN_FRAMES

    def _update_game_state(self):
        # Player
        self.player["vel"] *= self.PLAYER_FRICTION
        speed = np.linalg.norm(self.player["vel"])
        if speed > self.PLAYER_MAX_SPEED:
            self.player["vel"] = (self.player["vel"] / speed) * self.PLAYER_MAX_SPEED
        self.player["pos"] += self.player["vel"]
        self.player["pos"][0] %= self.WIDTH
        self.player["pos"][1] %= self.HEIGHT

        # Bullets
        for b in self.bullets:
            b["pos"] += b["vel"]
            b["lifespan"] -= 1
        self.bullets = [b for b in self.bullets if b["lifespan"] > 0]

        # Asteroids
        for a in self.asteroids:
            a["pos"] += a["vel"]
            a["angle"] += a["rot_speed"]
            a["pos"][0] %= self.WIDTH
            a["pos"][1] %= self.HEIGHT

        # Particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Bullet-Asteroid
        destroyed_asteroids = set()
        used_bullets = set()
        for i, bullet in enumerate(self.bullets):
            for j, asteroid in enumerate(self.asteroids):
                if j in destroyed_asteroids: continue
                dist = np.linalg.norm(bullet["pos"] - asteroid["pos"])
                if dist < asteroid["radius"]:
                    destroyed_asteroids.add(j)
                    used_bullets.add(i)
                    self._create_explosion(asteroid["pos"], asteroid["radius"])
                    # sfx: asteroid_explosion.wav
                    reward += 1.0
                    self.score += 1
                    if self.np_random.random() < self.FUEL_DROP_CHANCE:
                        self.fuel_canisters.append({"pos": asteroid["pos"].copy(), "blink_timer": 0})
                    break
        
        self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in destroyed_asteroids]
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in used_bullets]

        # Player-Fuel
        remaining_fuel = []
        for fuel_can in self.fuel_canisters:
            dist = np.linalg.norm(self.player["pos"] - fuel_can["pos"])
            if dist < self.PLAYER_RADIUS + 10:
                self.fuel = min(self.WIN_FUEL, self.fuel + 5.0)
                reward += 0.1
                self.score += 5
                # sfx: fuel_pickup.wav
            else:
                remaining_fuel.append(fuel_can)
        self.fuel_canisters = remaining_fuel
        
        # Player-Asteroid (terminates game, checked in _check_termination)
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player["pos"] - asteroid["pos"])
            if dist < self.PLAYER_RADIUS + asteroid["radius"] * 0.9: # 0.9 for forgiveness
                return reward - 100 # Immediate terminal penalty
        return reward

    def _check_termination(self):
        # Win condition
        if self.fuel >= self.WIN_FUEL:
            self.win = True
            return True, 100.0
        
        # Lose condition (collision)
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player["pos"] - asteroid["pos"])
            if dist < self.PLAYER_RADIUS + asteroid["radius"] * 0.9:
                self._create_explosion(self.player["pos"], self.PLAYER_RADIUS * 2)
                # sfx: player_death.wav
                return True, 0 # Penalty already applied in collision check

        # Lose condition (out of fuel)
        if self.fuel <= 0:
            return True, -50.0

        return False, 0.0

    def _spawn_new_asteroids(self):
        if self.np_random.random() < self.asteroid_spawn_rate:
            self._spawn_asteroid()

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.asteroid_spawn_rate = min(0.1, self.asteroid_spawn_rate + 0.01)
            self.asteroid_base_speed = min(3.0, self.asteroid_base_speed + 0.1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "fuel": self.fuel}

    # --- Spawning Methods ---
    def _spawn_asteroid(self):
        edge = self.np_random.integers(4)
        if edge == 0: pos = np.array([self.np_random.uniform(0, self.WIDTH), -self.ASTEROID_MAX_RADIUS], dtype=np.float64)
        elif edge == 1: pos = np.array([self.WIDTH + self.ASTEROID_MAX_RADIUS, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float64)
        elif edge == 2: pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ASTEROID_MAX_RADIUS], dtype=np.float64)
        else: pos = np.array([-self.ASTEROID_MAX_RADIUS, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float64)
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.asteroid_base_speed + self.np_random.uniform(-0.5, 0.5)
        vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * speed
        
        radius = self.np_random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)
        num_vertices = self.np_random.integers(self.ASTEROID_MIN_VERTICES, self.ASTEROID_MAX_VERTICES + 1)
        shape_angles = sorted([self.np_random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
        shape = [
            (math.cos(a) * (radius + self.np_random.uniform(-radius*0.3, radius*0.3)),
             math.sin(a) * (radius + self.np_random.uniform(-radius*0.3, radius*0.3)))
            for a in shape_angles
        ]

        self.asteroids.append({
            "pos": pos, "vel": vel, "radius": radius, "shape": shape,
            "angle": 0, "rot_speed": self.np_random.uniform(-1, 1)
        })

    def _spawn_bullet(self):
        angle_rad = math.radians(self.player["angle"])
        vel = np.array([math.cos(angle_rad), math.sin(angle_rad)]) * self.BULLET_SPEED
        pos = self.player["pos"] + np.array([math.cos(angle_rad), math.sin(angle_rad)]) * self.PLAYER_RADIUS
        self.bullets.append({"pos": pos, "vel": vel, "lifespan": self.BULLET_LIFESPAN})

    def _create_explosion(self, position, size):
        num_particles = int(size)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": position.copy(), "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": random.choice(self.PARTICLE_COLORS),
                "radius": self.np_random.uniform(2, 5)
            })
    
    # --- Rendering Methods ---
    def _render_stars(self):
        for x, y, size in self.stars:
            self.screen.set_at((x, y), (200, 200, 200))
            if size > 1:
                self.screen.set_at((x+1, y), (150, 150, 150))

    def _render_game(self):
        # Fuel canisters
        for can in self.fuel_canisters:
            can["blink_timer"] = (can["blink_timer"] + 1) % 30
            if can["blink_timer"] < 20:
                pygame.draw.circle(self.screen, self.COLOR_FUEL, can["pos"].astype(int), 7)
                pygame.draw.circle(self.screen, (255, 255, 255), can["pos"].astype(int), 3)

        # Asteroids
        for a in self.asteroids:
            points = []
            for sx, sy in a["shape"]:
                rotated_x = sx * math.cos(math.radians(a["angle"])) - sy * math.sin(math.radians(a["angle"]))
                rotated_y = sx * math.sin(math.radians(a["angle"])) + sy * math.cos(math.radians(a["angle"]))
                points.append((int(a["pos"][0] + rotated_x), int(a["pos"][1] + rotated_y)))
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, points, (200, 200, 200))

        # Bullets
        for b in self.bullets:
            pygame.draw.circle(self.screen, self.COLOR_BULLET, b["pos"].astype(int), 3)

        # Player
        if not (self.game_over and not self.win):
            p_pos = self.player["pos"]
            angle_rad = math.radians(self.player["angle"])
            points = [
                (p_pos[0] + self.PLAYER_RADIUS * math.cos(angle_rad), p_pos[1] + self.PLAYER_RADIUS * math.sin(angle_rad)),
                (p_pos[0] + self.PLAYER_RADIUS * math.cos(angle_rad + 2.5), p_pos[1] + self.PLAYER_RADIUS * math.sin(angle_rad + 2.5)),
                (p_pos[0] + self.PLAYER_RADIUS * math.cos(angle_rad - 2.5), p_pos[1] + self.PLAYER_RADIUS * math.sin(angle_rad - 2.5)),
            ]
            points_int = [(int(x), int(y)) for x, y in points]
            pygame.gfxdraw.filled_trigon(self.screen, *points_int[0], *points_int[1], *points_int[2], self.COLOR_PLAYER)
            pygame.gfxdraw.aatrigon(self.screen, *points_int[0], *points_int[1], *points_int[2], self.COLOR_PLAYER)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = p["color"] + (alpha,)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (p["pos"] - p["radius"]).astype(int))

    def _render_ui(self):
        # Fuel bar
        fuel_ratio = max(0, self.fuel / self.WIN_FUEL)
        bar_width = (self.WIDTH - 20) * fuel_ratio
        bar_color = (255, 0, 0) if fuel_ratio < 0.2 else ((255, 255, 0) if fuel_ratio < 0.5 else (0, 255, 0))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, 10, self.WIDTH - 20, 20))
        if bar_width > 0:
            pygame.draw.rect(self.screen, bar_color, (10, 10, bar_width, 20))
        
        fuel_text = self.font_small.render("FUEL", True, self.COLOR_TEXT)
        self.screen.blit(fuel_text, (15, 12))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 35))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        message = "YOU WIN!" if self.win else "GAME OVER"
        text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
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
        assert not trunc # Truncated is handled inside step, but step returns False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()