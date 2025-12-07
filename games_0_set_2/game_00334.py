
# Generated: 2025-08-27T13:20:32.305054
# Source Brief: brief_00334.md
# Brief Index: 334

        
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
        "Controls: ↑↓←→ to move. Hold Space to fire your mining laser. Avoid the red enemy ships!"
    )

    game_description = (
        "Pilot a mining ship, blast asteroids for valuable ore, and avoid deadly collisions in this top-down arcade space miner. Collect 100 ore to win!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width = 640
        self.height = 400

        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_STAR = (50, 60, 80)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 50, 50, 50)
        self.COLOR_ASTEROID = (120, 130, 140)
        self.COLOR_ORE = (255, 220, 0)
        self.COLOR_LASER = (255, 255, 255)
        self.COLOR_UI_TEXT = (230, 230, 230)

        # --- Game Parameters ---
        self.MAX_STEPS = 5000
        self.WIN_ORE_COUNT = 100
        self.INITIAL_LIVES = 3
        self.PLAYER_ACCELERATION = 0.3
        self.PLAYER_FRICTION = 0.96
        self.PLAYER_MAX_SPEED = 5
        self.PLAYER_RADIUS = 12
        self.INITIAL_ASTEROIDS = 15
        self.MAX_ASTEROIDS = 20
        self.INITIAL_ENEMIES = 2
        self.ENEMY_RADIUS = 10
        self.ORE_RADIUS = 4
        self.LASER_RANGE = 150
        self.LASER_COOLDOWN_FRAMES = 5
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player = {
            "x": self.width / 2,
            "y": self.height / 2,
            "vx": 0,
            "vy": 0,
            "angle": -90,
            "lives": self.INITIAL_LIVES,
            "ore_collected": 0,
        }

        self.laser_active = False
        self.laser_cooldown = 0

        self.stars = [
            (self.np_random.integers(0, self.width), self.np_random.integers(0, self.height), self.np_random.integers(1, 3))
            for _ in range(100)
        ]
        self.asteroids = [self._create_asteroid() for _ in range(self.INITIAL_ASTEROIDS)]
        self.enemies = [self._create_enemy() for _ in range(self.INITIAL_ENEMIES)]
        self.ores = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for existing
        
        self._handle_input(action)
        self._update_player()
        self._update_enemies()
        self._update_ores()
        self._update_particles()
        
        reward += self._handle_collisions()
        
        self._spawn_entities()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.win:
                reward += 100
            elif self.player["lives"] <= 0:
                reward -= 50
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Movement
        if movement == 1:  # Up
            self.player["vy"] -= self.PLAYER_ACCELERATION
            self.player["angle"] = -90
        elif movement == 2:  # Down
            self.player["vy"] += self.PLAYER_ACCELERATION
            self.player["angle"] = 90
        elif movement == 3:  # Left
            self.player["vx"] -= self.PLAYER_ACCELERATION
            self.player["angle"] = 180
        elif movement == 4:  # Right
            self.player["vx"] += self.PLAYER_ACCELERATION
            self.player["angle"] = 0
            
        # Firing Laser
        if self.laser_cooldown > 0:
            self.laser_cooldown -= 1
            self.laser_active = False
        
        if space_held and self.laser_cooldown == 0:
            self.laser_active = True
            # sfx: laser_fire.wav
            self.laser_cooldown = self.LASER_COOLDOWN_FRAMES
        else:
            self.laser_active = False

    def _update_player(self):
        # Apply friction
        self.player["vx"] *= self.PLAYER_FRICTION
        self.player["vy"] *= self.PLAYER_FRICTION
        
        # Clamp speed
        speed = math.hypot(self.player["vx"], self.player["vy"])
        if speed > self.PLAYER_MAX_SPEED:
            self.player["vx"] = (self.player["vx"] / speed) * self.PLAYER_MAX_SPEED
            self.player["vy"] = (self.player["vy"] / speed) * self.PLAYER_MAX_SPEED

        # Update position and wrap around screen
        self.player["x"] = (self.player["x"] + self.player["vx"]) % self.width
        self.player["y"] = (self.player["y"] + self.player["vy"]) % self.height

        # Engine trail particles
        if abs(self.player["vx"]) > 0.1 or abs(self.player["vy"]) > 0.1:
            self._create_particles(self.player["x"], self.player["y"], 1, self.COLOR_PLAYER, 2, 10, -self.player["vx"], -self.player["vy"])


    def _update_enemies(self):
        # Difficulty scaling
        speed_multiplier = 1.0 + (self.steps // 500) * 0.05

        for enemy in self.enemies:
            enemy["angle"] += enemy["speed"] * speed_multiplier
            enemy["x"] = enemy["center_x"] + enemy["radius"] * math.cos(enemy["angle"])
            enemy["y"] = enemy["center_y"] + enemy["radius"] * math.sin(enemy["angle"])
            # Wrap patrol center if needed (for very large patrols)
            enemy["center_x"] %= self.width
            enemy["center_y"] %= self.height


    def _update_ores(self):
        for ore in self.ores[:]:
            ore["x"] += ore["vx"]
            ore["y"] += ore["vy"]
            ore["vx"] *= 0.98
            ore["vy"] *= 0.98
            ore["lifespan"] -= 1
            if ore["lifespan"] <= 0:
                self.ores.remove(ore)

    def _update_particles(self):
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Enemy
        for enemy in self.enemies:
            dist = math.hypot(self.player["x"] - enemy["x"], self.player["y"] - enemy["y"])
            if dist < self.PLAYER_RADIUS + self.ENEMY_RADIUS:
                self.player["lives"] -= 1
                reward -= 10
                # sfx: player_explosion.wav
                self._create_particles(self.player["x"], self.player["y"], 50, self.COLOR_PLAYER, 5, 40)
                self.player["x"], self.player["y"] = self.width / 2, self.height / 2
                self.player["vx"], self.player["vy"] = 0, 0
                if self.player["lives"] <= 0:
                    self.game_over = True
                break

        # Player vs Ore
        for ore in self.ores[:]:
            dist = math.hypot(self.player["x"] - ore["x"], self.player["y"] - ore["y"])
            if dist < self.PLAYER_RADIUS + self.ORE_RADIUS:
                self.ores.remove(ore)
                self.player["ore_collected"] += 1
                reward += 0.1
                # sfx: ore_collect.wav
                self._create_particles(ore["x"], ore["y"], 5, self.COLOR_ORE, 2, 15)
                if self.player["ore_collected"] >= self.WIN_ORE_COUNT:
                    self.win = True
                    self.game_over = True

        # Laser vs Asteroid
        if self.laser_active:
            rad_angle = math.radians(self.player["angle"])
            laser_end_x = self.player["x"] + self.LASER_RANGE * math.cos(rad_angle)
            laser_end_y = self.player["y"] + self.LASER_RANGE * math.sin(rad_angle)

            for asteroid in self.asteroids[:]:
                # Simple line-circle intersection
                dist = self._dist_point_to_segment(
                    asteroid["x"], asteroid["y"],
                    self.player["x"], self.player["y"],
                    laser_end_x, laser_end_y
                )
                if dist < asteroid["size"]:
                    asteroid["health"] -= 1
                    # sfx: laser_hit.wav
                    self._create_particles(asteroid["x"], asteroid["y"], 2, self.COLOR_LASER, 1, 5)
                    if asteroid["health"] <= 0:
                        reward += 1
                        self._spawn_ore_from_asteroid(asteroid)
                        self.asteroids.remove(asteroid)
                        # sfx: asteroid_explosion.wav
                        self._create_particles(asteroid["x"], asteroid["y"], int(asteroid["size"]), self.COLOR_ASTEROID, 4, 30)
                    break # Laser hits only one asteroid per frame
        return reward

    def _spawn_entities(self):
        if len(self.asteroids) < self.MAX_ASTEROIDS and self.np_random.random() < 0.02:
            self.asteroids.append(self._create_asteroid())

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_ores()
        self._render_asteroids()
        self._render_enemies()
        self._render_player()
        if self.laser_active:
            self._render_laser()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player["lives"],
            "ore_collected": self.player["ore_collected"],
        }
        
    # --- Entity Creation ---
    def _create_asteroid(self):
        size = self.np_random.uniform(10, 30)
        # Ensure asteroids don't spawn on the player
        while True:
            x = self.np_random.uniform(0, self.width)
            y = self.np_random.uniform(0, self.height)
            if math.hypot(x - self.player["x"], y - self.player["y"]) > 100:
                break
        
        points = []
        num_points = self.np_random.integers(7, 12)
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points)
            dist = self.np_random.uniform(size * 0.8, size * 1.2)
            points.append((dist * math.cos(angle), dist * math.sin(angle)))

        return {"x": x, "y": y, "size": size, "health": size / 5, "points": points}
        
    def _create_enemy(self):
        return {
            "center_x": self.np_random.uniform(0, self.width),
            "center_y": self.np_random.uniform(0, self.height),
            "radius": self.np_random.uniform(50, 150),
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "speed": self.np_random.uniform(0.01, 0.03),
            "x": 0, "y": 0 # Will be calculated in first update
        }

    def _spawn_ore_from_asteroid(self, asteroid):
        num_ore = int(asteroid["size"] / 5)
        for _ in range(num_ore):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.ores.append({
                "x": asteroid["x"],
                "y": asteroid["y"],
                "vx": speed * math.cos(angle),
                "vy": speed * math.sin(angle),
                "lifespan": self.np_random.integers(200, 400)
            })

    def _create_particles(self, x, y, count, color, max_speed, max_life, vx_bias=0, vy_bias=0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            self.particles.append({
                "x": x, "y": y,
                "vx": speed * math.cos(angle) + vx_bias,
                "vy": speed * math.sin(angle) + vy_bias,
                "lifespan": self.np_random.uniform(max_life / 2, max_life),
                "color": color
            })

    # --- Rendering ---
    def _render_stars(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)
            
    def _render_asteroids(self):
        for a in self.asteroids:
            points = [(a["x"] + px, a["y"] + py) for px, py in a["points"]]
            pygame.draw.polygon(self.screen, self.COLOR_ASTEROID, points)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_ores(self):
        for o in self.ores:
            pygame.gfxdraw.filled_circle(self.screen, int(o["x"]), int(o["y"]), self.ORE_RADIUS, self.COLOR_ORE)
            pygame.gfxdraw.aacircle(self.screen, int(o["x"]), int(o["y"]), self.ORE_RADIUS, self.COLOR_ORE)
            
    def _render_enemies(self):
        for e in self.enemies:
            p1 = (e["x"] + self.ENEMY_RADIUS, e["y"])
            p2 = (e["x"] - self.ENEMY_RADIUS / 2, e["y"] - self.ENEMY_RADIUS * 0.866)
            p3 = (e["x"] - self.ENEMY_RADIUS / 2, e["y"] + self.ENEMY_RADIUS * 0.866)
            # Simple rotation effect
            angle = math.atan2(e["y"] - e["center_y"], e["x"] - e["center_x"]) + math.pi/2
            points = [self._rotate_point(p, (e["x"], e["y"]), angle) for p in [p1, p2, p3]]
            
            # Glow effect
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_ENEMY_GLOW)
            # Main shape
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
            
    def _render_player(self):
        px, py = self.player["x"], self.player["y"]
        angle_rad = math.radians(self.player["angle"])
        
        p1 = (px + self.PLAYER_RADIUS, py)
        p2 = (px - self.PLAYER_RADIUS / 2, py - self.PLAYER_RADIUS * 0.866)
        p3 = (px - self.PLAYER_RADIUS / 2, py + self.PLAYER_RADIUS * 0.866)
        
        points = [self._rotate_point(p, (px, py), angle_rad) for p in [p1, p2, p3]]
        
        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2), self.PLAYER_RADIUS * 1.5)
        self.screen.blit(glow_surf, (int(px - self.PLAYER_RADIUS*2), int(py - self.PLAYER_RADIUS*2)))

        # Main ship
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_laser(self):
        rad_angle = math.radians(self.player["angle"])
        start_pos = (int(self.player["x"]), int(self.player["y"]))
        end_pos = (
            int(self.player["x"] + self.LASER_RANGE * math.cos(rad_angle)),
            int(self.player["y"] + self.LASER_RANGE * math.sin(rad_angle))
        )
        pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)
        pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos, end_pos, blend=1)
        
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            if alpha > 0:
                size = int(p["lifespan"] / 10)
                if size > 0:
                    color_with_alpha = p["color"] + (alpha,)
                    temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color_with_alpha, (size, size), size)
                    self.screen.blit(temp_surf, (int(p["x"] - size), int(p["y"] - size)))

    def _render_ui(self):
        # Ore display
        ore_text = self.font_small.render(f"ORE: {self.player['ore_collected']}/{self.WIN_ORE_COUNT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ore_text, (10, 10))
        
        # Lives display
        lives_text = self.font_small.render("LIVES:", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.width - 150, 10))
        for i in range(self.player["lives"]):
            points = [
                (self.width - 80 + i * 20, 18),
                (self.width - 80 + i * 20 - 5, 18 - 8.66),
                (self.width - 80 + i * 20 + 5, 18 - 8.66)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
        
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ENEMY
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(end_text, text_rect)

    # --- Utility Functions ---
    def _rotate_point(self, point, center, angle):
        ox, oy = center
        px, py = point
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def _dist_point_to_segment(self, px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        
        t = ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)
        t = max(0, min(1, t))
        
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return math.hypot(px - closest_x, py - closest_y)
        
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part allows a human to play the game using the keyboard.
    # It demonstrates how actions are mapped and the environment responds.
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Set up a display window
    pygame.display.set_caption("Space Miner")
    display_screen = pygame.display.set_mode((env.width, env.height))
    
    running = True
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        # Movement
        mov = 0 # No-op
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        # Space button
        space = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift button (unused in this game)
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = np.array([mov, space, shift])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Ore: {info['ore_collected']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()