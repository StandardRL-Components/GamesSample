
# Generated: 2025-08-27T15:49:54.188228
# Source Brief: brief_01088.md
# Brief Index: 1088

        
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
        "Controls: ↑ to thrust, ←→ to turn, and ↓ to brake. "
        "Hold shift to drift. Press space to fire your mining laser."
    )

    game_description = (
        "Pilot a mining ship in a dangerous asteroid field. Blast asteroids to "
        "collect valuable ore, but watch out for hostile alien patrols. "
        "Collect 100 ore to win, but lose all 3 lives and you're space dust."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    WIN_SCORE = 100
    STARTING_LIVES = 3

    # Player
    PLAYER_ACCEL = 0.4
    PLAYER_FRICTION = 0.96
    PLAYER_TURN_SPEED = 0.1
    PLAYER_DRIFT_FRICTION = 0.99
    PLAYER_DRIFT_TURN_MULT = 0.5
    PLAYER_BRAKE_FRICTION = 0.9
    PLAYER_RESPAWN_INVINCIBILITY = 90  # frames

    # Entities
    MAX_ASTEROIDS = 12
    MAX_ENEMIES = 2
    ORE_SPEED = 2.5
    ORE_LIFETIME = 450 # 15 seconds

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_THRUST = (200, 255, 255)
    COLOR_ENEMY = (255, 50, 100)
    COLOR_ASTEROID = (150, 150, 150)
    COLOR_ORE = (255, 220, 50)
    COLOR_LASER = (255, 100, 255)
    COLOR_EXPLOSION = [(255, 200, 50), (255, 100, 50), (200, 50, 50)]
    COLOR_TEXT = (220, 220, 240)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.np_random = None # Will be seeded in reset

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.STARTING_LIVES
        self.game_over = False
        self.last_ore_step = 0

        self.player = {
            "pos": np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64),
            "vel": np.array([0.0, 0.0], dtype=np.float64),
            "angle": -math.pi / 2,
            "invincible_timer": self.PLAYER_RESPAWN_INVINCIBILITY,
        }

        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.uniform(0.5, 1.5),
            )
            for _ in range(150)
        ]
        
        self.asteroids = []
        for _ in range(self.MAX_ASTEROIDS):
            self._spawn_asteroid()

        self.enemies = []
        for _ in range(self.MAX_ENEMIES):
            self._spawn_enemy()
            
        self.ore_particles = []
        self.fx_particles = []
        self.laser_active = False
        self.laser_hit_pos = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.02  # Time penalty

        if not self.game_over:
            self._handle_input(movement, shift_held)
            self._update_player(shift_held)
            self._update_enemies()
            self._update_ore_particles()
            self._update_fx_particles()
            
            laser_reward = self._handle_laser(space_held)
            reward += laser_reward

            collision_reward = self._handle_collisions()
            reward += collision_reward

            self._maintain_asteroids()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
            elif self.lives <= 0:
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    # --- State Update Helpers ---

    def _handle_input(self, movement, shift_held):
        turn_speed = self.PLAYER_TURN_SPEED * (self.PLAYER_DRIFT_TURN_MULT if shift_held else 1.0)
        if movement == 1:  # Up
            self.player["vel"] += np.array([math.cos(self.player["angle"]), math.sin(self.player["angle"])]) * self.PLAYER_ACCEL
            self._create_thrust_particles()
        elif movement == 2:  # Down
            self.player["vel"] *= self.PLAYER_BRAKE_FRICTION
        if movement == 3:  # Left
            self.player["angle"] -= turn_speed
        elif movement == 4:  # Right
            self.player["angle"] += turn_speed

    def _update_player(self, shift_held):
        friction = self.PLAYER_DRIFT_FRICTION if shift_held else self.PLAYER_FRICTION
        self.player["vel"] *= friction
        self.player["pos"] += self.player["vel"]

        # World wrapping
        self.player["pos"][0] %= self.WIDTH
        self.player["pos"][1] %= self.HEIGHT
        
        if self.player["invincible_timer"] > 0:
            self.player["invincible_timer"] -= 1

    def _update_enemies(self):
        speed_mult = 1.0 + (0.05 * (self.score // 25))
        for enemy in self.enemies:
            enemy["orbit_angle"] += enemy["orbit_speed"] * speed_mult
            enemy["pos"][0] = enemy["orbit_center"][0] + math.cos(enemy["orbit_angle"]) * enemy["orbit_radius"]
            enemy["pos"][1] = enemy["orbit_center"][1] + math.sin(enemy["orbit_angle"]) * enemy["orbit_radius"]

    def _update_ore_particles(self):
        for ore in self.ore_particles[:]:
            ore["life"] -= 1
            if ore["life"] <= 0:
                self.ore_particles.remove(ore)
                continue
            
            direction = self.player["pos"] - ore["pos"]
            dist_sq = direction[0]**2 + direction[1]**2
            if dist_sq > 1:
                direction /= np.sqrt(dist_sq)
            
            # Gravity pull
            pull_strength = min(5, 1000 / (dist_sq + 100))
            ore["vel"] = ore["vel"] * 0.95 + direction * self.ORE_SPEED * pull_strength * 0.1
            ore["pos"] += ore["vel"]

    def _update_fx_particles(self):
        for p in self.fx_particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.fx_particles.remove(p)

    def _handle_laser(self, space_held):
        self.laser_active = space_held
        self.laser_hit_pos = None
        if not space_held:
            return 0
        
        # sfx: laser_fire.wav
        reward = 0
        laser_start = self.player["pos"]
        laser_dir = np.array([math.cos(self.player["angle"]), math.sin(self.player["angle"])])
        
        closest_hit = None
        closest_dist_sq = float('inf')

        for asteroid in self.asteroids:
            vec_to_asteroid = asteroid["pos"] - laser_start
            proj = vec_to_asteroid.dot(laser_dir)
            if proj <= 0: continue
            
            dist_sq = np.sum(vec_to_asteroid**2) - proj**2
            if dist_sq < asteroid["radius"]**2:
                if proj**2 < closest_dist_sq:
                    closest_dist_sq = proj**2
                    closest_hit = asteroid
        
        if closest_hit:
            # sfx: asteroid_hit.wav
            hit_dist = np.sqrt(closest_dist_sq) - closest_hit["radius"]
            self.laser_hit_pos = laser_start + laser_dir * hit_dist
            
            closest_hit["hp"] -= 1
            if closest_hit["hp"] <= 0:
                # sfx: explosion_large.wav
                reward += closest_hit["reward"]
                self._create_explosion(closest_hit["pos"], closest_hit["radius"] * 2, self.COLOR_EXPLOSION)
                self._spawn_ore(closest_hit["pos"], closest_hit["value"])
                self.asteroids.remove(closest_hit)
            else: # Hit effect
                self._create_explosion(self.laser_hit_pos, 3, [self.COLOR_LASER])

        return reward

    def _handle_collisions(self):
        reward = 0
        # Player vs Enemy
        if self.player["invincible_timer"] <= 0:
            for enemy in self.enemies:
                dist = np.linalg.norm(self.player["pos"] - enemy["pos"])
                if dist < 10 + 10: # Player radius + enemy radius
                    # sfx: player_hit.wav
                    self.lives -= 1
                    reward -= 5
                    self._create_explosion(self.player["pos"], 30, self.COLOR_EXPLOSION)
                    self.player["pos"] = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
                    self.player["vel"] = np.array([0.0, 0.0], dtype=np.float64)
                    self.player["invincible_timer"] = self.PLAYER_RESPAWN_INVINCIBILITY
                    if self.lives <= 0:
                        self.game_over = True
                    break # Only one collision per frame
        
        # Player vs Ore
        for ore in self.ore_particles[:]:
            dist = np.linalg.norm(self.player["pos"] - ore["pos"])
            if dist < 10 + 3: # Player radius + ore radius
                # sfx: ore_collect.wav
                self.score = min(self.WIN_SCORE, self.score + 1)
                reward += 0.1
                self.last_ore_step = self.steps
                self.ore_particles.remove(ore)
        
        return reward

    def _maintain_asteroids(self):
        while len(self.asteroids) < self.MAX_ASTEROIDS:
            self._spawn_asteroid()
            
    def _check_termination(self):
        return self.score >= self.WIN_SCORE or self.lives <= 0 or self.steps >= self.MAX_STEPS

    # --- Spawning Helpers ---

    def _spawn_asteroid(self):
        size_roll = self.np_random.random()
        if size_roll < 0.5: # Small
            size, value, reward, hp = "small", 1, 1, 3
        elif size_roll < 0.85: # Medium
            size, value, reward, hp = "medium", 3, 3, 6
        else: # Large
            size, value, reward, hp = "large", 5, 5, 10
        
        radius = {"small": 12, "medium": 20, "large": 30}[size]
        
        # Spawn away from center
        pos = np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)])
        while np.linalg.norm(pos - self.player["pos"]) < 100:
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)])

        # Generate procedural shape
        num_vertices = self.np_random.integers(7, 12)
        angles = np.linspace(0, 2 * math.pi, num_vertices, endpoint=False)
        shape_rand = self.np_random.uniform(0.7, 1.3, num_vertices)
        points = [
            (math.cos(a) * radius * r, math.sin(a) * radius * r)
            for a, r in zip(angles, shape_rand)
        ]

        self.asteroids.append({
            "pos": pos, "radius": radius, "size": size, "value": value,
            "reward": reward, "hp": hp, "points": points, "angle": self.np_random.uniform(0, 2*math.pi)
        })

    def _spawn_enemy(self):
        self.enemies.append({
            "pos": np.array([0.0, 0.0]),
            "orbit_center": np.array([self.np_random.uniform(100, self.WIDTH-100), self.np_random.uniform(100, self.HEIGHT-100)]),
            "orbit_radius": self.np_random.uniform(50, 150),
            "orbit_angle": self.np_random.uniform(0, 2 * math.pi),
            "orbit_speed": self.np_random.uniform(0.01, 0.02) * (1 if self.np_random.random() > 0.5 else -1),
        })

    def _spawn_ore(self, pos, value):
        for _ in range(value):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.ore_particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle), math.sin(angle)]) * speed,
                "life": self.ORE_LIFETIME,
            })
            
    def _create_explosion(self, pos, count, colors):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.fx_particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle), math.sin(angle)]) * speed,
                "life": self.np_random.integers(10, 20),
                "color": random.choice(colors),
            })
            
    def _create_thrust_particles(self):
        if self.steps % 2 == 0:
            angle = self.player["angle"] + math.pi + self.np_random.uniform(-0.3, 0.3)
            speed = self.np_random.uniform(1, 3)
            pos_offset = np.array([math.cos(self.player["angle"]), math.sin(self.player["angle"])]) * -10
            self.fx_particles.append({
                "pos": self.player["pos"] + pos_offset,
                "vel": np.array([math.cos(angle), math.sin(angle)]) * speed + self.player["vel"]*0.5,
                "life": self.np_random.integers(8, 15),
                "color": self.COLOR_PLAYER_THRUST,
            })

    # --- Rendering Helpers ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_ore_particles()
        self._render_asteroids()
        self._render_enemies()
        self._render_player()
        if self.laser_active: self._render_laser()
        self._render_fx_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for x, y, speed in self.stars:
            px = int((x - self.player["pos"][0] * speed * 0.1) % self.WIDTH)
            py = int((y - self.player["pos"][1] * speed * 0.1) % self.HEIGHT)
            c = int(100 * speed)
            self.screen.set_at((px, py), (c,c,c))

    def _render_ore_particles(self):
        for ore in self.ore_particles:
            alpha = max(0, min(255, int(ore["life"] * 2)))
            color = (*self.COLOR_ORE, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(ore["pos"][0]), int(ore["pos"][1]), 3, color)

    def _render_asteroids(self):
        for ast in self.asteroids:
            points_rotated = []
            for px, py in ast["points"]:
                x_rot = px * math.cos(ast["angle"]) - py * math.sin(ast["angle"]) + ast["pos"][0]
                y_rot = px * math.sin(ast["angle"]) + py * math.cos(ast["angle"]) + ast["pos"][1]
                points_rotated.append((int(x_rot), int(y_rot)))

            pygame.gfxdraw.aapolygon(self.screen, points_rotated, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points_rotated, self.COLOR_ASTEROID)

    def _render_enemies(self):
        for enemy in self.enemies:
            p = enemy["pos"]
            angle = enemy["orbit_angle"] + math.pi/2
            points = [
                (p[0] + 10 * math.cos(angle), p[1] + 10 * math.sin(angle)),
                (p[0] + 5 * math.cos(angle + 2.5), p[1] + 5 * math.sin(angle + 2.5)),
                (p[0] + 5 * math.cos(angle - 2.5), p[1] + 5 * math.sin(angle - 2.5)),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

    def _render_player(self):
        p = self.player["pos"]
        a = self.player["angle"]
        
        # Invincibility blink
        if self.player["invincible_timer"] > 0 and self.steps % 10 < 5:
            return

        points = [
            (p[0] + 12 * math.cos(a), p[1] + 12 * math.sin(a)),
            (p[0] + 10 * math.cos(a + 2.5), p[1] + 10 * math.sin(a + 2.5)),
            (p[0] + 4 * math.cos(a + math.pi), p[1] + 4 * math.sin(a + math.pi)),
            (p[0] + 10 * math.cos(a - 2.5), p[1] + 10 * math.sin(a - 2.5)),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_laser(self):
        start_pos = self.player["pos"] + np.array([math.cos(self.player["angle"]), math.sin(self.player["angle"])]) * 12
        end_pos = self.laser_hit_pos if self.laser_hit_pos is not None else start_pos + np.array([math.cos(self.player["angle"]), math.sin(self.player["angle"])]) * 1000
        
        pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)
        pygame.gfxdraw.filled_circle(self.screen, int(start_pos[0]), int(start_pos[1]), 3, self.COLOR_LASER)
        if self.laser_hit_pos is not None:
             pygame.gfxdraw.filled_circle(self.screen, int(end_pos[0]), int(end_pos[1]), 4, self.COLOR_LASER)

    def _render_fx_particles(self):
        for p in self.fx_particles:
            alpha = int(255 * (p["life"] / 20))
            color = (*p["color"], alpha)
            size = int(p["life"] / 5)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            ship_points = [
                (self.WIDTH - 25 - i * 25, 15),
                (self.WIDTH - 35 - i * 25, 30),
                (self.WIDTH - 15 - i * 25, 30),
            ]
            pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_PLAYER)
        
        if self.game_over:
            msg = "MISSION COMPLETE" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    # --- Gymnasium Helpers ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    # --- Validation ---

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Set up Pygame window for human interaction
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()
    pygame.quit()