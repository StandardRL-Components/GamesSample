
# Generated: 2025-08-27T22:04:45.785688
# Source Brief: brief_03002.md
# Brief Index: 3002

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Hold space to mine the nearest asteroid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Mine asteroids for valuable minerals while dodging hostile alien ships in a top-down space arcade."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WIN_SCORE = 50
        self.MAX_LIVES = 3
        self.MAX_STEPS = 5000
        self.INITIAL_ASTEROIDS = 10
        self.MIN_ASTEROIDS = 8
        self.INITIAL_ENEMIES = 2
        
        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ASTEROID = (120, 120, 140)
        self.COLOR_MINERAL = (255, 220, 50)
        self.COLOR_LASER = (255, 100, 255)
        self.COLOR_EXPLOSION = (255, 150, 0)
        self.COLOR_TEXT = (220, 220, 240)

        # Player settings
        self.PLAYER_ACCELERATION = 0.4
        self.PLAYER_DRAG = 0.96
        self.PLAYER_MAX_SPEED = 5
        self.PLAYER_SIZE = 12
        self.PLAYER_INVINCIBILITY_STEPS = 90  # 3 seconds at 30fps

        # Game entity settings
        self.MINING_RANGE = 150
        self.MINING_POWER = 1
        self.ENEMY_PROXIMITY_RADIUS = 100
        self.ENEMY_BASE_SPEED = 1.0

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Internal State ---
        # These are initialized in reset()
        self.steps = 0
        self.mineral_count = 0
        self.player_lives = 0
        self.game_over = False
        self.player = {}
        self.enemies = []
        self.asteroids = []
        self.minerals = []
        self.particles = []
        self.stars = []
        self.mining_target = None
        self.last_damage_step = -self.PLAYER_INVINCIBILITY_STEPS

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.mineral_count = 0
        self.player_lives = self.MAX_LIVES
        self.game_over = False
        self.last_damage_step = -self.PLAYER_INVINCIBILITY_STEPS

        self.player = {
            "pos": pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2),
            "vel": pygame.Vector2(0, 0),
            "angle": -90,
        }

        self.stars = [
            (
                self.np_random.integers(0, self.SCREEN_WIDTH),
                self.np_random.integers(0, self.SCREEN_HEIGHT),
                self.np_random.uniform(0.5, 1.5),
            )
            for _ in range(150)
        ]

        self.enemies = []
        for _ in range(self.INITIAL_ENEMIES):
            self._spawn_enemy()

        self.asteroids = []
        for _ in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid()

        self.minerals = []
        self.particles = []
        self.mining_target = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        self._handle_input(action)
        self._update_player()
        self._update_enemies()
        self._update_asteroids()
        self._update_minerals()
        self._update_particles()
        
        mining_reward = self._handle_mining(action)
        collision_reward = self._handle_collisions()
        proximity_penalty = self._calculate_proximity_penalty()
        
        reward += mining_reward + collision_reward + proximity_penalty
        
        self._respawn_logic()
        self._update_difficulty()

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.mineral_count >= self.WIN_SCORE:
                reward += 100
            elif self.player_lives <= 0:
                reward -= 100
        
        # Small penalty for existing to encourage efficiency
        reward -= 0.001

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        accel = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            accel.y = -1
        elif movement == 2:  # Down
            accel.y = 1
        elif movement == 3:  # Left
            accel.x = -1
        elif movement == 4:  # Right
            accel.x = 1

        if accel.length() > 0:
            accel.scale_to_length(self.PLAYER_ACCELERATION)
            self.player["vel"] += accel
            self.player["angle"] = math.degrees(math.atan2(-accel.y, accel.x))


    def _update_player(self):
        # Apply drag and clamp speed
        self.player["vel"] *= self.PLAYER_DRAG
        if self.player["vel"].length() > self.PLAYER_MAX_SPEED:
            self.player["vel"].scale_to_length(self.PLAYER_MAX_SPEED)

        self.player["pos"] += self.player["vel"]
        self._wrap_position(self.player["pos"])

    def _update_enemies(self):
        speed_multiplier = 1.0 + (self.ENEMY_BASE_SPEED * (self.steps // 1000) * 0.05)
        for enemy in self.enemies:
            if enemy["pattern"] == "patrol":
                direction = (enemy["target"] - enemy["pos"])
                if direction.length() < 5:
                    enemy["target"] = self._get_random_pos_on_edge()
                else:
                    direction.scale_to_length(self.ENEMY_BASE_SPEED * speed_multiplier)
                    enemy["pos"] += direction
            elif enemy["pattern"] == "circle":
                enemy["angle"] += enemy["angular_vel"] * speed_multiplier
                enemy["pos"].x = enemy["center"].x + math.cos(enemy["angle"]) * enemy["radius"]
                enemy["pos"].y = enemy["center"].y + math.sin(enemy["angle"]) * enemy["radius"]
            
            self._wrap_position(enemy["pos"])


    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid["angle"] += asteroid["rot_speed"]

    def _update_minerals(self):
        for mineral in self.minerals:
            mineral["pos"] += mineral["vel"]
            mineral["vel"] *= 0.98 # Slow down
            self._wrap_position(mineral["pos"])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

    def _handle_mining(self, action):
        _, space_held, _ = action
        reward = 0
        self.mining_target = None

        if space_held and self.asteroids:
            # Find closest asteroid
            closest_asteroid = None
            min_dist_sq = self.MINING_RANGE ** 2
            
            for asteroid in self.asteroids:
                dist_sq = self.player["pos"].distance_squared_to(asteroid["pos"])
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_asteroid = asteroid
            
            if closest_asteroid:
                self.mining_target = closest_asteroid
                closest_asteroid["health"] -= self.MINING_POWER
                # sfx: mining_laser_loop.wav

                if closest_asteroid["health"] <= 0:
                    reward += 1.0 # Reward for destroying asteroid
                    self._create_explosion(closest_asteroid["pos"], self.COLOR_EXPLOSION, 30)
                    for _ in range(closest_asteroid["value"]):
                        self._spawn_mineral(closest_asteroid["pos"])
                    self.asteroids.remove(closest_asteroid)
                    self.mining_target = None
                    # sfx: explosion_medium.wav
        return reward

    def _handle_collisions(self):
        reward = 0
        # Player vs Enemy
        is_invincible = self.steps - self.last_damage_step < self.PLAYER_INVINCIBILITY_STEPS
        if not is_invincible:
            for enemy in self.enemies:
                if self.player["pos"].distance_to(enemy["pos"]) < self.PLAYER_SIZE + enemy["size"]:
                    self.player_lives -= 1
                    self.last_damage_step = self.steps
                    self._create_explosion(self.player["pos"], self.COLOR_PLAYER, 40)
                    self.player["pos"] = pygame.Vector2(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)
                    self.player["vel"] = pygame.Vector2(0,0)
                    # sfx: player_hit.wav
                    break
        
        # Player vs Mineral
        for mineral in self.minerals[:]:
            if self.player["pos"].distance_to(mineral["pos"]) < self.PLAYER_SIZE + 5:
                self.minerals.remove(mineral)
                self.mineral_count += 1
                reward += 0.1 # Reward for collecting mineral
                # sfx: collect_mineral.wav
        
        return reward

    def _calculate_proximity_penalty(self):
        for enemy in self.enemies:
            if self.player["pos"].distance_to(enemy["pos"]) < self.ENEMY_PROXIMITY_RADIUS:
                return -0.01
        return 0

    def _respawn_logic(self):
        if len(self.asteroids) < self.MIN_ASTEROIDS and self.np_random.random() < 0.02:
            self._spawn_asteroid()

    def _update_difficulty(self):
        # Handled inside _update_enemies via speed multiplier
        pass

    def _check_termination(self):
        if self.player_lives <= 0:
            return True
        if self.mineral_count >= self.WIN_SCORE:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_minerals()
        self._render_asteroids()
        self._render_enemies()
        self._render_player()
        self._render_laser()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.mineral_count,
            "steps": self.steps,
            "lives": self.player_lives
        }

    # --- Spawning Methods ---

    def _spawn_asteroid(self):
        pos = self._get_random_pos_on_edge()
        size = self.np_random.uniform(15, 30)
        num_vertices = self.np_random.integers(7, 12)
        base_shape = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(0.7, 1.1) * size
            base_shape.append(pygame.Vector2(math.cos(angle) * dist, math.sin(angle) * dist))
        
        self.asteroids.append({
            "pos": pos,
            "size": size,
            "angle": self.np_random.uniform(0, 360),
            "rot_speed": self.np_random.uniform(-0.5, 0.5),
            "health": size * 2,
            "value": self.np_random.integers(1, 6),
            "base_shape": base_shape
        })

    def _spawn_enemy(self):
        pos = self._get_random_pos_on_edge()
        pattern = self.np_random.choice(["patrol", "circle"])
        enemy = {
            "pos": pos,
            "size": 10,
            "pattern": pattern
        }
        if pattern == "patrol":
            enemy["target"] = self._get_random_pos_on_edge()
        else: # circle
            enemy["center"] = pygame.Vector2(self.np_random.uniform(100, self.SCREEN_WIDTH-100), self.np_random.uniform(100, self.SCREEN_HEIGHT-100))
            enemy["radius"] = self.np_random.uniform(50, 150)
            enemy["angle"] = self.np_random.uniform(0, 2 * math.pi)
            enemy["angular_vel"] = self.np_random.uniform(0.01, 0.03) * self.np_random.choice([-1, 1])
        self.enemies.append(enemy)

    def _spawn_mineral(self, pos):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 3)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.minerals.append({
            "pos": pos.copy(),
            "vel": vel
        })

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color
            })

    # --- Rendering Methods ---

    def _render_stars(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 255), (x, y), size)

    def _render_player(self):
        is_invincible = self.steps - self.last_damage_step < self.PLAYER_INVINCIBILITY_STEPS
        if is_invincible and self.steps % 10 < 5:
            return # Blink effect

        angle_rad = math.radians(self.player["angle"])
        p1 = self.player["pos"] + pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) * self.PLAYER_SIZE
        p2 = self.player["pos"] + pygame.Vector2(math.cos(angle_rad + 2.2), -math.sin(angle_rad + 2.2)) * self.PLAYER_SIZE
        p3 = self.player["pos"] + pygame.Vector2(math.cos(angle_rad - 2.2), -math.sin(angle_rad - 2.2)) * self.PLAYER_SIZE
        
        points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_enemies(self):
        for enemy in self.enemies:
            p1 = enemy["pos"] + pygame.Vector2(0, -enemy["size"])
            p2 = enemy["pos"] + pygame.Vector2(-enemy["size"]*0.866, enemy["size"]*0.5)
            p3 = enemy["pos"] + pygame.Vector2(enemy["size"]*0.866, enemy["size"]*0.5)
            points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            angle_rad = math.radians(asteroid["angle"])
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            points = []
            for p in asteroid["base_shape"]:
                x = p.x * cos_a - p.y * sin_a + asteroid["pos"].x
                y = p.x * sin_a + p.y * cos_a + asteroid["pos"].y
                points.append((x, y))
            
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_minerals(self):
        for mineral in self.minerals:
            pos = (int(mineral["pos"].x), int(mineral["pos"].y))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, (*self.COLOR_MINERAL, 50))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_MINERAL)

    def _render_laser(self):
        if self.mining_target:
            start_pos = (int(self.player["pos"].x), int(self.player["pos"].y))
            end_pos = (int(self.mining_target["pos"].x), int(self.mining_target["pos"].y))
            width = self.np_random.integers(1, 4)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, width)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p["lifespan"] / 6))
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y)), size)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"MINERALS: {self.mineral_count}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_small.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 130, 10))
        for i in range(self.player_lives):
            pos = (self.SCREEN_WIDTH - 70 + i * 20, 18)
            points = [
                (pos[0], pos[1] - 8),
                (pos[0] - 6, pos[1] + 4),
                (pos[0] + 6, pos[1] + 4)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg = "YOU WIN!" if self.mineral_count >= self.WIN_SCORE else "GAME OVER"
        text = self.font_large.render(msg, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    # --- Utility Methods ---

    def _wrap_position(self, pos):
        pos.x = pos.x % self.SCREEN_WIDTH
        pos.y = pos.y % self.SCREEN_HEIGHT

    def _get_random_pos_on_edge(self):
        side = self.np_random.integers(4)
        if side == 0: # top
            return pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -30)
        elif side == 1: # bottom
            return pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 30)
        elif side == 2: # left
            return pygame.Vector2(-30, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else: # right
            return pygame.Vector2(self.SCREEN_WIDTH + 30, self.np_random.uniform(0, self.SCREEN_HEIGHT))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This requires a window, so we'll re-init pygame for display
    pygame.display.init()
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS

    env.close()
    pygame.quit()