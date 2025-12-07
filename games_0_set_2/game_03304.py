
# Generated: 2025-08-27T22:57:26.904650
# Source Brief: brief_03304.md
# Brief Index: 3304

        
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
        "Controls: Use arrow keys (↑↓←→) to move your ship. "
        "Hold Space to activate your mining laser on nearby asteroids. Avoid collisions!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship through an asteroid field, collecting valuable ore while avoiding collisions. "
        "Collect 50 ore to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SIZE = 12
    PLAYER_SPEED = 4
    PLAYER_HEALTH_MAX = 100
    ASTEROID_MIN_SIZE, ASTEROID_MAX_SIZE = 15, 35
    ASTEROID_MIN_ORE, ASTEROID_MAX_ORE = 1, 5
    INITIAL_ASTEROID_COUNT = 15
    MAX_ASTEROIDS = 40
    ASTEROID_SPAWN_RATE = 50  # steps
    DIFFICULTY_INTERVAL = 100  # steps
    DIFFICULTY_ASTEROID_INCREASE = 1
    WIN_SCORE = 50
    MAX_STEPS = 2000
    MINING_RANGE = 90
    MINING_RATE = 0.1 # ore per step
    COLLISION_DAMAGE = 20

    # --- Colors ---
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128)
    COLOR_ASTEROID = (120, 120, 120)
    COLOR_ASTEROID_OUTLINE = (160, 160, 160)
    COLOR_ORE = (255, 220, 0)
    COLOR_LASER = (255, 255, 0)
    COLOR_TEXT = (220, 220, 255)
    COLOR_HEALTH_BAR = (0, 200, 80)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        self.player_pos = None
        self.player_health = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.asteroids = []
        self.particles = []
        self.mining_target = None
        self.mining_progress = 0
        self.starfield = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.asteroids.clear()
        self.particles.clear()
        self.mining_target = None
        self.mining_progress = 0

        self._generate_starfield()
        self._spawn_initial_asteroids()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Time penalty

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._update_player(movement)
        self._update_asteroids()
        self._update_particles()
        
        reward += self._handle_mining(space_held)
        reward += self._handle_collisions()

        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.player_health <= 0:
                reward += -100 # Loss penalty
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_starfield()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health}

    def _update_player(self, movement):
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED # Right

        # World wrapping
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT
        
        # Thruster particles
        if movement != 0:
            for _ in range(2):
                offset = np.array([random.uniform(-5, 5), random.uniform(-5, 5)])
                vel = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
                self.particles.append({
                    "pos": self.player_pos.copy() + offset, "vel": vel,
                    "color": (100, 100, 255), "lifespan": 10, "size": random.randint(1, 3)
                })

    def _update_asteroids(self):
        num_asteroids_target = min(self.MAX_ASTEROIDS, self.INITIAL_ASTEROID_COUNT + (self.steps // self.DIFFICULTY_INTERVAL) * self.DIFFICULTY_ASTEROID_INCREASE)

        if self.steps % self.ASTEROID_SPAWN_RATE == 0 and len(self.asteroids) < num_asteroids_target:
            self._spawn_asteroid()

        for asteroid in self.asteroids:
            asteroid["angle"] += asteroid["rotation_speed"]

    def _handle_mining(self, space_held):
        self.mining_target = None
        reward = 0
        if not space_held:
            self.mining_progress = 0
            return reward

        closest_asteroid = None
        min_dist = float('inf')

        for asteroid in self.asteroids:
            if asteroid["ore"] > 0:
                dist = np.linalg.norm(self.player_pos - asteroid["pos"])
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid
        
        if closest_asteroid and min_dist < self.MINING_RANGE:
            # sfx: mining_laser_start
            self.mining_target = closest_asteroid
            self.mining_progress += self.MINING_RATE
            
            if self.mining_progress >= 1:
                ore_mined = int(self.mining_progress)
                self.mining_progress -= ore_mined
                
                actual_mined = min(ore_mined, closest_asteroid["ore"])
                if actual_mined > 0:
                    # sfx: collect_ore
                    closest_asteroid["ore"] -= actual_mined
                    self.score += actual_mined
                    reward += 0.1 * actual_mined

                    for _ in range(actual_mined * 3):
                        self._create_ore_particle(closest_asteroid["pos"])

                    if closest_asteroid["ore"] <= 0:
                        reward += 1 # Bonus for depleting an asteroid
        else:
            self.mining_progress = 0
            # sfx: mining_laser_stop

        return reward

    def _handle_collisions(self):
        reward = 0
        asteroids_to_remove = []
        for i, asteroid in enumerate(self.asteroids):
            dist = np.linalg.norm(self.player_pos - asteroid["pos"])
            if dist < self.PLAYER_SIZE + asteroid["size"]:
                # sfx: ship_hit
                self.player_health -= self.COLLISION_DAMAGE
                self.player_health = max(0, self.player_health)
                asteroids_to_remove.append(i)
                self._create_explosion(self.player_pos)
        
        # Remove collided asteroids from back to front
        for i in sorted(asteroids_to_remove, reverse=True):
            self.asteroids.pop(i)
            
        return reward

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _check_termination(self):
        return self.player_health <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _render_game(self):
        self._draw_asteroids()
        self._draw_particles()
        if self.player_health > 0:
            self._draw_player()
            if self.mining_target:
                self._draw_laser()

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health Bar
        health_percent = self.player_health / self.PLAYER_HEALTH_MAX
        bar_width = 150
        bar_height = 15
        health_bar_rect = pygame.Rect(self.WIDTH - bar_width - 10, 10, bar_width, bar_height)
        current_health_width = int(bar_width * health_percent)
        current_health_rect = pygame.Rect(self.WIDTH - bar_width - 10, 10, current_health_width, bar_height)

        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, current_health_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, health_bar_rect, 1)

        if self.game_over:
            message = "MISSION COMPLETE" if self.score >= self.WIN_SCORE else "SHIP DESTROYED"
            text_surface = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surface, text_rect)

    def _generate_starfield(self):
        self.starfield.clear()
        for _ in range(200):
            self.starfield.append({
                "pos": (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                "size": self.np_random.integers(1, 3),
                "brightness": self.np_random.integers(50, 150)
            })

    def _draw_starfield(self):
        for star in self.starfield:
            c = star["brightness"]
            pygame.draw.circle(self.screen, (c, c, c), star["pos"], star["size"])

    def _spawn_initial_asteroids(self):
        for _ in range(self.INITIAL_ASTEROID_COUNT):
            self._spawn_asteroid()

    def _spawn_asteroid(self):
        while True:
            pos = self.np_random.uniform(low=0, high=[self.WIDTH, self.HEIGHT])
            if np.linalg.norm(pos - self.player_pos) > self.MINING_RANGE: # Don't spawn on player
                break
        
        size = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        ore = self.np_random.integers(self.ASTEROID_MIN_ORE, self.ASTEROID_MAX_ORE + 1)
        angle = self.np_random.uniform(0, 2 * math.pi)
        rotation_speed = self.np_random.uniform(-0.02, 0.02)
        
        num_vertices = self.np_random.integers(7, 12)
        vertices = []
        for i in range(num_vertices):
            a = 2 * math.pi * i / num_vertices
            r = size + self.np_random.uniform(-size * 0.2, size * 0.2)
            vertices.append((r * math.cos(a), r * math.sin(a)))

        self.asteroids.append({
            "pos": pos, "size": size, "ore": ore, "angle": angle,
            "rotation_speed": rotation_speed, "base_vertices": vertices
        })

    def _draw_player(self):
        # Glow effect
        glow_size = self.PLAYER_SIZE * 2.5
        glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_PLAYER_GLOW, 30), (glow_size, glow_size), glow_size)
        pygame.draw.circle(glow_surface, (*self.COLOR_PLAYER_GLOW, 50), (glow_size, glow_size), glow_size * 0.6)
        self.screen.blit(glow_surface, (int(self.player_pos[0] - glow_size), int(self.player_pos[1] - glow_size)), special_flags=pygame.BLEND_RGBA_ADD)

        # Ship body
        points = [
            (self.player_pos[0], self.player_pos[1] - self.PLAYER_SIZE),
            (self.player_pos[0] - self.PLAYER_SIZE / 1.5, self.player_pos[1] + self.PLAYER_SIZE / 2),
            (self.player_pos[0] + self.PLAYER_SIZE / 1.5, self.player_pos[1] + self.PLAYER_SIZE / 2),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _draw_asteroids(self):
        for asteroid in self.asteroids:
            rotated_vertices = []
            for vx, vy in asteroid["base_vertices"]:
                rot_x = vx * math.cos(asteroid["angle"]) - vy * math.sin(asteroid["angle"])
                rot_y = vx * math.sin(asteroid["angle"]) + vy * math.cos(asteroid["angle"])
                rotated_vertices.append((int(rot_x + asteroid["pos"][0]), int(rot_y + asteroid["pos"][1])))
            
            if len(rotated_vertices) > 2:
                pygame.gfxdraw.aapolygon(self.screen, rotated_vertices, self.COLOR_ASTEROID_OUTLINE)
                pygame.gfxdraw.filled_polygon(self.screen, rotated_vertices, self.COLOR_ASTEROID)

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 10))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_laser(self):
        target_pos = self.mining_target["pos"]
        # sfx: mining_laser_loop
        pygame.draw.aaline(self.screen, self.COLOR_LASER, self.player_pos, target_pos, 1)
        # Add a thicker, more transparent line for a glow effect
        pygame.draw.line(self.screen, (*self.COLOR_LASER, 100), self.player_pos, target_pos, 4)

    def _create_ore_particle(self, asteroid_pos):
        direction = self.player_pos - asteroid_pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction /= dist
        
        vel = direction * 3 + self.np_random.uniform(-0.5, 0.5, 2)
        self.particles.append({
            "pos": asteroid_pos.copy() + self.np_random.uniform(-5, 5, 2),
            "vel": vel, "color": self.COLOR_ORE, "lifespan": 30, "size": random.randint(2, 4)
        })

    def _create_explosion(self, pos):
        # sfx: explosion
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            color = random.choice([(255, 100, 0), (255, 200, 0), (200, 50, 0)])
            self.particles.append({
                "pos": pos.copy() + self.np_random.uniform(-10, 10, 2),
                "vel": vel, "color": color, "lifespan": random.randint(15, 30), "size": random.randint(2, 5)
            })

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and visualize the environment
    env = GameEnv(render_mode="rgb_array")
    
    # Override screen to be a display surface
    env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")

    obs, info = env.reset()
    done = False
    
    # --- Human Controls ---
    # Map keyboard keys to the MultiDiscrete action space
    key_to_action = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4
    }
    
    print(env.user_guide)

    while not done:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break # Prioritize first key in dict order if multiple are pressed
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation from the environment to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if done:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Wait 3 seconds before closing
            
    pygame.quit()