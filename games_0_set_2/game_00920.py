
# Generated: 2025-08-27T15:13:20.781942
# Source Brief: brief_00920.md
# Brief Index: 920

        
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
        "Controls: Arrow keys to move your ship. Hold Space to activate the mining beam on the nearest asteroid. "
        "Avoid colliding with asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship through a hazardous asteroid field. Extract ore from different types of asteroids "
        "to reach your quota, but watch your ship's integrity. The field gets denser and faster over time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width = 640
        self.height = 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # --- Game Constants ---
        self.WIN_SCORE = 100
        self.MAX_STEPS = 5000
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_ACCELERATION = 0.5
        self.PLAYER_FRICTION = 0.98
        self.PLAYER_MAX_SPEED = 6
        self.PLAYER_RADIUS = 12
        self.MINING_RANGE = 120
        self.MINING_RATE = 0.5
        self.INITIAL_ASTEROIDS = 3
        self.DIFFICULTY_INTERVAL = 200

        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_COMMON = (140, 140, 150)
        self.COLOR_RARE = (160, 110, 60)
        self.COLOR_LEGENDARY = (255, 215, 0)
        self.COLOR_ORE = (255, 255, 0)
        self.COLOR_EXPLOSION = (255, 100, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_BEAM = (255, 255, 255, 100)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.is_mining = False
        self.mining_target = None
        self.initial_spawn_rate = 0.01
        self.initial_speed_mod = 1.0
        self.spawn_rate = None
        self.speed_mod = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.width / 2, self.height / 2], dtype=np.float64)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float64)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False

        self.asteroids = []
        self.particles = []
        self.is_mining = False
        self.mining_target = None
        
        self.spawn_rate = self.initial_spawn_rate
        self.speed_mod = self.initial_speed_mod

        if not self.stars:
            for _ in range(150):
                self.stars.append(
                    (
                        self.np_random.integers(0, self.width),
                        self.np_random.integers(0, self.height),
                        self.np_random.integers(1, 3),
                    )
                )

        for _ in range(self.INITIAL_ASTEROIDS):
            self.asteroids.append(self._create_asteroid())

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Not used per design brief

        self._handle_input(movement)
        self._update_player()
        self._update_asteroids()
        
        mining_reward, depleted_bonus = self._handle_mining(space_held)
        reward += mining_reward + depleted_bonus
        
        collision_penalty = self._handle_collisions()
        reward += collision_penalty

        self._update_particles()
        self._spawn_new_asteroids()
        self._update_difficulty()

        self.steps += 1
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                self.win = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _toroidal_distance_sq(self, pos1, pos2):
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        tor_dx = min(dx, self.width - dx)
        tor_dy = min(dy, self.height - dy)
        return tor_dx**2 + tor_dy**2

    def _handle_input(self, movement):
        if movement == 1: # Up
            self.player_vel[1] -= self.PLAYER_ACCELERATION
        elif movement == 2: # Down
            self.player_vel[1] += self.PLAYER_ACCELERATION
        elif movement == 3: # Left
            self.player_vel[0] -= self.PLAYER_ACCELERATION
        elif movement == 4: # Right
            self.player_vel[0] += self.PLAYER_ACCELERATION

    def _update_player(self):
        self.player_vel = np.clip(self.player_vel, -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos[0] %= self.width
        self.player_pos[1] %= self.height

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"] * self.speed_mod
            asteroid["pos"][0] %= self.width
            asteroid["pos"][1] %= self.height

    def _handle_mining(self, space_held):
        self.is_mining = False
        self.mining_target = None
        if not space_held:
            return 0.0, 0.0

        closest_asteroid = None
        min_dist_sq = self.MINING_RANGE**2

        for asteroid in self.asteroids:
            dist_sq = self._toroidal_distance_sq(self.player_pos, asteroid["pos"])
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_asteroid = asteroid
        
        if closest_asteroid:
            self.is_mining = True
            self.mining_target = closest_asteroid
            
            mined_amount = min(self.MINING_RATE, closest_asteroid["ore"])
            closest_asteroid["ore"] -= mined_amount
            self.score += mined_amount
            
            # Create ore particles
            if self.steps % 3 == 0:
                angle_to_player = math.atan2(self.player_pos[1] - closest_asteroid["pos"][1], self.player_pos[0] - closest_asteroid["pos"][0])
                p_vel = np.array([math.cos(angle_to_player), math.sin(angle_to_player)]) * 2.5
                self.particles.append({
                    "pos": closest_asteroid["pos"].copy(),
                    "vel": p_vel + self.np_random.uniform(-0.5, 0.5, 2),
                    "life": 40, "color": self.COLOR_ORE, "size": 3
                })
            
            depleted_bonus = 0.0
            if closest_asteroid["ore"] <= 0:
                if closest_asteroid["type"] == "rare":
                    depleted_bonus = 1.0
                elif closest_asteroid["type"] == "legendary":
                    depleted_bonus = 5.0
                self.asteroids.remove(closest_asteroid)
            
            return mined_amount * 0.1, depleted_bonus
        return 0.0, 0.0

    def _handle_collisions(self):
        penalty = 0.0
        for asteroid in self.asteroids[:]:
            dist_sq = self._toroidal_distance_sq(self.player_pos, asteroid["pos"])
            if dist_sq < (self.PLAYER_RADIUS + asteroid["radius"])**2:
                damage = asteroid["radius"] # More damage from bigger asteroids
                self.player_health -= damage
                penalty -= damage * 0.1
                
                # Create explosion particles
                for _ in range(20):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                    self.particles.append({
                        "pos": self.player_pos.copy(),
                        "vel": vel,
                        "life": self.np_random.integers(20, 40),
                        "color": self.COLOR_EXPLOSION, "size": self.np_random.integers(2, 5)
                    })
                
                # sound_placeholder: play("explosion.wav")
                self.asteroids.remove(asteroid)
        return penalty

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _spawn_new_asteroids(self):
        if self.np_random.random() < self.spawn_rate:
            self.asteroids.append(self._create_asteroid(on_edge=True))

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.spawn_rate += 0.001
            self.speed_mod += 0.05
    
    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True, 100.0
        if self.player_health <= 0:
            return True, -100.0
        if self.steps >= self.MAX_STEPS:
            return True, 0.0
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_starfield()
        self._render_particles()
        self._render_asteroids()
        self._render_player()
        if self.is_mining and self.mining_target:
            self._render_mining_beam()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_starfield(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, (200, 200, 220), (x, y, size, size))

    def _render_particles(self):
        for p in self.particles:
            pos = p["pos"].astype(int)
            size = int(p["size"] * (p["life"] / 40.0))
            if size > 0:
                pygame.draw.rect(self.screen, p["color"], (pos[0] - size//2, pos[1] - size//2, size, size))

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = []
            for i in range(asteroid["shape_points"]):
                angle = 2 * math.pi * i / asteroid["shape_points"]
                rad = asteroid["shape_radii"][i]
                x = asteroid["pos"][0] + rad * math.cos(angle)
                y = asteroid["pos"][1] + rad * math.sin(angle)
                points.append((int(x), int(y)))
            pygame.gfxdraw.aapolygon(self.screen, points, asteroid["color"])
            pygame.gfxdraw.filled_polygon(self.screen, points, asteroid["color"])

    def _render_player(self):
        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2), self.PLAYER_RADIUS * 1.5)
        self.screen.blit(glow_surf, (int(self.player_pos[0] - self.PLAYER_RADIUS*2), int(self.player_pos[1] - self.PLAYER_RADIUS*2)))

        # Ship body
        angle = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else -math.pi/2
        p1 = (
            self.player_pos[0] + self.PLAYER_RADIUS * math.cos(angle),
            self.player_pos[1] + self.PLAYER_RADIUS * math.sin(angle)
        )
        p2 = (
            self.player_pos[0] + self.PLAYER_RADIUS * 0.8 * math.cos(angle + 2.5),
            self.player_pos[1] + self.PLAYER_RADIUS * 0.8 * math.sin(angle + 2.5)
        )
        p3 = (
            self.player_pos[0] + self.PLAYER_RADIUS * 0.8 * math.cos(angle - 2.5),
            self.player_pos[1] + self.PLAYER_RADIUS * 0.8 * math.sin(angle - 2.5)
        )
        points = [p1, p2, p3]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

    def _render_mining_beam(self):
        # sound_placeholder: play_loop("mining_beam.wav")
        start_pos = self.player_pos.astype(int)
        end_pos = self.mining_target["pos"].astype(int)
        for i in range(10):
            alpha = 50 + (math.sin(self.steps * 0.5 + i * 0.5) * 40)
            pygame.gfxdraw.line(self.screen, start_pos[0], start_pos[1], end_pos[0], end_pos[1], (*self.COLOR_BEAM[:3], int(alpha)))

    def _render_ui(self):
        # Ore collected
        ore_text = self.font_ui.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ore_text, (10, 10))

        # Health bar
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        health_bar_width = 150
        health_bar_height = 20
        health_bar_x = self.width - health_bar_width - 10
        health_bar_y = 10
        
        health_color = (int(255 * (1 - health_pct)), int(255 * health_pct), 0)
        pygame.draw.rect(self.screen, (50, 50, 50), (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, health_color, (health_bar_x, health_bar_y, int(health_bar_width * health_pct), health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (health_bar_x, health_bar_y, health_bar_width, health_bar_height), 1)

    def _render_game_over(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        msg = "MISSION COMPLETE" if self.win else "GAME OVER"
        text = self.font_game_over.render(msg, True, self.COLOR_LEGENDARY if self.win else self.COLOR_EXPLOSION)
        text_rect = text.get_rect(center=(self.width / 2, self.height / 2))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _create_asteroid(self, on_edge=False):
        roll = self.np_random.random()
        if roll < 0.1:
            atype, color, ore_range, rad_range, speed_range = "legendary", self.COLOR_LEGENDARY, (50, 60), (25, 35), (0.5, 1.0)
        elif roll < 0.4:
            atype, color, ore_range, rad_range, speed_range = "rare", self.COLOR_RARE, (30, 40), (20, 30), (0.8, 1.5)
        else:
            atype, color, ore_range, rad_range, speed_range = "common", self.COLOR_COMMON, (10, 20), (15, 25), (1.0, 2.0)

        radius = self.np_random.uniform(*rad_range)
        shape_points = 12
        shape_radii = [self.np_random.uniform(radius * 0.8, radius * 1.2) for _ in range(shape_points)]

        if on_edge:
            edge = self.np_random.integers(0, 4)
            if edge == 0: pos = np.array([-radius, self.np_random.uniform(0, self.height)])
            elif edge == 1: pos = np.array([self.width+radius, self.np_random.uniform(0, self.height)])
            elif edge == 2: pos = np.array([self.np_random.uniform(0, self.width), -radius])
            else: pos = np.array([self.np_random.uniform(0, self.width), self.height+radius])
        else:
            pos = self.np_random.uniform([0,0], [self.width, self.height])

        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(*speed_range)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed

        return {
            "pos": pos.astype(np.float64), "vel": vel.astype(np.float64), "radius": radius,
            "ore": self.np_random.uniform(*ore_range), "type": atype, "color": color,
            "shape_points": shape_points, "shape_radii": shape_radii
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the game directly to test it
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    terminated = False
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Asteroid Miner")
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
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()