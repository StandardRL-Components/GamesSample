
# Generated: 2025-08-28T06:23:57.170067
# Source Brief: brief_02907.md
# Brief Index: 2907

        
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
        "Controls: Hold Space to jump and navigate the asteroid field. Your goal is to survive and reach the 500m finish line."
    )

    game_description = (
        "A fast-paced, side-scrolling arcade game. Pilot your ship through a dangerous asteroid field, using precisely timed jumps to avoid collision and reach the finish line."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self._setup_constants()
        self._setup_fonts()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        self.validate_implementation()

    def _setup_constants(self):
        # Colors
        self.COLOR_BG = (10, 10, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 220, 255)
        self.COLOR_ASTEROID_1 = (140, 140, 150)
        self.COLOR_ASTEROID_2 = (200, 80, 80)
        self.COLOR_ASTEROID_3 = (80, 150, 200)
        self.COLOR_FINISH_LINE = (0, 255, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_THRUSTER = (255, 200, 100)
        self.COLOR_EXPLOSION = (255, 150, 50)
        
        # Physics & Gameplay
        self.GRAVITY = 0.4
        self.JUMP_STRENGTH = -8
        self.MAX_VEL_Y = 10
        self.PLAYER_X_POS = 120
        self.PIXELS_PER_METER = 10
        self.FINISH_DISTANCE_M = 500
        self.MAX_STEPS = 10000
        self.SAFE_HEIGHT_THRESHOLD = self.HEIGHT - 100
        
    def _setup_fonts(self):
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except pygame.error:
            self.font_large = None
            self.font_small = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = [self.PLAYER_X_POS, self.HEIGHT / 2]
        self.player_vel_y = 0
        self.player_angle = 0
        
        self.distance_traveled = 0.0
        self.world_x_offset = 0.0
        
        self.asteroids = []
        self.particles = []
        self.stars = self._generate_stars(200)
        
        self.prev_space_held = False
        
        self.base_scroll_speed = 2.0
        self.asteroid_speed_modifier = 1.0
        self.asteroid_spawn_chance = 0.02
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        terminated = False
        
        if not self.game_over:
            # --- Handle Input ---
            if space_held and not self.prev_space_held:
                self.player_vel_y = self.JUMP_STRENGTH
                # sound: player_jump.wav
                self._create_thruster_particles()
            self.prev_space_held = space_held

            # --- Update Difficulty ---
            self._update_difficulty()

            # --- Update Player ---
            self.player_vel_y += self.GRAVITY
            self.player_vel_y = np.clip(self.player_vel_y, -self.MAX_VEL_Y, self.MAX_VEL_Y)
            self.player_pos[1] += self.player_vel_y
            self.player_angle = np.clip(self.player_vel_y * 2.5, -30, 45)
            
            # Boundary checks
            if self.player_pos[1] < 0:
                self.player_pos[1] = 0
                self.player_vel_y = 0
            if self.player_pos[1] > self.HEIGHT:
                self.player_pos[1] = self.HEIGHT
                self.player_vel_y = 0

            # --- Update World ---
            scroll_speed = self.base_scroll_speed * (1 + self.steps * 0.00005)
            meters_this_step = scroll_speed / self.PIXELS_PER_METER
            self.distance_traveled += meters_this_step
            self.world_x_offset += scroll_speed
            reward += 0.1 * meters_this_step # Reward for traveling

            # --- Update Entities ---
            self._update_asteroids(scroll_speed)
            self._update_particles(scroll_speed)
            self._spawn_asteroids()

            # --- Check Collisions & Terminations ---
            asteroid_reward, collision_detected = self._check_collisions()
            reward += asteroid_reward

            if collision_detected:
                terminated = True
                self.game_over = True
                reward = -10.0
                # sound: explosion.wav
                self._create_explosion(self.player_pos)
            
            if self.distance_traveled >= self.FINISH_DISTANCE_M:
                terminated = True
                self.game_over = True
                self.win = True
                reward = 100.0
                # sound: win_fanfare.wav
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # --- Penalty for being too low ---
        if self.player_pos[1] > self.SAFE_HEIGHT_THRESHOLD:
             reward -= 0.02
        
        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_difficulty(self):
        self.asteroid_spawn_chance = min(0.1, 0.02 + self.steps * 0.00002)
        if self.steps > 0 and self.steps % 200 == 0:
            self.asteroid_speed_modifier += 0.01

    def _check_collisions(self):
        reward = 0
        collision_detected = False
        player_rect = pygame.Rect(self.player_pos[0] - 8, self.player_pos[1] - 8, 16, 16)
        
        for asteroid in self.asteroids:
            if player_rect.colliderect(asteroid["rect"]):
                collision_detected = True
                break
            
            if not asteroid["passed"] and asteroid["rect"].right < self.player_pos[0]:
                asteroid["passed"] = True
                reward += 1.0 # Reward for successfully passing an asteroid
                # sound: pass_asteroid.wav
                
        return reward, collision_detected

    # --- Spawning and Updating Entities ---

    def _generate_stars(self, n=200):
        return [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.choice([1, 2, 3]))
            for _ in range(n)
        ]

    def _spawn_asteroids(self):
        if self.np_random.random() < self.asteroid_spawn_chance:
            asteroid_type = self.np_random.choice([1, 2, 3])
            y_pos = self.np_random.integers(20, self.HEIGHT - 20)
            size = self.np_random.integers(15, 40)
            
            speed_multipliers = {1: 1.0, 2: 1.5, 3: 2.0}
            speed = speed_multipliers[asteroid_type] * self.asteroid_speed_modifier
            
            self.asteroids.append({
                "pos": [self.WIDTH + size, y_pos],
                "type": asteroid_type,
                "size": size,
                "speed": speed,
                "shape": self._generate_asteroid_shape(size),
                "rect": pygame.Rect(0, 0, 0, 0),
                "passed": False
            })

    def _generate_asteroid_shape(self, size):
        num_vertices = self.np_random.integers(7, 12)
        shape = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(0.7, 1.0) * size / 2
            shape.append((dist * math.cos(angle), dist * math.sin(angle)))
        return shape

    def _update_asteroids(self, scroll_speed):
        for asteroid in self.asteroids:
            asteroid["pos"][0] -= scroll_speed + asteroid["speed"]
            asteroid["rect"] = pygame.Rect(
                asteroid["pos"][0] - asteroid["size"] / 2,
                asteroid["pos"][1] - asteroid["size"] / 2,
                asteroid["size"],
                asteroid["size"]
            )
        self.asteroids = [a for a in self.asteroids if a["rect"].right > 0]

    def _create_thruster_particles(self):
        angle_rad = math.radians(self.player_angle + 180)
        for _ in range(10):
            offset_angle = self.np_random.uniform(-0.3, 0.3)
            vel_mag = self.np_random.uniform(2, 5)
            vel_x = vel_mag * math.cos(angle_rad + offset_angle)
            vel_y = vel_mag * math.sin(angle_rad + offset_angle)
            self.particles.append({
                "pos": [self.player_pos[0] - 10 * math.cos(angle_rad), self.player_pos[1] - 10 * math.sin(angle_rad)],
                "vel": [vel_x, vel_y],
                "life": self.np_random.integers(10, 20),
                "color": self.COLOR_THRUSTER,
                "size": self.np_random.uniform(2, 5)
            })

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(1, 8)
            vel_x = vel_mag * math.cos(angle)
            vel_y = vel_mag * math.sin(angle)
            self.particles.append({
                "pos": list(pos),
                "vel": [vel_x, vel_y],
                "life": self.np_random.integers(20, 40),
                "color": self.np_random.choice([self.COLOR_EXPLOSION, self.COLOR_THRUSTER, self.COLOR_PLAYER]),
                "size": self.np_random.uniform(2, 6)
            })

    def _update_particles(self, scroll_speed):
        for p in self.particles:
            p["pos"][0] += p["vel"][0] - scroll_speed
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]
        
    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_finish_line()
        self._render_particles()
        self._render_asteroids()
        if not (self.game_over and not self.win):
            self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, size in self.stars:
            # Parallax scrolling
            mod_x = (x - self.world_x_offset / (size * 2)) % self.WIDTH
            color_val = 50 * size
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(mod_x), int(y)), size / 2)

    def _render_finish_line(self):
        finish_x_world = self.FINISH_DISTANCE_M * self.PIXELS_PER_METER
        finish_x_screen = finish_x_world - self.world_x_offset
        if -self.HEIGHT < finish_x_screen < self.WIDTH:
            for y in range(0, self.HEIGHT, 20):
                color = self.COLOR_FINISH_LINE if (y // 20) % 2 == 0 else (0, 150, 0)
                pygame.draw.rect(self.screen, color, (finish_x_screen, y, 10, 20))

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = [(p[0] + asteroid["pos"][0], p[1] + asteroid["pos"][1]) for p in asteroid["shape"]]
            colors = {1: self.COLOR_ASTEROID_1, 2: self.COLOR_ASTEROID_2, 3: self.COLOR_ASTEROID_3}
            color = colors[asteroid["type"]]
            darker_color = tuple(max(0, c - 40) for c in color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, darker_color)

    def _render_player(self):
        # Glow effect
        for i in range(4, 0, -1):
            glow_alpha = 50 - i * 10
            glow_color = (*self.COLOR_PLAYER_GLOW, glow_alpha)
            s = pygame.Surface((i * 10, i * 10), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (i*5, i*5), i*5)
            self.screen.blit(s, (self.player_pos[0] - i*5, self.player_pos[1] - i*5), special_flags=pygame.BLEND_RGBA_ADD)

        # Ship body
        rotated_points = self._get_rotated_player_points()
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)

    def _get_rotated_player_points(self):
        points = [(12, 0), (-8, -7), (-8, 7)]
        angle_rad = math.radians(self.player_angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        rotated_points = []
        for x, y in points:
            new_x = x * cos_a - y * sin_a + self.player_pos[0]
            new_y = x * sin_a + y * cos_a + self.player_pos[1]
            rotated_points.append((new_x, new_y))
        return rotated_points

    def _render_particles(self):
        for p in self.particles:
            alpha = p["life"] * (255 / 40)
            color_with_alpha = (*p["color"], alpha)
            s = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color_with_alpha, (p["size"], p["size"]), p["size"])
            self.screen.blit(s, (p["pos"][0] - p["size"], p["pos"][1] - p["size"]), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        if not self.font_large: return

        dist_text = self.font_large.render(f"Distance: {int(self.distance_traveled)}m / {self.FINISH_DISTANCE_M}m", True, self.COLOR_TEXT)
        self.screen.blit(dist_text, (10, 10))

        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))

        if self.game_over:
            msg = "MISSION COMPLETE" if self.win else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_FINISH_LINE if self.win else self.COLOR_ASTEROID_2)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_traveled": self.distance_traveled,
            "win": self.win,
        }

    def close(self):
        pygame.quit()

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Jumper")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        space_pressed = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space_pressed = True
        
        # Map keys to action space
        # Movement (action[0]) is unused in this game for direct control
        # Shift (action[2]) is unused
        action = [0, 1 if space_pressed else 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render the observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Distance: {info['distance_traveled']:.2f}m")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS
        
    env.close()