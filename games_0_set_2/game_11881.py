import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:44:21.544945
# Source Brief: brief_01881.md
# Brief Index: 1881
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a drone collects minerals while dodging lasers.
    
    The agent controls a drone with the goal of collecting 15 units of three
    different mineral types. The environment is populated with oscillating laser
    turrets that pose a threat. The drone has a limited shield capacity, and
    the mission is timed.

    The visual style is a clean, high-contrast, retro-arcade look with smooth
    animations and particle effects for clear feedback.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Collect different types of minerals with your drone while dodging oscillating laser turrets."
    user_guide = "Use the arrow keys to move your drone. Press space when near a mineral deposit to collect it."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For physics updates, not rendering speed
        self.TIME_LIMIT_SECONDS = 180
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        self.TARGET_MINERAL_COUNT = 15
        self.INITIAL_SHIELDS = 5
        self.NUM_MINERAL_TYPES = 3
        self.NUM_MINERAL_DEPOSITS_PER_TYPE = 8
        self.NUM_TURRETS = 6

        # Player properties
        self.PLAYER_ACCELERATION = 1.2
        self.PLAYER_FRICTION = 0.90
        self.PLAYER_MAX_SPEED = 10
        self.PLAYER_RADIUS = 10
        self.PLAYER_INVINCIBILITY_FRAMES = 60 # 2 seconds

        # Turret properties
        self.TURRET_LASER_LENGTH = 150
        self.TURRET_LASER_WIDTH = 3

        # Colors
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_LASER = (255, 20, 20)
        self.COLOR_LASER_GLOW = (255, 100, 100)
        self.COLOR_TURRET = (100, 110, 120)
        self.MINERAL_COLORS = [
            (50, 255, 50),   # Green
            (50, 150, 255),  # Blue
            (255, 255, 50)   # Yellow
        ]
        self.UI_COLOR = (220, 220, 240)
        self.UI_SUCCESS = (100, 255, 100)
        self.UI_FAIL = (255, 100, 100)
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- State Variables ---
        # These are initialized properly in reset()
        self.player_pos = None
        self.player_vel = None
        self.minerals = None
        self.turrets = None
        self.particles = None
        self.mineral_counts = None
        self.shields = None
        self.invincibility_timer = None
        self.time_remaining = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_needed_mineral_dist = None
        self.last_reward = 0.0

        # self.reset() is called by the wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.invincibility_timer = 0
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shields = self.INITIAL_SHIELDS
        self.mineral_counts = [0] * self.NUM_MINERAL_TYPES
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.last_reward = 0.0
        
        # Procedural generation
        self._generate_world()
        self.particles = []
        
        # Reward state
        self.last_needed_mineral_dist = self._get_closest_needed_mineral_dist()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game State ---
        self._handle_player_movement(movement)
        self._update_turrets()
        self._update_particles()
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        
        # --- Handle Interactions & Rewards ---
        reward = 0
        
        # Mineral collection
        if space_held:
            collected_mineral, mineral_type = self._check_mineral_collection()
            if collected_mineral:
                # SFX: Mineral collect
                reward += 1.0
                if self.mineral_counts[mineral_type] == self.TARGET_MINERAL_COUNT - 1:
                    reward += 5.0 # Bonus for completing a type
                self.mineral_counts[mineral_type] = min(self.TARGET_MINERAL_COUNT, self.mineral_counts[mineral_type] + 1)
        
        # Turret collision
        if self.invincibility_timer == 0 and self._check_turret_collision():
            # SFX: Player hit
            self.shields -= 1
            self.invincibility_timer = self.PLAYER_INVINCIBILITY_FRAMES
            reward -= 10.0
            self._spawn_particles(self.player_pos, (255, 50, 50), 30, 4.0)
        
        # Proximity reward
        dist = self._get_closest_needed_mineral_dist()
        if dist < self.last_needed_mineral_dist:
            reward += 0.1 # Small reward for getting closer
        self.last_needed_mineral_dist = dist

        # --- Check Termination ---
        terminated = False
        win = all(count >= self.TARGET_MINERAL_COUNT for count in self.mineral_counts)
        
        if win:
            # SFX: Mission success
            reward += 100.0
            terminated = True
        elif self.time_remaining <= 0 or self.shields <= 0 or self.steps >= self.MAX_STEPS:
            # SFX: Mission fail
            reward -= 100.0
            terminated = True
            
        self.game_over = terminated
        self.score += reward
        self.last_reward = reward

        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_world(self):
        self.minerals = []
        for type_idx in range(self.NUM_MINERAL_TYPES):
            for _ in range(self.NUM_MINERAL_DEPOSITS_PER_TYPE):
                self.minerals.append({
                    "pos": self.np_random.uniform(low=[50, 50], high=[self.WIDTH - 50, self.HEIGHT - 50]),
                    "type": type_idx,
                    "size": 8,
                    "color": self.MINERAL_COLORS[type_idx]
                })

        self.turrets = []
        for _ in range(self.NUM_TURRETS):
            self.turrets.append({
                "pos": self.np_random.uniform(low=[30, 30], high=[self.WIDTH - 30, self.HEIGHT - 30]),
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "speed": self.np_random.uniform(0.01, 0.03) * self.np_random.choice([-1, 1])
            })
    
    def _handle_player_movement(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        accel = np.array([0.0, 0.0])
        if movement == 1: accel[1] = -self.PLAYER_ACCELERATION
        elif movement == 2: accel[1] = self.PLAYER_ACCELERATION
        elif movement == 3: accel[0] = -self.PLAYER_ACCELERATION
        elif movement == 4: accel[0] = self.PLAYER_ACCELERATION
        
        self.player_vel += accel
        self.player_vel *= self.PLAYER_FRICTION
        
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED

        self.player_pos += self.player_vel
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _update_turrets(self):
        for turret in self.turrets:
            turret["angle"] += turret["speed"]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] = max(0, p["size"] - 0.1)

    def _check_mineral_collection(self):
        collect_radius = self.PLAYER_RADIUS + 20 # Generous collection range
        for i, mineral in reversed(list(enumerate(self.minerals))):
            if self.mineral_counts[mineral["type"]] < self.TARGET_MINERAL_COUNT:
                dist_sq = np.sum((self.player_pos - mineral["pos"])**2)
                if dist_sq < collect_radius**2:
                    self._spawn_particles(mineral["pos"], mineral["color"], 20, 3.0)
                    mineral_type = mineral["type"]
                    self.minerals.pop(i)
                    return True, mineral_type
        return False, -1

    def _check_turret_collision(self):
        for turret in self.turrets:
            laser_start = turret["pos"]
            laser_end = turret["pos"] + self.TURRET_LASER_LENGTH * np.array([math.cos(turret["angle"]), math.sin(turret["angle"])])
            
            # Line segment to point distance check
            d_sq = self._point_segment_dist_sq(self.player_pos, laser_start, laser_end)
            if d_sq < self.PLAYER_RADIUS**2:
                return True
        return False

    def _get_closest_needed_mineral_dist(self):
        needed_minerals = [m for m in self.minerals if self.mineral_counts[m["type"]] < self.TARGET_MINERAL_COUNT]
        if not needed_minerals:
            return 0
        
        distances = [np.linalg.norm(self.player_pos - m["pos"]) for m in needed_minerals]
        return min(distances) if distances else float('inf')

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render minerals
        for mineral in self.minerals:
            pos = mineral["pos"].astype(int)
            size = int(mineral["size"])
            color = mineral["color"]
            pygame.draw.rect(self.screen, color, (pos[0] - size, pos[1] - size, size*2, size*2))

        # Render turrets and lasers
        for turret in self.turrets:
            pos = turret["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_TURRET)
            
            end_pos = turret["pos"] + self.TURRET_LASER_LENGTH * np.array([math.cos(turret["angle"]), math.sin(turret["angle"])])
            pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, pos, end_pos.astype(int), self.TURRET_LASER_WIDTH + 3)
            pygame.draw.line(self.screen, self.COLOR_LASER, pos, end_pos.astype(int), self.TURRET_LASER_WIDTH)

        # Render particles
        for p in self.particles:
            pos = p["pos"].astype(int)
            size = int(p["size"])
            if size > 0:
                alpha = int(255 * (p["life"] / p["max_life"]))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (pos[0] - size, pos[1] - size))

        # Render player
        is_invincible_flash = self.invincibility_timer > 0 and (self.steps // 3) % 2 == 0
        if not is_invincible_flash:
            player_pos_int = self.player_pos.astype(int)
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], 
                                          self.PLAYER_RADIUS + 5, (*self.COLOR_PLAYER_GLOW, 50))
            # Player triangle
            angle = math.atan2(self.player_vel[1], self.player_vel[0]) if np.any(self.player_vel) else -math.pi/2
            points = []
            for i in range(3):
                a = angle + i * 2 * math.pi / 3
                p = self.player_pos + np.array([math.cos(a), math.sin(a)]) * self.PLAYER_RADIUS
                points.append(p.astype(int))
            pygame.gfxdraw.filled_trigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_PLAYER)
            pygame.gfxdraw.aatrigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_PLAYER)

    def _render_ui(self):
        # Mineral counts
        y_offset = 10
        for i in range(self.NUM_MINERAL_TYPES):
            text = f"{self.mineral_counts[i]}/{self.TARGET_MINERAL_COUNT}"
            color = self.MINERAL_COLORS[i]
            surf = self.font_main.render(text, True, self.UI_COLOR)
            pygame.draw.rect(self.screen, color, (10, y_offset, 15, 15))
            self.screen.blit(surf, (30, y_offset - 2))
            y_offset += 25
            
        # Timer
        mins, secs = divmod(max(0, self.time_remaining), 60)
        timer_text = f"TIME {int(mins):02}:{int(secs):02}"
        timer_surf = self.font_main.render(timer_text, True, self.UI_COLOR)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))
        
        # Shields
        shield_text = "SHIELDS"
        shield_surf = self.font_main.render(shield_text, True, self.UI_COLOR)
        text_w = shield_surf.get_width()
        start_x = (self.WIDTH - text_w - (self.shields * 20)) / 2
        self.screen.blit(shield_surf, (start_x, self.HEIGHT - 30))
        for i in range(self.shields):
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (start_x + text_w + 10 + i*20, self.HEIGHT - 28, 15, 15))

        # Game Over Message
        if self.game_over:
            win = all(count >= self.TARGET_MINERAL_COUNT for count in self.mineral_counts)
            message = "MISSION COMPLETE" if win else "MISSION FAILED"
            color = self.UI_SUCCESS if win else self.UI_FAIL
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "shields": self.shields,
            "time_remaining": self.time_remaining,
            "mineral_counts": self.mineral_counts,
            "last_reward": self.last_reward
        }

    def _spawn_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color,
                "size": self.np_random.uniform(3, 6)
            })

    def _point_segment_dist_sq(self, p, a, b):
        l2 = np.sum((a - b)**2)
        if l2 == 0.0: return np.sum((p - a)**2)
        t = max(0, min(1, np.dot(p - a, b - a) / l2))
        projection = a + t * (b - a)
        return np.sum((p - projection)**2)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will not work in a headless environment
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_over_display_timer = 150 # Frames to show game over screen

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Mineral Collector Drone")
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation from the environment to the display screen
        # Need to transpose it back to Pygame's (width, height, channels) format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            game_over_display_timer -= 1
            if game_over_display_timer <= 0:
                obs, info = env.reset()
                game_over_display_timer = 150
        
        clock.tick(env.FPS)
        
    env.close()