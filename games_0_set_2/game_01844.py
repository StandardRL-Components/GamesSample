import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your character. "
        "Collect gems to score points and avoid the red obstacles."
    )

    game_description = (
        "Race against time to collect gems in an isometric world. "
        "Dodge moving obstacles, which get faster over time. "
        "Collecting gems near obstacles grants bonus points."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds * 30 FPS

    # Grid dimensions
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    TILE_WIDTH_ISO, TILE_HEIGHT_ISO = 48, 24
    ORIGIN_X, ORIGIN_Y = WIDTH // 2, 80

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 50)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_TEXT = (240, 240, 240)
    GEM_COLORS = [
        (0, 200, 255),  # Cyan
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 128, 0),  # Orange
    ]

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_pixel_pos = None
        self.gems = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.time_remaining = None
        self.gems_collected = None
        self.last_dist_to_gem = None
        self.gem_respawn_cooldown = None
        self.obstacle_speed = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_STEPS / self.FPS
        self.gems_collected = 0
        self.game_over = False
        self.particles = []
        self.gem_respawn_cooldown = 0
        self.obstacle_speed = 0.5 # grid units per second

        # Player
        self.player_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2], dtype=float)
        self.player_pixel_pos = np.array(self._iso_to_screen(self.player_pos), dtype=float)

        # Obstacles
        self.obstacles = []
        obstacle_paths = [
            # Horizontal
            (np.array([2, 2]), np.array([self.GRID_WIDTH - 3, 2])),
            (np.array([2, self.GRID_HEIGHT - 3]), np.array([self.GRID_WIDTH - 3, self.GRID_HEIGHT - 3])),
            # Vertical
            (np.array([4, 4]), np.array([4, self.GRID_HEIGHT - 5])),
            (np.array([self.GRID_WIDTH - 5, 4]), np.array([self.GRID_WIDTH - 5, self.GRID_HEIGHT - 5])),
        ]
        for start, end in obstacle_paths:
            self.obstacles.append({
                "pos": start.astype(float),
                "start": start.astype(float),
                "end": end.astype(float),
                "direction": 1,
            })

        # Gems
        self.gems = []
        self._spawn_gems(5)
        
        self.last_dist_to_gem = self._get_dist_to_nearest_gem()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Update Game State ---
        self._update_timer()
        self._update_difficulty()
        self._handle_input(action)
        self._update_positions()
        
        # --- Handle Collections & Collisions ---
        collection_reward = self._handle_gem_collection()
        reward += collection_reward

        # --- Calculate Rewards ---
        dist_to_gem = self._get_dist_to_nearest_gem()
        if dist_to_gem < self.last_dist_to_gem:
            reward += 0.1
        elif dist_to_gem > self.last_dist_to_gem:
            reward -= 0.2
        self.last_dist_to_gem = dist_to_gem

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.gems_collected >= 50:
                reward = 100.0 # Victory
            elif self.time_remaining <= 0:
                reward = -100.0 # Timeout
            else: # Collision
                reward = -100.0

        # --- Update Effects ---
        self._update_particles()
        self._update_gem_spawner()

        obs = self._get_observation()
        return obs, reward, terminated, False, self._get_info()

    # --- Game Logic Sub-methods ---
    def _handle_input(self, action):
        movement = action[0]
        target_pos = self.player_pos.copy()
        if movement == 1: target_pos[1] -= 1  # Up
        elif movement == 2: target_pos[1] += 1  # Down
        elif movement == 3: target_pos[0] -= 1  # Left
        elif movement == 4: target_pos[0] += 1  # Right
        
        # Clamp to grid boundaries
        target_pos[0] = np.clip(target_pos[0], 0, self.GRID_WIDTH - 1)
        target_pos[1] = np.clip(target_pos[1], 0, self.GRID_HEIGHT - 1)
        self.player_pos = target_pos

    def _update_positions(self):
        # Interpolate player pixel position
        target_pixel_pos = self._iso_to_screen(self.player_pos)
        self.player_pixel_pos += (target_pixel_pos - self.player_pixel_pos) * 0.5

        # Update obstacle positions
        for obs in self.obstacles:
            path_vector = obs["end"] - obs["start"]
            path_length = np.linalg.norm(path_vector)
            if path_length == 0: continue

            move_dist = (self.obstacle_speed / self.FPS) * obs["direction"]
            
            # Project current position onto the path to find progress
            current_vec = obs["pos"] - obs["start"]
            progress = np.dot(current_vec, path_vector) / (path_length ** 2) if path_length > 0 else 0
            
            new_progress = progress + (move_dist / path_length if path_length > 0 else 0)
            
            if not (0 <= new_progress <= 1):
                obs["direction"] *= -1
                new_progress = np.clip(new_progress, 0, 1)

            obs["pos"] = obs["start"] + new_progress * path_vector

    def _handle_gem_collection(self):
        reward = 0
        collected_any = False
        for gem in self.gems[:]:
            if np.linalg.norm(self.player_pos - gem["pos"]) < 0.5:
                self.gems.remove(gem)
                self.gems_collected += 1
                self.score += 1
                reward += 1.0
                collected_any = True
                
                # Bonus for risky collection
                dist_to_obs = self._get_dist_to_nearest_obstacle(self.player_pos)
                if dist_to_obs < 2.0:
                    self.score += 2
                    reward += 2.0
                    
                # SFX: Gem collect
                self._spawn_particles(self._iso_to_screen(gem["pos"]), gem["color"])
        
        if collected_any:
            self.gem_respawn_cooldown = self.FPS // 2 # 0.5 second cooldown
        return reward

    def _update_timer(self):
        self.time_remaining = max(0, self.time_remaining - 1.0 / self.FPS)

    def _update_difficulty(self):
        # Increase obstacle speed every 10 seconds
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.obstacle_speed += 0.05

    def _check_termination(self):
        if self.gems_collected >= 50: return True
        if self.time_remaining <= 0: return True
        if self.steps >= self.MAX_STEPS: return True
        
        player_grid_rounded = np.round(self.player_pos).astype(int)
        for obs in self.obstacles:
            obs_grid_rounded = np.round(obs["pos"]).astype(int)
            if np.linalg.norm(self.player_pos - obs["pos"]) < 0.5:
                # SFX: Player hit
                return True
        return False

    # --- Spawning and Effects ---
    def _spawn_gems(self, num_gems):
        occupied_spaces = {tuple(np.round(g["pos"]).astype(int)) for g in self.gems}
        occupied_spaces.add(tuple(np.round(self.player_pos).astype(int)))
        for _ in range(num_gems):
            attempts = 0
            while attempts < 100:
                pos = np.array([
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT)
                ])
                if tuple(pos) not in occupied_spaces:
                    occupied_spaces.add(tuple(pos))
                    self.gems.append({
                        "pos": pos.astype(float),
                        "color": random.choice(self.GEM_COLORS),
                        "bob_offset": self.np_random.random() * math.pi * 2
                    })
                    break
                attempts += 1

    def _update_gem_spawner(self):
        if self.gem_respawn_cooldown > 0:
            self.gem_respawn_cooldown -= 1
            if self.gem_respawn_cooldown == 0:
                needed = 5 - len(self.gems)
                if needed > 0:
                    self._spawn_gems(needed)

    def _spawn_particles(self, pos, color):
        # SFX: Particle burst
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": np.array(pos, dtype=float),
                "vel": velocity,
                "life": self.np_random.integers(15, 25),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    # --- Helper Functions ---
    def _get_dist_to_nearest_gem(self):
        if not self.gems: return float('inf')
        return min(np.linalg.norm(self.player_pos - gem["pos"]) for gem in self.gems)

    def _get_dist_to_nearest_obstacle(self, pos):
        if not self.obstacles: return float('inf')
        return min(np.linalg.norm(pos - obs["pos"]) for obs in self.obstacles)

    def _iso_to_screen(self, pos):
        x, y = pos
        screen_x = self.ORIGIN_X + (x - y) * (self.TILE_WIDTH_ISO / 2)
        screen_y = self.ORIGIN_Y + (x + y) * (self.TILE_HEIGHT_ISO / 2)
        return int(screen_x), int(screen_y)

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid()
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._iso_to_screen((x, y))
                p2 = self._iso_to_screen((x + 1, y))
                p3 = self._iso_to_screen((x + 1, y + 1))
                p4 = self._iso_to_screen((x, y + 1))
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

    def _render_game(self):
        # Combine all dynamic objects and sort by y-position for correct draw order
        render_list = []
        
        # Player
        render_list.append({"type": "player", "pos": self.player_pos, "pixel_pos": self.player_pixel_pos})
        
        # Gems
        for gem in self.gems:
            # FIX: Add 'pos' key explicitly for consistent sorting
            render_list.append({"type": "gem", "pos": gem["pos"], "data": gem})
        
        # Obstacles
        for obs in self.obstacles:
            render_list.append({"type": "obstacle", "pos": obs["pos"]})

        # Sort by isometric depth (y+x is a good approximation)
        # FIX: Use a simplified, correct key now that all items have 'pos'
        render_list.sort(key=lambda item: item["pos"][0] + item["pos"][1])

        for item in render_list:
            if item["type"] == "player":
                self._render_player(item["pixel_pos"])
            elif item["type"] == "gem":
                self._render_gem(item["data"])
            elif item["type"] == "obstacle":
                self._render_obstacle(item["pos"])

    def _render_player(self, pixel_pos):
        px, py = int(pixel_pos[0]), int(pixel_pos[1])
        size = self.TILE_WIDTH_ISO // 2
        # Glow
        glow_radius = int(size * 1.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (px - glow_radius, py - glow_radius))
        # Player body
        points = [
            (px, py - size // 2),
            (px + size // 2, py),
            (px, py + size // 2),
            (px - size // 2, py)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_gem(self, gem):
        bob = math.sin(self.steps / 10.0 + gem["bob_offset"]) * 3
        px, py = self._iso_to_screen(gem["pos"])
        py += int(bob)
        size = self.TILE_WIDTH_ISO // 4
        points = [
            (px, py - size),
            (px + size, py),
            (px, py + size),
            (px - size, py)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, gem["color"])
        pygame.gfxdraw.aapolygon(self.screen, points, gem["color"])

    def _render_obstacle(self, pos):
        px, py = self._iso_to_screen(pos)
        size = self.TILE_WIDTH_ISO // 2
        points = [
            (px, py - size // 2),
            (px + size // 2, py),
            (px, py + size // 2),
            (px - size // 2, py)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / 25.0
            radius = int(life_ratio * 5)
            if radius > 0:
                color = (*p["color"], int(life_ratio * 255))
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (int(p["pos"][0]) - radius, int(p["pos"][1]) - radius))
    
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_str = f"{int(self.time_remaining // 60):02}:{int(self.time_remaining % 60):02}"
        time_text = self.font_main.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Gem Counter
        gem_text = self.font_small.render(f"Gems: {self.gems_collected} / 50", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (self.WIDTH // 2 - gem_text.get_width() // 2, self.HEIGHT - 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "time_remaining": self.time_remaining,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # To run and play the game manually
    render_mode = "human"
    if render_mode == "human":
        # Unset the dummy video driver if we want to visualize
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for visualization
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)
        
    env.close()