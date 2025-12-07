import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:26:30.862112
# Source Brief: brief_00415.md
# Brief Index: 415
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a digital grid to collect all data packets while avoiding patrol drones, then return to the central extraction point."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move. Collect all data packets and return to the center to win."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (10, 10, 26)
    COLOR_GRID = (30, 30, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_DRONE = (255, 50, 50)
    COLOR_DRONE_GLOW = (255, 50, 50, 50)
    COLOR_PACKET = (50, 255, 100)
    COLOR_PACKET_GLOW = (50, 255, 100, 70)
    COLOR_TARGET = (255, 200, 0)
    COLOR_TARGET_GLOW = (255, 200, 0, 40)
    COLOR_TEXT = (220, 220, 240)

    # Game World
    WORLD_SIZE_X = 30
    WORLD_SIZE_Y = 30
    PACKET_COUNT = 20
    DRONE_COUNT = 10

    # Player Physics
    PLAYER_ACCELERATION = 0.3
    PLAYER_MAX_SPEED = 4.0
    PLAYER_DRAG = 0.96
    PLAYER_SIZE = 0.6
    PLAYER_HEIGHT = 1.5

    # Drone Physics
    DRONE_SIZE = 0.5
    DRONE_HEIGHT = 1.0

    # Collision Radii
    PLAYER_COLLISION_RADIUS = 0.8
    DRONE_COLLISION_RADIUS = 0.8
    PACKET_COLLISION_RADIUS = 1.0
    TARGET_COLLISION_RADIUS = 2.0

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
        self.font_timer = pygame.font.SysFont("Consolas", 24, bold=True)

        # Isometric projection vectors
        self.world_origin = pygame.Vector2(self.SCREEN_WIDTH // 2, 80)
        self.tile_vec_x = pygame.Vector2(16, 8)
        self.tile_vec_y = pygame.Vector2(-16, 8)

        # self.reset() # This is called by the wrapper, no need to call it here.
        # self.validate_implementation() # This is for debugging, not needed in the final version.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player = {
            "pos": pygame.Vector3(self.WORLD_SIZE_X / 2, self.WORLD_SIZE_Y / 2, 0),
            "vel": pygame.Vector3(0, 0, 0)
        }

        self.packets = []
        for _ in range(self.PACKET_COUNT):
            self.packets.append({
                "pos": pygame.Vector3(
                    self.np_random.uniform(2, self.WORLD_SIZE_X - 2),
                    self.np_random.uniform(2, self.WORLD_SIZE_Y - 2),
                    self.np_random.uniform(0.5, 2.5)
                ),
                "collected": False
            })

        self.drones = []
        for _ in range(self.DRONE_COUNT):
            self.drones.append({
                "center": pygame.Vector3(
                    self.np_random.uniform(0, self.WORLD_SIZE_X),
                    self.np_random.uniform(0, self.WORLD_SIZE_Y),
                    self.np_random.uniform(1.0, 4.0)
                ),
                "radius": self.np_random.uniform(3, 8),
                "speed": self.np_random.uniform(0.02, 0.05),
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "pos": pygame.Vector3(0, 0, 0)
            })

        self.target_zone = {
            "pos": pygame.Vector3(self.WORLD_SIZE_X / 2, self.WORLD_SIZE_Y / 2, 0)
        }
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.01  # Survival reward

        # --- 1. Update Player ---
        accel = pygame.Vector3(0, 0, 0)
        if movement == 1: accel.y = -self.PLAYER_ACCELERATION # Up
        elif movement == 2: accel.y = self.PLAYER_ACCELERATION # Down
        elif movement == 3: accel.x = -self.PLAYER_ACCELERATION # Left
        elif movement == 4: accel.x = self.PLAYER_ACCELERATION # Right

        self.player["vel"] += accel
        if self.player["vel"].length() > self.PLAYER_MAX_SPEED:
            self.player["vel"].scale_to_length(self.PLAYER_MAX_SPEED)
        
        self.player["pos"] += self.player["vel"]
        self.player["vel"] *= self.PLAYER_DRAG

        # Boundary checks
        self.player["pos"].x = np.clip(self.player["pos"].x, 0, self.WORLD_SIZE_X)
        self.player["pos"].y = np.clip(self.player["pos"].y, 0, self.WORLD_SIZE_Y)

        # --- 2. Update Drones ---
        for drone in self.drones:
            drone["angle"] += drone["speed"]
            offset_x = math.cos(drone["angle"]) * drone["radius"]
            offset_y = math.sin(drone["angle"]) * drone["radius"]
            drone["pos"] = drone["center"] + pygame.Vector3(offset_x, offset_y, 0)

        # --- 3. Update Particles ---
        self._update_particles()

        # --- 4. Check Interactions ---
        # Player -> Packets
        for packet in self.packets:
            if not packet["collected"]:
                dist = self.player["pos"].distance_to(packet["pos"])
                if dist < self.PACKET_COLLISION_RADIUS:
                    packet["collected"] = True
                    self.score += 1
                    reward += 1.0
                    # Sound: Packet collect
                    self._spawn_particles(packet["pos"], self.COLOR_PACKET, 15, 0.5)

        # Player -> Drones
        for drone in self.drones:
            dist = self.player["pos"].distance_to(drone["pos"])
            if dist < self.DRONE_COLLISION_RADIUS:
                self.game_over = True
                reward -= 100.0
                # Sound: Explosion
                self._spawn_particles(self.player["pos"], self.COLOR_DRONE, 50, 1.0)
                break
        
        # --- 5. Check Win/Loss Conditions ---
        self.steps += 1
        terminated = False
        
        # Win condition
        if self.score >= self.PACKET_COUNT:
            dist_to_target = self.player["pos"].distance_to(self.target_zone["pos"])
            if dist_to_target < self.TARGET_COLLISION_RADIUS:
                self.win = True
                self.game_over = True
                terminated = True
                reward += 100.0

        # Loss conditions
        if self.game_over and not self.win:
            terminated = True
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            if not self.win:
                reward -= 100.0
            self.game_over = True
            terminated = True # Per Gymnasium API, truncated can be true with terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _project(self, pos: pygame.Vector3) -> pygame.Vector2:
        screen_pos = self.world_origin + pos.x * self.tile_vec_x + pos.y * self.tile_vec_y
        screen_pos.y -= pos.z * 10 # Z-scaling
        return screen_pos

    def _draw_iso_cube(self, surface, pos, size, height, color, glow_color):
        points = [
            pygame.Vector3(-size, -size, 0), pygame.Vector3(size, -size, 0),
            pygame.Vector3(size, size, 0), pygame.Vector3(-size, size, 0)
        ]
        
        base_pts = [self._project(pos + p) for p in points]
        top_pts = [self._project(pos + p + pygame.Vector3(0, 0, height)) for p in points]

        # Glow
        pygame.gfxdraw.filled_polygon(surface, top_pts, glow_color)
        
        # Faces (draw order matters)
        # Top
        pygame.gfxdraw.filled_polygon(surface, top_pts, color)
        pygame.gfxdraw.aapolygon(surface, top_pts, color)
        # Sides
        right_face = [base_pts[1], top_pts[1], top_pts[2], base_pts[2]]
        left_face = [base_pts[2], top_pts[2], top_pts[3], base_pts[3]]
        
        darker_color = tuple(max(0, c - 40) for c in color)
        pygame.gfxdraw.filled_polygon(surface, right_face, darker_color)
        pygame.gfxdraw.aapolygon(surface, right_face, darker_color)
        
        darkest_color = tuple(max(0, c - 60) for c in color)
        pygame.gfxdraw.filled_polygon(surface, left_face, darkest_color)
        pygame.gfxdraw.aapolygon(surface, left_face, darkest_color)

    def _draw_iso_sphere(self, surface, pos, radius, color, glow_color):
        screen_pos = self._project(pos)
        
        # Glow effect
        glow_radius = int(radius * 1.8)
        pygame.gfxdraw.filled_circle(surface, int(screen_pos.x), int(screen_pos.y), glow_radius, glow_color)

        # Main sphere
        pygame.gfxdraw.filled_circle(surface, int(screen_pos.x), int(screen_pos.y), int(radius), color)
        pygame.gfxdraw.aacircle(surface, int(screen_pos.x), int(screen_pos.y), int(radius), color)

    def _spawn_particles(self, pos, color, count, speed_mult):
        for _ in range(count):
            vel = pygame.Vector3(
                self.np_random.uniform(-1, 1),
                self.np_random.uniform(-1, 1),
                self.np_random.uniform(0.5, 2)
            )
            vel.scale_to_length(self.np_random.uniform(0.2, 1.0) * speed_mult)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": self.np_random.integers(20, 40),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["vel"].z -= 0.05 # Gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        # --- Draw Grid ---
        for i in range(self.WORLD_SIZE_X + 1):
            start = self._project(pygame.Vector3(i, 0, 0))
            end = self._project(pygame.Vector3(i, self.WORLD_SIZE_Y, 0))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for i in range(self.WORLD_SIZE_Y + 1):
            start = self._project(pygame.Vector3(0, i, 0))
            end = self._project(pygame.Vector3(self.WORLD_SIZE_X, i, 0))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # --- Prepare and Sort Renderables ---
        renderables = []
        # Target Zone
        renderables.append({"type": "target", "pos": self.target_zone["pos"]})
        # Packets
        for p in self.packets:
            if not p["collected"]:
                renderables.append({"type": "packet", "pos": p["pos"]})
        # Drones
        for d in self.drones:
            renderables.append({"type": "drone", "pos": d["pos"]})
        # Player
        renderables.append({"type": "player", "pos": self.player["pos"]})
        # Particles
        for p in self.particles:
            renderables.append({"type": "particle", "pos": p["pos"], "lifespan": p["lifespan"], "color": p["color"]})
        
        # Sort by y-coordinate for correct draw order
        renderables.sort(key=lambda r: r["pos"].y + r["pos"].x)

        # --- Draw Game Objects ---
        for r in renderables:
            if r["type"] == "target":
                if self.score >= self.PACKET_COUNT:
                    self._draw_iso_cube(self.screen, r["pos"], self.TARGET_COLLISION_RADIUS, 0.2, self.COLOR_TARGET, self.COLOR_TARGET_GLOW)
            elif r["type"] == "packet":
                self._draw_iso_sphere(self.screen, r["pos"], 6, self.COLOR_PACKET, self.COLOR_PACKET_GLOW)
            elif r["type"] == "drone":
                self._draw_iso_cube(self.screen, r["pos"], self.DRONE_SIZE, self.DRONE_HEIGHT, self.COLOR_DRONE, self.COLOR_DRONE_GLOW)
            elif r["type"] == "player":
                self._draw_iso_cube(self.screen, r["pos"], self.PLAYER_SIZE, self.PLAYER_HEIGHT, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
            elif r["type"] == "particle":
                alpha = max(0, min(255, int(r["lifespan"] * 12)))
                color = (*r["color"], alpha)
                spos = self._project(r["pos"])
                pygame.gfxdraw.filled_circle(self.screen, int(spos.x), int(spos.y), 2, color)

        # --- Draw UI ---
        # Packets collected
        score_text = self.font_ui.render(f"DATA: {self.score}/{self.PACKET_COUNT}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Speed
        speed = self.player["vel"].length() * 10
        speed_text = self.font_ui.render(f"SPEED: {speed:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH - speed_text.get_width() - 10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_DRONE
        time_text = self.font_timer.render(f"{time_left:.2f}", True, time_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH // 2 - time_text.get_width() // 2, 10))

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block is for human play and debugging.
    # It will not work in a headless environment.
    # To run, you might need to unset the SDL_VIDEODRIVER variable.
    # e.g., `SDL_VIDEODRIVER=x11 python your_game_file.py` on Linux.
    
    # Unset the dummy driver for local rendering
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Runner")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)
        
    env.close()