import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:22:45.887973
# Source Brief: brief_01029.md
# Brief Index: 1029
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for managing drone state
class Drone:
    def __init__(self, path, speed):
        self.path = path
        self.path_index = 0
        self.pos = np.array(self.path[self.path_index], dtype=float)
        self.target = np.array(self.path[self.path_index + 1], dtype=float)
        self.speed = speed
        self.rotation = 0

    def update(self):
        direction = self.target - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = self.target
            self.path_index = (self.path_index + 1) % (len(self.path) - 1)
            self.target = np.array(self.path[self.path_index + 1], dtype=float)
            # Reverse path for patrol
            if self.path_index == len(self.path) - 2:
                self.path.reverse()
                self.path_index = 0
                self.target = np.array(self.path[1], dtype=float)
        else:
            self.pos += (direction / distance) * self.speed
        
        self.rotation = (self.rotation + 2) % 360


# Helper class for visual particles
class Particle:
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95 # friction
        self.radius *= 0.97
        self.lifetime -= 1
        return self.lifetime > 0 and self.radius > 0.5


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A top-down stealth game. Change your color to match the floor, terraform tiles to create safe paths, and avoid patrolling drones to reach the exit."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to change the color of the tile in front of you. Press shift to cycle your own color."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.TILE_SIZE = 20
        self.MAX_STEPS = 1000
        self.DRONE_DETECTION_RADIUS = 3.5

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_FLOOR = (30, 35, 55)
        self.COLOR_WALL = (50, 60, 90)
        self.COLOR_EXIT = (255, 220, 0)
        self.COLOR_DRONE = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 255, 255)
        self.CAMO_COLORS = [
            (30, 200, 80),   # Green
            (150, 80, 220),  # Purple
            (0, 200, 200),   # Cyan
            self.COLOR_FLOOR # Default floor color
        ]

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
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = None
        self.walls = None
        self.player_pos = None
        self.player_color_idx = 0
        self.player_facing_dir = np.array([0, 1])
        self.exit_pos = None
        self.drones = []
        self.particles = []
        self.detection_meter = 0.0
        self.drone_base_speed = 0.08
        self.prev_shift_held = False

        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # validation is done by tests

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.detection_meter = 0.0
        self.particles = []
        self.prev_shift_held = False

        self._generate_level()

        self.player_color_idx = self.np_random.integers(0, len(self.CAMO_COLORS))
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        # --- Handle Player Actions ---
        self._handle_color_cycle(shift_held)
        terraform_reward = self._handle_terraform(space_held)
        reward += terraform_reward
        self._handle_movement(movement)

        # --- Update Game World ---
        self.steps += 1
        current_drone_speed = self.drone_base_speed + 0.05 * (self.steps // 500)
        for drone in self.drones:
            drone.speed = current_drone_speed
            drone.update()
        
        self._update_particles()

        # --- Core Logic: Detection & Rewards ---
        detection_reward, is_detected = self._update_detection()
        reward += detection_reward

        if self.detection_meter > 50:
            reward -= 1.0

        # --- Check Termination Conditions ---
        if self.detection_meter >= 100:
            reward = -100.0
            terminated = True
            # sfx: detection alarm
            self._create_particles(self._grid_to_pixel(self.player_pos), 50, self.COLOR_DRONE, 5.0)

        if tuple(self.player_pos) == self.exit_pos:
            reward = 100.0
            self.score += 1
            terminated = True
            # sfx: level complete
            self._create_particles(self._grid_to_pixel(self.exit_pos), 50, self.COLOR_EXIT, 5.0)

        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        truncated = False # No truncation condition in this game
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_color_cycle(self, shift_held):
        if shift_held and not self.prev_shift_held:
            self.player_color_idx = (self.player_color_idx + 1) % len(self.CAMO_COLORS)
            # sfx: color change pop
            self._create_particles(self._grid_to_pixel(self.player_pos), 15, self.CAMO_COLORS[self.player_color_idx], 2.0)
        self.prev_shift_held = shift_held

    def _handle_terraform(self, space_held):
        if space_held:
            target_pos = self.player_pos + self.player_facing_dir
            tx, ty = int(target_pos[0]), int(target_pos[1])

            if 0 <= tx < self.GRID_WIDTH and 0 <= ty < self.GRID_HEIGHT and (tx, ty) not in self.walls:
                current_tile_color = self.grid[ty, tx]
                player_color = self.CAMO_COLORS[self.player_color_idx]
                if tuple(current_tile_color) != player_color:
                    self.grid[ty, tx] = player_color
                    # sfx: terraform fizz
                    self._create_particles(self._grid_to_pixel(target_pos), 10, player_color, 1.0)
                    return 1.0
        return 0.0

    def _handle_movement(self, movement):
        move_map = {
            1: np.array([0, -1]),  # Up
            2: np.array([0, 1]),   # Down
            3: np.array([-1, 0]),  # Left
            4: np.array([1, 0]),   # Right
        }
        if movement in move_map:
            self.player_facing_dir = move_map[movement]
            next_pos = self.player_pos + self.player_facing_dir
            
            # Wraparound logic
            next_pos[0] %= self.GRID_WIDTH
            next_pos[1] %= self.GRID_HEIGHT

            if tuple(next_pos) not in self.walls:
                self.player_pos = next_pos
                # sfx: player step
            else:
                # sfx: bump into wall
                pass

    def _update_detection(self):
        reward = 0
        is_detected_this_step = False
        player_on_tile_color = tuple(self.grid[int(self.player_pos[1]), int(self.player_pos[0])])
        player_camo_color = self.CAMO_COLORS[self.player_color_idx]
        is_camouflaged = player_on_tile_color == player_camo_color

        for drone in self.drones:
            dist = np.linalg.norm(self.player_pos - drone.pos)
            if dist < self.DRONE_DETECTION_RADIUS:
                is_detected_this_step = True
                if is_camouflaged:
                    self.detection_meter -= 2.0
                    reward += 0.1
                else:
                    self.detection_meter += 5.0
                    reward -= 0.01
                break
        
        if not is_detected_this_step:
            self.detection_meter -= 0.5
        
        self.detection_meter = np.clip(self.detection_meter, 0, 100)
        return reward, is_detected_this_step

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _generate_level(self):
        self.grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH, 3), self.COLOR_FLOOR, dtype=np.uint8)
        self.walls = set()

        # Create a central block of walls
        for y in range(5, self.GRID_HEIGHT - 5):
            for x in range(8, self.GRID_WIDTH - 8):
                if self.np_random.random() > 0.4:
                    self.walls.add((x, y))

        # Define start and exit, ensuring they are not in walls
        self.player_pos = np.array([2, self.GRID_HEIGHT // 2])
        self.exit_pos = (self.GRID_WIDTH - 3, self.GRID_HEIGHT // 2)
        if tuple(self.player_pos) in self.walls: self.walls.remove(tuple(self.player_pos))
        if self.exit_pos in self.walls: self.walls.remove(self.exit_pos)

        # Create drones with patrol paths
        self.drones = []
        drone_path1 = [(5, 2), (5, self.GRID_HEIGHT - 3)]
        drone_path2 = [(self.GRID_WIDTH - 6, 2), (self.GRID_WIDTH - 6, self.GRID_HEIGHT - 3)]
        drone_path3 = [(10, 2), (self.GRID_WIDTH - 11, 2)]
        self.drones.append(Drone(drone_path1, self.drone_base_speed))
        self.drones.append(Drone(drone_path2, self.drone_base_speed))
        self.drones.append(Drone(drone_path3, self.drone_base_speed))

    def _render_game(self):
        # Render grid tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.grid[y, x], rect)

        # Render walls
        for x, y in self.walls:
            rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1) # Outline

        # Render exit
        exit_px = self._grid_to_pixel(self.exit_pos)
        self._draw_glow(exit_px, self.TILE_SIZE * 0.8, self.COLOR_EXIT)
        exit_rect = pygame.Rect(0, 0, self.TILE_SIZE, self.TILE_SIZE)
        exit_rect.center = exit_px
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect.inflate(-4, -4))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p.lifetime / p.max_lifetime))
            color = (*p.color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), int(p.radius), color)

        # Render drones
        for drone in self.drones:
            drone_px = self._grid_to_pixel(drone.pos)
            # Detection radius
            pygame.gfxdraw.filled_circle(self.screen, drone_px[0], drone_px[1], int(self.DRONE_DETECTION_RADIUS * self.TILE_SIZE), (100, 0, 0, 50))
            # Drone body
            self._draw_rotated_triangle(drone_px, self.TILE_SIZE * 0.6, drone.rotation, self.COLOR_DRONE)

        # Render player
        player_px = self._grid_to_pixel(self.player_pos)
        player_color = self.CAMO_COLORS[self.player_color_idx]
        self._draw_glow(player_px, self.TILE_SIZE * 0.7, self.COLOR_PLAYER_GLOW)
        player_rect = pygame.Rect(0, 0, self.TILE_SIZE, self.TILE_SIZE)
        player_rect.center = player_px
        pygame.draw.rect(self.screen, player_color, player_rect.inflate(-4, -4))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, player_rect.inflate(-4, -4), 1)

    def _render_ui(self):
        # Detection Meter
        bar_width = self.WIDTH - 20
        bar_height = 15
        fill_width = (self.detection_meter / 100) * bar_width
        pygame.draw.rect(self.screen, (80, 0, 0), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_DRONE, (10, 10, fill_width, bar_height))
        pygame.draw.rect(self.screen, (255, 255, 255), (10, 10, bar_width, bar_height), 1)
        
        # Camo Indicator
        pygame.draw.rect(self.screen, self.CAMO_COLORS[self.player_color_idx], (10, 35, 30, 30))
        pygame.draw.rect(self.screen, (255, 255, 255), (10, 35, 30, 30), 1)
        
        # Score and Steps Text
        score_text = self.font.render(f"SCORE: {self.score}", True, (255, 255, 255))
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 35))

    def _grid_to_pixel(self, grid_pos):
        px = int((grid_pos[0] + 0.5) * self.TILE_SIZE)
        py = int((grid_pos[1] + 0.5) * self.TILE_SIZE)
        return px, py

    def _create_particles(self, pos, count, color, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(20, 40)
            self.particles.append(Particle(pos, vel, radius, color, lifetime))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _draw_glow(self, center, max_radius, color):
        for i in range(int(max_radius), 0, -2):
            alpha = int(50 * (1 - (i / max_radius)))
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], i, (*color, alpha))

    def _draw_rotated_triangle(self, center, size, angle, color):
        points = []
        for i in range(3):
            rad = math.radians(angle + i * 120)
            x = center[0] + size * math.cos(rad)
            y = center[1] + size * math.sin(rad)
            points.append((x, y))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def validate_implementation(self):
        # This method is for self-testing and can be removed or commented out.
        # The test harness will perform these checks externally.
        print("Running internal validation...")
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # This block is for manual testing and will not be run by the evaluation system.
    # It requires a display.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Use arrow keys for movement, space to terraform, left shift to cycle color
    pygame.display.set_caption("Synthetic Escape")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        # Poll events for key presses
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        # Check for held keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Limit to 30 FPS for human playability

    pygame.quit()
    env.close()