import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


# Set for headless operation, required by the test environment
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = "Stack cubes as high as you can to build a tower, but be careful of the increasing wind and the tower's sway."
    user_guide = "Use the arrow keys (↑↓←→) to position the next cube and press space to drop it into place."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30

    # Colors
    COLOR_BG = (15, 19, 23)
    COLOR_GROUND = (40, 45, 50)
    COLOR_GROUND_LINE = (60, 65, 70)
    COLOR_TEXT = (220, 220, 220)
    COLOR_WIND = (180, 180, 200, 50)
    CUBE_PALETTE = [
        (255, 87, 34),   # Deep Orange
        (3, 169, 244),   # Light Blue
        (255, 235, 59),  # Yellow
        (76, 175, 80),   # Green
        (233, 30, 99),   # Pink
        (156, 39, 176),  # Purple
    ]

    # Game settings
    CUBE_SIZE = 40
    MAX_CUBES = 20
    MAX_STEPS = 1000
    GHOST_CUBE_SPEED = 5
    VICTORY_LEVEL = 20
    COLLAPSE_ANGLE_DEG = 20

    # Physics
    WIND_INCREMENT = 0.005
    SWAY_SPRING_CONSTANT = 0.001
    SWAY_DAMPING = 0.95

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # --- Game State Initialization ---
        self.cubes = []
        self.ghost_cube = None
        self.steps = 0
        self.score = 0
        self.level = 0
        self.game_over = False
        self.wind_speed = 0.0
        self.wind_particles = []
        self.tower_sway_offset = 0.0
        self.tower_sway_velocity = 0.0
        self.center_of_mass = np.array([0.0, 0.0, 0.0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state variables
        self.steps = 0
        self.score = 0
        self.level = 0
        self.game_over = False
        self.wind_speed = 0.0
        self.tower_sway_offset = 0.0
        self.tower_sway_velocity = 0.0

        # Create the base cube
        self.cubes = [{
            "pos": np.array([0.0, 0.0, self.CUBE_SIZE / 2]),
            "color": self.COLOR_GROUND,
            "is_base": True
        }]
        
        # Create the first ghost cube
        self._create_ghost_cube()
        self._update_center_of_mass()

        # Initialize wind particles
        self.wind_particles = []
        for _ in range(50):
            self.wind_particles.append([
                self.np_random.uniform(0, self.SCREEN_WIDTH),
                self.np_random.uniform(0, self.SCREEN_HEIGHT)
            ])

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement, space_pressed, shift_pressed = action
        is_drop_action = space_pressed == 1

        if is_drop_action:
            # Solidify the ghost cube
            new_cube = {
                "pos": self.ghost_cube["pos"].copy(),
                "color": self.ghost_cube["color"],
                "is_base": False
            }
            self.cubes.append(new_cube)
            self.level += 1
            self.score += 1

            # Update game state
            self.wind_speed = self.level * self.WIND_INCREMENT
            self._update_center_of_mass()
            
            # Rewards for placement
            reward += 1.0  # Level completed
            reward += 0.1  # Successful placement
            
            # Check for victory
            if self.level >= self.VICTORY_LEVEL:
                self.game_over = True
                reward += 100.0 # Victory bonus
            else:
                self._create_ghost_cube()

        else: # Not a drop action, so handle movement
            if movement == 1: # Up
                self.ghost_cube["pos"][1] -= self.GHOST_CUBE_SPEED
            elif movement == 2: # Down
                self.ghost_cube["pos"][1] += self.GHOST_CUBE_SPEED
            elif movement == 3: # Left
                self.ghost_cube["pos"][0] -= self.GHOST_CUBE_SPEED
            elif movement == 4: # Right
                self.ghost_cube["pos"][0] += self.GHOST_CUBE_SPEED

            # Clamp ghost cube position to a reasonable area
            max_offset = self.CUBE_SIZE * 2
            self.ghost_cube["pos"][0] = np.clip(self.ghost_cube["pos"][0], -max_offset, max_offset)
            self.ghost_cube["pos"][1] = np.clip(self.ghost_cube["pos"][1], -max_offset, max_offset)


        # --- Physics Simulation ---
        if not self.game_over and len(self.cubes) > 1:
            # Calculate forces
            wind_force = self.wind_speed * (len(self.cubes) - 1)
            com_horizontal_offset = self.center_of_mass[0] + self.tower_sway_offset
            restoring_force = -com_horizontal_offset * self.SWAY_SPRING_CONSTANT * len(self.cubes)

            # Update sway velocity and position
            total_force = wind_force + restoring_force
            sway_acceleration = total_force / len(self.cubes)
            self.tower_sway_velocity += sway_acceleration
            self.tower_sway_velocity *= self.SWAY_DAMPING
            self.tower_sway_offset += self.tower_sway_velocity

        # --- Check for Termination ---
        tower_angle_rad = self._get_tower_angle()
        if abs(math.degrees(tower_angle_rad)) > self.COLLAPSE_ANGLE_DEG:
            self.game_over = True
            reward = -100.0 # Collapse penalty
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True # End episode if it runs too long

        # Penalty for high sway
        if abs(math.degrees(tower_angle_rad)) > 5:
            reward -= 0.01

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_ground()
        self._render_wind()
        self._render_tower()
        
        # Render UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "wind_speed": self.wind_speed,
            "tower_sway_angle_deg": math.degrees(self._get_tower_angle())
        }

    # --- Helper and Rendering Methods ---

    def _create_ghost_cube(self):
        top_cube_pos = self.cubes[-1]["pos"]
        self.ghost_cube = {
            "pos": np.array([top_cube_pos[0], top_cube_pos[1], top_cube_pos[2] + self.CUBE_SIZE]),
            "color": self.CUBE_PALETTE[self.level % len(self.CUBE_PALETTE)],
            "is_base": False,
            "is_ghost": True
        }

    def _update_center_of_mass(self):
        if len(self.cubes) <= 1:
            self.center_of_mass = self.cubes[0]["pos"].copy()
            return

        com = np.zeros(3)
        # Exclude the base cube from CoM calculation for sway dynamics
        tower_cubes = self.cubes[1:]
        for cube in tower_cubes:
            com += cube["pos"]
        self.center_of_mass = com / len(tower_cubes)

    def _get_tower_angle(self):
        if len(self.cubes) <= 1:
            return 0.0
        com_height = self.center_of_mass[2] - self.cubes[0]["pos"][2]
        if com_height <= 0:
            return 0.0
        # Add a small epsilon to avoid division by zero
        return math.atan(self.tower_sway_offset / (com_height + 1e-6))

    def _project(self, pos):
        x, y, z = pos
        iso_x = (x - y) * 0.866 # cos(30)
        iso_y = (x + y) * 0.5   # sin(30)
        iso_y -= z
        
        # Center on screen
        screen_x = self.SCREEN_WIDTH / 2 + iso_x
        screen_y = self.SCREEN_HEIGHT * 0.8 - iso_y
        return int(screen_x), int(screen_y)

    def _draw_cube(self, surface, pos, angle, color, size, is_ghost=False):
        s = size / 2
        
        # Cube vertices in local space
        points_3d = [
            np.array([-s, -s, -s]), np.array([s, -s, -s]), np.array([s, s, -s]), np.array([-s, s, -s]),
            np.array([-s, -s, s]), np.array([s, -s, s]), np.array([s, s, s]), np.array([-s, s, s])
        ]

        # Rotation matrix for sway
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot_matrix = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
        
        # Transform points
        transformed_points = []
        for p in points_3d:
            rotated_p = np.dot(p, rot_matrix)
            final_pos = rotated_p + pos
            transformed_points.append(self._project(final_pos))

        # Simplified face drawing for fixed isometric view
        top_face = [transformed_points[i] for i in [7, 6, 2, 3]]
        right_face = [transformed_points[i] for i in [6, 5, 1, 2]]
        left_face = [transformed_points[i] for i in [7, 4, 0, 3]]

        # Shading
        light_color = tuple(min(255, int(c * 1.0)) for c in color)
        mid_color = tuple(min(255, int(c * 0.8)) for c in color)
        dark_color = tuple(min(255, int(c * 0.6)) for c in color)

        alpha = 100 if is_ghost else 255
        
        pygame.gfxdraw.filled_polygon(surface, top_face, (*light_color, alpha))
        pygame.gfxdraw.filled_polygon(surface, right_face, (*mid_color, alpha))
        pygame.gfxdraw.filled_polygon(surface, left_face, (*dark_color, alpha))

        if not is_ghost:
            pygame.gfxdraw.aapolygon(surface, top_face, light_color)
            pygame.gfxdraw.aapolygon(surface, right_face, mid_color)
            pygame.gfxdraw.aapolygon(surface, left_face, dark_color)

    def _render_ground(self):
        for i in range(-8, 9):
            start_3d = (i * self.CUBE_SIZE, -8 * self.CUBE_SIZE, 0)
            end_3d = (i * self.CUBE_SIZE, 8 * self.CUBE_SIZE, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GROUND_LINE, self._project(start_3d), self._project(end_3d))
            
            start_3d = (-8 * self.CUBE_SIZE, i * self.CUBE_SIZE, 0)
            end_3d = (8 * self.CUBE_SIZE, i * self.CUBE_SIZE, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GROUND_LINE, self._project(start_3d), self._project(end_3d))

    def _render_wind(self):
        wind_strength = self.wind_speed * 1000
        for p in self.wind_particles:
            p[0] += wind_strength * 0.1 + 0.1 # Move based on wind + constant drift
            if p[0] > self.SCREEN_WIDTH:
                p[0] = 0
                p[1] = self.np_random.uniform(0, self.SCREEN_HEIGHT)
            
            # Draw wind line
            start_pos = (int(p[0]), int(p[1]))
            end_pos = (int(p[0] - max(1, wind_strength)), int(p[1]))
            pygame.draw.aaline(self.screen, self.COLOR_WIND, start_pos, end_pos)

    def _render_tower(self):
        tower_angle = self._get_tower_angle()
        
        all_drawable_cubes = self.cubes
        if self.ghost_cube and not self.game_over:
            all_drawable_cubes = self.cubes + [self.ghost_cube]
            
        all_drawable_cubes.sort(key=lambda c: c["pos"][0] + c["pos"][1])

        # Draw drop shadow first
        for cube in all_drawable_cubes:
            if cube.get("is_base"): continue
            
            shadow_pos = cube["pos"].copy()
            shadow_pos[2] = 0 
            
            rotated_shadow_pos = shadow_pos.copy()
            rotated_shadow_pos[0] += self.tower_sway_offset
            
            self._draw_cube(self.screen, rotated_shadow_pos, 0, (0,0,0), self.CUBE_SIZE, is_ghost=True)

        # Draw cubes
        for cube in all_drawable_cubes:
            is_ghost = cube.get("is_ghost", False)
            angle = 0 if cube.get("is_base") else tower_angle
            
            render_pos = cube["pos"].copy()
            if not cube.get("is_base"):
                render_pos[0] += self.tower_sway_offset
            
            self._draw_cube(self.screen, render_pos, angle, cube["color"], self.CUBE_SIZE, is_ghost)

    def _render_ui(self):
        # Level / Score
        level_text = self.font_main.render(f"LEVEL: {self.level}/{self.VICTORY_LEVEL}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Wind Indicator
        wind_label = self.font_small.render("WIND", True, self.COLOR_TEXT)
        self.screen.blit(wind_label, (self.SCREEN_WIDTH - 120, 15))
        
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (self.SCREEN_WIDTH - 125, 35, 110, 20))
        wind_bar_width = min(100, self.wind_speed * 2000)
        pygame.draw.rect(self.screen, self.COLOR_WIND[:3], (self.SCREEN_WIDTH - 120, 40, wind_bar_width, 10))
        
        # Game Over / Victory Text
        if self.game_over:
            if self.level >= self.VICTORY_LEVEL:
                msg = "TOWER COMPLETE!"
                color = self.CUBE_PALETTE[1]
            else:
                msg = "TOWER COLLAPSED!"
                color = self.CUBE_PALETTE[0]
            
            end_text = self.font_main.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == '__main__':
    # To run this example, you might need to comment out the line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # This is because the example code directly creates a pygame display window.
    
    env = GameEnv()
    
    # Check if we can create a display
    try:
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Cube Stacker")
        human_render = True
    except pygame.error:
        print("Pygame display could not be initialized (likely in a headless environment). Running without visualization.")
        human_render = False

    obs, info = env.reset()
    done = False
    
    movement = 0 # 0:none, 1:up, 2:down, 3:left, 4:right
    space = 0 # 0:released, 1:pressed
    
    running = True
    while running:
        if human_render:
            # Pygame event handling for manual control
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        done = False
                    if event.key == pygame.K_SPACE:
                        space = 1 # Trigger drop for one frame
            
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
        else: # Simple agent for headless mode
            action = env.action_space.sample()
            movement, space, _ = action
            if env.steps % 10 == 0: # Drop a block every 10 steps
                space = 1
            else:
                space = 0


        # Construct action and step environment
        action = [movement, space, 0] # Shift is not used
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if human_render:
            # Render observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        # Reset one-shot actions
        space = 0
        
        if done:
            print(f"Episode finished. Final Score: {info['score']}, Level: {info['level']}")
            if human_render:
                pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
        
        env.clock.tick(GameEnv.TARGET_FPS)
        
    env.close()