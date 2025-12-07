
# Generated: 2025-08-28T03:12:52.250890
# Source Brief: brief_04858.md
# Brief Index: 4858

        
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
    """
    A Gymnasium environment where a robot navigates a grid with rotating lasers.

    **Game Objective:**
    Guide the robot to the green goal tile while avoiding the red laser beams.

    **Gameplay:**
    The game is turn-based. Each action moves the robot one tile. The lasers
    rotate 90 degrees clockwise at a fixed interval of steps. The game becomes
    progressively harder as the laser rotation speed increases with each level.

    **Scoring:**
    - Reaching the goal: +10 points
    - Hitting a laser: -10 points
    - Each step taken: -0.2 points
    - Being adjacent to the goal: +0.1 points (shaping reward)
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the robot. Avoid the red lasers."
    )

    game_description = (
        "Guide a robot through rotating laser grids to its destination. "
        "Plan your path carefully to reach the goal in the fewest steps."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.TILE_SIZE = 40
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_ROBOT = (0, 150, 255)
        self.COLOR_ROBOT_GLOW = (0, 150, 255, 50)
        self.COLOR_GOAL = (0, 255, 100)
        self.COLOR_GOAL_GLOW = (0, 255, 100, 70)
        self.COLOR_EMITTER = (180, 0, 0)
        self.COLOR_LASER = (255, 20, 20)
        self.COLOR_LASER_GLOW = (255, 20, 20, 100)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_SPARK = (255, 220, 180)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 0
        self.laser_rotation_speed = 0
        self.robot_pos = [0, 0]
        self.goal_pos = [0, 0]
        self.lasers = []
        self.particles = []
        self.np_random = None

        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # On the very first reset, or if a full game reset is intended
        if options is None or not options.get("level_complete", False):
            self.level = 1
            self.laser_rotation_speed = 5  # Start with a slow rotation
            self.score = 0
        
        self.steps = 0
        self.game_over = False
        self.particles = []
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Procedurally generates a new level layout."""
        self.lasers = []
        
        # Place robot and goal
        self.robot_pos = [1, self.np_random.integers(1, self.GRID_HEIGHT - 1)]
        self.goal_pos = [self.GRID_WIDTH - 2, self.np_random.integers(1, self.GRID_HEIGHT - 1)]

        # Place laser emitters
        num_lasers = min(3 + self.level, 8)
        for _ in range(num_lasers):
            while True:
                pos = [
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT),
                ]
                # Avoid placing on start/goal or overwriting another emitter
                is_safe_pos = pos != self.robot_pos and pos != self.goal_pos
                is_unique_pos = all(emitter["pos"] != pos for emitter in self.lasers)
                if is_safe_pos and is_unique_pos:
                    break
            
            angle = self.np_random.integers(0, 4) * 90  # 0, 90, 180, 270
            self.lasers.append({"pos": pos, "angle": angle})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.2  # Base penalty for taking a step
        self.steps += 1
        
        # --- 1. Update Game Logic ---
        # Move robot
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.robot_pos[0] = np.clip(self.robot_pos[0] + dx, 0, self.GRID_WIDTH - 1)
            self.robot_pos[1] = np.clip(self.robot_pos[1] + dy, 0, self.GRID_HEIGHT - 1)

        # Rotate lasers
        if self.steps > 0 and self.steps % self.laser_rotation_speed == 0:
            for laser in self.lasers:
                laser["angle"] = (laser["angle"] + 90) % 360
            # sfx: laser_rotate.wav

        # Update particles
        self._update_particles()

        # --- 2. Check for Win/Loss Conditions ---
        terminated = False
        
        # Win: Reached goal (priority over laser collision)
        if self.robot_pos == self.goal_pos:
            reward = 10.0
            terminated = True
            self.game_over = True
            self.level += 1
            self.laser_rotation_speed = max(1, self.laser_rotation_speed - 1)
            # sfx: goal_reached.wav
            # Reset for the next level
            obs, info = self.reset(options={"level_complete": True})
            return obs, reward, False, False, info # Not terminated yet, just new level

        # Loss: Hit by a laser
        for laser in self.lasers:
            if self.robot_pos in self._get_laser_beam_tiles(laser):
                reward = -10.0
                terminated = True
                self.game_over = True
                self._spawn_particles(self._grid_to_pixel(self.robot_pos), 50, self.COLOR_SPARK)
                # sfx: robot_destroyed.wav
                break
        
        # Loss: Max steps reached
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            # sfx: timeout.wav

        # --- 3. Calculate Final Reward ---
        if not terminated:
            # Reward for being near the goal
            dist_to_goal = abs(self.robot_pos[0] - self.goal_pos[0]) + abs(self.robot_pos[1] - self.goal_pos[1])
            if dist_to_goal == 1:
                reward += 0.1

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
        }

    # --- Rendering ---
    def _render_game(self):
        self._draw_grid()
        self._draw_lasers()
        self._draw_goal()
        if not (self.game_over and self.robot_pos not in self._get_laser_beam_tiles_flat()):
            self._draw_robot()
        self._draw_particles()

    def _draw_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.WIDTH, py))

    def _draw_lasers(self):
        for laser in self.lasers:
            # Emitter
            center_px = self._grid_to_pixel(laser["pos"])
            pygame.gfxdraw.filled_trigon(self.screen, 
                center_px[0], center_px[1] - 8,
                center_px[0] - 8, center_px[1] + 8,
                center_px[0] + 8, center_px[1] + 8,
                self.COLOR_EMITTER)
            
            # Beam
            start_px = center_px
            end_pos = self._get_laser_endpoint(laser)
            end_px = self._grid_to_pixel(end_pos)

            # Adjust endpoints to align with grid lines
            if laser["angle"] == 0: end_px = (end_px[0], -self.TILE_SIZE // 2)
            elif laser["angle"] == 90: end_px = (self.WIDTH + self.TILE_SIZE // 2, end_px[1])
            elif laser["angle"] == 180: end_px = (end_px[0], self.HEIGHT + self.TILE_SIZE // 2)
            elif laser["angle"] == 270: end_px = (-self.TILE_SIZE // 2, end_px[1])
            
            # Draw glow
            pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, start_px, end_px, 7)
            # Draw core beam
            pygame.draw.line(self.screen, self.COLOR_LASER, start_px, end_px, 3)

    def _draw_goal(self):
        self._draw_glowing_rect(self.goal_pos, self.COLOR_GOAL, self.COLOR_GOAL_GLOW)

    def _draw_robot(self):
        self._draw_glowing_rect(self.robot_pos, self.COLOR_ROBOT, self.COLOR_ROBOT_GLOW)

    def _draw_glowing_rect(self, grid_pos, color, glow_color):
        center_px = self._grid_to_pixel(grid_pos)
        size = self.TILE_SIZE * 0.6
        rect = pygame.Rect(center_px[0] - size / 2, center_px[1] - size / 2, size, size)
        
        # Glow
        glow_radius = int(size)
        pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], glow_radius, glow_color)
        
        # Core shape
        pygame.draw.rect(self.screen, color, rect, border_radius=3)

    def _render_ui(self):
        steps_text = self.font_ui.render(f"STEPS: {self.steps}", True, self.COLOR_TEXT)
        level_text = self.font_ui.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)

        self.screen.blit(steps_text, (10, 5))
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 5))
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 5))

    # --- Particle System ---
    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({"pos": list(pos), "vel": velocity, "life": lifetime, "max_life": lifetime, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _draw_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            radius = int(life_ratio * 5)
            alpha = int(life_ratio * 255)
            color = (*p["color"], alpha)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, color)

    # --- Helper Functions ---
    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel center coordinates."""
        px = int((grid_pos[0] + 0.5) * self.TILE_SIZE)
        py = int((grid_pos[1] + 0.5) * self.TILE_SIZE)
        return px, py

    def _get_laser_beam_tiles(self, laser):
        """Returns a list of grid coordinates covered by a laser beam."""
        tiles = []
        x, y = laser["pos"]
        angle = laser["angle"]
        if angle == 0:  # Up
            for i in range(y - 1, -1, -1): tiles.append([x, i])
        elif angle == 90:  # Right
            for i in range(x + 1, self.GRID_WIDTH): tiles.append([i, y])
        elif angle == 180:  # Down
            for i in range(y + 1, self.GRID_HEIGHT): tiles.append([x, i])
        elif angle == 270:  # Left
            for i in range(x - 1, -1, -1): tiles.append([i, y])
        return tiles
    
    def _get_laser_beam_tiles_flat(self):
        """Returns a flat list of all tiles currently occupied by any laser."""
        all_tiles = []
        for laser in self.lasers:
            all_tiles.extend(self._get_laser_beam_tiles(laser))
        return all_tiles

    def _get_laser_endpoint(self, laser):
        """Calculates the grid coordinate where a laser beam ends."""
        x, y = laser["pos"]
        angle = laser["angle"]
        if angle == 0: return [x, 0]
        if angle == 90: return [self.GRID_WIDTH - 1, y]
        if angle == 180: return [x, self.GRID_HEIGHT - 1]
        if angle == 270: return [0, y]
        return [x, y]

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up Pygame window for display
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Laser Grid Robot")
    clock = pygame.time.Clock()

    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("Quit: ESC or close window")
    print("="*30 + "\n")

    while not done:
        # --- Action Mapping for Manual Control ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used

        # --- Event Handling ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated
                    action_taken = True
        
        # If no key was pressed, we still need to step with a no-op action
        # because auto_advance is False. However, for a better human experience,
        # we only step when a key is pressed.
        
        # --- Rendering ---
        # The observation is the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS

    print(f"Game Over! Final Score: {info['score']:.1f}, Level: {info['level']}")
    env.close()