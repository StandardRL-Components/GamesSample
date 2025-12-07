import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:57:58.331001
# Source Brief: brief_00781.md
# Brief Index: 781
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a drone collects power-ups on a grid.
    The drone's speed increases with each power-up collected.
    The episode ends upon collecting all power-ups, hitting a wall, or reaching the step limit.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Pilot a high-speed drone to collect all the power-ups in a walled arena. "
        "Each collection increases your speed, but be careful not to crash into the walls."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to move the drone."
    auto_advance = False

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WALL_THICKNESS = 15
    GRID_SPACING = 40
    MAX_STEPS = 2000
    NUM_POWERUPS = 10

    # Drone properties
    DRONE_SIZE = 12
    INITIAL_DRONE_SPEED = 2.0
    SPEED_INCREMENT = 1.0
    TRAIL_LENGTH = 20

    # Power-up properties
    POWERUP_RADIUS = 8
    POWERUP_GLOW_FACTOR = 2.5

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_WALL = (80, 80, 100)
    COLOR_DRONE = (0, 191, 255)
    COLOR_DRONE_GLOW = (0, 191, 255, 50)
    COLOR_POWERUP = (50, 255, 100)
    COLOR_POWERUP_GLOW = (50, 255, 100, 60)
    COLOR_UI_TEXT = (220, 220, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)

        # Initialize state variables
        self.drone_pos = None
        self.drone_speed = None
        self.powerups = []
        self.powerups_collected = None
        self.trail = None
        self.steps = None
        self.score = None
        self.game_over = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.powerups_collected = 0
        self.drone_speed = self.INITIAL_DRONE_SPEED

        # Place drone in the center, ensuring it's not in a wall
        self.drone_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0], dtype=float)

        # Spawn power-ups
        self._spawn_powerups()

        # Initialize trail
        self.trail = deque(maxlen=self.TRAIL_LENGTH)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and return the last state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- Game Logic ---
        dist_before = self._get_closest_powerup_dist()

        # 1. Unpack and process action
        movement = action[0]  # 0-4: none/up/down/left/right
        self._update_drone(movement)

        # 2. Check for collisions and events
        hit_wall = self._check_wall_collision()
        collected_powerup_event = self._check_powerup_collection()
        
        # 3. Update game state based on events
        if collected_powerup_event:
            self.powerups_collected += 1
            self.drone_speed += self.SPEED_INCREMENT
            # sfx: Powerup collect sound

        if hit_wall:
            self.game_over = True
            # sfx: Wall collision sound

        # 4. Check for termination conditions
        won_game = self.powerups_collected == self.NUM_POWERUPS
        max_steps_reached = self.steps >= self.MAX_STEPS
        terminated = hit_wall or won_game or max_steps_reached
        self.game_over = terminated

        # --- Reward Calculation ---
        dist_after = self._get_closest_powerup_dist()
        reward = self._calculate_reward(dist_before, dist_after, collected_powerup_event, hit_wall, won_game)
        self.score += reward # Accumulate total reward for info dict

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_reward(self, dist_before, dist_after, collected_powerup, hit_wall, won_game):
        reward = 0.0
        
        if hit_wall:
            return -100.0
        
        if won_game:
            return 100.0
        
        if collected_powerup:
            reward += 10.0
        
        # Continuous reward for moving towards the nearest power-up
        if dist_after < dist_before:
            reward += 0.1 # Small reward for getting closer
        else:
            reward -= 0.1 # Small penalty for moving away

        return reward

    def _update_drone(self, movement):
        velocity = np.array([0.0, 0.0])
        if movement == 1:  # Up
            velocity[1] = -self.drone_speed
        elif movement == 2:  # Down
            velocity[1] = self.drone_speed
        elif movement == 3:  # Left
            velocity[0] = -self.drone_speed
        elif movement == 4:  # Right
            velocity[0] = self.drone_speed
        
        # sfx: Drone hum, pitch increases with speed
        
        # Update trail before moving
        if np.any(velocity): # Only add to trail if moving
            self.trail.append(self.drone_pos.copy())

        self.drone_pos += velocity

    def _check_wall_collision(self):
        half_size = self.DRONE_SIZE / 2
        return (self.drone_pos[0] - half_size < self.WALL_THICKNESS or
                self.drone_pos[0] + half_size > self.SCREEN_WIDTH - self.WALL_THICKNESS or
                self.drone_pos[1] - half_size < self.WALL_THICKNESS or
                self.drone_pos[1] + half_size > self.SCREEN_HEIGHT - self.WALL_THICKNESS)

    def _check_powerup_collection(self):
        collected_any = False
        for i in range(len(self.powerups) - 1, -1, -1):
            powerup_pos = self.powerups[i]
            dist = np.linalg.norm(self.drone_pos - powerup_pos)
            if dist < (self.DRONE_SIZE / 2 + self.POWERUP_RADIUS):
                self.powerups.pop(i)
                collected_any = True
        return collected_any

    def _get_closest_powerup_dist(self):
        if not self.powerups:
            return 0
        distances = [np.linalg.norm(self.drone_pos - p) for p in self.powerups]
        return min(distances) if distances else 0

    def _spawn_powerups(self):
        self.powerups = []
        
        # Create a grid of possible spawn points
        valid_spawns = []
        min_x = self.WALL_THICKNESS + self.POWERUP_RADIUS + self.GRID_SPACING
        max_x = self.SCREEN_WIDTH - self.WALL_THICKNESS - self.POWERUP_RADIUS - self.GRID_SPACING
        min_y = self.WALL_THICKNESS + self.POWERUP_RADIUS + self.GRID_SPACING
        max_y = self.SCREEN_HEIGHT - self.WALL_THICKNESS - self.POWERUP_RADIUS - self.GRID_SPACING
        
        for x in range(int(min_x), int(max_x), self.GRID_SPACING):
            for y in range(int(min_y), int(max_y), self.GRID_SPACING):
                valid_spawns.append(np.array([x, y], dtype=float))

        # Randomly select spawn points from the grid
        if len(valid_spawns) >= self.NUM_POWERUPS:
            chosen_indices = self.np_random.choice(len(valid_spawns), self.NUM_POWERUPS, replace=False)
            for i in chosen_indices:
                self.powerups.append(valid_spawns[i])
        else:
            # Fallback if grid is too small for the number of powerups
            for _ in range(self.NUM_POWERUPS):
                 self.powerups.append(random.choice(valid_spawns))


    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "powerups_collected": self.powerups_collected,
            "drone_speed": self.drone_speed,
        }

    def _render_game(self):
        # 1. Render grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SPACING):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SPACING):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # 2. Render walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.SCREEN_HEIGHT - self.WALL_THICKNESS, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # 3. Render trail
        if len(self.trail) > 1:
            for i in range(len(self.trail) - 1):
                start_pos = self.trail[i]
                end_pos = self.trail[i+1]
                alpha = int(255 * (i / self.TRAIL_LENGTH))
                color = (*self.COLOR_DRONE[:3], alpha)
                pygame.draw.line(self.screen, color, start_pos.astype(int), end_pos.astype(int), width=max(1, int(self.DRONE_SIZE * (i / self.TRAIL_LENGTH))))

        # 4. Render power-ups
        for pos in self.powerups:
            self._draw_glowing_circle(self.screen, pos, self.POWERUP_RADIUS, self.COLOR_POWERUP, self.COLOR_POWERUP_GLOW)

        # 5. Render drone
        if not self.game_over:
            drone_rect = pygame.Rect(
                self.drone_pos[0] - self.DRONE_SIZE / 2,
                self.drone_pos[1] - self.DRONE_SIZE / 2,
                self.DRONE_SIZE, self.DRONE_SIZE
            )
            # Glow
            glow_surf = pygame.Surface((self.DRONE_SIZE * 2, self.DRONE_SIZE * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_DRONE_GLOW, (self.DRONE_SIZE, self.DRONE_SIZE), self.DRONE_SIZE)
            self.screen.blit(glow_surf, (drone_rect.centerx - self.DRONE_SIZE, drone_rect.centery - self.DRONE_SIZE))
            # Core
            pygame.draw.rect(self.screen, self.COLOR_DRONE, drone_rect)
            
    def _draw_glowing_circle(self, surface, center, radius, color, glow_color):
        """Helper to draw a circle with a neon glow."""
        center_int = center.astype(int)
        
        # Glow
        glow_radius = int(radius * self.POWERUP_GLOW_FACTOR)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(s, (center_int[0] - glow_radius, center_int[1] - glow_radius))

        # Core circle (using anti-aliased drawing for smoothness)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), color)


    def _render_ui(self):
        ui_text = f"Power-ups: {self.powerups_collected} / {self.NUM_POWERUPS}"
        text_surface = self.font.render(ui_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.WALL_THICKNESS + 10, self.WALL_THICKNESS + 5))
        
        if self.game_over:
            if self.powerups_collected == self.NUM_POWERUPS:
                end_text = "MISSION COMPLETE"
            else:
                end_text = "DRONE DESTROYED"
            
            end_font = pygame.font.SysFont("Consolas", 60, bold=True)
            end_surf = end_font.render(end_text, True, self.COLOR_UI_TEXT)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_surf, end_rect)


    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        # print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Drone Collector")
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode finished! Score: {info['score']:.2f}, Steps: {info['steps']}")

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata["render_fps"])

    env.close()