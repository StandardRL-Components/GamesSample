import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:18:42.212248
# Source Brief: brief_00338.md
# Brief Index: 338
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a submarine navigates a 3x3 minefield.

    The agent controls the submarine's movement one tile at a time. Each move
    consumes fuel. Colliding with a mine causes significant fuel loss but removes
    the mine. The goal is to reach the exit tile with at least 50% fuel.

    Visuals are a key focus, with smooth animations for movement, particle effects
    for collisions, and a clean, high-contrast UI.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a submarine through a minefield to reach the exit. "
        "Colliding with mines drains fuel, and you must finish with at least 50% fuel to win."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move your submarine one tile at a time. "
        "Reach the green exit tile to complete the level."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 3
    CELL_SIZE = 100
    GRID_LINE_WIDTH = 2
    MAX_STEPS = 1000
    MAX_OBSTACLES = 6
    MIN_OBSTACLES = 2

    # --- Colors ---
    COLOR_BG = (15, 23, 42)          # Dark Slate Blue
    COLOR_GRID = (51, 65, 85)        # Lighter Slate
    COLOR_PLAYER = (59, 130, 246)    # Bright Blue
    COLOR_PLAYER_GLOW = (37, 99, 235) # Darker Bright Blue
    COLOR_OBSTACLE = (239, 68, 68)   # Bright Red
    COLOR_OBSTACLE_PARTICLE = (251, 113, 133) # Lighter Red
    COLOR_EXIT = (34, 197, 94)       # Bright Green
    COLOR_UI_TEXT = (226, 232, 240)  # Off-white
    COLOR_FUEL_BAR_BG = (71, 85, 105)
    COLOR_FUEL_HIGH = (74, 222, 128) # Green
    COLOR_FUEL_MED = (250, 204, 21)  # Yellow
    COLOR_FUEL_LOW = (248, 113, 113) # Red

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Grid & Positioning ---
        self.grid_width = self.GRID_SIZE * self.CELL_SIZE
        self.grid_height = self.GRID_SIZE * self.CELL_SIZE
        self.grid_top_left = (
            (self.SCREEN_WIDTH - self.grid_width) // 2,
            (self.SCREEN_HEIGHT - self.grid_height) // 2
        )

        # --- Persistent State (across episodes) ---
        self.successful_episodes = 0
        self.num_obstacles = self.MIN_OBSTACLES

        # --- Episode State ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fuel = 100.0
        self.player_pos = [0, 0]
        self.player_visual_pos = [0.0, 0.0]
        self.obstacle_positions = []
        self.exit_pos = [0, 0]
        self.particles = []
        self.last_action_time = 0
        self.message = ""
        self.message_alpha = 0

        # self.reset() # Removed to align with standard gym practice of calling reset externally

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Episode Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fuel = 100.0
        self.particles = []
        self.message = ""
        self.message_alpha = 0

        # --- Determine Obstacle Count for this Episode ---
        self.num_obstacles = min(self.MAX_OBSTACLES, self.MIN_OBSTACLES + self.successful_episodes // 500)

        # --- Place Game Elements ---
        # Generate all possible grid cells, shuffle, and assign
        all_positions = [[x, y] for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_positions)

        self.player_pos = all_positions.pop()
        self.exit_pos = all_positions.pop()
        self.obstacle_positions = [all_positions.pop() for _ in range(self.num_obstacles)]
        
        # Initialize visual position to logical position
        self.player_visual_pos = list(self._grid_to_pixels(self.player_pos))

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If game is over, do nothing but return current state
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]  # 0:none, 1:up, 2:down, 3:left, 4:right
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = 0
        self.steps += 1

        # --- Process Movement ---
        if movement != 0:
            reward -= 0.1  # Cost for taking a move action
            self.fuel = max(0, self.fuel - 10) # Fuel cost for any move attempt
            # // Sound effect: Submarine engine whir

            target_pos = list(self.player_pos)
            if movement == 1:   # Up
                target_pos[1] -= 1
            elif movement == 2: # Down
                target_pos[1] += 1
            elif movement == 3: # Left
                target_pos[0] -= 1
            elif movement == 4: # Right
                target_pos[0] += 1
            
            # Clamp to grid boundaries
            target_pos[0] = np.clip(target_pos[0], 0, self.GRID_SIZE - 1)
            target_pos[1] = np.clip(target_pos[1], 0, self.GRID_SIZE - 1)
            
            self.player_pos = target_pos

            # --- Collision Check ---
            collided_obstacle_index = -1
            for i, obs_pos in enumerate(self.obstacle_positions):
                if self.player_pos == obs_pos:
                    collided_obstacle_index = i
                    break
            
            if collided_obstacle_index != -1:
                # // Sound effect: Explosion
                self.fuel = max(0, self.fuel - 25)
                reward -= 2.5
                collided_pos = self.obstacle_positions.pop(collided_obstacle_index)
                self._create_particles(self._grid_to_pixels(collided_pos))

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_pos == self.exit_pos:
            terminated = True
            if self.fuel >= 50:
                reward += 100
                self.score += 1
                self.successful_episodes += 1
                self._set_message("VICTORY!", self.COLOR_EXIT)
            else:
                reward -= 100 # Penalty for failing the fuel requirement
                self._set_message("LOW FUEL!", self.COLOR_FUEL_LOW)
        
        if self.fuel <= 0:
            terminated = True
            reward -= 100
            self._set_message("OUT OF FUEL", self.COLOR_FUEL_LOW)
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self._set_message("TIME LIMIT", self.COLOR_UI_TEXT)

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        # --- Update Visuals (Interpolation & Particles) ---
        self._update_player_visuals()
        self._update_particles()
        
        # --- Render Game ---
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_elements()
        self._render_particles()
        self._render_player()
        
        # --- Render UI ---
        self._render_ui()
        self._render_message()

        # --- Convert to Numpy Array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "player_pos": self.player_pos,
            "obstacles_remaining": len(self.obstacle_positions),
        }

    # --- Helper & Rendering Methods ---

    def _set_message(self, text, color):
        self.message = text
        self.message_color = color
        self.message_alpha = 255

    def _grid_to_pixels(self, grid_pos):
        """Converts grid coordinates (e.g., [0, 1]) to pixel coordinates for the center of the cell."""
        col, row = grid_pos
        x = self.grid_top_left[0] + (col * self.CELL_SIZE) + (self.CELL_SIZE / 2)
        y = self.grid_top_left[1] + (row * self.CELL_SIZE) + (self.CELL_SIZE / 2)
        return int(x), int(y)

    def _lerp(self, start, end, t):
        """Linear interpolation."""
        return start + t * (end - start)

    def _update_player_visuals(self):
        """Smoothly interpolates the player's visual position towards its logical position."""
        target_px = self._grid_to_pixels(self.player_pos)
        self.player_visual_pos[0] = self._lerp(self.player_visual_pos[0], target_px[0], 0.25)
        self.player_visual_pos[1] = self._lerp(self.player_visual_pos[1], target_px[1], 0.25)

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_x = self.grid_top_left[0] + i * self.CELL_SIZE
            start_y = self.grid_top_left[1]
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (start_x, start_y + self.grid_height), self.GRID_LINE_WIDTH)
            # Horizontal lines
            start_x = self.grid_top_left[0]
            start_y = self.grid_top_left[1] + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (start_x + self.grid_width, start_y), self.GRID_LINE_WIDTH)

    def _render_elements(self):
        # Exit Tile
        exit_px = self._grid_to_pixels(self.exit_pos)
        exit_rect = pygame.Rect(0, 0, self.CELL_SIZE - 10, self.CELL_SIZE - 10)
        exit_rect.center = exit_px
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=8)

        # Obstacles
        for obs_pos in self.obstacle_positions:
            obs_px = self._grid_to_pixels(obs_pos)
            pygame.draw.circle(self.screen, self.COLOR_OBSTACLE, obs_px, self.CELL_SIZE // 3)

    def _render_player(self):
        player_px = (int(self.player_visual_pos[0]), int(self.player_visual_pos[1]))
        
        # Glow effect
        for i in range(15, 0, -2):
            alpha = 100 - (i * 6)
            if alpha > 0:
                color = (*self.COLOR_PLAYER_GLOW, alpha)
                temp_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (self.CELL_SIZE // 2, self.CELL_SIZE // 2), self.CELL_SIZE // 3 + i)
                self.screen.blit(temp_surf, (player_px[0] - self.CELL_SIZE // 2, player_px[1] - self.CELL_SIZE // 2))

        # Player square
        player_rect = pygame.Rect(0, 0, self.CELL_SIZE // 2, self.CELL_SIZE // 2)
        player_rect.center = player_px
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=6)

    def _render_ui(self):
        # Fuel Bar
        fuel_bar_width = 200
        fuel_bar_height = 20
        fuel_bar_x = 20
        fuel_bar_y = 20
        
        # Determine fuel color
        if self.fuel > 60:
            fuel_color = self.COLOR_FUEL_HIGH
        elif self.fuel > 25:
            fuel_color = self.COLOR_FUEL_MED
        else:
            fuel_color = self.COLOR_FUEL_LOW

        # Background
        pygame.draw.rect(self.screen, self.COLOR_FUEL_BAR_BG, (fuel_bar_x, fuel_bar_y, fuel_bar_width, fuel_bar_height), border_radius=5)
        # Foreground (current fuel)
        current_fuel_width = int(fuel_bar_width * (self.fuel / 100.0))
        pygame.draw.rect(self.screen, fuel_color, (fuel_bar_x, fuel_bar_y, max(0, current_fuel_width), fuel_bar_height), border_radius=5)
        
        # Fuel Text
        fuel_text = self.font_ui.render(f"FUEL: {int(self.fuel)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(fuel_text, (fuel_bar_x + fuel_bar_width + 10, fuel_bar_y))

        # Score Text
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

    def _render_message(self):
        if self.message_alpha > 0:
            msg_surf = self.font_msg.render(self.message, True, self.message_color)
            msg_surf.set_alpha(self.message_alpha)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)
            self.message_alpha = max(0, self.message_alpha - 2)

    def _create_particles(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": velocity,
                "life": self.np_random.integers(20, 40),
                "size": self.np_random.uniform(5, 12),
            }
            self.particles.append(particle)
            
    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["size"] *= 0.95
        self.particles = [p for p in self.particles if p["life"] > 0 and p["size"] > 1]

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = int(p["size"])
            if size > 0:
                rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
                # Fade color with life
                alpha = int(255 * (p["life"] / 40.0))
                color = (*self.COLOR_OBSTACLE_PARTICLE, max(0, min(255, alpha)))
                
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, color, (0, 0, size, size), border_radius=2)
                self.screen.blit(temp_surf, rect.topleft)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Need to reset first to initialize everything
        _ = self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert 'score' in info and 'steps' in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == '__main__':
    # Un-comment the line below to run with a display window
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This loop allows a human to play the game.
    # Use arrow keys to move. Press 'R' to reset.
    
    obs, info = env.reset()
    done = False
    
    # Setup a display window
    pygame.display.set_caption("Submarine Minefield")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        # For auto-advancing environments, we step even with no-op actions
        # to allow animations and other time-based events to process.
        
        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    done = True # Signal to reset the env
                
                # Map keys to actions for the next step
                if not env.game_over:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the environment to the display window
        # The observation is the frame, so we just need to display it.
        # Pygame uses (W, H) but obs is (H, W, C), so we transpose.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done:
            print(f"Episode Finished. Score: {info['score']}, Fuel: {info['fuel']:.1f}")
            pygame.time.wait(2000) # Pause for 2 seconds on game over
            obs, info = env.reset()
            done = False

        env.clock.tick(30) # Limit frame rate

    env.close()