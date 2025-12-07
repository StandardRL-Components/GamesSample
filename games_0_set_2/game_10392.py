import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:18:04.465463
# Source Brief: brief_00392.md
# Brief Index: 392
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Tectonic Dominance: A Gymnasium environment where the agent manipulates tectonic plates
    to reshape a world, aiming to control the majority of the landmass.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manipulate tectonic plates to reshape the world. Shift continents to raise land "
        "above sea level and achieve continental dominance."
    )
    user_guide = (
        "Controls: Use arrow keys to move the selected tectonic plate. "
        "Press space to cycle to the next plate, or shift to cycle to the previous one."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 64, 40
    CELL_WIDTH = SCREEN_WIDTH // GRID_WIDTH
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT

    # Colors
    COLOR_BG = (25, 45, 90)  # Deep Ocean Blue
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_UI_BAR_BG = (50, 50, 50)
    COLOR_UI_BAR_FG = (60, 200, 255)
    COLOR_PLATE_BORDER = (255, 255, 0, 100)
    COLOR_PLATE_SELECTED = (255, 0, 0, 200)

    # Land color gradient from low to high elevation
    LAND_COLORS = [
        (34, 139, 34),   # Forest Green
        (85, 107, 47),   # Dark Olive Green
        (139, 69, 19),   # Saddle Brown
        (160, 82, 45),   # Sienna
        (245, 245, 245), # White Smoke (Peaks)
    ]
    
    # Game parameters
    SEA_LEVEL = 50
    MAX_ELEVATION = 255
    STABILITY_THRESHOLD = 30  # How much higher a cell can be than its neighbor before collapsing
    EROSION_ITERATIONS = 3    # How many times to run the landslide simulation per step
    EROSION_RATE = 0.25       # How much material moves in a landslide event
    WIN_PERCENTAGE = 80.0
    MAX_STEPS = 500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game State Variables ---
        self.terrain_elevation = None
        self.plates = []
        self.selected_plate_index = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_land_percentage = 0.0
        self.last_space_held = False
        self.last_shift_held = False
        self.shockwaves = [] # For visual effects

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shockwaves = []

        self._generate_initial_terrain()
        self._define_plates()
        
        self.selected_plate_index = 0
        self.last_land_percentage = self._calculate_land_percentage()
        
        # Reset action press trackers
        self.last_space_held = True # Prevent action on first frame
        self.last_shift_held = True 
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # On subsequent steps after termination, just return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack and Process Actions ---
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held

        reward = 0
        
        # Action: Cycle plates
        if space_press:
            # sfx: UI_SELECT
            self.selected_plate_index = (self.selected_plate_index + 1) % len(self.plates)
        if shift_press:
            # sfx: UI_SELECT
            self.selected_plate_index = (self.selected_plate_index - 1 + len(self.plates)) % len(self.plates)

        # Action: Move selected plate
        if movement != 0:
            # sfx: EARTHQUAKE_RUMBLE
            self._apply_plate_movement(movement)
            self._simulate_erosion_and_landslides()
            
        # --- Update Game Logic ---
        self.steps += 1
        reward += self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()

        # Update last held states
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        self._render_effects()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "land_percentage": self._calculate_land_percentage(),
        }

    # --- Core Game Mechanics ---

    def _generate_initial_terrain(self):
        self.terrain_elevation = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), self.SEA_LEVEL - 20, dtype=np.float32)
        num_continents = self.np_random.integers(3, 6)
        
        for _ in range(num_continents):
            cx, cy = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
            max_h = self.np_random.uniform(150, 250)
            size_x = self.np_random.uniform(self.GRID_WIDTH / 4, self.GRID_WIDTH / 2)
            size_y = self.np_random.uniform(self.GRID_HEIGHT / 4, self.GRID_HEIGHT / 2)

            y, x = np.ogrid[:self.GRID_HEIGHT, :self.GRID_WIDTH]
            dist_sq = ((x - cx) / size_x)**2 + ((y - cy) / size_y)**2
            continent = max_h * np.exp(-dist_sq)
            self.terrain_elevation += continent

        self.terrain_elevation = np.clip(self.terrain_elevation, 0, self.MAX_ELEVATION)

    def _define_plates(self):
        self.plates = []
        plate_configs = [
            (0.25, 0.25, 0.3, 0.3), (0.75, 0.25, 0.3, 0.3),
            (0.25, 0.75, 0.3, 0.3), (0.75, 0.75, 0.3, 0.3),
            (0.5, 0.5, 0.4, 0.4)
        ]
        for cx_ratio, cy_ratio, w_ratio, h_ratio in plate_configs:
            w = int(self.GRID_WIDTH * w_ratio)
            h = int(self.GRID_HEIGHT * h_ratio)
            x = int(self.GRID_WIDTH * cx_ratio - w / 2)
            y = int(self.GRID_HEIGHT * cy_ratio - h / 2)
            self.plates.append(pygame.Rect(x, y, w, h))

    def _apply_plate_movement(self, direction):
        plate_rect = self.plates[self.selected_plate_index]
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[direction]

        # Get the slice of terrain under the plate
        sub_terrain = self.terrain_elevation[plate_rect.top:plate_rect.bottom, plate_rect.left:plate_rect.right]
        
        # Roll the terrain within the plate's area, wrapping around
        rolled_terrain = np.roll(sub_terrain, shift=(dy, dx), axis=(0, 1))

        # Apply the rolled terrain back to the main grid
        self.terrain_elevation[plate_rect.top:plate_rect.bottom, plate_rect.left:plate_rect.right] = rolled_terrain
        
        # sfx: EARTHQUAKE_LARGE
        # Create a visual shockwave
        center_x = (plate_rect.centerx * self.CELL_WIDTH)
        center_y = (plate_rect.centery * self.CELL_HEIGHT)
        self.shockwaves.append({'pos': (center_x, center_y), 'radius': 10, 'max_radius': 150, 'life': 1.0, 'width': 15})

    def _simulate_erosion_and_landslides(self):
        # sfx: LANDSLIDE_SMALL
        for _ in range(self.EROSION_ITERATIONS):
            elevation_copy = self.terrain_elevation.copy()
            # Padded copy to handle boundaries gracefully
            padded = np.pad(elevation_copy, 1, mode='edge')
            
            for r in range(self.GRID_HEIGHT):
                for c in range(self.GRID_WIDTH):
                    # Padded coordinates
                    pr, pc = r + 1, c + 1
                    
                    # Check 4 neighbors
                    neighbors = [(pr-1, pc), (pr+1, pc), (pr, pc-1), (pr, pc+1)]
                    current_h = padded[pr, pc]

                    for nr, nc in neighbors:
                        neighbor_h = padded[nr, nc]
                        if current_h > neighbor_h + self.STABILITY_THRESHOLD:
                            # Landslide occurs
                            transfer_amount = (current_h - neighbor_h) * self.EROSION_RATE
                            # Update the original grid (non-padded coordinates)
                            self.terrain_elevation[r, c] -= transfer_amount
                            # Convert neighbor's padded coords back to original
                            or_r, or_c = max(0, min(self.GRID_HEIGHT-1, nr-1)), max(0, min(self.GRID_WIDTH-1, nc-1))
                            self.terrain_elevation[or_r, or_c] += transfer_amount
            
            # Ensure elevation stays within bounds
            np.clip(self.terrain_elevation, 0, self.MAX_ELEVATION, out=self.terrain_elevation)

    def _calculate_land_percentage(self):
        land_cells = np.sum(self.terrain_elevation > self.SEA_LEVEL)
        total_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        return (land_cells / total_cells) * 100 if total_cells > 0 else 0

    def _calculate_reward(self):
        current_land_percentage = self._calculate_land_percentage()
        reward = (current_land_percentage - self.last_land_percentage) * 0.1
        
        if current_land_percentage > self.last_land_percentage and self.steps > 0:
            reward += 1.0 # Event-based reward for increasing land

        self.last_land_percentage = current_land_percentage
        
        if current_land_percentage >= self.WIN_PERCENTAGE:
            reward += 100.0
        elif current_land_percentage <= 0.1: # Effectively all land is gone
            reward -= 100.0
            
        return reward

    def _check_termination(self):
        land_percentage = self._calculate_land_percentage()
        if land_percentage >= self.WIN_PERCENTAGE:
            self.game_over = True
            # sfx: VICTORY_FANFARE
            return True
        if land_percentage <= 0.1 and self.steps > 1: # Small buffer to not lose on frame 1
            self.game_over = True
            # sfx: DEFEAT_SOUND
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    # --- Rendering ---

    def _render_game(self):
        color_levels = np.linspace(self.SEA_LEVEL, self.MAX_ELEVATION, len(self.LAND_COLORS)).tolist()

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                elevation = self.terrain_elevation[r, c]
                if elevation > self.SEA_LEVEL:
                    # Determine color based on elevation
                    color_index = 0
                    for i, level in enumerate(color_levels):
                        if elevation >= level:
                            color_index = i
                    color = self.LAND_COLORS[color_index]
                    
                    rect = pygame.Rect(c * self.CELL_WIDTH, r * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
                    pygame.draw.rect(self.screen, color, rect)

        # Draw plate boundaries
        for i, plate_rect in enumerate(self.plates):
            screen_rect = pygame.Rect(
                plate_rect.left * self.CELL_WIDTH, plate_rect.top * self.CELL_HEIGHT,
                plate_rect.width * self.CELL_WIDTH, plate_rect.height * self.CELL_HEIGHT
            )
            if i == self.selected_plate_index:
                pygame.draw.rect(self.screen, self.COLOR_PLATE_SELECTED, screen_rect, 4)
            else:
                s = pygame.Surface((screen_rect.width, screen_rect.height), pygame.SRCALPHA)
                pygame.draw.rect(s, self.COLOR_PLATE_BORDER, s.get_rect(), 2)
                self.screen.blit(s, screen_rect.topleft)

    def _render_effects(self):
        # Update and draw shockwaves
        for i in range(len(self.shockwaves) - 1, -1, -1):
            wave = self.shockwaves[i]
            wave['radius'] += (wave['max_radius'] - wave['radius']) * 0.1
            wave['life'] -= 0.03
            
            if wave['life'] <= 0:
                self.shockwaves.pop(i)
                continue
            
            alpha = int(255 * wave['life'])
            color = (255, 200, 50, alpha)
            
            # Use gfxdraw for anti-aliased circles
            pygame.gfxdraw.aacircle(self.screen, int(wave['pos'][0]), int(wave['pos'][1]), int(wave['radius']), color)
            if wave['radius'] > 1:
                pygame.gfxdraw.aacircle(self.screen, int(wave['pos'][0]), int(wave['pos'][1]), int(wave['radius'] - 1), color)
    
    def _render_ui(self):
        # Land Control Bar
        land_percentage = self._calculate_land_percentage()
        bar_width = 200
        bar_height = 20
        bar_x, bar_y = 15, 15
        
        # Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        # Foreground
        fill_width = int(bar_width * (land_percentage / 100.0))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FG, (bar_x, bar_y, fill_width, bar_height))
        # Border
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Text
        text_surf = self.font_small.render(f"Land: {land_percentage:.1f}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (bar_x + bar_width + 10, bar_y + 2))
        
        # Steps
        steps_text = self.font_small.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 15, 15))

        # Game Over Text
        if self.game_over:
            land_percentage = self._calculate_land_percentage()
            if land_percentage >= self.WIN_PERCENTAGE:
                msg = "CONTINENTAL DOMINANCE ACHIEVED"
                color = (100, 255, 100)
            else:
                msg = "WORLD SUBMERGED"
                color = (255, 100, 100)
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(text_surf, text_rect)


    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == '__main__':
    # This block will not run in the test environment, but is useful for local development.
    # To run it, you'll need to unset the dummy video driver.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tectonic Dominance")
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # [movement, space, shift]
    
    print("\n--- Manual Control ---")
    print("Arrows: Move Plate")
    print("Space: Next Plate")
    print("Shift: Previous Plate")
    print("R: Reset")
    print("Q: Quit")

    running = True
    while running:
        # --- Event Handling ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                
                # Update actions based on key presses
                if not done:
                    if event.key == pygame.K_UP: action[0] = 1; action_taken = True
                    elif event.key == pygame.K_DOWN: action[0] = 2; action_taken = True
                    elif event.key == pygame.K_LEFT: action[0] = 3; action_taken = True
                    elif event.key == pygame.K_RIGHT: action[0] = 4; action_taken = True
                    
                    if event.key == pygame.K_SPACE: action[1] = 1; action_taken = True
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1; action_taken = True
        
        # --- Step the Environment ---
        # The game is turn-based, so we only step when an action is taken.
        if action_taken and not done:
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            print(f"Step: {info['steps']}, Land: {info['land_percentage']:.2f}%, Reward: {reward:.2f}, Terminated: {terminated}")
            if terminated or truncated:
                done = True
                print("Game Over. Press 'R' to reset or 'Q' to quit.")
        
        # --- Reset actions for next frame ---
        action = [0, 0, 0]

        # --- Rendering ---
        # The environment's observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS

    env.close()