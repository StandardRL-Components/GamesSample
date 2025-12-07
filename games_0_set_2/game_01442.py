
# Generated: 2025-08-27T17:09:23.752828
# Source Brief: brief_01442.md
# Brief Index: 1442

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import pygame


class GameEnv(gym.Env):
    """
    A minimalist puzzle game where the player pushes colored pixels into a target zone.
    The game is turn-based, with a limited number of moves. The goal is to get all
    pixels into the target zone before running out of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to push all pixels one step in that direction."
    )

    # User-facing game description
    game_description = (
        "A strategic puzzle game. Push all colored pixels into the green target zone "
        "before you run out of moves. Plan your pushes carefully, as every pixel moves at once!"
    )

    # Frames only advance when an action is received.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        """Initializes the game environment."""
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_PIXELS = 5
        self.MAX_MOVES = 20

        # --- Visuals & Layout ---
        self.CELL_SIZE = 36
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2
        self.PIXEL_SIZE = self.CELL_SIZE - 6
        self.PIXEL_OFFSET = (self.CELL_SIZE - self.PIXEL_SIZE) // 2

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_TARGET = (60, 200, 120)
        self.COLOR_TEXT = (230, 230, 230)
        self.PIXEL_COLORS = [
            (255, 87, 87),   # Red
            (87, 134, 255),  # Blue
            (255, 255, 87),  # Yellow
            (87, 255, 183),  # Cyan
            (255, 87, 255),  # Magenta
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 30)
            self.font_small = pygame.font.SysFont(None, 24)

        # --- Game State (initialized in reset) ---
        self.pixels = []
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.np_random = None

        # Define target zone in grid coordinates
        tz_w, tz_h = 5, 4  # 20% of 100 cells
        tz_x = (self.GRID_SIZE - tz_w) // 2
        tz_y = (self.GRID_SIZE - tz_h) // 2
        self.target_zone_coords = pygame.Rect(tz_x, tz_y, tz_w, tz_h)
        self.target_center = (
            self.target_zone_coords.x + self.target_zone_coords.w / 2,
            self.target_zone_coords.y + self.target_zone_coords.h / 2
        )

        # Final check
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Reset game state variables
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False

        # Generate new pixel layout
        self._generate_pixels()

        return self._get_observation(), self._get_info()

    def step(self, action):
        """Processes an action and advances the game state by one step."""
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # No-op action does not consume a move and gives no reward
        if movement == 0:
            return self._get_observation(), 0, False, False, self._get_info()

        # A valid move is made
        self.moves_left -= 1
        reward = self._perform_push(movement)

        # Check for termination conditions
        all_in_zone = all(self._is_in_target(p['pos']) for p in self.pixels)
        win = all_in_zone
        loss = self.moves_left <= 0 and not win
        terminated = win or loss

        if win:
            reward += 100  # Large reward for winning
            self.game_over = True
        elif loss:
            reward -= 100  # Large penalty for losing
            self.game_over = True

        self.score += reward
        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _perform_push(self, movement):
        """Handles the core game logic of pushing pixels and calculating rewards."""
        if movement not in [1, 2, 3, 4]:
            return 0

        # Store pre-move state for reward calculation
        pixels_in_zone_before = sum(1 for p in self.pixels if self._is_in_target(p['pos']))
        dist_before = sum(self._manhattan_distance(p['pos'], self.target_center) for p in self.pixels)

        # --- Push Logic ---
        # Determine direction and sorting order for collision handling
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[movement]
        
        # Sort pixels to process those at the "front" of the push first
        # This ensures correct chain reactions (A pushes B, B pushes C)
        sort_key = lambda p: p['pos'][0] if dx != 0 else p['pos'][1]
        reverse_sort = dx > 0 or dy > 0
        sorted_pixels = sorted(self.pixels, key=sort_key, reverse=reverse_sort)

        occupied_coords = {p['pos'] for p in self.pixels}

        for pixel in sorted_pixels:
            current_pos = pixel['pos']
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)

            # Check boundaries
            if not (0 <= next_pos[0] < self.GRID_SIZE and 0 <= next_pos[1] < self.GRID_SIZE):
                continue  # Blocked by wall

            # Check collision with other pixels
            if next_pos in occupied_coords:
                continue # Blocked by another pixel

            # If move is valid, update state
            pixel['pos'] = next_pos
            occupied_coords.remove(current_pos)
            occupied_coords.add(next_pos)
            # Sound: 'push.wav'

        # --- Reward Calculation ---
        pixels_in_zone_after = sum(1 for p in self.pixels if self._is_in_target(p['pos']))
        dist_after = sum(self._manhattan_distance(p['pos'], self.target_center) for p in self.pixels)

        # Reward for moving pixels into the target zone
        reward = (pixels_in_zone_after - pixels_in_zone_before) * 1.0
        # Sound: if reward > 0: 'zone_enter.wav'

        # Reward for moving pixels closer to the target zone center
        distance_change = dist_before - dist_after
        reward += distance_change * 0.1

        return reward

    def _generate_pixels(self):
        """Generates a new set of pixels, ensuring they are not in the target zone."""
        self.pixels.clear()
        occupied_positions = set()
        
        for i in range(self.NUM_PIXELS):
            while True:
                pos = (
                    self.np_random.integers(0, self.GRID_SIZE),
                    self.np_random.integers(0, self.GRID_SIZE),
                )
                if pos not in occupied_positions and not self._is_in_target(pos):
                    occupied_positions.add(pos)
                    self.pixels.append({"pos": pos, "color": self.PIXEL_COLORS[i]})
                    break
    
    def _is_in_target(self, pos):
        """Checks if a grid coordinate is within the target zone."""
        return self.target_zone_coords.collidepoint(pos)

    def _manhattan_distance(self, pos1, pos2):
        """Calculates the Manhattan distance between two grid points."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_observation(self):
        """Renders the current game state to a NumPy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        # Convert to numpy array and transpose for Gymnasium's expected format (H, W, C)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        """Returns a dictionary with auxiliary diagnostic information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "pixels_in_zone": sum(1 for p in self.pixels if self._is_in_target(p['pos'])),
        }

    def _render_game(self):
        """Renders the grid, target zone, and pixels."""
        # Render target zone (semi-transparent)
        target_surface = pygame.Surface((self.GRID_WIDTH, self.GRID_HEIGHT), pygame.SRCALPHA)
        tz_pixel_rect = pygame.Rect(
            self.target_zone_coords.x * self.CELL_SIZE,
            self.target_zone_coords.y * self.CELL_SIZE,
            self.target_zone_coords.w * self.CELL_SIZE,
            self.target_zone_coords.h * self.CELL_SIZE
        )
        pygame.draw.rect(target_surface, self.COLOR_TARGET + (80,), tz_pixel_rect, border_radius=5)
        self.screen.blit(target_surface, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET))

        # Render grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Render pixels
        for pixel in self.pixels:
            grid_x, grid_y = pixel['pos']
            pixel_rect = pygame.Rect(
                self.GRID_X_OFFSET + grid_x * self.CELL_SIZE + self.PIXEL_OFFSET,
                self.GRID_Y_OFFSET + grid_y * self.CELL_SIZE + self.PIXEL_OFFSET,
                self.PIXEL_SIZE,
                self.PIXEL_SIZE
            )
            # Draw a subtle shadow/border
            shadow_rect = pixel_rect.copy()
            shadow_rect.move_ip(1, 1)
            pygame.draw.rect(self.screen, (0, 0, 0, 100), shadow_rect, border_radius=4)
            # Draw the main pixel
            pygame.draw.rect(self.screen, pixel['color'], pixel_rect, border_radius=4)

    def _render_ui(self):
        """Renders the UI elements like score and moves left."""
        # Moves Left
        moves_text = self.font_main.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 15))

        # Pixels in Target
        pixels_in_zone = sum(1 for p in self.pixels if self._is_in_target(p['pos']))
        target_text = self.font_main.render(f"IN ZONE: {pixels_in_zone}/{self.NUM_PIXELS}", True, self.COLOR_TEXT)
        text_rect = target_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(target_text, text_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            
            all_in_zone = pixels_in_zone == self.NUM_PIXELS
            message = "LEVEL CLEAR!" if all_in_zone else "OUT OF MOVES"
            color = (150, 255, 150) if all_in_zone else (255, 150, 150)
            
            end_text = self.font_main.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, end_rect)


    def close(self):
        """Cleans up Pygame resources."""
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # and demonstrates how to use the environment.
    
    # Set this to False to see the game window
    HEADLESS = True
    if HEADLESS:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    if not HEADLESS:
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Pixel Pusher")

    running = True
    total_reward = 0
    
    print("--- Pixel Pusher ---")
    print(GameEnv.user_guide)
    print("Press ESC or close the window to quit.")
    
    while running:
        # --- Human Controls ---
        movement_action = 0 # No-op by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    print("--- Resetting Environment ---")
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
                
                # Map keys to actions only if the game is not over
                if not done:
                    if event.key == pygame.K_UP:
                        movement_action = 1
                    elif event.key == pygame.K_DOWN:
                        movement_action = 2
                    elif event.key == pygame.K_LEFT:
                        movement_action = 3
                    elif event.key == pygame.K_RIGHT:
                        movement_action = 4

        # --- Environment Step ---
        if movement_action != 0 and not done:
            # Construct the action for the MultiDiscrete space
            action = [movement_action, 0, 0] # Space and Shift are not used
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            print(f"Move: {info['steps']}, Action: {movement_action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

            if done:
                print(f"--- Game Over ---")
                print(f"Final Score: {info['score']:.2f}, Moves Left: {info['moves_left']}")
                print("Press 'R' to play again.")

        # --- Rendering ---
        if not HEADLESS:
            # The observation is the rendered game frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30) # Limit frame rate

    env.close()