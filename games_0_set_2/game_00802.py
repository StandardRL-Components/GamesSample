
# Generated: 2025-08-27T14:50:53.202846
# Source Brief: brief_00802.md
# Brief Index: 802

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    An isometric puzzle game where the player pushes colored boxes onto matching targets.
    The goal is to solve the puzzle in the fewest moves possible.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to select a box. Hold space and press an arrow key to push the selected box."
    )

    # Short, user-facing description of the game
    game_description = (
        "Push colored boxes onto their matching targets in this isometric puzzle. Plan your moves carefully to solve the puzzle before you run out of moves."
    )

    # Frames only advance when an action is received
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        """Initializes the game environment."""
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Game Constants ---
        self.grid_size = 5
        self.max_moves = 15
        self.num_boxes = 4

        # --- Visuals ---
        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 62)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_SELECTION = (255, 255, 255)
        self.COLOR_WIN = (152, 251, 152)
        self.COLOR_LOSE = (255, 105, 97)
        
        self.box_colors = [
            ((255, 99, 71), (139, 0, 0)),    # Red (Tomato, DarkRed)
            ((60, 179, 113), (0, 100, 0)),   # Green (MediumSeaGreen, DarkGreen)
            ((100, 149, 237), (0, 0, 139)),  # Blue (CornflowerBlue, DarkBlue)
            ((255, 215, 0), (184, 134, 11))  # Yellow (Gold, DarkGoldenrod)
        ]
        
        # Fonts
        try:
            self.ui_font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
            self.ui_font_small = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.ui_font_large = pygame.font.SysFont("arial", 36)
            self.ui_font_small = pygame.font.SysFont("arial", 24)

        # Isometric projection parameters
        self.tile_width = 80
        self.tile_height = 40
        self.box_height = 35
        self.origin_x = self.screen_width // 2
        self.origin_y = self.screen_height // 2 - (self.grid_size * self.tile_height // 4)

        # --- State Variables ---
        self.np_random = None
        self.boxes = []
        self.targets = []
        self.boxes_on_target = []
        self.selected_box_idx = 0
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_flash_timer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        
        # Initialize RNG
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Reset game state
        self.moves_left = self.max_moves
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_flash_timer = 0
        self.selected_box_idx = 0
        self.boxes_on_target = [False] * self.num_boxes

        # Generate puzzle layout
        possible_coords = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        shuffled_coords = self.np_random.choice(possible_coords, size=self.num_boxes * 2, replace=False)
        
        self.targets = [tuple(coord) for coord in shuffled_coords[:self.num_boxes]]
        self.boxes = [tuple(coord) for coord in shuffled_coords[self.num_boxes:]]

        # Ensure selected box is valid
        if self.selected_box_idx >= self.num_boxes:
            self.selected_box_idx = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Processes an action and updates the game state.
        - Arrow keys change box selection.
        - Space + Arrow key pushes the selected box.
        """
        # Unpack factorized action
        movement_action = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        
        is_push_action = space_held and movement_action > 0
        is_select_action = not space_held and movement_action > 0

        if is_push_action:
            # --- PUSH LOGIC ---
            if self.moves_left > 0:
                self.moves_left -= 1
                reward -= 0.1  # Cost for making a move
                
                # Get push direction vector
                direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
                dx, dy = direction_map[movement_action]
                
                # Calculate target position
                box_pos = self.boxes[self.selected_box_idx]
                target_pos = (box_pos[0] + dx, box_pos[1] + dy)
                
                # Check for collisions (walls or other boxes)
                is_wall_collision = not (0 <= target_pos[0] < self.grid_size and 0 <= target_pos[1] < self.grid_size)
                is_box_collision = any(b_pos == target_pos for i, b_pos in enumerate(self.boxes))
                
                if not is_wall_collision and not is_box_collision:
                    self.boxes[self.selected_box_idx] = target_pos
                    # sfx: push_sfx
                else:
                    # sfx: bump_sfx
                    pass
            
            # --- POST-MOVE REWARD & STATE UPDATE ---
            all_on_target = True
            for i in range(self.num_boxes):
                box_on_its_target = self.boxes[i] == self.targets[i]
                
                if box_on_its_target and not self.boxes_on_target[i]:
                    reward += 1.0  # Reward for placing a box correctly
                    self.boxes_on_target[i] = True
                    # sfx: target_complete_sfx
                elif not box_on_its_target and self.boxes_on_target[i]:
                    reward -= 1.0  # Penalty for moving a box off its target
                    self.boxes_on_target[i] = False
                
                if not box_on_its_target:
                    all_on_target = False

            # --- CHECK TERMINATION ---
            if all_on_target:
                reward += 100  # Victory bonus
                terminated = True
                self.game_over = True
                self.win_flash_timer = 15 # Start victory flash
                # sfx: victory_sfx
            elif self.moves_left <= 0:
                reward -= 50  # Defeat penalty
                terminated = True
                self.game_over = True
                # sfx: failure_sfx

        elif is_select_action:
            # --- SELECTION LOGIC ---
            # Cycle selection
            if movement_action == 4: # Right
                self.selected_box_idx = (self.selected_box_idx + 1) % self.num_boxes
            elif movement_action == 3: # Left
                self.selected_box_idx = (self.selected_box_idx - 1 + self.num_boxes) % self.num_boxes
            elif movement_action == 2: # Down
                 self.selected_box_idx = (self.selected_box_idx + 2) % self.num_boxes
            elif movement_action == 1: # Up
                 self.selected_box_idx = (self.selected_box_idx - 2 + self.num_boxes) % self.num_boxes
            # sfx: select_sfx
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _grid_to_screen(self, grid_x, grid_y):
        """Converts grid coordinates to screen coordinates for isometric projection."""
        screen_x = self.origin_x + (grid_x - grid_y) * (self.tile_width / 2)
        screen_y = self.origin_y + (grid_x + grid_y) * (self.tile_height / 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, surface, color, grid_pos, y_offset=0):
        """Draws a flat isometric tile on the grid."""
        cx, cy = self._grid_to_screen(grid_pos[0], grid_pos[1])
        cy += y_offset
        points = [
            (cx, cy - self.tile_height / 2),
            (cx + self.tile_width / 2, cy),
            (cx, cy + self.tile_height / 2),
            (cx - self.tile_width / 2, cy)
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_iso_box(self, surface, colors, grid_pos, is_selected):
        """Draws a 3D isometric box."""
        top_color, side_color = colors
        cx, cy = self._grid_to_screen(grid_pos[0], grid_pos[1])
        
        # Draw selection glow first
        if is_selected:
            glow_radius = int(self.tile_width * 0.6)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_SELECTION, 60), (glow_radius, glow_radius), glow_radius)
            pygame.draw.circle(glow_surf, (*self.COLOR_SELECTION, 80), (glow_radius, glow_radius), int(glow_radius * 0.8))
            surface.blit(glow_surf, (cx - glow_radius, cy - glow_radius + int(self.tile_height * 0.25)), special_flags=pygame.BLEND_RGBA_ADD)

        # Top face
        top_points = [
            (cx, cy - self.box_height - self.tile_height / 2),
            (cx + self.tile_width / 2, cy - self.box_height),
            (cx, cy - self.box_height + self.tile_height / 2),
            (cx - self.tile_width / 2, cy - self.box_height)
        ]
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, side_color)
        
        # Right side face
        right_points = [
            (cx, cy + self.tile_height / 2),
            (cx + self.tile_width / 2, cy),
            (cx + self.tile_width / 2, cy - self.box_height),
            (cx, cy - self.box_height + self.tile_height / 2)
        ]
        pygame.gfxdraw.filled_polygon(surface, right_points, side_color)
        pygame.gfxdraw.aapolygon(surface, right_points, side_color)

        # Left side face
        left_points = [
            (cx, cy + self.tile_height / 2),
            (cx - self.tile_width / 2, cy),
            (cx - self.tile_width / 2, cy - self.box_height),
            (cx, cy - self.box_height + self.tile_height / 2)
        ]
        pygame.gfxdraw.filled_polygon(surface, left_points, side_color)
        pygame.gfxdraw.aapolygon(surface, left_points, side_color)

    def _render_game(self):
        """Renders all game elements (grid, targets, boxes)."""
        # Draw grid lines
        for i in range(self.grid_size + 1):
            start_g = self._grid_to_screen(i, 0)
            end_g = self._grid_to_screen(i, self.grid_size)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_g, end_g)
            start_g = self._grid_to_screen(0, i)
            end_g = self._grid_to_screen(self.grid_size, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_g, end_g)

        # Draw targets
        for i, pos in enumerate(self.targets):
            color = self.box_colors[i][0]
            self._draw_iso_tile(self.screen, color, pos, y_offset=2)

        # Draw boxes
        for i, pos in enumerate(self.boxes):
            is_selected = (i == self.selected_box_idx)
            self._draw_iso_box(self.screen, self.box_colors[i], pos, is_selected)
            
            # Flash on completion
            if self.boxes[i] == self.targets[i]:
                flash_alpha = 100 + 100 * math.sin(pygame.time.get_ticks() * 0.01)
                self._draw_iso_tile(self.screen, (*self.COLOR_SELECTION, flash_alpha), pos, y_offset=-self.box_height)


    def _render_ui(self):
        """Renders the UI overlay (moves left, game over text)."""
        # Moves Left
        moves_text = self.ui_font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            
            is_win = all(self.boxes_on_target)
            if is_win:
                if self.win_flash_timer > 0:
                    flash_alpha = self.win_flash_timer * 17 # 15*17 ~ 255
                    overlay.fill((*self.COLOR_WIN, flash_alpha))
                    self.win_flash_timer -= 1
                
                msg = "PUZZLE SOLVED!"
                color = self.COLOR_WIN
            else:
                overlay.fill((*self.COLOR_LOSE, 128))
                msg = "OUT OF MOVES"
                color = self.COLOR_LOSE
            
            self.screen.blit(overlay, (0, 0))
            
            text_surf = self.ui_font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_observation(self):
        """Renders the current game state to an RGB array."""
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game and UI
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array and transpose for Gymnasium
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        """Returns a dictionary with auxiliary diagnostic information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "selected_box": self.selected_box_idx,
        }

    def close(self):
        """Clean up Pygame resources."""
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run headlessly
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:", info)

    # Simulate a few random steps
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}, Terminated: {terminated}")
        if terminated:
            print("--- Episode Finished ---")
            obs, info = env.reset()
            print("--- New Episode Started ---", info)

    # Example of saving a frame as an image
    try:
        from PIL import Image
        img = Image.fromarray(obs)
        img.save("game_frame.png")
        print("\nSaved a sample frame to game_frame.png")
    except ImportError:
        print("\nPIL/Pillow not installed, skipping frame save.")

    env.close()