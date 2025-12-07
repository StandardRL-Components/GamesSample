
# Generated: 2025-08-28T00:51:01.590620
# Source Brief: brief_03918.md
# Brief Index: 3918

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move shape, ↑↓ to rotate. Press space to place the shape."
    )

    game_description = (
        "An isometric puzzle game. Place falling shapes to fill the 10x10 grid. "
        "You have a limited number of moves to complete the board. Plan your placements carefully!"
    )

    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 10
    MAX_MOVES = 15
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (30, 35, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_PANEL = (45, 50, 60, 180)
    COLOR_GHOST = (200, 200, 200, 50)
    
    SHAPE_COLORS = [
        (52, 152, 219),  # Blue (I)
        (241, 196, 15),   # Yellow (O)
        (155, 89, 182),  # Purple (T)
        (230, 126, 34),  # Orange (L)
        (46, 204, 113),  # Green (S)
        (231, 76, 60),   # Red (Z)
        (26, 188, 156)   # Teal (J)
    ]

    # --- Shape Definitions (Tetrominoes) ---
    SHAPES = {
        0: [[(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (0, 1), (0, 2), (0, 3)]], # I
        1: [[(0, 0), (1, 0), (0, 1), (1, 1)]], # O
        2: [[(0, 0), (1, 0), (2, 0), (1, 1)], [(1, 0), (0, 1), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (1, 0)], [(0, 0), (0, 1), (1, 1), (0, 2)]], # T
        3: [[(0, 0), (0, 1), (0, 2), (1, 2)], [(0, 1), (1, 1), (2, 1), (0, 0)], [(0, 0), (1, 0), (1, 1), (1, 2)], [(2, 0), (0, 1), (1, 1), (2, 1)]], # L
        4: [[(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]], # S
        5: [[(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 0), (1, 1), (0, 1), (0, 2)]], # Z
        6: [[(1, 0), (1, 1), (1, 2), (0, 2)], [(0, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (1, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0), (2, 1)]] # J
    }
    
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
        
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 36)
        
        # Isometric projection parameters
        self.iso_cell_width = 28
        self.iso_cell_height = 14
        self.iso_block_height = 10
        self.grid_origin_x = self.SCREEN_WIDTH // 2 - self.iso_cell_width
        self.grid_origin_y = 120
        
        self.reset()
        
        # This is a stub for the validation check.
        # It's good practice but not strictly part of the env itself.
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.placed_blocks = [] # List of (x, y, color_index, placement_order)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.last_action_feedback = None # For visual feedback
        
        self._choose_next_shape()
        self._spawn_new_shape()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        
        movement, space_press, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Placement ---
        if space_press:
            # Sfx: Place piece
            reward, placed = self._place_current_shape()
            if placed:
                self.moves_left -= 1
                if np.all(self.grid == 1): # Win condition
                    self.win = True
                    self.game_over = True
                    reward += 50
                    self.last_action_feedback = ("WIN!", (0, 255, 0))
                elif self.moves_left <= 0: # Loss condition
                    self.game_over = True
                    reward -= 50
                    self.last_action_feedback = ("GAME OVER", (255, 0, 0))
                else:
                    self._spawn_new_shape()
            else:
                # Sfx: Invalid placement
                reward = -0.1 # Penalty for trying to place in an invalid spot
                self.last_action_feedback = ("INVALID", (255, 100, 0))

        # --- Handle Movement & Rotation ---
        else:
            reward = -0.01 # Small penalty for taking a step without placing
            self.last_action_feedback = None
            
            old_pos = self.current_pos
            old_rot = self.current_rot
            
            if movement == 1: # Rotate CW (Up Arrow)
                self.current_rot = (self.current_rot + 1) % len(self.SHAPES[self.current_shape_id])
                # Sfx: Rotate
            elif movement == 2: # Rotate CCW (Down Arrow)
                self.current_rot = (self.current_rot - 1 + len(self.SHAPES[self.current_shape_id])) % len(self.SHAPES[self.current_shape_id])
                # Sfx: Rotate
            elif movement == 3: # Move Left
                self.current_pos = (self.current_pos[0] - 1, self.current_pos[1])
                # Sfx: Move
            elif movement == 4: # Move Right
                self.current_pos = (self.current_pos[0] + 1, self.current_pos[1])
                # Sfx: Move

            if not self._is_valid_position(self.get_current_shape_coords()):
                self.current_pos = old_pos
                self.current_rot = old_rot
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        terminated = self.game_over

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
        return {"score": int(self.score), "steps": self.steps, "moves_left": self.moves_left}

    # --- Game Logic Helpers ---

    def _choose_next_shape(self):
        self.next_shape_id = self.np_random.integers(0, len(self.SHAPES))

    def _spawn_new_shape(self):
        self.current_shape_id = self.next_shape_id
        self._choose_next_shape()
        self.current_rot = 0
        self.current_pos = (self.GRID_WIDTH // 2 - 1, self.GRID_HEIGHT // 2 - 1)
        if not self._is_valid_position(self.get_current_shape_coords()):
            # No space left to spawn a piece
            self.game_over = True
            self.last_action_feedback = ("NO SPACE!", (255, 0, 0))

    def get_current_shape_coords(self, offset=(0,0), rot=None):
        shape_template = self.SHAPES[self.current_shape_id][self.current_rot if rot is None else rot]
        return [(c[0] + self.current_pos[0] + offset[0], c[1] + self.current_pos[1] + offset[1]) for c in shape_template]

    def _is_valid_position(self, shape_coords):
        for x, y in shape_coords:
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return False
            if self.grid[x, y] == 1:
                return False
        return True

    def _place_current_shape(self):
        shape_coords = self.get_current_shape_coords()
        if not self._is_valid_position(shape_coords):
            return 0, False

        # Calculate reward before updating grid
        placement_reward = len(shape_coords) # +1 per cell
        
        neighbor_count = 0
        for x, y in shape_coords:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if self.grid[nx, ny] == 1:
                        neighbor_count += 1
        
        if neighbor_count <= 1:
            placement_reward += 5 # Risky placement bonus
            self.last_action_feedback = ("RISKY! +5", (50, 255, 150))
        elif neighbor_count > 2:
            placement_reward -= 0.2 # Safe placement penalty
            self.last_action_feedback = ("SAFE", (200, 200, 100))
        else:
            self.last_action_feedback = ("PLACED", (220, 220, 220))

        # Update grid and state
        for x, y in shape_coords:
            self.grid[x, y] = 1
            self.placed_blocks.append((x, y, self.current_shape_id, self.MAX_MOVES - self.moves_left))
        
        self.score += placement_reward
        return placement_reward, True

    # --- Rendering Helpers ---
    
    def _iso_to_screen(self, x, y):
        screen_x = self.grid_origin_x + (x - y) * self.iso_cell_width
        screen_y = self.grid_origin_y + (x + y) * self.iso_cell_height
        return int(screen_x), int(screen_y)

    def _draw_iso_block(self, surface, x, y, color, height_offset=0):
        top_point = self._iso_to_screen(x, y)
        top_point = (top_point[0], top_point[1] - height_offset)
        
        p = [
            (top_point[0] + self.iso_cell_width, top_point[1]),
            (top_point[0] + self.iso_cell_width * 2, top_point[1] + self.iso_cell_height),
            (top_point[0] + self.iso_cell_width, top_point[1] + self.iso_cell_height * 2),
            (top_point[0], top_point[1] + self.iso_cell_height)
        ]
        
        side_color1 = tuple(max(0, c - 40) for c in color)
        side_color2 = tuple(max(0, c - 60) for c in color)

        # Right face
        pygame.gfxdraw.filled_polygon(surface, [p[0], p[1], (p[1][0], p[1][1] + self.iso_block_height), (p[0][0], p[0][1] + self.iso_block_height)], side_color1)
        # Left face
        pygame.gfxdraw.filled_polygon(surface, [p[3], p[0], (p[0][0], p[0][1] + self.iso_block_height), (p[3][0], p[3][1] + self.iso_block_height)], side_color2)
        # Top face
        pygame.gfxdraw.filled_polygon(surface, p, color)
        pygame.gfxdraw.aapolygon(surface, p, color)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            p1 = self._iso_to_screen(i, 0)
            p2 = self._iso_to_screen(i, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for i in range(self.GRID_HEIGHT + 1):
            p1 = self._iso_to_screen(0, i)
            p2 = self._iso_to_screen(self.GRID_WIDTH, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

        # Draw placed blocks
        for x, y, color_idx, placement_order in self.placed_blocks:
            base_color = self.SHAPE_COLORS[color_idx % len(self.SHAPE_COLORS)]
            # Darken based on placement order
            darken_factor = 1.0 - 0.6 * (placement_order / self.MAX_MOVES)
            color = tuple(int(c * darken_factor) for c in base_color)
            self._draw_iso_block(self.screen, x, y, color)
            
        if self.game_over:
            return

        # Draw ghost piece
        ghost_coords = self.get_current_shape_coords() # No need to calculate drop, pieces are placed on the plane
        if self._is_valid_position(ghost_coords):
            for x, y in ghost_coords:
                self._draw_iso_block(self.screen, x, y, self.COLOR_GHOST)

        # Draw current piece
        shape_coords = self.get_current_shape_coords()
        color = self.SHAPE_COLORS[self.current_shape_id % len(self.SHAPE_COLORS)]
        for x, y in shape_coords:
            self._draw_iso_block(self.screen, x, y, color, height_offset=self.iso_block_height * 2)

    def _render_ui(self):
        # UI Panel
        panel_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 60)
        panel_surface = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_surface.fill(self.COLOR_UI_PANEL)
        self.screen.blit(panel_surface, (0,0))
        pygame.draw.line(self.screen, (0,0,0,100), (0, 60), (self.SCREEN_WIDTH, 60), 1)

        # Score
        score_text = self.font_title.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        # Moves Left
        moves_text = self.font_title.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 15))
        
        # Next Shape Preview
        next_text = self.font_main.render("NEXT:", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 150, 80))
        
        next_shape_template = self.SHAPES[self.next_shape_id][0]
        next_color = self.SHAPE_COLORS[self.next_shape_id % len(self.SHAPE_COLORS)]
        for dx, dy in next_shape_template:
            px = self.SCREEN_WIDTH - 120 + dx * 15
            py = 110 + dy * 15
            pygame.draw.rect(self.screen, next_color, (px, py, 14, 14))
            pygame.draw.rect(self.screen, self.COLOR_BG, (px, py, 14, 14), 1)

        # Game Over / Win Message
        if self.game_over:
            msg, color = self.last_action_feedback
            end_text = self.font_title.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)
        elif self.last_action_feedback:
            msg, color = self.last_action_feedback
            feedback_text = self.font_main.render(msg, True, color)
            text_rect = feedback_text.get_rect(center=(self.SCREEN_WIDTH // 2, 85))
            self.screen.blit(feedback_text, text_rect)

    def close(self):
        pygame.quit()

# Example usage and validation
if __name__ == '__main__':
    def validate_implementation(env_instance):
        print("--- Running Implementation Validation ---")
        # Test action space
        assert env_instance.action_space.shape == (3,)
        assert env_instance.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {env_instance.action_space.nvec.tolist()}"
        print("✓ Action space validated.")
        
        # Test observation space  
        test_obs = env_instance._get_observation()
        assert test_obs.shape == (400, 640, 3), f"Observation shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8, f"Observation dtype is {test_obs.dtype}"
        print("✓ Observation space validated.")
        
        # Test reset
        obs, info = env_instance.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        print("✓ reset() validated.")
        
        # Test step
        test_action = env_instance.action_space.sample()
        obs, reward, term, trunc, info = env_instance.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ step() validated.")
        
        print("\n✓ Implementation validated successfully")

    # Create and validate the environment
    env = GameEnv()
    validate_implementation(env)

    # --- Interactive Gameplay Loop ---
    print("\n--- Starting Interactive Gameplay ---")
    print(GameEnv.user_guide)
    
    env.reset()
    running = True
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Puzzle Environment")
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Reset action after one step
        action.fill(0)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            env.reset()

        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                if event.key == pygame.K_q:
                    running = False

        # Get key presses for next action
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1 # Rotate CW
        elif keys[pygame.K_DOWN]:
            action[0] = 2 # Rotate CCW
        elif keys[pygame.K_LEFT]:
            action[0] = 3 # Move Left
        elif keys[pygame.K_RIGHT]:
            action[0] = 4 # Move Right
        
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Place
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1 # Shift (no-op in this game)

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()