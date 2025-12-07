import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:43:21.380888
# Source Brief: brief_00590.md
# Brief Index: 590
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "A strategic puzzle game where you shift groups of colored blocks. "
        "Consolidate blocks of the same color to form a large 3x3 square to win."
    )
    user_guide = (
        "Controls: Use Shift to cycle which color to move. Hold Space and press an "
        "arrow key (↑↓←→) to shift all blocks of the selected color."
    )
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame and Display Setup ---
        self.render_mode = render_mode
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

        # --- Visual Design ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (50, 60, 80)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_BLOCKS = {
            1: (220, 50, 50),   # Red
            2: (50, 150, 220),  # Blue
            3: (50, 220, 150),  # Green
        }
        self.COLOR_HIGHLIGHT = (255, 255, 100)
        self.FONT_UI = pygame.font.SysFont("Consolas", 24)
        self.FONT_MSG = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Game Mechanics & State ---
        self.GRID_SIZE = 5
        self.CELL_SIZE = 60
        self.GRID_MARGIN_X = (self.width - self.GRID_SIZE * self.CELL_SIZE) / 2
        self.GRID_MARGIN_Y = (self.height - self.GRID_SIZE * self.CELL_SIZE) / 2
        self.MAX_STEPS = 1000
        self.BLOCK_COUNT_PER_COLOR = 6

        # State variables are initialized in reset()
        self.grid = None
        self.prev_grid = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.selected_color = 1
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_score_metric = 0
        self.last_2x2_squares = set()
        self.animation_progress = 1.0
        self.animation_duration = 0.3 # seconds

        # Initialize state
        # self.reset() is called by the environment wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.selected_color = 1
        self.prev_space_held = False
        self.prev_shift_held = False
        self.animation_progress = 1.0

        # Generate a valid initial grid
        while True:
            self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
            empty_cells = list(np.ndindex(self.grid.shape))
            self.np_random.shuffle(empty_cells)
            
            for color_id in self.COLOR_BLOCKS.keys():
                for _ in range(self.BLOCK_COUNT_PER_COLOR):
                    if not empty_cells: break
                    r, c = empty_cells.pop()
                    self.grid[r, c] = color_id
            
            if not self._check_for_squares(3): # Ensure no immediate win
                break
        
        self.prev_grid = np.copy(self.grid)

        # Initialize reward metrics
        self.last_score_metric = self._calculate_potential_score()
        self.last_2x2_squares = self._check_for_squares(2)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01 # Small penalty for waiting
        action_taken = False

        # Handle color selection: cycle on SHIFT press
        if shift_held and not self.prev_shift_held:
            # sfx: UI_Cycle.wav
            self.selected_color = (self.selected_color % 3) + 1
        
        # Handle shift action: on SPACE held + direction
        if space_held and movement in [1, 2, 3, 4]:
            action_taken = True
            self.steps += 1
            
            # Store previous grid for animation
            self.prev_grid = np.copy(self.grid)
            
            # Perform the shift
            self._perform_shift(self.selected_color, movement)
            
            # Start animation
            self.animation_progress = 0.0

            # Calculate rewards
            reward = self._calculate_reward()
            self.score += reward

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _perform_shift(self, color, direction):
        # sfx: Block_Shift.wav
        new_grid = np.copy(self.grid)
        components = self._get_all_components(color)
        
        # Clear moving blocks from their original positions to avoid self-collision
        for r, c in np.argwhere(self.grid == color):
            new_grid[r, c] = 0

        for comp in components:
            dist = 1 + (len(comp) - 1) // 2
            
            dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[direction]
            
            for r, c in comp:
                new_r = (r + dr * dist) % self.GRID_SIZE
                new_c = (c + dc * dist) % self.GRID_SIZE
                new_grid[new_r, new_c] = color # Overwrites destination
        
        self.grid = new_grid

    def _calculate_reward(self):
        reward = 0
        
        # Continuous reward for forming larger groups
        current_potential = self._calculate_potential_score()
        reward += current_potential - self.last_score_metric
        self.last_score_metric = current_potential
        
        # Event reward for forming 2x2 squares
        current_2x2_squares = self._check_for_squares(2)
        newly_formed_2x2 = current_2x2_squares - self.last_2x2_squares
        if newly_formed_2x2:
            # sfx: Reward_Small.wav
            reward += 5 * len(newly_formed_2x2)
        self.last_2x2_squares = current_2x2_squares

        # Goal-oriented rewards for terminal states
        if self._check_for_squares(3):
            # sfx: Win.wav
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            # sfx: Lose.wav
            reward -= 10
            
        return reward

    def _calculate_potential_score(self):
        sizes = self._get_largest_group_sizes()
        # Square sizes to heavily reward consolidation
        return sum(size ** 2 for size in sizes.values()) / 10.0

    def _check_termination(self):
        if self.game_over:
            return True
        
        if self._check_for_squares(3):
            self.game_over = True
            self.win_state = True
            return True
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        # Update animation progress
        if self.animation_progress < 1.0:
            self.animation_progress += 1.0 / (self.metadata["render_fps"] * self.animation_duration)
            self.animation_progress = min(1.0, self.animation_progress)

        # Render all game elements
        self._render_all()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_blocks()
        self._render_ui()
        if self.game_over:
            self._render_game_over_message()

    def _render_grid_and_blocks(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_MARGIN_X + i * self.CELL_SIZE, self.GRID_MARGIN_Y)
            end_pos = (self.GRID_MARGIN_X + i * self.CELL_SIZE, self.GRID_MARGIN_Y + self.GRID_SIZE * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_MARGIN_X, self.GRID_MARGIN_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_MARGIN_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_MARGIN_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Find where blocks were and where they are now
        moved_blocks = []
        static_blocks = []
        
        # Easing function for smooth animation
        anim_t = 1 - (1 - self.animation_progress) ** 3 # Ease-out cubic

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color = self.grid[r, c]
                if color == 0: continue

                # Find where this block came from
                prev_pos = None
                if np.array_equal(self.grid, self.prev_grid):
                    prev_pos = (r, c)
                else:
                    # This is a simplification; for complex shuffles it might not be perfect
                    # but works for this game's deterministic movement.
                    # We search for a block of the same color that is now empty.
                    possible_origins = np.argwhere((self.prev_grid == color) & (self.grid != color))
                    if len(possible_origins) > 0:
                        # Find the most likely origin based on shift direction
                        # For simplicity, we just take the first one found.
                        # A better method would be to track moves explicitly.
                        min_dist = float('inf')
                        best_origin = None
                        for pr, pc in possible_origins:
                            dist_sq = (pr-r)**2 + (pc-c)**2
                            if dist_sq < min_dist:
                                min_dist = dist_sq
                                best_origin = (pr, pc)
                        if best_origin:
                            prev_pos = best_origin
                
                if prev_pos is None or (prev_pos[0] == r and prev_pos[1] == c):
                    static_blocks.append(((r, c), color))
                else:
                    moved_blocks.append((prev_pos, (r, c), color))

        # Draw static blocks
        for (r, c), color in static_blocks:
            self._draw_block(r, c, color, 1.0)
        
        # Draw moving blocks with interpolation
        for (pr, pc), (r, c), color in moved_blocks:
            interp_r = pr + (r - pr) * anim_t
            interp_c = pc + (c - pc) * anim_t
            self._draw_block(interp_r, interp_c, color, 1.0)

    def _draw_block(self, r, c, color_id, alpha_mult=1.0):
        block_rect = pygame.Rect(
            self.GRID_MARGIN_X + c * self.CELL_SIZE,
            self.GRID_MARGIN_Y + r * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        
        padding = 4
        inner_rect = block_rect.inflate(-padding*2, -padding*2)
        
        color = self.COLOR_BLOCKS[color_id]
        
        # Draw highlight for selected color
        if color_id == self.selected_color and not self.game_over:
            highlight_rect = block_rect.inflate(-padding, -padding)
            pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, highlight_rect, border_radius=8)

        # Draw the block itself
        pygame.draw.rect(self.screen, color, inner_rect, border_radius=6)
        
        # Add a subtle 3D effect
        light_color = tuple(min(255, x + 25) for x in color)
        dark_color = tuple(max(0, x - 25) for x in color)
        top_left_pts = [inner_rect.topleft, inner_rect.topright, (inner_rect.right-2, inner_rect.top+2), (inner_rect.left+2, inner_rect.top+2)]
        bottom_right_pts = [inner_rect.bottomleft, inner_rect.bottomright, (inner_rect.right-2, inner_rect.bottom-2), (inner_rect.left+2, inner_rect.bottom-2)]
        pygame.draw.polygon(self.screen, light_color, [inner_rect.topleft, inner_rect.bottomleft, (inner_rect.left+2, inner_rect.bottom-2), (inner_rect.left+2, inner_rect.top+2)])
        pygame.draw.polygon(self.screen, dark_color, [inner_rect.topright, inner_rect.bottomright, (inner_rect.right-2, inner_rect.bottom-2), (inner_rect.right-2, inner_rect.top+2)])


    def _render_ui(self):
        # Turns display
        turns_text = self.FONT_UI.render(f"TURN: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(turns_text, (20, 20))
        
        # Score display
        score_text = self.FONT_UI.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 50))
        
        # Selected color indicator
        sel_text = self.FONT_UI.render("SHIFTING:", True, self.COLOR_TEXT)
        self.screen.blit(sel_text, (self.width - 160, 20))
        
        color_rect = pygame.Rect(self.width - 160, 50, 40, 40)
        pygame.draw.rect(self.screen, self.COLOR_BLOCKS[self.selected_color], color_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, color_rect, 3, border_radius=8)

        # Controls hint
        controls1 = self.FONT_UI.render("SHIFT: Cycle Color", True, self.COLOR_GRID)
        controls2 = self.FONT_UI.render("SPACE+ARROWS: Move", True, self.COLOR_GRID)
        self.screen.blit(controls1, (self.width - 200, self.height - 60))
        self.screen.blit(controls2, (self.width - 200, self.height - 35))

    def _render_game_over_message(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "YOU WIN!" if self.win_state else "TIME UP!"
        color = (100, 255, 150) if self.win_state else (255, 100, 100)
        
        text_surf = self.FONT_MSG.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.width / 2, self.height / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    # --- Helper methods for game logic ---
    def _find_connected_component(self, r, c, grid, visited):
        if not (0 <= r < self.GRID_SIZE and 0 <= c < self.GRID_SIZE) or visited[r, c]:
            return set()
        
        color = grid[r, c]
        if color == 0:
            return set()

        component = set()
        q = deque([(r, c)])
        visited[r, c] = True
        
        while q:
            row, col = q.popleft()
            component.add((row, col))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and
                        not visited[nr, nc] and grid[nr, nc] == color):
                    visited[nr, nc] = True
                    q.append((nr, nc))
        return component

    def _get_all_components(self, color):
        visited = np.zeros_like(self.grid, dtype=bool)
        components = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == color and not visited[r, c]:
                    component = self._find_connected_component(r, c, self.grid, visited)
                    if component:
                        components.append(component)
        return components

    def _get_largest_group_sizes(self):
        sizes = {c: 0 for c in self.COLOR_BLOCKS.keys()}
        for color in self.COLOR_BLOCKS.keys():
            components = self._get_all_components(color)
            if components:
                sizes[color] = max(len(comp) for comp in components)
        return sizes

    def _check_for_squares(self, size):
        colors_with_squares = set()
        for color in self.COLOR_BLOCKS.keys():
            for r in range(self.GRID_SIZE - size + 1):
                for c in range(self.GRID_SIZE - size + 1):
                    subgrid = self.grid[r:r+size, c:c+size]
                    if np.all(subgrid == color):
                        colors_with_squares.add(color)
                        break
                if color in colors_with_squares:
                    break
        return colors_with_squares

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example of how to run the environment ---
    # This main block is for human play and debugging, not used by the tests
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Manual play loop
    running = True
    
    # Use a separate display for human play
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Color Shift Puzzle")
    
    # Action state
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = 0
    shift_held = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle key presses for manual control
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
            
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0
                elif event.key == pygame.K_SPACE: space_held = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0

        # Construct action from key states
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a bit before auto-resetting for the player
            pygame.time.wait(3000)
            obs, info = env.reset()

        env.clock.tick(env.metadata["render_fps"])

    env.close()