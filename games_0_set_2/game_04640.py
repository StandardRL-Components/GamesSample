
# Generated: 2025-08-28T03:01:02.591474
# Source Brief: brief_04640.md
# Brief Index: 4640

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your avatar (the white circle). "
        "Push colored blocks onto their matching target squares."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist block-pushing puzzle (Sokoban). "
        "Solve the puzzle by moving all blocks to their targets in the fewest moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    CELL_SIZE = 40
    MAX_MOVES = 25
    NUM_BLOCKS = 3
    SHUFFLE_STEPS = 50

    COLOR_BG = (34, 38, 54)
    COLOR_GRID = (54, 58, 74)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_WALL = (14, 18, 34)
    COLOR_TEXT = (230, 230, 230)
    COLOR_UI_BG = (44, 48, 64, 200)

    BLOCK_COLORS = [
        (255, 95, 95),   # Red
        (95, 255, 95),   # Green
        (95, 175, 255),  # Blue
        (255, 255, 95),  # Yellow
    ]
    
    TARGET_COLORS = [
        (120, 45, 45),
        (45, 120, 45),
        (45, 80, 120),
        (120, 120, 45),
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables
        self.player_pos = None
        self.blocks = None
        self.targets = None
        self.walls = None
        self.initial_player_pos = None
        self.initial_blocks = None
        self.blocks_on_target_indices = None
        
        self.steps = 0
        self.moves_taken = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        # Initialize state by generating a puzzle
        self.reset()
        
        # Run self-check
        self.validate_implementation()

    def _generate_puzzle(self):
        """Generates a new, solvable puzzle by starting from a solved state and shuffling."""
        self.walls = set()
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((0, y))
            self.walls.add((self.GRID_WIDTH - 1, y))

        valid_cells = [
            (x, y) for x in range(1, self.GRID_WIDTH - 1) for y in range(1, self.GRID_HEIGHT - 1)
        ]
        self.np_random.shuffle(valid_cells)
        
        # Place targets and blocks on them (solved state)
        self.targets = [valid_cells.pop() for _ in range(self.NUM_BLOCKS)]
        self.blocks = list(self.targets)
        
        # Place player
        self.player_pos = valid_cells.pop()

        # Shuffle the puzzle with random moves
        for _ in range(self.SHUFFLE_STEPS):
            move_dir_idx = self.np_random.integers(1, 5)
            self._apply_move_logic(move_dir_idx, update_state=True)

        # Save the starting state for future resets
        self.initial_player_pos = self.player_pos
        self.initial_blocks = list(self.blocks)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.moves_taken = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        if self.initial_player_pos is None:
            self._generate_puzzle()
        else:
            self.player_pos = self.initial_player_pos
            self.blocks = list(self.initial_blocks)
        
        self.blocks_on_target_indices = self._get_blocks_on_target()

        return self._get_observation(), self._get_info()
    
    def _apply_move_logic(self, movement_action, update_state=False):
        """
        Calculates the result of a move without changing the game state unless specified.
        Returns a tuple (did_move, new_player_pos, new_blocks_list).
        """
        if movement_action == 0:
            return False, self.player_pos, list(self.blocks)

        px, py = self.player_pos
        dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement_action]
        
        next_pos = (px + dx, py + dy)
        
        # Case 1: Move into a wall
        if next_pos in self.walls:
            return False, self.player_pos, list(self.blocks)
        
        # Case 2: Move into a block
        if next_pos in self.blocks:
            block_idx = self.blocks.index(next_pos)
            next_block_pos = (next_pos[0] + dx, next_pos[1] + dy)
            
            # If block is blocked by a wall or another block, no move
            if next_block_pos in self.walls or next_block_pos in self.blocks:
                return False, self.player_pos, list(self.blocks)
            else:
                # Valid push
                new_blocks = list(self.blocks)
                new_blocks[block_idx] = next_block_pos
                if update_state:
                    self.player_pos = next_pos
                    self.blocks = new_blocks
                return True, next_pos, new_blocks
        
        # Case 3: Move into empty space
        else:
            if update_state:
                self.player_pos = next_pos
            return True, next_pos, list(self.blocks)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False
        
        # Only directional inputs count as a "move"
        if movement > 0:
            self.moves_taken += 1
            reward -= 0.1  # Penalty for each move
            
            # Store which blocks were on target before the move
            prev_on_target = self.blocks_on_target_indices

            did_move, _, _ = self._apply_move_logic(movement, update_state=True)
            
            # Check for rewards related to block placement
            if did_move:
                # # sound: move_sfx
                current_on_target = self._get_blocks_on_target()
                
                newly_on_target = current_on_target - prev_on_target
                newly_off_target = prev_on_target - current_on_target
                
                if newly_on_target:
                    # # sound: block_on_target_sfx
                    reward += len(newly_on_target) * 5.0
                if newly_off_target:
                    reward -= len(newly_off_target) * 5.0 # Discourage moving blocks off targets
                
                self.blocks_on_target_indices = current_on_target

        self.steps += 1
        self.score += reward
        
        # Check for termination conditions
        if len(self.blocks_on_target_indices) == self.NUM_BLOCKS:
            # # sound: win_sfx
            reward += 50
            self.score += 50
            terminated = True
            self.game_over = True
            self.win_message = "PUZZLE SOLVED!"
        elif self.moves_taken >= self.MAX_MOVES:
            # # sound: lose_sfx
            reward -= 50
            self.score -= 50
            terminated = True
            self.game_over = True
            self.win_message = "OUT OF MOVES"

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_blocks_on_target(self):
        on_target = set()
        for i, block_pos in enumerate(self.blocks):
            if block_pos == self.targets[i]:
                on_target.add(i)
        return on_target

    def _world_to_screen(self, x, y):
        """Converts grid coordinates to pixel coordinates for rendering."""
        px = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) / 2 + x * self.CELL_SIZE
        py = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) / 2 + y * self.CELL_SIZE
        return int(px), int(py)

    def _render_game(self):
        """Renders the main game elements (grid, targets, blocks, player)."""
        offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) / 2
        offset_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) / 2
        
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            start = (offset_x + x * self.CELL_SIZE, offset_y)
            end = (offset_x + x * self.CELL_SIZE, offset_y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (offset_x, offset_y + y * self.CELL_SIZE)
            end = (offset_x + self.GRID_WIDTH * self.CELL_SIZE, offset_y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
            
        # Draw targets
        for i, (tx, ty) in enumerate(self.targets):
            px, py = self._world_to_screen(tx, ty)
            rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.TARGET_COLORS[i], rect, border_radius=4)
            pygame.draw.rect(self.screen, self.BLOCK_COLORS[i], rect, width=3, border_radius=4)

        # Draw blocks
        for i, (bx, by) in enumerate(self.blocks):
            px, py = self._world_to_screen(bx, by)
            shadow_rect = pygame.Rect(px + 4, py + 4, self.CELL_SIZE, self.CELL_SIZE)
            block_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            
            is_on_target = i in self.blocks_on_target_indices
            
            pygame.draw.rect(self.screen, (0,0,0,50), shadow_rect, border_radius=8)
            pygame.draw.rect(self.screen, self.BLOCK_COLORS[i], block_rect, border_radius=8)
            
            # Visual feedback for on-target blocks
            if is_on_target:
                pygame.draw.rect(self.screen, (255,255,255), block_rect, width=3, border_radius=8)
                
        # Draw player
        px, py = self._world_to_screen(self.player_pos[0], self.player_pos[1])
        center_x, center_y = px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2
        radius = self.CELL_SIZE // 2 - 4
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        """Renders the UI overlay."""
        # Info panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        
        # Moves text
        moves_text = f"Moves: {self.moves_taken:02d} / {self.MAX_MOVES}"
        moves_surf = self.font_main.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (15, 10))
        
        # Score text
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(right=self.SCREEN_WIDTH - 15, top=10)
        self.screen.blit(score_surf, score_rect)
        
        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.font_big.render(self.win_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

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
            "moves_taken": self.moves_taken,
            "blocks_on_target": len(self.blocks_on_target_indices),
        }

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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless

    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:")
    print(f"  Info: {info}")

    # --- To display the game in a window ---
    # This part is for human testing and visualization
    # It requires a display and is not part of the core headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Sokoban Gym Environment")
        clock = pygame.time.Clock()
        
        running = True
        while running:
            action = [0, 0, 0] # Default to no-op
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    elif event.key == pygame.K_r: # Reset
                        obs, info = env.reset()
                    elif event.key == pygame.K_q: # Quit
                        running = False

            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Move: {info['moves_taken']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
                if terminated:
                    print("Game Over. Press 'R' to reset or 'Q' to quit.")

            # Draw the observation to the display window
            frame = env._get_observation()
            frame = np.transpose(frame, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(30) # Limit frame rate
            
        env.close()

    except pygame.error as e:
        print("\nCould not create display window. Pygame might be in headless mode.")
        print("This is normal if you don't have a display attached.")
        env.close()