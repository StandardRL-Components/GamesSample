
# Generated: 2025-08-27T19:33:37.564424
# Source Brief: brief_02192.md
# Brief Index: 2192

        
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


# Block class for state management
class Block:
    def __init__(self, id, initial_pos, target_pos):
        self.id = id
        self.pos = initial_pos
        self.target_pos = target_pos

    def is_in_place(self):
        return self.pos == self.target_pos

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to slide all blocks simultaneously. "
        "Try to arrange them in numerical order."
    )
    game_description = (
        "A minimalist sliding puzzle. All blocks move at once. "
        "Arrange the numbered tiles in order before you run out of moves."
    )

    # The game is turn-based. State is static until an action is received.
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (25, 28, 32)
    COLOR_GRID = (50, 56, 64)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    
    # Block Colors
    COLOR_BLOCK_CORRECT = (76, 175, 80) # Green
    COLOR_BLOCK_WRONG = (211, 47, 47) # Red

    # Grid settings
    GRID_ROWS = 4
    GRID_COLS = 4
    GRID_MARGIN_X = 120
    GRID_MARGIN_Y = 50
    BLOCK_SIZE = 70
    BLOCK_SPACING = 10
    BLOCK_CORNER_RADIUS = 8
    
    MAX_EPISODE_STEPS = 1000

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
        
        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_block = pygame.font.Font(None, 48)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Game state variables are initialized in reset()
        self.blocks = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.level = 0
        self.win_streak = 0
        self.moves_left = 0
        self.max_moves = 0
        self.num_blocks = 0
        
        # Used to ensure the reset method is called first
        self._needs_reset = True
        
        # RNG
        self.np_random = None
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._needs_reset = False
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        
        # Level progression
        if options and "level" in options:
             self.level = options["level"]
             self.win_streak = (self.level -1) * 5
        else:
             self.level = 1
             self.win_streak = 0

        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes the board for the current level."""
        self.level = math.floor(self.win_streak / 5) + 1
        
        # Scale difficulty
        self.num_blocks = min(8 + self.level - 1, 15)
        self.max_moves = 10 + (self.level - 1) * 2
        self.moves_left = self.max_moves
        
        # Create target positions
        target_positions = []
        for i in range(self.num_blocks):
            target_positions.append((i % self.GRID_COLS, i // self.GRID_COLS))
            
        # Create shuffled initial positions
        possible_positions = [(c, r) for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS)]
        initial_positions = self.np_random.permutation(possible_positions).tolist()[:self.num_blocks]

        # Create blocks
        self.blocks = []
        for i in range(self.num_blocks):
            block_id = i + 1
            self.blocks.append(Block(block_id, tuple(initial_positions[i]), tuple(target_positions[i])))
            
        # Ensure puzzle is not solved on start
        if self._is_solved():
            self._setup_level() # Recurse until we get an unsolved puzzle
            
    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Cannot call step before reset. Call env.reset() first.")
            
        # If game is over, a new action should effectively reset to the next level
        if self.game_over:
            if self.win_state:
                self.win_streak += 1
            else:
                self.win_streak = 0 # Reset streak on loss
            self._setup_level()
            self.game_over = False
            self.win_state = False
            
        reward = 0
        terminated = False
        
        movement = action[0]
        
        # Only process a move if it's a directional input
        if movement in [1, 2, 3, 4]:
            # A move is made
            self.moves_left -= 1
            # sfx: block_slide_start
            
            blocks_in_place_before = {b.id for b in self.blocks if b.is_in_place()}
            
            self._apply_move(movement)
            
            blocks_in_place_after = {b.id for b in self.blocks if b.is_in_place()}
            
            # Reward for new blocks in place
            newly_placed_blocks = blocks_in_place_after - blocks_in_place_before
            reward += len(newly_placed_blocks) * 5.0
            
            # Continuous reward for blocks in correct position
            reward += len(blocks_in_place_after) * 0.1
            
        # Update game state
        self.steps += 1
        self.score += reward
        
        # Check for termination
        terminated, win = self._check_termination()
        if terminated:
            self.game_over = True
            self.win_state = win
            if win:
                reward += 100.0 # Win bonus
                # sfx: puzzle_solved
            else:
                reward -= 100.0 # Lose penalty
                # sfx: puzzle_failed
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_move(self, direction):
        """
        Applies a slide move to all blocks.
        Direction: 1=up, 2=down, 3=left, 4=right
        """
        if direction == 1: # Up
            move_vec = (0, -1)
            sort_key = lambda b: b.pos[1]
            sort_reverse = False
        elif direction == 2: # Down
            move_vec = (0, 1)
            sort_key = lambda b: b.pos[1]
            sort_reverse = True
        elif direction == 3: # Left
            move_vec = (-1, 0)
            sort_key = lambda b: b.pos[0]
            sort_reverse = False
        else: # Right
            move_vec = (1, 0)
            sort_key = lambda b: b.pos[0]
            sort_reverse = True

        # Sort blocks to process them in the correct order for pushing
        sorted_blocks = sorted(self.blocks, key=sort_key, reverse=sort_reverse)
        
        # Get current positions of all blocks for collision detection
        current_positions = {b.pos for b in self.blocks}

        for block in sorted_blocks:
            # Start from current position
            new_pos = block.pos
            
            while True:
                next_pos = (new_pos[0] + move_vec[0], new_pos[1] + move_vec[1])
                
                # Check grid boundaries
                if not (0 <= next_pos[0] < self.GRID_COLS and 0 <= next_pos[1] < self.GRID_ROWS):
                    break
                    
                # Check for collision with other blocks
                if next_pos in current_positions and next_pos != block.pos:
                    break
                
                new_pos = next_pos

            # Update positions for collision detection in this step
            current_positions.remove(block.pos)
            current_positions.add(new_pos)
            
            # Update block's final position
            block.pos = new_pos

    def _is_solved(self):
        return all(b.is_in_place() for b in self.blocks)
        
    def _check_termination(self):
        """Returns (terminated, win)"""
        if self._is_solved():
            return True, True
        if self.moves_left <= 0:
            return True, False
        if self.steps >= self.MAX_EPISODE_STEPS:
            return True, False
        return False, False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_width = self.GRID_COLS * self.BLOCK_SIZE + (self.GRID_COLS - 1) * self.BLOCK_SPACING
        grid_height = self.GRID_ROWS * self.BLOCK_SIZE + (self.GRID_ROWS - 1) * self.BLOCK_SPACING
        grid_x = self.GRID_MARGIN_X
        grid_y = self.GRID_MARGIN_Y
        
        pygame.draw.rect(
            self.screen, self.COLOR_GRID,
            (grid_x - self.BLOCK_SPACING, grid_y - self.BLOCK_SPACING, 
             grid_width + 2 * self.BLOCK_SPACING, grid_height + 2 * self.BLOCK_SPACING),
            border_radius=self.BLOCK_CORNER_RADIUS
        )
        
        # Draw empty slots
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                slot_x = grid_x + c * (self.BLOCK_SIZE + self.BLOCK_SPACING)
                slot_y = grid_y + r * (self.BLOCK_SIZE + self.BLOCK_SPACING)
                pygame.draw.rect(
                    self.screen, self.COLOR_BG,
                    (slot_x, slot_y, self.BLOCK_SIZE, self.BLOCK_SIZE),
                    border_radius=self.BLOCK_CORNER_RADIUS
                )

        # Draw blocks
        if self._needs_reset: return
        
        for block in self.blocks:
            pixel_x = grid_x + block.pos[0] * (self.BLOCK_SIZE + self.BLOCK_SPACING)
            pixel_y = grid_y + block.pos[1] * (self.BLOCK_SIZE + self.BLOCK_SPACING)
            
            color = self._get_color_for_block(block)
            
            # Draw block
            self._draw_rounded_rect(
                self.screen,
                (pixel_x, pixel_y, self.BLOCK_SIZE, self.BLOCK_SIZE),
                color, self.BLOCK_CORNER_RADIUS
            )
            
            # Draw number
            self._draw_text(
                str(block.id), self.font_block, self.COLOR_TEXT,
                pixel_x + self.BLOCK_SIZE / 2,
                pixel_y + self.BLOCK_SIZE / 2
            )
            
    def _render_ui(self):
        # Draw Moves Left
        self._draw_text(f"Moves: {self.moves_left}", self.font_main, self.COLOR_TEXT, 10, 10, align="topleft")
        
        # Draw Level
        self._draw_text(f"Level: {self.level}", self.font_main, self.COLOR_TEXT, self.SCREEN_WIDTH - 10, 10, align="topright")

        # Draw Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_state:
                msg = "PUZZLE SOLVED!"
                color = self.COLOR_WIN
            else:
                msg = "OUT OF MOVES"
                color = self.COLOR_LOSE
            
            self._draw_text(msg, self.font_game_over, color, self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20)
            self._draw_text("Next move starts new puzzle", self.font_main, self.COLOR_TEXT, self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_left": self.moves_left,
            "win_streak": self.win_streak,
        }
        
    def _get_color_for_block(self, block):
        if block.is_in_place():
            return self.COLOR_BLOCK_CORRECT
        
        # Manhattan distance
        dist = abs(block.pos[0] - block.target_pos[0]) + abs(block.pos[1] - block.target_pos[1])
        max_dist = self.GRID_COLS + self.GRID_ROWS - 2
        
        # Interpolate color from wrong to correct
        # High distance -> closer to WRONG, Low distance -> closer to CORRECT
        ratio = min(1.0, dist / max_dist)
        
        r = self.COLOR_BLOCK_CORRECT[0] + ratio * (self.COLOR_BLOCK_WRONG[0] - self.COLOR_BLOCK_CORRECT[0])
        g = self.COLOR_BLOCK_CORRECT[1] + ratio * (self.COLOR_BLOCK_WRONG[1] - self.COLOR_BLOCK_CORRECT[1])
        b = self.COLOR_BLOCK_CORRECT[2] + ratio * (self.COLOR_BLOCK_WRONG[2] - self.COLOR_BLOCK_CORRECT[2])

        return (int(r), int(g), int(b))

    def _draw_text(self, text, font, color, x, y, align="center"):
        # Shadow
        text_surface_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect_shadow = text_surface_shadow.get_rect()
        if align == "center":
            text_rect_shadow.center = (int(x) + 1, int(y) + 1)
        elif align == "topleft":
            text_rect_shadow.topleft = (int(x) + 1, int(y) + 1)
        elif align == "topright":
            text_rect_shadow.topright = (int(x) + 1, int(y) + 1)
        self.screen.blit(text_surface_shadow, text_rect_shadow)
        
        # Main text
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = (int(x), int(y))
        elif align == "topleft":
            text_rect.topleft = (int(x), int(y))
        elif align == "topright":
            text_rect.topright = (int(x), int(y))
        self.screen.blit(text_surface, text_rect)

    def _draw_rounded_rect(self, surface, rect, color, radius):
        """Draw a rectangle with rounded corners."""
        rect = pygame.Rect(rect)
        pygame.gfxdraw.aacircle(surface, rect.left + radius, rect.top + radius, radius, color)
        pygame.gfxdraw.aacircle(surface, rect.right - radius - 1, rect.top + radius, radius, color)
        pygame.gfxdraw.aacircle(surface, rect.left + radius, rect.bottom - radius - 1, radius, color)
        pygame.gfxdraw.aacircle(surface, rect.right - radius - 1, rect.bottom - radius - 1, radius, color)
        pygame.gfxdraw.filled_circle(surface, rect.left + radius, rect.top + radius, radius, color)
        pygame.gfxdraw.filled_circle(surface, rect.right - radius - 1, rect.top + radius, radius, color)
        pygame.gfxdraw.filled_circle(surface, rect.left + radius, rect.bottom - radius - 1, radius, color)
        pygame.gfxdraw.filled_circle(surface, rect.right - radius - 1, rect.bottom - radius - 1, radius, color)
        pygame.draw.rect(surface, color, (rect.left + radius, rect.top, rect.width - 2 * radius, rect.height))
        pygame.draw.rect(surface, color, (rect.left, rect.top + radius, rect.width, rect.height - 2 * radius))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Temporarily set up a dummy state for observation testing
        self._needs_reset = False
        self.reset(seed=0)

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
        
        self._needs_reset = True # Reset flag
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sliding Blocks")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
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
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        # Only step if a move was made
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Draw the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate
        
    env.close()