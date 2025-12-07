
# Generated: 2025-08-28T03:14:34.610378
# Source Brief: brief_01971.md
# Brief Index: 1971

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    A grid-based puzzle game where the player pushes a red block to an exit.

    The player must navigate a red block through a maze by pushing other
    movable blocks. The goal is to reach the green exit square within a
    limited number of moves. Each push action, successful or not, consumes
    one move. The game is turn-based, and the state only advances when a
    valid action is provided.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to push the red block. Try to reach the green exit in under 25 moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A block-pushing puzzle game. Strategically move blocks to clear a path for the red block to reach the green exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Grid elements
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    BLOCK = 3
    EXIT = 4

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 45, 50)
    COLOR_WALL = (70, 80, 90)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_BLOCK = (50, 120, 255)
    COLOR_EXIT = (50, 255, 120)
    COLOR_TEXT = (230, 230, 230)
    
    # Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 12
    GRID_ROWS = 8
    CELL_SIZE = 50
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2
    
    # Game parameters
    MAX_MOVES = 25
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables (initialized in reset)
        self.grid = None
        self.player_pos = None
        self.exit_pos = None
        self.moves_left = 0
        self.last_move_info = None # For rendering ghost blocks
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def _get_level(self):
        """Returns a predefined level layout."""
        level = [
            "WWWWWWWWWWWW",
            "WP..B....B.W",
            "W.WW.W..W..W",
            "W.B...BW...W",
            "W..W.B...B.W",
            "W.B.W......W",
            "W.....B..GEW",
            "WWWWWWWWWWWW",
        ]
        char_to_id = {'W': self.WALL, 'P': self.PLAYER, 'B': self.BLOCK, 'E': self.EXIT, '.': self.EMPTY, 'G': self.EXIT}
        grid = np.array([[char_to_id[c] for c in row] for row in level], dtype=np.uint8)
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = self._get_level()
        player_coords = np.argwhere(self.grid == self.PLAYER)
        self.player_pos = tuple(player_coords[0]) if len(player_coords) > 0 else (1,1)
        
        exit_coords = np.argwhere(self.grid == self.EXIT)
        self.exit_pos = tuple(exit_coords[0]) if len(exit_coords) > 0 else (6,9)

        self.moves_left = self.MAX_MOVES
        self.last_move_info = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.last_move_info = None # Clear ghost effect from previous turn
        
        reward = 0
        terminated = False
        
        # Only movement actions (1-4) consume a turn. No-op (0) does nothing.
        if movement > 0:
            self.steps += 1
            self.moves_left -= 1
            reward -= 0.1 # Cost for taking a turn
            
            move_succeeded, move_details, move_reward = self._do_push(movement)
            
            if move_succeeded:
                self.last_move_info = move_details
                reward += move_reward
                # Game end condition: player reached exit
                if self.player_pos == self.exit_pos:
                    terminated = True
                    reward += 100
                    
            # Game end condition: out of moves
            if self.moves_left <= 0 and not terminated:
                terminated = True
                reward -= 100
        
        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _do_push(self, direction):
        """
        Handles the core block pushing logic.
        Returns (success, move_details, reward).
        """
        # 1:up, 2:down, 3:left, 4:right
        moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = moves[direction]

        chain = []
        r, c = self.player_pos
        
        # 1. Determine the chain of blocks to be pushed
        while 0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS:
            block_type = self.grid[r, c]
            if block_type == self.EMPTY or block_type == self.EXIT:
                # The chain can move into this space
                break
            if block_type == self.WALL:
                # The chain is blocked by a wall
                return False, [], 0
            
            chain.append({'type': block_type, 'pos': (r, c)})
            r, c = r + dr, c + dc
        else:
            # The chain is blocked by the edge of the grid
            return False, [], 0

        # 2. If the chain can move, update the grid
        move_details = []
        move_reward = 0
        
        # Manhattan distance function for reward calculation
        def manhattan_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Iterate in reverse to move blocks into empty spaces
        for block in reversed(chain):
            old_r, old_c = block['pos']
            new_r, new_c = old_r + dr, old_c + dc
            block_type = block['type']

            # Calculate reward for moving blue blocks
            if block_type == self.BLOCK:
                dist_before = manhattan_dist((old_r, old_c), self.exit_pos)
                dist_after = manhattan_dist((new_r, new_c), self.exit_pos)
                if dist_after < dist_before:
                    move_reward += 1.0
                elif dist_after > dist_before:
                    move_reward -= 1.0

            # Update grid state
            self.grid[new_r, new_c] = block_type
            self.grid[old_r, old_c] = self.EMPTY
            
            # Update player position if it's the player block
            if block_type == self.PLAYER:
                self.player_pos = (new_r, new_c)
            
            # Store details for rendering
            move_details.append({'from': (old_r, old_c), 'to': (new_r, new_c), 'type': block_type})
        
        return True, move_details, move_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT))

        # Draw "ghost" of last move for visual feedback
        if self.last_move_info:
            for move in self.last_move_info:
                r, c = move['from']
                block_type = move['type']
                x = self.GRID_OFFSET_X + c * self.CELL_SIZE
                y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
                
                color = self.COLOR_WALL
                if block_type == self.PLAYER: color = self.COLOR_PLAYER
                elif block_type == self.BLOCK: color = self.COLOR_BLOCK
                
                ghost_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                ghost_surface.fill((*color, 60))
                self.screen.blit(ghost_surface, (x, y))

        # Draw grid elements
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                block_type = self.grid[r, c]
                if block_type == self.EMPTY:
                    continue
                
                color = self.COLOR_BG
                if block_type == self.WALL: color = self.COLOR_WALL
                elif block_type == self.PLAYER: color = self.COLOR_PLAYER
                elif block_type == self.BLOCK: color = self.COLOR_BLOCK
                elif block_type == self.EXIT: color = self.COLOR_EXIT
                
                x = self.GRID_OFFSET_X + c * self.CELL_SIZE
                y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                
                if block_type == self.EXIT:
                    pygame.draw.rect(self.screen, color, rect.inflate(-8, -8))
                else:
                    pygame.draw.rect(self.screen, color, rect)
                    # Add a subtle inner shadow for depth
                    pygame.draw.rect(self.screen, (0,0,0,30), rect.inflate(-4,-4))

    def _render_ui(self):
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 10))

        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            
            end_text_str = "YOU WIN!" if self.player_pos == self.exit_pos else "OUT OF MOVES"
            font_end = pygame.font.SysFont("monospace", 60, bold=True)
            end_text = font_end.render(end_text_str, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            self.screen.blit(overlay, (0,0))
            self.screen.blit(end_text, end_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "player_pos": self.player_pos,
        }

    def close(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Pusher")
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
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
        
        # Only step if a move was made
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate
        
    env.close()