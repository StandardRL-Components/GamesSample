
# Generated: 2025-08-27T13:34:54.407394
# Source Brief: brief_00414.md
# Brief Index: 414

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to push all blocks simultaneously. "
        "The goal is to get each block to its matching color target."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based puzzle game. Slide colored blocks to their matching "
        "targets within a limited number of moves. Plan your pushes carefully to avoid getting stuck!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 12
    GRID_ROWS = 8
    CELL_SIZE = 40
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT) // 2
    
    MAX_MOVES = 30
    NUM_BLOCKS = 4

    # Colors (Clean, high-contrast palette)
    COLOR_BG = (34, 39, 46)        # Dark grey-blue
    COLOR_GRID = (60, 68, 81)      # Lighter grey-blue
    COLOR_TEXT = (233, 236, 239)   # Off-white
    BLOCK_COLORS = [
        (231, 76, 60),   # Red
        (52, 152, 219),  # Blue
        (46, 204, 113),  # Green
        (241, 196, 15),  # Yellow
        (155, 89, 182),  # Purple
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
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        
        # State variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.moves_left = None
        self.blocks = None
        self.targets = None
        self.np_random = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        
        self._generate_level()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Creates a new puzzle layout."""
        self.blocks = []
        self.targets = []
        
        occupied_positions = set()
        
        colors = self.np_random.choice(len(self.BLOCK_COLORS), self.NUM_BLOCKS, replace=False)
        chosen_colors = [self.BLOCK_COLORS[i] for i in colors]

        for color_tuple in chosen_colors:
            # Place target
            target_pos = self._get_random_empty_pos(occupied_positions)
            self.targets.append({"pos": target_pos, "color": color_tuple})
            occupied_positions.add(target_pos)

            # Place block
            block_pos = self._get_random_empty_pos(occupied_positions)
            self.blocks.append({"pos": block_pos, "color": color_tuple, "locked": False})
            occupied_positions.add(block_pos)

    def _get_random_empty_pos(self, occupied_positions):
        """Finds a random empty grid position."""
        while True:
            pos = (
                self.np_random.integers(0, self.GRID_COLS),
                self.np_random.integers(0, self.GRID_ROWS),
            )
            if pos not in occupied_positions:
                return pos

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if movement == 0: # No-op action
            return self._get_observation(), 0, False, False, self._get_info()

        # A real move was made
        self.moves_left -= 1
        self.steps += 1
        reward = self._process_push(movement)

        # Check for win condition
        all_locked = all(b['locked'] for b in self.blocks)
        if all_locked:
            reward += 100  # Win bonus
            self.game_over = True
            
        # Check for loss condition
        if self.moves_left <= 0 and not all_locked:
            reward -= 10 # Loss penalty
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated always False
            self._get_info()
        )

    def _process_push(self, movement):
        """Processes a push action, moving all blocks and calculating rewards."""
        if movement == 1:  # Up
            direction = (0, -1)
            sort_key = lambda b: b['pos'][1]
            sort_reverse = False
        elif movement == 2:  # Down
            direction = (0, 1)
            sort_key = lambda b: b['pos'][1]
            sort_reverse = True
        elif movement == 3:  # Left
            direction = (-1, 0)
            sort_key = lambda b: b['pos'][0]
            sort_reverse = False
        else:  # Right (movement == 4)
            direction = (1, 0)
            sort_key = lambda b: b['pos'][0]
            sort_reverse = True
        
        movable_blocks = sorted([b for b in self.blocks if not b['locked']], key=sort_key, reverse=sort_reverse)
        
        occupied_positions = {b['pos'] for b in self.blocks}
        
        move_reward = 0

        for block in movable_blocks:
            old_pos = block['pos']
            target_for_block = next(t for t in self.targets if t['color'] == block['color'])
            old_dist = self._manhattan_distance(old_pos, target_for_block['pos'])

            new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1])
            
            can_move = True
            if not (0 <= new_pos[0] < self.GRID_COLS and 0 <= new_pos[1] < self.GRID_ROWS):
                can_move = False
            elif new_pos in occupied_positions:
                can_move = False
            
            if can_move:
                # Sound effect: Block slide
                block['pos'] = new_pos
                occupied_positions.remove(old_pos)
                occupied_positions.add(new_pos)
                
                new_dist = self._manhattan_distance(new_pos, target_for_block['pos'])
                if new_dist < old_dist:
                    move_reward += 1
                elif new_dist > old_dist:
                    move_reward -= 1
                
                if new_pos == target_for_block['pos']:
                    block['locked'] = True
                    move_reward += 5
                    # Sound effect: Block lock
        
        return move_reward

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the grid, targets, and blocks."""
        # Draw grid lines
        for x in range(self.GRID_COLS + 1):
            px = self.GRID_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y_OFFSET), (px, self.GRID_Y_OFFSET + self.GRID_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, py), (self.GRID_X_OFFSET + self.GRID_WIDTH, py))

        # Draw targets (hollow squares)
        for target in self.targets:
            rect = self._get_grid_rect(target['pos'])
            pygame.draw.rect(self.screen, target['color'], rect, width=3)
        
        # Draw blocks (solid squares)
        for block in self.blocks:
            rect = self._get_grid_rect(block['pos'])
            pygame.draw.rect(self.screen, block['color'], rect)
            if block['locked']:
                inner_rect = rect.inflate(-self.CELL_SIZE * 0.4, -self.CELL_SIZE * 0.4)
                pygame.draw.rect(self.screen, self.COLOR_BG, inner_rect)

    def _get_grid_rect(self, grid_pos):
        """Converts grid coordinates to a pygame.Rect."""
        x, y = grid_pos
        px = self.GRID_X_OFFSET + x * self.CELL_SIZE
        py = self.GRID_Y_OFFSET + y * self.CELL_SIZE
        return pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)

    def _render_ui(self):
        """Renders UI elements like score and moves left."""
        moves_text = f"Moves: {self.moves_left}"
        text_surface = self.font_main.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (20, 20))

        score_text = f"Score: {self.score}"
        text_surface = self.font_main.render(score_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(text_surface, text_rect)
        
        if self.game_over:
            all_locked = all(b['locked'] for b in self.blocks)
            message = "PUZZLE SOLVED!" if all_locked else "OUT OF MOVES"
            color = self.BLOCK_COLORS[2] if all_locked else self.BLOCK_COLORS[0]
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            
            end_text_surface = self.font_main.render(message, True, color)
            end_text_rect = end_text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text_surface, end_text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "blocks_locked": sum(1 for b in self.blocks if b['locked']),
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Slider Puzzle")
    clock = pygame.time.Clock()
    
    print("\n--- Manual Control ---")
    print(env.user_guide)
    
    action = [0, 0, 0]

    while not done:
        movement_action = 0 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement_action = 1
                elif event.key == pygame.K_DOWN:
                    movement_action = 2
                elif event.key == pygame.K_LEFT:
                    movement_action = 3
                elif event.key == pygame.K_RIGHT:
                    movement_action = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                elif event.key == pygame.K_q:
                    done = True

        if movement_action != 0:
            action[0] = movement_action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    print("Game Over!")
    env.close()