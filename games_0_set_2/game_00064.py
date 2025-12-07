import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



class GameEnv(gym.Env):
    """
    A puzzle game where the player pushes colored blocks onto matching targets.
    The goal is to solve the puzzle in the fewest moves possible.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to push all blocks in one direction. Solve the puzzle before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push colored blocks onto matching targets in a grid-based puzzle to clear the stage before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Game Configuration ---
    GRID_DIM = (16, 10)  # Width, Height in cells
    CELL_SIZE = 40
    SCREEN_WIDTH = GRID_DIM[0] * CELL_SIZE
    SCREEN_HEIGHT = GRID_DIM[1] * CELL_SIZE

    # --- Colors ---
    COLOR_BG = (32, 34, 37)          # Dark grey
    COLOR_GRID = (66, 69, 74)        # Lighter grey
    COLOR_TEXT = (220, 221, 222)     # Off-white
    COLOR_TEXT_SHADOW = (0, 0, 0, 150)
    
    BLOCK_COLORS = {
        'red': (237, 67, 55),
        'blue': (60, 134, 240),
        'green': (88, 189, 78),
        'yellow': (250, 195, 15)
    }
    TARGET_COLORS = {
        'red': (74, 38, 35),
        'blue': (38, 62, 94),
        'green': (44, 82, 47),
        'yellow': (92, 78, 34)
    }

    # --- Level Data ---
    LEVELS = [
        {
            "moves": 20,
            "blocks": [
                {'pos': (4, 4), 'color': 'red'},
                {'pos': (6, 6), 'color': 'blue'}
            ],
            "targets": [
                {'pos': (11, 4), 'color': 'red'},
                {'pos': (9, 6), 'color': 'blue'}
            ]
        },
        {
            "moves": 35,
            "blocks": [
                {'pos': (3, 2), 'color': 'red'},
                {'pos': (4, 2), 'color': 'blue'},
                {'pos': (5, 2), 'color': 'green'}
            ],
            "targets": [
                {'pos': (12, 7), 'color': 'red'},
                {'pos': (11, 7), 'color': 'blue'},
                {'pos': (10, 7), 'color': 'green'}
            ]
        },
        {
            "moves": 50,
            "blocks": [
                {'pos': (7, 2), 'color': 'red'},
                {'pos': (8, 2), 'color': 'blue'},
                {'pos': (7, 4), 'color': 'green'},
                {'pos': (8, 4), 'color': 'yellow'}
            ],
            "targets": [
                {'pos': (2, 8), 'color': 'blue'},
                {'pos': (13, 8), 'color': 'red'},
                {'pos': (2, 1), 'color': 'yellow'},
                {'pos': (13, 1), 'color': 'green'}
            ]
        }
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Ensure pygame runs headless
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        self.current_level_index = 0
        self.blocks = []
        self.targets = []
        self.moves_remaining = 0
        self.max_moves = 0
        self.steps = 0
        self.cumulative_reward = 0.0
        self.game_over = False
        self.win_state = False
        self.last_move_dir = (0, 0)
        
        # self.reset() is called by validate_implementation
        self.validate_implementation()

    def _load_level(self, level_index):
        level_data = self.LEVELS[level_index % len(self.LEVELS)]
        self.max_moves = level_data["moves"]
        self.moves_remaining = self.max_moves

        self.blocks = []
        for i, b in enumerate(level_data["blocks"]):
            block_rect = pygame.Rect(
                b['pos'][0] * self.CELL_SIZE,
                b['pos'][1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            self.blocks.append({'rect': block_rect, 'color': b['color'], 'id': i})

        self.targets = []
        for i, t in enumerate(level_data["targets"]):
            target_rect = pygame.Rect(
                t['pos'][0] * self.CELL_SIZE,
                t['pos'][1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            # Find the matching block id
            block_id = next(b['id'] for b in self.blocks if b['color'] == t['color'])
            self.targets.append({'rect': target_rect, 'color': t['color'], 'block_id': block_id})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.game_over and self.win_state:
            self.current_level_index = (self.current_level_index + 1) % len(self.LEVELS)
        else: # On loss or first start
            self.current_level_index = 0

        self._load_level(self.current_level_index)

        self.steps = 0
        self.cumulative_reward = 0.0
        self.game_over = False
        self.win_state = False
        self.last_move_dir = (0, 0)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        terminated = False
        self.steps += 1
        
        move_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
        self.last_move_dir = move_dir
        
        if movement == 0: # No-op
            reward -= 0.1 # Small penalty for wasting a turn
        else:
            self.moves_remaining -= 1
            
            pre_move_distances = self._get_all_distances()
            
            # --- Core Push Logic ---
            # Sort blocks to handle chain reactions correctly
            # For pushes right (dx=1), sort by x descending. For left (dx=-1), ascending.
            # For pushes down (dy=1), sort by y descending. For up (dy=-1), ascending.
            sort_key = lambda b: b['rect'].x
            reverse_sort = bool(move_dir[0] > 0)
            if move_dir[1] != 0:
                sort_key = lambda b: b['rect'].y
                reverse_sort = bool(move_dir[1] > 0)

            sorted_blocks = sorted(self.blocks, key=sort_key, reverse=reverse_sort)
            
            occupied_rects = [b['rect'].copy() for b in self.blocks]

            for block in sorted_blocks:
                new_pos_x = block['rect'].x + move_dir[0] * self.CELL_SIZE
                new_pos_y = block['rect'].y + move_dir[1] * self.CELL_SIZE

                # Boundary checks
                if not (0 <= new_pos_x < self.SCREEN_WIDTH and 0 <= new_pos_y < self.SCREEN_HEIGHT):
                    continue

                # Collision check with other blocks
                potential_rect = block['rect'].move(
                    move_dir[0] * self.CELL_SIZE, move_dir[1] * self.CELL_SIZE
                )
                
                collision = False
                for other_block_rect in occupied_rects:
                    if other_block_rect != block['rect'] and potential_rect.colliderect(other_block_rect):
                        collision = True
                        break
                
                if not collision:
                    # Find the index of the current block in the main list.
                    idx = next(i for i, b in enumerate(self.blocks) if b['id'] == block['id'])
                    
                    # Store the original rect to find its index in occupied_rects.
                    # This must be done before self.blocks is updated, because that update
                    # also modifies the 'block' variable used in this loop (it's a reference).
                    original_rect_for_lookup = block['rect']
                    
                    # Update the rect in the main list.
                    self.blocks[idx]['rect'] = potential_rect
                    
                    # Find the original rect's index in the collision list and update it.
                    c_idx = occupied_rects.index(original_rect_for_lookup)
                    occupied_rects[c_idx] = potential_rect

            # --- Reward Calculation ---
            post_move_distances = self._get_all_distances()
            
            num_solved_pre = sum(1 for d in pre_move_distances if d == 0)
            num_solved_post = sum(1 for d in post_move_distances if d == 0)

            # Reward for getting a block on target
            if num_solved_post > num_solved_pre:
                reward += 5.0 * (num_solved_post - num_solved_pre)

            # Reward for moving closer/further
            for i in range(len(self.blocks)):
                dist_change = pre_move_distances[i] - post_move_distances[i]
                reward += dist_change * 0.1 # Reward is proportional to distance change in cells
        
        # --- Termination Check ---
        is_win = all(d == 0 for d in self._get_all_distances())
        is_loss = self.moves_remaining <= 0 and not is_win

        if is_win:
            reward += 100.0
            terminated = True
            self.game_over = True
            self.win_state = True
        elif is_loss:
            reward -= 50.0
            terminated = True
            self.game_over = True

        self.cumulative_reward += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_all_distances(self):
        distances = []
        # Sort blocks by ID to ensure consistent distance calculation order
        sorted_blocks = sorted(self.blocks, key=lambda b: b['id'])
        for block in sorted_blocks:
            # Find the corresponding target
            target = next((t for t in self.targets if t['block_id'] == block['id']), None)
            if target:
                dist_x = abs(block['rect'].centerx - target['rect'].centerx) // self.CELL_SIZE
                dist_y = abs(block['rect'].centery - target['rect'].centery) // self.CELL_SIZE
                distances.append(dist_x + dist_y)
        return distances

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, 3), but we want (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw targets
        for target in self.targets:
            color = self.TARGET_COLORS[target['color']]
            pygame.draw.rect(self.screen, color, target['rect'])
            
            # Check if solved and add a glow
            block = next(b for b in self.blocks if b['id'] == target['block_id'])
            if block['rect'] == target['rect']:
                glow_color = self.BLOCK_COLORS[target['color']]
                # Use gfxdraw for anti-aliased circle
                center = target['rect'].center
                for r in range(self.CELL_SIZE // 4, 0, -2):
                    alpha = 100 - int((r / (self.CELL_SIZE // 4)) * 100)
                    if alpha > 0:
                        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], r, (*glow_color, alpha))

        # Draw blocks
        for block in self.blocks:
            main_color = self.BLOCK_COLORS[block['color']]
            shadow_color = tuple(max(0, c-40) for c in main_color)
            
            # Draw shadow/border for depth
            shadow_rect = block['rect'].move(0, 4)
            pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_radius=4)
            
            # Draw main block
            pygame.draw.rect(self.screen, main_color, block['rect'], border_radius=4)

    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, font, color, pos):
            shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, pos)

        # Draw moves remaining
        moves_text = f"Moves: {self.moves_remaining}/{self.max_moves}"
        draw_text(moves_text, self.font_main, self.COLOR_TEXT, (10, 10))

        # Draw level
        level_text = f"Level: {self.current_level_index + 1}"
        text_width = self.font_main.size(level_text)[0]
        draw_text(level_text, self.font_main, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 10, 10))

        # Draw game over screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "LEVEL COMPLETE" if self.win_state else "OUT OF MOVES"
            color = (152, 251, 152) if self.win_state else (255, 105, 97)
            
            msg_width, msg_height = self.font_large.size(msg)
            draw_text(msg, self.font_large, color, 
                      ((self.SCREEN_WIDTH - msg_width) // 2, (self.SCREEN_HEIGHT - msg_height) // 2))

    def _get_info(self):
        return {
            "score": self.cumulative_reward,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "level": self.current_level_index + 1,
            "is_win": self.win_state
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Beginning implementation validation...")
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test action space
        assert hasattr(self, 'action_space'), "action_space attribute not set"
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        assert hasattr(self, 'observation_space'), "observation_space attribute not set"
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
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
    # The environment is designed for headless operation.
    # The following code demonstrates how to interact with it.
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    print("Environment reset.")
    print(f"Initial Info: {info}")

    # --- Automated Test Run ---
    for i in range(5):
        action = env.action_space.sample()
        print(f"\nStep {i+1}, Action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = env.reset()

    env.close()
    print("\nEnvironment closed.")