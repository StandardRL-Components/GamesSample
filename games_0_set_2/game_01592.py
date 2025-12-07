
# Generated: 2025-08-27T17:37:58.088645
# Source Brief: brief_01592.md
# Brief Index: 1592

        
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
        "Get all colored blocks to their matching targets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist grid-based puzzle. Push colored blocks onto their target locations "
        "within a limited number of moves. Plan your pushes carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 28, 32)
    COLOR_GRID = (50, 55, 60)
    COLOR_UI_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (231, 76, 60),   # Red
        (52, 152, 219),  # Blue
        (46, 204, 113),  # Green
        (241, 196, 15),  # Yellow
        (155, 89, 182),  # Purple
    ]

    # Grid
    GRID_WIDTH = 10
    GRID_HEIGHT = 7
    CELL_SIZE = 50
    GRID_MARGIN_X = (640 - GRID_WIDTH * CELL_SIZE) // 2
    GRID_MARGIN_Y = (400 - GRID_HEIGHT * CELL_SIZE) // 2
    
    # Game mechanics
    NUM_BLOCKS = 5
    MAX_MOVES = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("dejavusans", 24, bold=True)
        self.font_win = pygame.font.SysFont("dejavusans", 48, bold=True)
        
        # State variables (initialized in reset)
        self.blocks = []
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # No-op action
        if movement == 0:
            return self._get_observation(), 0.0, False, False, self._get_info()

        # A real move is attempted
        self.steps += 1
        self.moves_left -= 1
        
        reward = self._apply_push(movement)
        self.score += reward

        terminated = self._check_termination()
        
        if terminated:
            if self._check_win_condition():
                # Win bonus based on remaining moves
                win_bonus = 50 + (self.moves_left * 2)
                self.score += win_bonus
                reward += win_bonus
            else:
                # Loss penalty
                self.score -= 100
                reward = -100.0
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        """Generates a new puzzle layout."""
        self.blocks = []
        all_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_cells)

        # Place targets
        target_positions = all_cells[:self.NUM_BLOCKS]
        
        # Place blocks
        available_block_cells = all_cells[self.NUM_BLOCKS:]
        block_positions = available_block_cells[:self.NUM_BLOCKS]
        
        # Ensure at least one block is adjacent to its target to guarantee a starting move
        # We do this by taking the first block/target pair and moving the block
        target_pos = target_positions[0]
        adj_cells = [
            (target_pos[0] + dx, target_pos[1] + dy)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
        ]
        valid_adj_cells = [
            p for p in adj_cells 
            if 0 <= p[0] < self.GRID_WIDTH and 0 <= p[1] < self.GRID_HEIGHT and p not in target_positions
        ]
        
        if valid_adj_cells:
            # Find which block position needs to be replaced
            pos_to_replace = block_positions[0]
            # Replace it with a valid adjacent cell
            block_positions[0] = self.np_random.choice(valid_adj_cells, 1).tolist()[0]
            # Put the replaced position back into the pool if it wasn't the chosen adjacent cell
            if pos_to_replace != block_positions[0] and pos_to_replace not in block_positions:
                 available_block_cells.append(pos_to_replace)


        for i in range(self.NUM_BLOCKS):
            self.blocks.append({
                "id": i,
                "color": self.BLOCK_COLORS[i],
                "pos": tuple(block_positions[i]),
                "target_pos": tuple(target_positions[i]),
                "locked": False,
            })

    def _apply_push(self, movement_action):
        """Calculates rewards and applies block movement for a given push."""
        # Map movement action to a direction vector
        # 1=up, 2=down, 3=left, 4=right
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[movement_action]

        # Sort blocks to process them from the direction of the push
        # e.g., for a right push, process rightmost blocks first
        sort_reverse = (dx > 0 or dy > 0)
        sorted_indices = sorted(
            range(len(self.blocks)),
            key=lambda i: (self.blocks[i]['pos'][0] * dx + self.blocks[i]['pos'][1] * dy),
            reverse=sort_reverse
        )
        
        current_positions = {b['pos'] for b in self.blocks}
        pre_move_distances = {
            i: self._manhattan_distance(self.blocks[i]['pos'], self.blocks[i]['target_pos'])
            for i in sorted_indices if not self.blocks[i]['locked']
        }
        
        reward = 0.0
        moved_blocks = {}

        for i in sorted_indices:
            block = self.blocks[i]
            if block['locked']:
                continue

            next_pos = (block['pos'][0] + dx, block['pos'][1] + dy)

            # Check boundaries
            if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                continue
            
            # Check collision with other blocks
            if next_pos in current_positions:
                continue
            
            # This block can move
            moved_blocks[i] = next_pos
            current_positions.remove(block['pos'])
            current_positions.add(next_pos)
        
        # Now, calculate rewards and apply state changes
        for i, new_pos in moved_blocks.items():
            block = self.blocks[i]
            block['pos'] = new_pos

            # Check for locking
            if new_pos == block['target_pos']:
                block['locked'] = True
                reward += 5.0  # Placed a block on target
            else:
                # Distance-based reward
                post_move_dist = self._manhattan_distance(new_pos, block['target_pos'])
                if post_move_dist < pre_move_distances[i]:
                    reward += 0.5  # Moved closer
                elif post_move_dist > pre_move_distances[i]:
                    reward -= 0.2  # Moved further

        return reward

    def _check_termination(self):
        if self.moves_left <= 0:
            return True
        return self._check_win_condition()

    def _check_win_condition(self):
        return all(b['locked'] for b in self.blocks)

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
            "moves_left": self.moves_left,
            "blocks_locked": sum(1 for b in self.blocks if b['locked']),
        }
    
    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_MARGIN_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_MARGIN_Y), (px, self.GRID_MARGIN_Y + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_MARGIN_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_MARGIN_X, py), (self.GRID_MARGIN_X + self.GRID_WIDTH * self.CELL_SIZE, py))

        # Draw targets
        for block in self.blocks:
            self._draw_target(block['target_pos'], block['color'])

        # Draw blocks
        for block in self.blocks:
            self._draw_block(block['pos'], block['color'], block['locked'])

    def _draw_target(self, pos, color):
        """Draws a target location on the grid."""
        px = self.GRID_MARGIN_X + pos[0] * self.CELL_SIZE
        py = self.GRID_MARGIN_Y + pos[1] * self.CELL_SIZE
        
        target_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        # Faded color for the target
        faded_color = tuple(c // 4 for c in color)
        pygame.draw.rect(self.screen, faded_color, target_rect)
        
        # Inner outline for definition
        outline_color = tuple(min(255, c // 2) for c in color)
        pygame.draw.rect(self.screen, outline_color, target_rect, 2)


    def _draw_block(self, pos, color, locked):
        """Draws a block on the grid."""
        padding = 4
        px = self.GRID_MARGIN_X + pos[0] * self.CELL_SIZE + padding
        py = self.GRID_MARGIN_Y + pos[1] * self.CELL_SIZE + padding
        size = self.CELL_SIZE - 2 * padding
        
        block_rect = pygame.Rect(px, py, size, size)
        
        # Draw block with antialiasing
        pygame.gfxdraw.box(self.screen, block_rect, color)
        
        # Add a subtle border
        border_color = tuple(max(0, c - 40) for c in color)
        pygame.gfxdraw.rectangle(self.screen, block_rect, border_color)

        if locked:
            # Draw a 'locked' indicator (white inner square)
            lock_padding = 10
            lock_rect = pygame.Rect(px + lock_padding, py + lock_padding, size - 2 * lock_padding, size - 2 * lock_padding)
            pygame.gfxdraw.box(self.screen, lock_rect, (255, 255, 255, 200))

    def _render_ui(self):
        # Display moves left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (15, 10))

        # Display score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(640 - 15, 10))
        self.screen.blit(score_text, score_rect)

        # Display game over message
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self._check_win_condition():
                msg = "PUZZLE SOLVED!"
                color = (46, 204, 113) # Green
            else:
                msg = "OUT OF MOVES"
                color = (231, 76, 60) # Red
            
            end_text = self.font_win.render(msg, True, color)
            end_rect = end_text.get_rect(center=(320, 200))
            self.screen.blit(end_text, end_rect)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Use a real screen for manual play
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Block Pusher Puzzle")
    clock = pygame.time.Clock()

    print(env.game_description)
    print(env.user_guide)

    while not done:
        action = [0, 0, 0]  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
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
                    action = [0, 0, 0] # prevent action on reset frame
                
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {terminated}")
        
        # Render the observation to the screen
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()