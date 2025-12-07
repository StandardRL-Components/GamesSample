
# Generated: 2025-08-28T02:36:15.158985
# Source Brief: brief_01751.md
# Brief Index: 1751

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to set push direction. Shift to cycle selected block. Space to push."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist block-pushing puzzle. Move all blocks to their colored targets before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 16
        self.GRID_HEIGHT = 10
        self.CELL_SIZE = 40
        
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
        
        # Fonts
        self.ui_font = pygame.font.Font(None, 36)
        self.guide_font = pygame.font.Font(None, 20)
        self.end_font = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 60)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (224, 82, 99),    # Red
            (82, 166, 224),   # Blue
            (247, 198, 91),   # Yellow
        ]
        self.TARGET_COLORS = [
            (60, 30, 35),
            (30, 50, 60),
            (60, 55, 30),
        ]
        self.TARGET_LIT_COLORS = [
            (110, 60, 70),
            (60, 90, 110),
            (110, 100, 60),
        ]
        self.COLOR_SELECTED = (255, 255, 255)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # Game constants
        self.NUM_BLOCKS = 3
        self.MAX_MOVES = 15
        self.SCRAMBLE_MOVES = 10
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.blocks = []
        self.targets = []
        self.selected_block_idx = 0
        self.last_push_direction = (0, -1)  # Default up
        self.shift_pressed_last_frame = False
        self.blocks_on_target_state = set()

        self.validate_implementation()
    
    def _generate_puzzle(self):
        """Creates a new, solvable puzzle by starting from the solved state and reversing moves."""
        self.targets = []
        self.blocks = []
        
        possible_positions = []
        for x in range(1, self.GRID_WIDTH - 1):
            for y in range(1, self.GRID_HEIGHT - 1):
                possible_positions.append((x, y))
        
        # Place targets
        target_positions = self.np_random.choice(len(possible_positions), self.NUM_BLOCKS, replace=False)
        for i, pos_idx in enumerate(target_positions):
            pos = possible_positions[pos_idx]
            self.targets.append({'pos': pos, 'id': i})
            # Start blocks on targets (solved state)
            self.blocks.append({'pos': pos, 'id': i})

        # Scramble the puzzle with "pulls" (inverse pushes)
        for _ in range(self.SCRAMBLE_MOVES):
            block_idx = self.np_random.integers(0, self.NUM_BLOCKS)
            direction_idx = self.np_random.integers(1, 5) # 1-4 for directions
            
            # A "pull" is a push in the opposite direction
            pull_direction = self._get_direction_vector(direction_idx)
            push_direction = (-pull_direction[0], -pull_direction[1])
            
            self._execute_push(block_idx, push_direction, is_scramble=True)

        self.selected_block_idx = 0
        self.last_push_direction = (0, -1) # Reset push direction
        self._reset_selection() # Select top-leftmost block

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.shift_pressed_last_frame = False
        
        self._generate_puzzle()
        self.blocks_on_target_state = self._get_current_placements()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        move_executed = False

        # 1. Handle non-move-costing actions first (selection, direction)
        if shift_held and not self.shift_pressed_last_frame:
            # # sound_effect: "UI_Select_Cycle"
            self.selected_block_idx = (self.selected_block_idx + 1) % self.NUM_BLOCKS
        self.shift_pressed_last_frame = shift_held

        if movement > 0:
            new_direction = self._get_direction_vector(movement)
            if new_direction != self.last_push_direction:
                # # sound_effect: "UI_Direction_Change"
                self.last_push_direction = new_direction

        # 2. Handle move-costing action (push)
        if space_held:
            if self._execute_push(self.selected_block_idx, self.last_push_direction):
                # # sound_effect: "Block_Push_Success"
                move_executed = True
                self.moves_left -= 1
                self.steps += 1
                reward = -0.1
            else:
                # # sound_effect: "Block_Push_Fail"
                pass # Push was blocked, no move cost, no reward change

        # 3. Calculate rewards and check for termination after a move
        if move_executed:
            # Reward for new placements
            current_placements = self._get_current_placements()
            newly_placed = current_placements - self.blocks_on_target_state
            if newly_placed:
                # # sound_effect: "Block_On_Target"
                reward += len(newly_placed) * 1.0
            self.blocks_on_target_state = current_placements
            self.score = len(current_placements)

            # Check for win/loss conditions
            if len(self.blocks_on_target_state) == self.NUM_BLOCKS:
                # # sound_effect: "Puzzle_Solved"
                self.game_over = True
                reward += 50.0
            elif self.moves_left <= 0:
                # # sound_effect: "Game_Over_Moves"
                self.game_over = True
        
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_direction_vector(self, movement_action):
        if movement_action == 1: return (0, -1)  # Up
        if movement_action == 2: return (0, 1)   # Down
        if movement_action == 3: return (-1, 0)  # Left
        if movement_action == 4: return (1, 0)   # Right
        return (0, 0)

    def _execute_push(self, block_idx, direction, is_scramble=False):
        if direction == (0, 0):
            return False

        dx, dy = direction
        
        line_of_blocks = []
        block_to_check = self.blocks[block_idx]
        
        # Find all blocks in the push line
        temp_pos = block_to_check['pos']
        while True:
            found_block = self._get_block_at(temp_pos)
            if found_block:
                line_of_blocks.append(found_block)
                temp_pos = (temp_pos[0] + dx, temp_pos[1] + dy)
            else:
                break
        
        # The space just beyond the last block in the line
        final_pos = temp_pos
        
        # Check for boundary collision
        if not (0 <= final_pos[0] < self.GRID_WIDTH and 0 <= final_pos[1] < self.GRID_HEIGHT):
            return False # Blocked by wall

        # Execute move for all blocks in the line
        for block in reversed(line_of_blocks): # Move from back to front
            block['pos'] = (block['pos'][0] + dx, block['pos'][1] + dy)
            
        return True

    def _get_block_at(self, pos):
        for block in self.blocks:
            if block['pos'] == pos:
                return block
        return None
        
    def _get_current_placements(self):
        placements = set()
        for block in self.blocks:
            for target in self.targets:
                if block['id'] == target['id'] and block['pos'] == target['pos']:
                    placements.add(block['id'])
        return placements

    def _reset_selection(self):
        """Selects the block that is top-most, then left-most."""
        if not self.blocks: return
        self.blocks.sort(key=lambda b: (b['pos'][1], b['pos'][0]))
        self.selected_block_idx = 0

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
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw targets
        for target in self.targets:
            px, py = self._grid_to_pixel_center(target['pos'])
            is_lit = target['id'] in self.blocks_on_target_state
            color = self.TARGET_LIT_COLORS[target['id']] if is_lit else self.TARGET_COLORS[target['id']]
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(self.CELL_SIZE * 0.35), color)
            pygame.gfxdraw.aacircle(self.screen, px, py, int(self.CELL_SIZE * 0.35), color)

        # Draw blocks
        for i, block in enumerate(self.blocks):
            px, py = self._grid_to_pixel_topleft(block['pos'])
            color = self.BLOCK_COLORS[block['id']]
            rect = pygame.Rect(px + 4, py + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            
            # Highlight selected block
            if i == self.selected_block_idx and not self.game_over:
                pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 3, border_radius=4)

        # Draw push direction indicator
        if not self.game_over and self.selected_block_idx < len(self.blocks):
            selected_block = self.blocks[self.selected_block_idx]
            spx, spy = self._grid_to_pixel_center(selected_block['pos'])
            dx, dy = self.last_push_direction
            
            p1 = (spx + dx * 12, spy + dy * 12)
            p2 = (spx + dx * 18 - dy * 5, spy + dy * 18 + dx * 5)
            p3 = (spx + dx * 18 + dy * 5, spy + dy * 18 - dx * 5)
            pygame.draw.polygon(self.screen, self.COLOR_SELECTED, [p1, p2, p3])

    def _render_ui(self):
        # Render moves left
        moves_text = self.ui_font.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Render score
        score_text = self.ui_font.render(f"Score: {self.score}/{self.NUM_BLOCKS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Render user guide
        guide_text = self.guide_font.render(self.user_guide, True, self.COLOR_GRID)
        self.screen.blit(guide_text, (10, self.SCREEN_HEIGHT - guide_text.get_height() - 5))

        # Render win/loss message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if len(self.blocks_on_target_state) == self.NUM_BLOCKS:
                msg_text = self.end_font.render("PUZZLE SOLVED", True, self.COLOR_WIN)
            else:
                msg_text = self.end_font.render("OUT OF MOVES", True, self.COLOR_LOSE)
                
            text_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(msg_text, text_rect)

    def _grid_to_pixel_center(self, grid_pos):
        px = int((grid_pos[0] + 0.5) * self.CELL_SIZE)
        py = int((grid_pos[1] + 0.5) * self.CELL_SIZE)
        return px, py
        
    def _grid_to_pixel_topleft(self, grid_pos):
        px = int(grid_pos[0] * self.CELL_SIZE)
        py = int(grid_pos[1] * self.CELL_SIZE)
        return px, py

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
    # Set to "human" to play, "rgb_array" for headless
    render_mode = "human" 
    
    if render_mode == "human":
        # Modify the class to render to screen for human play
        GameEnv.metadata["render_modes"].append("human")
        _original_init = GameEnv.__init__
        _original_get_obs = GameEnv._get_observation

        def human_init(self, render_mode="human"):
            _original_init(self, render_mode=render_mode)
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Block Pusher")

        def human_get_obs(self):
            obs = _original_get_obs(self)
            if self.render_mode == "human":
                pygame.display.flip()
            return obs

        GameEnv.__init__ = human_init
        GameEnv._get_observation = human_get_obs

    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()
    done = False
    
    # --- Human Player Controls ---
    # Maps keyboard keys to the MultiDiscrete action space
    key_to_action = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0],
        pygame.K_LSHIFT: [0, 0, 1],
        pygame.K_RSHIFT: [0, 0, 1],
    }
    
    print("\n" + "="*30)
    print("      Block Pusher Controls")
    print("="*30)
    print(GameEnv.user_guide)
    print("R: Reset level")
    print("Q: Quit")
    print("="*30 + "\n")

    while not done:
        action = [0, 0, 0] # Default no-op action
        
        if render_mode == "human":
            # For human play, we poll events and construct the action
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        done = True
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
            
            # Check for held keys to build the MultiDiscrete action
            keys = pygame.key.get_pressed()
            action_changed = False
            for key, act_val in key_to_action.items():
                if keys[key]:
                    action[0] = max(action[0], act_val[0])
                    action[1] = max(action[1], act_val[1])
                    action[2] = max(action[2], act_val[2])
                    action_changed = True
            
            # Only step if an action is taken
            if action_changed:
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                # For turn-based, we still need to render even on no-op
                env._get_observation()

            if terminated:
                print(f"Game Over! Final Info: {info}")
                pygame.time.wait(2000) # Pause on game over
                obs, info = env.reset()

            env.clock.tick(30) # Limit FPS for human play
        
        else: # For RL agent (random actions)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode finished. Final Info: {info}")
                obs, info = env.reset()

    env.close()