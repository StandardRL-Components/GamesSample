
# Generated: 2025-08-27T20:05:06.929394
# Source Brief: brief_02340.md
# Brief Index: 2340

        
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
    A block-pushing puzzle environment. The goal is to push 5 colored blocks
    onto their corresponding colored targets within a 20-move limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to push the selected block. "
        "Press Space to cycle to the next block, and Shift to cycle to the previous block."
    )

    game_description = (
        "Push colored blocks onto matching targets in a grid-based puzzle. "
        "You have a limited number of moves to solve the puzzle. Plan your pushes carefully!"
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game Configuration ---
        self.GRID_WIDTH = 16
        self.GRID_HEIGHT = 10
        self.CELL_SIZE = 40
        self.X_OFFSET = (640 - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.Y_OFFSET = (400 - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        self.NUM_BLOCKS = 5
        self.MOVE_LIMIT = 20
        self.MAX_STEPS = 1000 # Fallback step limit

        # --- Colors ---
        self.COLOR_BG = (26, 33, 41)
        self.COLOR_GRID = (46, 53, 61)
        self.COLOR_WALL = (67, 76, 88)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SELECTED = (255, 255, 255)
        
        self.BLOCK_COLORS = [
            (231, 76, 60),   # Red
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
            (241, 196, 15),  # Yellow
            (155, 89, 182)   # Purple
        ]
        
        # --- State Variables ---
        self.game_over = False
        self.score = 0
        self.steps = 0
        self.moves_remaining = 0
        self.blocks = []
        self.targets = []
        self.walls = []
        self.selected_block_idx = 0
        self.last_action_was_selection = False
        
        self.reset()
        
        # --- Self-Validation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_over = False
        self.score = 0
        self.steps = 0
        self.moves_remaining = self.MOVE_LIMIT
        self.selected_block_idx = 0
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        action_taken = False
        
        # 1. Handle Block Selection
        if space_press:
            self.selected_block_idx = (self.selected_block_idx + 1) % self.NUM_BLOCKS
            action_taken = True
        elif shift_press:
            self.selected_block_idx = (self.selected_block_idx - 1 + self.NUM_BLOCKS) % self.NUM_BLOCKS
            action_taken = True

        self.last_action_was_selection = action_taken

        # 2. Handle Block Pushing
        if movement > 0 and not action_taken: # Prioritize selection over movement in same step
            self.moves_remaining -= 1
            reward -= 0.1 # Small penalty for making a move
            
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            
            completed_before = self._count_completed_targets()
            self._execute_push(dx, dy)
            completed_after = self._count_completed_targets()
            
            newly_completed = completed_after - completed_before
            if newly_completed > 0:
                reward += newly_completed * 5.0 # Reward for placing a block
                # sfx: success chime

        self.score += reward
        
        # 3. Check for Termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self._check_win_condition():
                terminal_reward = 50.0
                # sfx: victory fanfare
            else: # Lost due to move limit
                terminal_reward = -50.0
                # sfx: failure sound
            self.score += terminal_reward
            reward += terminal_reward

        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_puzzle(self):
        # Create walls around the border
        self.walls = []
        for x in range(self.GRID_WIDTH):
            self.walls.append((x, 0))
            self.walls.append((x, self.GRID_HEIGHT - 1))
        for y in range(1, self.GRID_HEIGHT - 1):
            self.walls.append((0, y))
            self.walls.append((self.GRID_WIDTH - 1, y))

        # Add some random internal walls
        num_internal_walls = self.np_random.integers(3, 7)
        for _ in range(num_internal_walls):
            wall_len = self.np_random.integers(2, 5)
            start_x = self.np_random.integers(2, self.GRID_WIDTH - 3)
            start_y = self.np_random.integers(2, self.GRID_HEIGHT - 3)
            is_horizontal = self.np_random.choice([True, False])
            for i in range(wall_len):
                if is_horizontal:
                    pos = (start_x + i, start_y)
                else:
                    pos = (start_x, start_y + i)
                if pos not in self.walls:
                    self.walls.append(pos)
        
        # Generate valid spawn points
        valid_positions = []
        for x in range(1, self.GRID_WIDTH - 1):
            for y in range(1, self.GRID_HEIGHT - 1):
                if (x, y) not in self.walls:
                    valid_positions.append((x, y))

        # Ensure we have enough space for blocks and targets
        if len(valid_positions) < self.NUM_BLOCKS * 2:
            # Failsafe if random walls block too much space
            return self._generate_puzzle()

        # Place targets
        self.targets = []
        target_indices = self.np_random.choice(len(valid_positions), self.NUM_BLOCKS, replace=False)
        for i, color in enumerate(self.BLOCK_COLORS):
            pos = valid_positions[target_indices[i]]
            self.targets.append({"pos": pos, "color": color})
        
        # Place blocks on targets (solved state)
        self.blocks = []
        for target in self.targets:
             self.blocks.append({"pos": target["pos"], "color": target["color"]})

        # Scramble the puzzle by working backwards
        num_scramble_moves = self.np_random.integers(15, 25)
        for _ in range(num_scramble_moves):
            block_idx = self.np_random.integers(0, self.NUM_BLOCKS)
            self.selected_block_idx = block_idx
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            self._execute_push(dx, dy)
        
        # Reset selection to first block
        self.selected_block_idx = 0


    def _execute_push(self, dx, dy):
        # sfx: block push sound
        block_to_push = self.blocks[self.selected_block_idx]
        
        # 1. Identify the chain of blocks to be pushed
        chain = []
        current_pos = block_to_push["pos"]
        
        while True:
            found_block = None
            for i, block in enumerate(self.blocks):
                if block["pos"] == current_pos:
                    found_block = block
                    chain.append(i)
                    break
            
            if found_block is None:
                break
            
            current_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        # 2. Check if the push is valid
        end_of_chain_pos = (self.blocks[chain[-1]]["pos"][0] + dx, self.blocks[chain[-1]]["pos"][1] + dy)
        
        is_occupied = any(block["pos"] == end_of_chain_pos for block in self.blocks)
        is_wall = end_of_chain_pos in self.walls
        
        if not is_occupied and not is_wall:
            # 3. Move all blocks in the chain
            for block_idx in reversed(chain):
                old_pos = self.blocks[block_idx]["pos"]
                self.blocks[block_idx]["pos"] = (old_pos[0] + dx, old_pos[1] + dy)

    def _check_win_condition(self):
        return self._count_completed_targets() == self.NUM_BLOCKS

    def _count_completed_targets(self):
        count = 0
        for block in self.blocks:
            for target in self.targets:
                if block["pos"] == target["pos"] and block["color"] == target["color"]:
                    count += 1
                    break
        return count

    def _check_termination(self):
        return self.moves_remaining <= 0 or self._check_win_condition()

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
            "moves_remaining": self.moves_remaining,
            "completed_targets": self._count_completed_targets()
        }
    
    def _grid_to_pixel(self, x, y):
        return (self.X_OFFSET + x * self.CELL_SIZE, self.Y_OFFSET + y * self.CELL_SIZE)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            start_pos = self._grid_to_pixel(x, 0)
            end_pos = self._grid_to_pixel(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_pos[0], 0), (end_pos[0], 400), 1)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = self._grid_to_pixel(0, y)
            end_pos = self._grid_to_pixel(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, start_pos[1]), (640, end_pos[1]), 1)

        # Draw walls
        for wall_pos in self.walls:
            px, py = self._grid_to_pixel(wall_pos[0], wall_pos[1])
            pygame.draw.rect(self.screen, self.COLOR_WALL, (px, py, self.CELL_SIZE, self.CELL_SIZE))

        # Draw targets
        for target in self.targets:
            px, py = self._grid_to_pixel(target["pos"][0], target["pos"][1])
            color = target["color"]
            # Desaturate color for targets
            target_color = tuple(int(c * 0.4) for c in color)
            pygame.gfxdraw.box(self.screen, (px + 4, py + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8), target_color)

        # Draw blocks
        for i, block in enumerate(self.blocks):
            px, py = self._grid_to_pixel(block["pos"][0], block["pos"][1])
            color = block["color"]
            
            # Draw shadow
            shadow_pos = (px + 6, py + 6, self.CELL_SIZE - 6, self.CELL_SIZE - 6)
            pygame.gfxdraw.box(self.screen, shadow_pos, (0,0,0,100))

            # Draw block
            block_rect = (px + 3, py + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6)
            pygame.gfxdraw.box(self.screen, block_rect, color)
            pygame.gfxdraw.rectangle(self.screen, block_rect, tuple(min(255, c+30) for c in color))

            # Highlight selected block
            if i == self.selected_block_idx:
                pygame.draw.rect(self.screen, self.COLOR_SELECTED, (px, py, self.CELL_SIZE, self.CELL_SIZE), 3)

    def _render_ui(self):
        # Moves Remaining
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 10))
        
        # Current Score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 50))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self._check_win_condition():
                end_text = self.font_large.render("PUZZLE SOLVED!", True, self.BLOCK_COLORS[1])
            else:
                end_text = self.font_large.render("OUT OF MOVES!", True, self.BLOCK_COLORS[0])
            
            text_rect = end_text.get_rect(center=(320, 200))
            self.screen.blit(end_text, text_rect)

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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.init()
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Block Pusher")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
        
        action = [movement, space, shift]
        
        # Only step if an action is taken
        if any(a > 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("Game Over! Press 'R' to reset.")
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()