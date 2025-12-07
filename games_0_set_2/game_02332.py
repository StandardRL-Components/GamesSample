
# Generated: 2025-08-27T20:03:04.624870
# Source Brief: brief_02332.md
# Brief Index: 2332

        
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
        "Controls: Arrows to push. Space/Shift to cycle selected block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A Sokoban-style puzzle game. Push the colored blocks onto their matching targets before you run out of moves. Use arrows to push the selected block (highlighted) and Space/Shift to change which block is selected."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    CELL_SIZE = 36
    GRID_WIDTH = (SCREEN_WIDTH // CELL_SIZE) - 2
    GRID_HEIGHT = (SCREEN_HEIGHT // CELL_SIZE)
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (26, 33, 54)
    COLOR_GRID = (45, 52, 71)
    COLOR_TEXT = (220, 220, 230)
    COLOR_SELECT_OUTLINE = (255, 255, 255)
    
    BLOCK_COLORS = [
        (52, 152, 219),  # Blue
        (231, 76, 60),   # Red
        (46, 204, 113),  # Green
        (241, 196, 15),  # Yellow
        (155, 89, 182),  # Purple
        (230, 126, 34),  # Orange
        (26, 188, 156),  # Turquoise
        (243, 156, 18),  # Sun Flower
    ]

    # Rewards
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0
    REWARD_MOVE = -0.1
    REWARD_ON_TARGET = 1.0
    
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
        self.font_main = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # Game state variables
        self.level = 0
        self.max_blocks = 5
        self.max_moves = 15
        self.blocks = []
        self.targets = []
        self.moves_remaining = 0
        self.selected_block_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Level progression
        self.max_blocks = min(8, 5 + self.level)
        self.max_moves = min(25, 15 + self.level * 2)
        self.moves_remaining = self.max_moves

        # Generate a new solvable puzzle
        self._generate_puzzle()

        self.selected_block_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0
        action_taken = False

        # --- Handle Block Selection ---
        # Cycle selection on button press (not hold)
        if space_held and not self.last_space_held:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.blocks)
            action_taken = True
        if shift_held and not self.last_shift_held:
            self.selected_block_idx = (self.selected_block_idx - 1 + len(self.blocks)) % len(self.blocks)
            action_taken = True
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Handle Push Action ---
        if movement > 0:
            action_taken = True
            self.moves_remaining -= 1
            reward += self.REWARD_MOVE

            # Check previous state for reward calculation
            blocks_on_target_before = self._count_blocks_on_target()

            # Attempt the push
            self._push_block(self.selected_block_idx, movement)

            # Check post-move state for reward
            blocks_on_target_after = self._count_blocks_on_target()
            
            # Reward for newly placed blocks
            if blocks_on_target_after > blocks_on_target_before:
                reward += (blocks_on_target_after - blocks_on_target_before) * self.REWARD_ON_TARGET

        self.steps += 1
        self.score += reward
        
        # --- Check Termination Conditions ---
        terminated = False
        if self._check_win_condition():
            self.score += self.REWARD_WIN
            terminated = True
            self.game_over = True
            self.level += 1 # Progress difficulty for next reset
        elif self.moves_remaining <= 0:
            self.score += self.REWARD_LOSE
            terminated = True
            self.game_over = True
            self.level = 0 # Reset difficulty on loss
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.level = 0
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw targets
        for target in self.targets:
            gx, gy = target["pos"]
            cx = self.GRID_OFFSET_X + int((gx + 0.5) * self.CELL_SIZE)
            cy = self.GRID_OFFSET_Y + int((gy + 0.5) * self.CELL_SIZE)
            radius = self.CELL_SIZE // 4
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, target["color"])
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, target["color"])

        # Draw blocks
        for i, block in enumerate(self.blocks):
            gx, gy = block["pos"]
            rect = pygame.Rect(
                self.GRID_OFFSET_X + gx * self.CELL_SIZE + 3,
                self.GRID_OFFSET_Y + gy * self.CELL_SIZE + 3,
                self.CELL_SIZE - 6,
                self.CELL_SIZE - 6
            )
            pygame.draw.rect(self.screen, block["color"], rect, border_radius=4)
            
            # Highlight selected block
            if i == self.selected_block_idx:
                pygame.draw.rect(self.screen, self.COLOR_SELECT_OUTLINE, rect, 2, border_radius=4)

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 15, 10))
        
        # Score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Level
        level_text = self.font_small.render(f"Level: {self.level + 1}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (15, 40))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_status = "PUZZLE SOLVED!" if self._check_win_condition() else "OUT OF MOVES"
            
            end_text = self.font_main.render(win_status, True, self.COLOR_SELECT_OUTLINE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "level": self.level,
            "blocks_on_target": self._count_blocks_on_target(),
        }

    def _generate_puzzle(self):
        self.blocks = []
        self.targets = []
        
        # Get all possible grid positions
        all_pos = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_pos)
        
        # Place targets
        for i in range(self.max_blocks):
            pos = all_pos.pop()
            color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
            self.targets.append({"pos": pos, "color": color, "id": i})
            # Initially place blocks on targets
            self.blocks.append({"pos": pos, "color": color, "id": i})
        
        # Shuffle the puzzle by making random moves from the solved state
        # This guarantees solvability
        shuffle_moves = self.max_blocks * 5
        for _ in range(shuffle_moves):
            block_idx = self.np_random.integers(0, self.max_blocks)
            move_dir = self.np_random.integers(1, 5)
            self._push_block(block_idx, move_dir, is_shuffle=True)

    def _push_block(self, block_idx, direction, is_shuffle=False):
        # sfx: push_attempt.wav
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[direction]
        
        start_block = self.blocks[block_idx]
        chain = [start_block]
        
        # Find all blocks in the push chain
        current_pos = start_block["pos"]
        while True:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check for wall collision
            if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                return # Push fails, chain hits a wall
            
            # Check if next cell is occupied by another block
            found_block = False
            for b in self.blocks:
                if b["pos"] == next_pos:
                    chain.append(b)
                    current_pos = next_pos
                    found_block = True
                    break
            if not found_block:
                break # Chain ends, next spot is empty

        # Move all blocks in the chain, starting from the end
        for block_in_chain in reversed(chain):
            old_pos = block_in_chain["pos"]
            block_in_chain["pos"] = (old_pos[0] + dx, old_pos[1] + dy)
        
        if not is_shuffle:
            # sfx: push_success.wav
            pass

    def _count_blocks_on_target(self):
        count = 0
        for block in self.blocks:
            for target in self.targets:
                if block["id"] == target["id"] and block["pos"] == target["pos"]:
                    count += 1
                    break
        return count

    def _check_win_condition(self):
        return self._count_blocks_on_target() == len(self.blocks)

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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Puzzler")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
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
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_q:
                    running = False
        
        # If an action was triggered, step the environment
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
        
        # Render the current observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print("Game Over! Press 'r' to restart or 'q' to quit.")

    env.close()