
# Generated: 2025-08-27T21:35:59.434430
# Source Brief: brief_02840.md
# Brief Index: 2840

        
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
        "Controls: Use arrow keys to push all blocks simultaneously. Goal: move blocks to their matching colored targets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A Sokoban-style puzzle where every block moves at once. Plan your pushes carefully to solve the puzzle within the move limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.CELL_SIZE = 40
        self.NUM_BLOCKS = 15
        self.MAX_MOVES = 100
        self.MAX_SHUFFLE_MOVES = 30

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_WHITE = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 87, 87), (255, 170, 87), (255, 255, 87),
            (170, 255, 87), (87, 255, 87), (87, 255, 170),
            (87, 255, 255), (87, 170, 255), (87, 87, 255),
            (170, 87, 255), (255, 87, 255), (255, 87, 170),
            (255, 200, 200), (200, 255, 200), (200, 200, 255)
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)

        # --- State Variables ---
        self.steps = 0
        self.moves_taken = 0
        self.score = 0.0
        self.game_over = False
        self.blocks = []
        self.targets = []

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.moves_taken = 0
        self.score = 0.0
        self.game_over = False

        # Generate fixed, non-overlapping target locations
        possible_coords = np.array([(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)])
        self.np_random.shuffle(possible_coords)
        self.targets = [tuple(c) for c in possible_coords[:self.NUM_BLOCKS]]

        # Generate a solvable block configuration by shuffling from the solved state
        self._generate_solvable_puzzle()

        return self._get_observation(), self._get_info()

    def _generate_solvable_puzzle(self):
        # Start with the solved state (each block is on its target)
        self.blocks = list(self.targets)

        # Apply a number of random pushes to shuffle the board
        num_shuffles = self.np_random.integers(20, self.MAX_SHUFFLE_MOVES + 1)
        for _ in range(num_shuffles):
            random_push = self.np_random.integers(1, 5)  # 1-4 for up, down, left, right
            self._perform_push(random_push)

        # In the unlikely event the shuffle solved it, re-shuffle
        if self._count_blocks_on_target() == self.NUM_BLOCKS:
            self._generate_solvable_puzzle()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        terminated = False

        # Only non-noop actions (1-4) constitute a move
        if movement > 0:
            self.moves_taken += 1
            blocks_on_target_before = self._count_blocks_on_target()

            # sfx_block_push
            self._perform_push(movement)

            blocks_on_target_after = self._count_blocks_on_target()

            # --- Reward Calculation ---
            reward = -0.1  # Cost per move
            newly_on_target_reward = (blocks_on_target_after - blocks_on_target_before) * 5.0
            reward += newly_on_target_reward
            self.score += reward

            # --- Termination Check ---
            if blocks_on_target_after == self.NUM_BLOCKS:
                # Win condition
                terminated = True
                self.game_over = True
                win_bonus = 100.0
                reward += win_bonus
                self.score += win_bonus
                # sfx_win_fanfare
            elif self.moves_taken >= self.MAX_MOVES:
                # Lose condition
                terminated = True
                self.game_over = True
                loss_penalty = -100.0
                reward += loss_penalty
                self.score += loss_penalty
                # sfx_lose_buzzer

        self.steps += 1
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _perform_push(self, movement_direction):
        if movement_direction == 1:  # Up
            dx, dy, sort_key, sort_reverse = 0, -1, lambda b: self.blocks[b][1], False
        elif movement_direction == 2:  # Down
            dx, dy, sort_key, sort_reverse = 0, 1, lambda b: self.blocks[b][1], True
        elif movement_direction == 3:  # Left
            dx, dy, sort_key, sort_reverse = -1, 0, lambda b: self.blocks[b][0], False
        elif movement_direction == 4:  # Right
            dx, dy, sort_key, sort_reverse = 1, 0, lambda b: self.blocks[b][0], True
        else:
            return

        block_indices = sorted(range(self.NUM_BLOCKS), key=sort_key, reverse=sort_reverse)
        occupied_cells = set(self.blocks)

        for i in block_indices:
            current_pos = self.blocks[i]
            occupied_cells.remove(current_pos)

            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            while (0 <= next_pos[0] < self.GRID_WIDTH and
                   0 <= next_pos[1] < self.GRID_HEIGHT and
                   next_pos not in occupied_cells):
                current_pos = next_pos
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)

            self.blocks[i] = current_pos
            occupied_cells.add(current_pos)

    def _count_blocks_on_target(self):
        count = 0
        for i in range(self.NUM_BLOCKS):
            if self.blocks[i] == self.targets[i]:
                count += 1
        return count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.WIDTH + 1, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT + 1, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw targets first so blocks are rendered on top
        for i, (tx, ty) in enumerate(self.targets):
            pixel_x = tx * self.CELL_SIZE
            pixel_y = ty * self.CELL_SIZE
            target_color = self.BLOCK_COLORS[i]
            
            # Create a darker, less saturated version for the target zone
            r, g, b = target_color
            desat_color = (max(0, r - 150), max(0, g - 150), max(0, b - 150))
            
            target_rect = pygame.Rect(pixel_x, pixel_y, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, desat_color, target_rect.inflate(-8, -8))

        # Draw blocks
        for i, (bx, by) in enumerate(self.blocks):
            pixel_x = bx * self.CELL_SIZE
            pixel_y = by * self.CELL_SIZE
            block_color = self.BLOCK_COLORS[i]
            
            block_rect = pygame.Rect(pixel_x + 4, pixel_y + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            pygame.draw.rect(self.screen, block_color, block_rect, border_radius=4)
            
            # Visual feedback for correctly placed blocks
            if self.blocks[i] == self.targets[i]:
                # Draw a bright, glowing center
                center = block_rect.center
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 5, self.COLOR_WHITE)
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], 5, self.COLOR_WHITE)

    def _render_ui(self):
        # Moves counter
        moves_text = f"Moves: {self.moves_taken}/{self.MAX_MOVES}"
        moves_surf = self.font_large.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_surf, (10, 10))

        # Score display
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            is_win = self._count_blocks_on_target() == self.NUM_BLOCKS
            message = "PUZZLE SOLVED!" if is_win else "OUT OF MOVES!"
            
            msg_surf = self.font_large.render(message, True, self.COLOR_WHITE)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(msg_surf, msg_rect)

            reset_surf = self.font_medium.render("Call reset() to play again", True, self.COLOR_UI_TEXT)
            reset_rect = reset_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(reset_surf, reset_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    # env.validate_implementation() # Optional check
    
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Block Pusher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
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
                elif event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_q:
                    running = False
                
                # If a move key was pressed, step the environment
                if action[0] != 0 and not done:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}, Info: {info}")

        # Render the environment observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate

    env.close()