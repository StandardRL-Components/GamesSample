
# Generated: 2025-08-27T21:47:14.763562
# Source Brief: brief_02906.md
# Brief Index: 2906

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A grid-based puzzle environment where the player must push blocks to clear a path
    to an exit within a time limit. This game is inspired by Sokoban, with a focus
    on planning and efficient movement.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move and push blocks. Clear a path to the red exit before the timer runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A grid-based puzzle where you push blocks to reach an exit. Each move costs time. Plan your pushes carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_SIZE = 40
        self.MAX_TIME = 30
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_HIGHLIGHT = (128, 255, 200)
        self.COLOR_BLOCK = (0, 128, 255)
        self.COLOR_BLOCK_HIGHLIGHT = (128, 200, 255)
        self.COLOR_EXIT = (255, 64, 64)
        self.COLOR_EXIT_HIGHLIGHT = (255, 150, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_WIN = (255, 255, 0)
        self.COLOR_LOSE = (255, 100, 100)

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
        self.font_small = pygame.font.Font(None, 28)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = 0
        self.player_pos = [0, 0]
        self.blocks = []
        self.exit_pos = [0, 0]
        self.np_random = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = self.MAX_TIME

        # Generate a fixed, solvable level layout
        self.player_pos = [2, 5]
        self.exit_pos = [14, 5]
        self.blocks = [
            [5, 3], [5, 4], [5, 5], [5, 6], [5, 7],
            [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8],
            [11, 4], [11, 5], [11, 6]
        ]
        self.blocks = [list(pos) for pos in self.blocks]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Cost for taking a step/turn

        if movement != 0:  # Any action other than no-op costs time
            self.time_remaining -= 1
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))

            if dx != 0 or dy != 0:
                target_x, target_y = self.player_pos[0] + dx, self.player_pos[1] + dy

                block_idx = self._get_block_at(target_x, target_y)
                if block_idx is not None:
                    # Attempt to push the block
                    push_reward, push_successful = self._handle_push(block_idx, dx, dy)
                    if push_successful:
                        reward += push_reward
                        self.player_pos = [target_x, target_y]
                        # sfx: block_slide.wav
                elif self._is_within_bounds(target_x, target_y):
                    # Move player into empty space
                    self.player_pos = [target_x, target_y]
                    # sfx: player_step.wav

        self.steps += 1
        self.score += reward

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.player_pos == self.exit_pos:
                reward += 100  # Reached exit
                self.score += 100
                # sfx: level_win.wav
            elif self.time_remaining <= 0:
                reward += -50  # Timed out
                self.score += -50
                # sfx: level_lose.wav

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_push(self, first_block_idx, dx, dy):
        line_of_blocks = []
        current_pos = list(self.blocks[first_block_idx])

        while True:
            idx = self._get_block_at(current_pos[0], current_pos[1])
            if idx is not None:
                line_of_blocks.append(idx)
                current_pos[0] += dx
                current_pos[1] += dy
            else:
                break

        last_block_pos = self.blocks[line_of_blocks[-1]]
        next_pos_x, next_pos_y = last_block_pos[0] + dx, last_block_pos[1] + dy

        if not self._is_within_bounds(next_pos_x, next_pos_y) or self._get_block_at(next_pos_x, next_pos_y) is not None:
            # sfx: push_fail.wav
            return 0, False

        push_reward = 0
        for block_idx in reversed(line_of_blocks):
            block_pos = self.blocks[block_idx]
            dist_before = self._manhattan_distance(block_pos, self.exit_pos)
            dist_after = self._manhattan_distance([block_pos[0] + dx, block_pos[1] + dy], self.exit_pos)

            if dist_after < dist_before:
                push_reward += 5
            elif dist_after > dist_before:
                push_reward -= 1

        for block_idx in reversed(line_of_blocks):
            self.blocks[block_idx][0] += dx
            self.blocks[block_idx][1] += dy

        return push_reward, True

    def _get_block_at(self, x, y):
        for i, block_pos in enumerate(self.blocks):
            if block_pos[0] == x and block_pos[1] == y:
                return i
        return None

    def _is_within_bounds(self, x, y):
        return 0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _check_termination(self):
        if self.player_pos == self.exit_pos:
            return True
        if self.time_remaining <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
                pygame.draw.rect(self.screen, self.COLOR_GRID, (px, py, self.CELL_SIZE, self.CELL_SIZE), 1)

        self._draw_beveled_rect(self.exit_pos, self.COLOR_EXIT, self.COLOR_EXIT_HIGHLIGHT)
        for block_pos in self.blocks:
            self._draw_beveled_rect(block_pos, self.COLOR_BLOCK, self.COLOR_BLOCK_HIGHLIGHT)
        self._draw_beveled_rect(self.player_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_HIGHLIGHT)

    def _draw_beveled_rect(self, grid_pos, color, highlight_color):
        px, py = grid_pos[0] * self.CELL_SIZE, grid_pos[1] * self.CELL_SIZE
        margin = 4
        highlight_size = 3
        
        pygame.draw.rect(self.screen, color, (px + margin, py + margin, self.CELL_SIZE - 2 * margin, self.CELL_SIZE - 2 * margin))
        pygame.draw.rect(self.screen, highlight_color, (px + margin, py + margin, self.CELL_SIZE - 2 * margin, highlight_size))

    def _render_ui(self):
        time_text = self.font_large.render(f"Time: {max(0, self.time_remaining)}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(time_text, time_rect)

        score_text = self.font_large.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topleft=(15, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.player_pos == self.exit_pos:
                msg, color = "LEVEL COMPLETE!", self.COLOR_WIN
            else:
                msg, color = "TIME UP!", self.COLOR_LOSE

            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(end_text, end_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score:.1f}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "player_pos": list(self.player_pos),
            "exit_pos": list(self.exit_pos),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the game directly for testing
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation() # Run validation on startup
    
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    print("\n" + env.user_guide)
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
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
                    done = False
                elif event.key == pygame.K_q:
                    running = False
        
        # Only step if an action was taken
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                print(f"Game Over! Final Score: {info['score']:.1f}. Press 'R' to restart or 'Q' to quit.")

        # Render the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()