import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:50:11.826742
# Source Brief: brief_03035.md
# Brief Index: 3035
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: A minimalist puzzle game where the player controls a transformable
    block on a 7x7 grid. The objective is to clear all 15 red squares by
    touching them, which can trigger satisfying chain reactions. The game is
    designed with a strong emphasis on visual clarity and game feel.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) - Currently unused.
    - actions[2]: Shift button (0=released, 1=held) - Transforms the player block.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +100 for clearing all red squares (winning).
    - -100 for getting stuck with no valid moves (losing).
    - +1.0 for each red square cleared directly by the player.
    - +0.1 for each red square cleared in a chain reaction.
    - -0.01 per step to encourage efficiency.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Clear all red squares by moving your transformable block over them on a 7x7 grid, triggering chain reactions."
    user_guide = "Use the arrow keys (↑↓←→) to move your block. Press Shift to transform its shape."
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 7
    NUM_RED_SQUARES = 15
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255)
    COLOR_RED = (255, 50, 50)
    COLOR_CLEARED = (200, 200, 220)
    COLOR_FLASH = (255, 255, 255)
    COLOR_UI = (240, 240, 240)

    # Game states
    STATE_EMPTY = 0
    STATE_RED = 1
    STATE_CLEARED = 2

    # Player shapes
    SHAPE_1x1 = 0
    SHAPE_2x1 = 1  # Horizontal
    SHAPE_1x2 = 2  # Vertical

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font = pygame.font.SysFont(None, 28, bold=True)


        # Grid rendering properties
        self.grid_area_size = 350
        self.cell_size = self.grid_area_size // self.GRID_SIZE
        self.grid_top_left_x = (self.SCREEN_WIDTH - self.grid_area_size) // 2
        self.grid_top_left_y = (self.SCREEN_HEIGHT - self.grid_area_size) // 2

        # Initialize state variables
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.player_pos = (0, 0)
        self.player_shape = self.SHAPE_1x1
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.flash_effects = []

        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # validation is for dev, not needed in production env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.flash_effects = []

        self.player_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.player_shape = self.SHAPE_1x1

        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), self.STATE_EMPTY, dtype=np.int8)
        
        available_cells = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (c, r) != self.player_pos:
                    available_cells.append((c, r))
        
        red_square_indices = self.np_random.choice(len(available_cells), self.NUM_RED_SQUARES, replace=False)
        for idx in red_square_indices:
            x, y = available_cells[idx]
            self.grid[y, x] = self.STATE_RED

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01

        movement, _, shift_held = action[0], action[1] == 1, action[2] == 1

        if shift_held:
            # sfx: transform_sound
            next_shape = (self.player_shape + 1) % 3
            if self._is_valid_config(self.player_pos, next_shape):
                self.player_shape = next_shape

        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if self._is_valid_config(new_pos, self.player_shape):
                self.player_pos = new_pos
                # sfx: move_sound

        player_cells = self._get_player_cells(self.player_pos, self.player_shape)
        
        cleared_this_step = []
        chain_reaction_queue = []

        for x, y in player_cells:
            if 0 <= y < self.GRID_SIZE and 0 <= x < self.GRID_SIZE and self.grid[y, x] == self.STATE_RED:
                self.grid[y, x] = self.STATE_CLEARED
                reward += 1.0
                cleared_this_step.append((x, y))
                chain_reaction_queue.append((x, y))
                # sfx: clear_direct_sound
        
        head = 0
        while head < len(chain_reaction_queue):
            cx, cy = chain_reaction_queue[head]
            head += 1

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                    if self.grid[ny, nx] == self.STATE_RED:
                        self.grid[ny, nx] = self.STATE_CLEARED
                        reward += 0.1
                        cleared_this_step.append((nx, ny))
                        chain_reaction_queue.append((nx, ny))
                        # sfx: clear_chain_sound
        
        for pos in cleared_this_step:
            self.flash_effects.append([pos, 10])

        terminated = False
        truncated = False
        remaining_red = np.sum(self.grid == self.STATE_RED)

        if remaining_red == 0:
            terminated = True
            reward += 100.0
            # sfx: win_sound
        elif not self._has_valid_moves():
            terminated = True
            reward += -100.0
            # sfx: lose_sound
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.game_over = terminated or truncated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_player_cells(self, pos, shape):
        x, y = pos
        cells = [(x, y)]
        if shape == self.SHAPE_2x1:
            cells.append((x + 1, y))
        elif shape == self.SHAPE_1x2:
            cells.append((x, y + 1))
        return cells

    def _is_valid_config(self, pos, shape):
        cells = self._get_player_cells(pos, shape)
        for x, y in cells:
            if not (0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE):
                return False
            if self.grid[y, x] == self.STATE_CLEARED:
                return False
        return True

    def _has_valid_moves(self):
        for next_shape in range(3):
            if next_shape != self.player_shape and self._is_valid_config(self.player_pos, next_shape):
                return True
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if self._is_valid_config(new_pos, self.player_shape):
                return True
        
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_red": np.sum(self.grid == self.STATE_RED),
        }

    def _get_observation(self):
        new_flashes = []
        for flash in self.flash_effects:
            flash[1] -= 1
            if flash[1] > 0:
                new_flashes.append(flash)
        self.flash_effects = new_flashes
        
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_top_left_x + i * self.cell_size, self.grid_top_left_y), (self.grid_top_left_x + i * self.cell_size, self.grid_top_left_y + self.grid_area_size), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_top_left_x, self.grid_top_left_y + i * self.cell_size), (self.grid_top_left_x + self.grid_area_size, self.grid_top_left_y + i * self.cell_size), 1)

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(self.grid_top_left_x + c * self.cell_size + 1, self.grid_top_left_y + r * self.cell_size + 1, self.cell_size - 1, self.cell_size - 1)
                if self.grid[r, c] == self.STATE_RED:
                    pygame.draw.rect(self.screen, self.COLOR_RED, cell_rect, border_radius=4)
                elif self.grid[r, c] == self.STATE_CLEARED:
                    pygame.draw.rect(self.screen, self.COLOR_CLEARED, cell_rect, border_radius=4)
        
        for pos, lifetime in self.flash_effects:
            x, y = pos
            alpha = int(255 * (lifetime / 10))
            flash_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_FLASH, alpha))
            self.screen.blit(flash_surface, (self.grid_top_left_x + x * self.cell_size, self.grid_top_left_y + y * self.cell_size))

        player_cells = self._get_player_cells(self.player_pos, self.player_shape)
        for x, y in player_cells:
            center_x = self.grid_top_left_x + x * self.cell_size + self.cell_size // 2
            center_y = self.grid_top_left_y + y * self.cell_size + self.cell_size // 2
            
            for i in range(10, 0, -1):
                alpha = max(0, 40 - i * 4)
                radius = self.cell_size // 2 + i
                pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), int(radius), (*self.COLOR_PLAYER_GLOW, alpha))
            
            player_rect = pygame.Rect(self.grid_top_left_x + x * self.cell_size + 2, self.grid_top_left_y + y * self.cell_size + 2, self.cell_size - 3, self.cell_size - 3)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=5)

    def _render_ui(self):
        remaining_text = f"REMAINING: {np.sum(self.grid == self.STATE_RED)}"
        text_surface = self.font.render(remaining_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (20, 20))
        
        score_text = f"SCORE: {self.score:.2f}"
        score_surface = self.font.render(score_text, True, self.COLOR_UI)
        score_rect = score_surface.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_surface, score_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a display window
    game_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Transform")
    clock = pygame.time.Clock()

    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Press 'R' to reset.")
    print("----------------------\n")

    while not done:
        movement = 0 # no-op
        shift = 0
        
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                movement = 1
            elif event.key == pygame.K_DOWN:
                movement = 2
            elif event.key == pygame.K_LEFT:
                movement = 3
            elif event.key == pygame.K_RIGHT:
                movement = 4
            elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                shift = 1
            elif event.key == pygame.K_r: # Reset on 'r'
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")
                continue # skip step
            
            action = [movement, 0, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Remaining: {info['remaining_red']}")
            done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done:
            print("\n--- GAME OVER ---")
            print(f"Final Score: {info['score']:.2f}")
            print("Press 'R' to play again or close the window.")
            
            waiting_for_reset = True
            while waiting_for_reset:
                 for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        done = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        done = False
                        waiting_for_reset = False
                        print("\n--- Game Reset ---\n")

        clock.tick(30)

    env.close()