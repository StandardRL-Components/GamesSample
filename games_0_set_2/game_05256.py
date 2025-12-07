
# Generated: 2025-08-28T04:26:53.355187
# Source Brief: brief_05256.md
# Brief Index: 5256

        
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
    """
    A Sokoban-inspired puzzle game where the player must push crates onto target
    locations within a limited number of steps. The game is designed with a
    clean, modern aesthetic and provides clear feedback for all actions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move. Push all brown crates onto the green targets."
    )

    # Short, user-facing description of the game
    game_description = (
        "A fast-paced puzzle game. Push all crates onto their targets before you run out of moves."
    )

    # Frames advance only when an action is received, suitable for turn-based games.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.MAX_STEPS = 200  # Represents the time limit in moves

        # --- Visuals ---
        self.COLOR_BG = (30, 40, 50)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_WALL = (100, 110, 120)
        self.COLOR_WALL_SHADOW = (80, 90, 100)
        self.COLOR_TARGET = (60, 150, 60)
        self.COLOR_TARGET_FILLED = (100, 200, 100)
        self.COLOR_PLAYER = (220, 50, 50)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_CRATE = (160, 110, 70)
        self.COLOR_CRATE_SHADOW = (130, 80, 40)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_OVERLAY = (0, 0, 0, 180)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game State Variables (initialized in reset) ---
        self.level_map = []
        self.player_pos = (0, 0)
        self.crates = []
        self.targets = []
        self.walls = []
        self.crates_on_target_state = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # --- Pre-defined Levels ---
        self._levels = [
            [
                "################",
                "#T             #",
                "# ###########  #",
                "# # P       #  #",
                "# # C       # T#",
                "# ###########  #",
                "#   C          #",
                "#   C   T      #",
                "#              #",
                "################",
            ],
            [
                "################",
                "#              #",
                "# T  C       T #",
                "# ## # ####### #",
                "#  C # #  P    #",
                "#    # # ##### #",
                "#  T # # C     #",
                "#    # #       #",
                "#      ####### #",
                "################",
            ]
        ]
        
        self.reset()
        self.validate_implementation()

    def _get_level(self):
        # Choose a level layout at random.
        return self.np_random.choice(self._levels)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.level_map = self._get_level()
        self.player_pos = (0, 0)
        self.crates = []
        self.targets = []
        self.walls = []

        for y, row in enumerate(self.level_map):
            for x, char in enumerate(row):
                if char == 'P':
                    self.player_pos = (x, y)
                elif char == 'C':
                    self.crates.append((x, y))
                elif char == 'T':
                    self.targets.append((x, y))
                elif char == '#':
                    self.walls.append((x, y))
        
        assert len(self.crates) == len(self.targets), "Mismatch in crate and target count in level design"

        self.crates_on_target_state = [False] * len(self.crates)
        self._update_crate_on_target_status()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        return self._get_observation(), self._get_info()

    def _is_wall(self, pos):
        return pos in self.walls

    def _is_crate(self, pos):
        return pos in self.crates
    
    def _update_crate_on_target_status(self):
        on_target_list = [False] * len(self.crates)
        available_targets = list(self.targets)
        for i, crate_pos in enumerate(self.crates):
            if crate_pos in available_targets:
                on_target_list[i] = True
                available_targets.remove(crate_pos)
        return on_target_list

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        reward = -0.1  # Cost of living/time penalty
        self.steps += 1

        # --- Movement Logic ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            player_x, player_y = self.player_pos
            next_player_pos = (player_x + dx, player_y + dy)

            if not self._is_wall(next_player_pos):
                if self._is_crate(next_player_pos):
                    crate_index = self.crates.index(next_player_pos)
                    next_crate_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                    
                    if not self._is_wall(next_crate_pos) and not self._is_crate(next_crate_pos):
                        self.crates[crate_index] = next_crate_pos
                        self.player_pos = next_player_pos
                        # sfx: push_crate.wav
                else:
                    self.player_pos = next_player_pos
                    # sfx: step.wav

        # --- Reward Calculation ---
        new_on_target_state = self._update_crate_on_target_status()
        
        num_newly_on_target = 0
        for i in range(len(self.crates)):
            if new_on_target_state[i] and not self.crates_on_target_state[i]:
                num_newly_on_target += 1
        
        if num_newly_on_target > 0:
            reward += 10 * num_newly_on_target
            self.score += 10 * num_newly_on_target
            # sfx: crate_on_target.wav

        self.crates_on_target_state = new_on_target_state

        # --- Termination Check ---
        terminated = False
        
        if all(self.crates_on_target_state):
            reward += 50
            self.score += 50
            self.game_over = True
            self.win = True
            terminated = True
            # sfx: win_level.wav

        if self.steps >= self.MAX_STEPS and not self.game_over:
            reward -= 50
            self.score -= 50
            self.game_over = True
            terminated = True
            # sfx: lose_level.wav
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw targets
        for tx, ty in self.targets:
            rect = pygame.Rect(tx * self.CELL_SIZE, ty * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            is_filled = (tx, ty) in self.crates
            color = self.COLOR_TARGET_FILLED if is_filled else self.COLOR_TARGET
            pygame.draw.rect(self.screen, color, rect.inflate(-8, -8), border_radius=4)

        # Draw walls
        for wx, wy in self.walls:
            rect = pygame.Rect(wx * self.CELL_SIZE, wy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL_SHADOW, rect)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect.inflate(-4, -4))

        # Draw crates
        for cx, cy in self.crates:
            rect = pygame.Rect(cx * self.CELL_SIZE, cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CRATE_SHADOW, rect.inflate(-8, -8))
            pygame.draw.rect(self.screen, self.COLOR_CRATE, rect.inflate(-12, -12))
            
        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(px * self.CELL_SIZE, py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect.inflate(-10, -10), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-14, -14), border_radius=3)
    
    def _render_ui(self):
        moves_left = max(0, self.MAX_STEPS - self.steps)
        time_text = f"Moves Left: {moves_left}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        crates_on_target = sum(self.crates_on_target_state)
        crates_text = f"Solved: {crates_on_target}/{len(self.targets)}"
        crates_surf = self.font_main.render(crates_text, True, self.COLOR_TEXT)
        self.screen.blit(crates_surf, (10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            end_text = "PUZZLE SOLVED!" if self.win else "OUT OF MOVES!"
            color = self.COLOR_WIN if self.win else self.COLOR_LOSE
                
            end_surf = self.font_large.render(end_text, True, color)
            text_rect = end_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_surf, text_rect)

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
            "crates_on_target": sum(self.crates_on_target_state),
        }

    def close(self):
        pygame.font.quit()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to interact with the game
    render_mode = "human" # "rgb_array" for headless
    
    if render_mode == 'human':
        pygame.display.set_caption("Sokoban Puzzle")
        screen = pygame.display.set_mode((640, 400))
    
    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()
    terminated = False
    
    # Mapping from Pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    print("\n" + env.game_description)
    print(env.user_guide)

    while not terminated:
        action = [0, 0, 0] # Default no-op action
        
        if render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key in key_to_action:
                        action[0] = key_to_action[event.key]
                    if event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        print("--- Game Reset ---")
                    if event.key == pygame.K_ESCAPE:
                        terminated = True
            
            # For human play, we only step when a key is pressed
            if action[0] == 0:
                # Update display without stepping the game logic
                frame = np.transpose(env._get_observation(), (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                env.clock.tick(30)
                continue

        obs, reward, term, trunc, info = env.step(action)
        terminated = term

        if reward != -0.1: # Print significant reward events
             print(f"Step: {info['steps']}, Reward: {reward:.1f}, Score: {info['score']}, Terminated: {term}")

        if render_mode == 'human':
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30)

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            if render_mode == 'human':
                pygame.time.wait(2000) # Pause for 2 seconds on game over
            obs, info = env.reset()
            terminated = False # Set to False to play again, or True to exit loop
    
    env.close()