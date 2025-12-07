
# Generated: 2025-08-28T03:49:30.065650
# Source Brief: brief_05052.md
# Brief Index: 5052

        
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
        "Controls: Use arrow keys to push the blue block. Guide the red block to the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based puzzle game. Slide blocks around a grid to guide a target block to the exit within a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_MOVES = 20
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("impact", 60)
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_EXIT = (60, 220, 120)
        self.COLOR_EXIT_BORDER = (40, 180, 90)
        self.COLOR_PLAYER = (80, 150, 255)
        self.COLOR_PLAYER_BORDER = (50, 100, 200)
        self.COLOR_TARGET = (255, 100, 100)
        self.COLOR_TARGET_BORDER = (200, 60, 60)
        self.OBSTACLE_COLORS = [
            ((160, 170, 180), (130, 140, 150)),
            ((140, 150, 160), (110, 120, 130)),
            ((120, 130, 140), (90, 100, 110)),
        ]
        
        # Calculate grid rendering properties
        self.CELL_SIZE = 36
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2
        
        # Initialize state variables
        self.grid = None
        self.player_pos = None
        self.target_pos = None
        self.exit_pos = None
        self.obstacle_pos_map = None
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        
        self._generate_solvable_puzzle()
        self.initial_target_dist = self._get_target_dist()
        
        return self._get_observation(), self._get_info()
    
    def _generate_solvable_puzzle(self):
        # Generate a puzzle by starting with a solved state and applying random valid moves.
        # This guarantees a solution exists within the scramble depth.
        
        # 1. Create a solved state
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        # Place exit (not on edge)
        self.exit_pos = (
            self.np_random.integers(1, self.GRID_SIZE - 1),
            self.np_random.integers(1, self.GRID_SIZE - 1)
        )
        
        # Place target on exit
        self.target_pos = self.exit_pos
        self.grid[self.target_pos[1], self.target_pos[0]] = 2 # 2: Target
        
        # Place player, ensuring it's not on the target
        while True:
            pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
            if self.grid[pos[1], pos[0]] == 0:
                self.player_pos = pos
                self.grid[self.player_pos[1], self.player_pos[0]] = 1 # 1: Player
                break
                
        # Place obstacles
        self.obstacle_pos_map = {}
        num_obstacles = self.np_random.integers(5, 11)
        for i in range(num_obstacles):
            while True:
                pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
                if self.grid[pos[1], pos[0]] == 0:
                    obstacle_id = 3 + i
                    self.grid[pos[1], pos[0]] = obstacle_id
                    self.obstacle_pos_map[obstacle_id] = pos
                    break
        
        # 2. Scramble the puzzle with random moves
        scramble_depth = self.np_random.integers(15, self.MAX_MOVES)
        for _ in range(scramble_depth):
            # Try random moves until one is successful
            for _ in range(10): # Max 10 attempts to find a valid move
                move_dir = self.np_random.integers(1, 5) # 1-4 for directions
                if self._apply_push(move_dir):
                    break # sfx: block_slide_quiet during setup

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        reward = 0
        terminated = False
        
        if movement != 0: # Action is a push
            self.steps += 1
            reward -= 0.1 # Cost per move
            self.moves_left -= 1
            
            prev_dist = self._get_target_dist()
            
            # sfx: block_slide
            move_successful = self._apply_push(movement)
            
            new_dist = self._get_target_dist()
            
            if new_dist < prev_dist:
                reward += 5.0 # Reward for moving target closer
            elif new_dist > prev_dist:
                reward -= 1.0 # Penalty for moving target further
        
        # Check for termination conditions
        if self.target_pos == self.exit_pos:
            # sfx: win_jingle
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0:
            # sfx: lose_buzzer
            reward -= 100.0
            terminated = True
            self.game_over = True
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_push(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[movement]
        
        px, py = self.player_pos
        
        # Find the line of blocks to push
        line_to_push = []
        nx, ny = px, py
        
        while True:
            nx, ny = nx + dx, ny + dy
            if not (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE):
                return False # Hit wall, cannot push
            
            block_id = self.grid[ny, nx]
            if block_id == 0:
                break # Found empty space, can push
            
            line_to_push.append((nx, ny, block_id))
        
        # Move all blocks in the line, starting from the back
        if not line_to_push and not (0 <= px + dx < self.GRID_SIZE and 0 <= py + dy < self.GRID_SIZE):
             return False # Player pushing against a wall

        for x, y, block_id in reversed(line_to_push):
            self.grid[y + dy, x + dx] = block_id
            if block_id == 2: self.target_pos = (x + dx, y + dy)
            elif block_id in self.obstacle_pos_map: self.obstacle_pos_map[block_id] = (x + dx, y + dy)

        # Move player
        self.grid[py, px] = 0
        self.grid[py + dy, px + dx] = 1
        self.player_pos = (px + dx, py + dy)
        
        return True

    def _get_target_dist(self):
        if self.target_pos is None or self.exit_pos is None:
            return 0
        return abs(self.target_pos[0] - self.exit_pos[0]) + abs(self.target_pos[1] - self.exit_pos[1])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            self.GRID_X_OFFSET + ex * self.CELL_SIZE,
            self.GRID_Y_OFFSET + ey * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT_BORDER, exit_rect)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect.inflate(-4, -4))

        # Draw Grid Lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID,
                (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET),
                (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID,
                (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE),
                (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + i * self.CELL_SIZE))

        # Draw Blocks
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                block_id = self.grid[y, x]
                if block_id != 0:
                    rect = pygame.Rect(
                        self.GRID_X_OFFSET + x * self.CELL_SIZE,
                        self.GRID_Y_OFFSET + y * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    
                    if block_id == 1: # Player
                        color, border_color = self.COLOR_PLAYER, self.COLOR_PLAYER_BORDER
                    elif block_id == 2: # Target
                        color, border_color = self.COLOR_TARGET, self.COLOR_TARGET_BORDER
                    else: # Obstacles
                        color_idx = (block_id - 3) % len(self.OBSTACLE_COLORS)
                        color, border_color = self.OBSTACLE_COLORS[color_idx]
                    
                    pygame.draw.rect(self.screen, border_color, rect)
                    pygame.draw.rect(self.screen, color, rect.inflate(-6, -6))

    def _render_ui(self):
        # Render Moves Left
        moves_text = self.font_ui.render(f"Moves Left: {self.moves_left}", True, (200, 200, 220))
        self.screen.blit(moves_text, (20, 20))
        
        # Render Score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, (200, 200, 220))
        score_rect = score_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Render Game Over screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.target_pos == self.exit_pos:
                msg = "YOU WIN!"
                color = self.COLOR_EXIT
            else:
                msg = "OUT OF MOVES"
                color = self.COLOR_TARGET
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "target_dist": self._get_target_dist(),
        }

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Playable Demo ---
    # This requires a window to be created.
    # To run, change `self.screen` to `pygame.display.set_mode(...)`
    # and add a `pygame.display.flip()` in the main loop.
    
    # For headless verification, we just check if the environment runs.
    print("Running a sample episode...")
    for i in range(env.MAX_MOVES + 5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished.")
            break
    
    env.close()
    print("Sample episode finished without errors.")