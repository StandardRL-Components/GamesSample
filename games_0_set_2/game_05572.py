import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to push adjacent blocks. Your goal is to move the red block to the green tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic block-pushing puzzle. Navigate a red block through a maze by pushing other blocks. "
        "Each push costs a move. Solve the puzzle before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.TILE_SIZE = 40
        self.SHUFFLE_STEPS = 8
        self.NUM_BLUE_BLOCKS = 5

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_WALL = (60, 60, 70)
        self.COLOR_WALL_ACCENT = (80, 80, 90)
        self.COLOR_GOAL = (50, 150, 50)
        self.COLOR_GOAL_ACCENT = (100, 200, 100)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_ACCENT = (255, 120, 120)
        self.COLOR_BLOCK = (50, 100, 255)
        self.COLOR_BLOCK_ACCENT = (120, 170, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_WIN = (100, 255, 100)
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
        self.font_main = pygame.font.Font(None, 32)
        self.font_big = pygame.font.Font(None, 72)
        
        # --- Game State (initialized in reset) ---
        self.grid = None
        self.player_pos = None
        self.goal_pos = None
        self.moves_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.last_moved_blocks = [] # For rendering after-images: [(old_pos, color), ...]

        self.direction_map = {
            1: (0, -1),  # Up
            2: (0, 1),   # Down
            3: (-1, 0),  # Left
            4: (1, 0),   # Right
        }
        
        # --- Final setup ---
        # self.reset() is called by validate_implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.last_moved_blocks = []
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.last_moved_blocks.clear()
        
        movement = action[0]
        reward = 0.0
        terminated = False

        if movement != 0: # An action was taken
            self.moves_remaining -= 1
            # Small penalty for taking a turn
            reward -= 0.1 

            push_successful, push_reward, moved_blocks = self._attempt_push(self.direction_map[movement])
            
            if push_successful:
                reward += push_reward
                for old_pos, _, color in moved_blocks:
                    self.last_moved_blocks.append((old_pos, color))
        
        self.score += reward

        # Check for termination conditions
        if self.player_pos == self.goal_pos:
            win_reward = 100.0
            self.score += win_reward
            reward += win_reward
            terminated = True
            self.game_over = True
            self.win_message = "YOU WIN!"
        elif self.moves_remaining <= 0:
            lose_penalty = -100.0
            self.score += lose_penalty
            reward += lose_penalty
            terminated = True
            self.game_over = True
            self.win_message = "OUT OF MOVES"

        # Cap max steps
        if self.steps >= 1000:
            terminated = True
            self.game_over = True
            if not self.win_message:
                 self.win_message = "TIME LIMIT"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _attempt_push(self, direction):
        px, py = self.player_pos
        dx, dy = direction
        target_x, target_y = px + dx, py + dy

        if not (0 <= target_x < self.GRID_W and 0 <= target_y < self.GRID_H) or \
           self.grid[target_y, target_x] not in [2, 3]:  # 2=blue_block, 3=player_block
            return False, 0.0, []

        chain = []
        curr_x, curr_y = target_x, target_y
        while 0 <= curr_x < self.GRID_W and 0 <= curr_y < self.GRID_H:
            tile_val = self.grid[curr_y, curr_x]
            if tile_val in [2, 3]:
                chain.append((curr_x, curr_y))
                curr_x += dx
                curr_y += dy
            else:
                break
        
        if not (0 <= curr_x < self.GRID_W and 0 <= curr_y < self.GRID_H) or \
           self.grid[curr_y, curr_x] == 1:  # Hit boundary or a wall
            return False, 0.0, []

        moved_blocks_info = []
        old_player_dist = self._manhattan_distance(self.player_pos, self.goal_pos)
        original_player_pos = self.player_pos

        for bx, by in reversed(chain):
            block_type = self.grid[by, bx]
            new_x, new_y = bx + dx, by + dy
            
            self.grid[new_y, new_x] = block_type
            
            # When a block moves, restore the tile underneath it.
            # If it was the goal, restore the goal tile (4), otherwise it's empty (0).
            if (bx, by) == self.goal_pos:
                 self.grid[by, bx] = 4
            else:
                 self.grid[by, bx] = 0

            if block_type == 3:
                self.player_pos = (new_x, new_y)
            
            color = self.COLOR_PLAYER if block_type == 3 else self.COLOR_BLOCK
            moved_blocks_info.append(((bx, by), (new_x, new_y), color))

        new_player_dist = self._manhattan_distance(self.player_pos, self.goal_pos)
        reward = float(old_player_dist - new_player_dist)

        if self.player_pos != original_player_pos:
            goal_dx = np.sign(self.goal_pos[0] - original_player_pos[0])
            goal_dy = np.sign(self.goal_pos[1] - original_player_pos[1])
            if (dx != 0 and dx == goal_dx) or (dy != 0 and dy == goal_dy):
                 reward += 5.0
                 
        return True, reward, moved_blocks_info

    def _generate_puzzle(self):
        while True:
            self.grid = np.zeros((self.GRID_H, self.GRID_W), dtype=np.int8)
            self.grid[0, :] = self.grid[-1, :] = self.grid[:, 0] = self.grid[:, -1] = 1 # Walls

            # FIX: Standardize coordinate system to (x, y) for all positions.
            # np.where returns (rows, cols), so we zip (cols, rows) to get (x, y).
            rows, cols = np.where(self.grid == 0)
            valid_pos = list(zip(cols, rows))
            
            # Place goal
            goal_idx = self.np_random.integers(len(valid_pos))
            self.goal_pos = valid_pos.pop(goal_idx) # self.goal_pos is now (x, y)
            self.grid[self.goal_pos[1], self.goal_pos[0]] = 4 # Goal tile (access grid as [y, x])

            # Place player on goal initially
            self.player_pos = self.goal_pos # self.player_pos is (x, y)
            self.grid[self.player_pos[1], self.player_pos[0]] = 3 # Player tile overwrites goal tile

            # Place blue blocks
            for _ in range(self.NUM_BLUE_BLOCKS):
                if not valid_pos: break
                block_idx = self.np_random.integers(len(valid_pos))
                bx, by = valid_pos.pop(block_idx) # Unpack as (x, y)
                self.grid[by, bx] = 2 # Blue block (access grid as [y, x])

            # Shuffle by reverse pushing (pulling)
            for _ in range(self.SHUFFLE_STEPS):
                # FIX: Use consistent (x, y) coordinates for movable blocks
                rows, cols = np.where((self.grid == 2) | (self.grid == 3))
                movable_blocks = list(zip(cols, rows))
                if not movable_blocks: break
                
                block_idx = self.np_random.integers(len(movable_blocks))
                # FIX: Unpack as (x, y)
                bx, by = movable_blocks[block_idx]
                
                pull_dirs = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = bx + dx, by + dy
                    if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.grid[ny, nx] == 0:
                        pull_dirs.append((dx, dy))
                
                if pull_dirs:
                    pdx, pdy = pull_dirs[self.np_random.integers(len(pull_dirs))]
                    block_type = self.grid[by, bx]
                    self.grid[by + pdy, bx + pdx] = block_type
                    self.grid[by, bx] = 0
                    if block_type == 3:
                        self.player_pos = (bx + pdx, by + pdy)

            if self.player_pos != self.goal_pos:
                break # Puzzle is valid
        
        self.moves_remaining = int(self.SHUFFLE_STEPS * 1.5) + self.NUM_BLUE_BLOCKS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_block(self, pos, color, accent_color):
        x, y = pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE
        inset = self.TILE_SIZE // 8
        pygame.draw.rect(self.screen, color, (x, y, self.TILE_SIZE, self.TILE_SIZE))
        pygame.draw.rect(self.screen, accent_color, (x + inset, y + inset, self.TILE_SIZE - 2 * inset, self.TILE_SIZE - 2 * inset))

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw after-images for moved blocks
        for pos, color in self.last_moved_blocks:
            x, y = pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE
            ghost_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
            ghost_surf.fill((*color, 80))
            self.screen.blit(ghost_surf, (x, y))

        # Draw grid elements
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                tile = self.grid[y, x]
                if tile == 1: # Wall
                    self._render_block((x, y), self.COLOR_WALL, self.COLOR_WALL_ACCENT)
                elif tile == 2: # Blue block
                    self._render_block((x, y), self.COLOR_BLOCK, self.COLOR_BLOCK_ACCENT)
                elif tile == 3: # Player block
                    self._render_block((x, y), self.COLOR_PLAYER, self.COLOR_PLAYER_ACCENT)
                elif tile == 4: # Goal
                    self._render_block((x, y), self.COLOR_GOAL, self.COLOR_GOAL_ACCENT)
    
    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 10))

        # Score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(score_text, score_rect)
        
        # Game over message
        if self.game_over:
            color = self.COLOR_WIN if self.player_pos == self.goal_pos else self.COLOR_LOSE
            end_text = self.font_big.render(self.win_message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = end_rect.inflate(40, 40)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect.topleft)
            pygame.draw.rect(s, (255,255,255), s.get_rect(), 2, border_radius=5)
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "player_pos": self.player_pos,
            "goal_pos": self.goal_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Beginning implementation validation...")
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        import pygame
        env = GameEnv()
        obs, info = env.reset()
        
        pygame.display.set_caption("Block Pusher")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        
        running = True
        terminated = False
        clock = pygame.time.Clock()
        
        print("\n--- Manual Control ---")
        print("Arrow Keys: Push blocks")
        print("R: Reset level")
        print("Q: Quit")
        print(env.user_guide)
        print("----------------------\n")
        
        while running:
            action_to_take = None
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and not terminated:
                    if event.key == pygame.K_UP:
                        action_to_take = np.array([1, 0, 0])
                    elif event.key == pygame.K_DOWN:
                        action_to_take = np.array([2, 0, 0])
                    elif event.key == pygame.K_LEFT:
                        action_to_take = np.array([3, 0, 0])
                    elif event.key == pygame.K_RIGHT:
                        action_to_take = np.array([4, 0, 0])
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: # Reset
                        obs, info = env.reset()
                        terminated = False
                        print("\n--- Level Reset ---")
                    elif event.key == pygame.K_q:
                        running = False

            if action_to_take is not None:
                obs, reward, terminated, _, info = env.step(action_to_take)
                print(f"Action: {action_to_take[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit frame rate

        env.close()

    except pygame.error as e:
        print(f"\nPygame display could not be initialized: {e}")
        print("This is expected in a headless environment. The code is likely correct.")
        print("Running a simple step test without rendering.")
        env = GameEnv()
        env.reset()
        env.step(env.action_space.sample())
        env.close()
        print("Headless test passed.")