
# Generated: 2025-08-27T23:00:04.266618
# Source Brief: brief_03320.md
# Brief Index: 3320

        
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
        "Controls: Use arrow keys to push all blocks simultaneously in a direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push colored blocks to their matching goal zones. You have a limited number of moves and time. Plan your pushes carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_BLOCKS = 5
        self.MAX_MOVES = 50
        self.MAX_TIME = 60 # In steps, since it's turn-based
        self.MAX_EPISODE_STEPS = 1000

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
        self.font_small = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 62)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.COLOR_WIN = (180, 255, 180)
        self.COLOR_LOSE = (255, 180, 180)

        self.BLOCK_COLORS = [
            (255, 90, 90),   # Red
            (90, 200, 90),   # Green
            (90, 150, 255),  # Blue
            (255, 220, 90),  # Yellow
            (200, 90, 255),  # Purple
        ]
        
        # Rendering constants
        self.GRID_AREA_SIZE = self.HEIGHT - 20
        self.CELL_SIZE = self.GRID_AREA_SIZE // self.GRID_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_AREA_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_AREA_SIZE) // 2
        self.BLOCK_SIZE = int(self.CELL_SIZE * 0.8)
        self.BLOCK_OFFSET = (self.CELL_SIZE - self.BLOCK_SIZE) // 2
        
        # Initialize state variables
        self.blocks = []
        self.goals = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_left = 0
        self.time_left = 0
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_left = self.MAX_MOVES
        self.time_left = self.MAX_TIME
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        """Generates a new puzzle layout for blocks and goals."""
        all_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_coords)

        self.goals = []
        self.blocks = []
        
        used_coords = set()

        # Place goals
        for i in range(self.NUM_BLOCKS):
            pos = all_coords.pop()
            self.goals.append({'pos': pos, 'color': self.BLOCK_COLORS[i]})
            used_coords.add(pos)

        # Place blocks, ensuring they don't start on their own goal
        for i in range(self.NUM_BLOCKS):
            goal_pos = self.goals[i]['pos']
            block_pos = None
            
            # Find a position for the block that is not its goal
            for j in range(len(all_coords)):
                if all_coords[j] != goal_pos:
                    block_pos = all_coords.pop(j)
                    break
            # If all remaining spots are this block's goal (highly unlikely), just take the last spot
            if block_pos is None:
                block_pos = all_coords.pop()

            self.blocks.append({'pos': block_pos, 'color': self.BLOCK_COLORS[i]})
            used_coords.add(block_pos)
    
    def step(self, action):
        if self.game_over:
            reward = 0.0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        movement = action[0]  # 0-4: none/up/down/left/right
        
        blocks_on_goal_before = self._count_blocks_on_goals()

        if movement > 0: # 0 is no-op
            self.moves_left -= 1
            reward -= 0.1 # Penalty for each move

            directions = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            self._perform_push(directions[movement])
        
        self.time_left -= 1
        
        blocks_on_goal_after = self._count_blocks_on_goals()
        
        # Reward for newly placed blocks
        reward += (blocks_on_goal_after - blocks_on_goal_before) * 1.0
        
        self.score += reward
        
        terminated, win = self._check_termination()

        if win:
            reward += 50.0
            self.score += 50.0

        if terminated:
            self.game_over = True
            self.win_state = win

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _perform_push(self, direction):
        """Pushes all blocks in the given direction until they hit a wall or another block."""
        dx, dy = direction
        
        # Sort blocks based on push direction to handle collisions correctly
        # E.g., when pushing right, process the rightmost blocks first.
        sort_key = lambda b: b['pos'][0] * dx + b['pos'][1] * dy
        sorted_blocks = sorted(self.blocks, key=sort_key, reverse=True)
        
        all_block_positions = {tuple(b['pos']) for b in self.blocks}

        for block in sorted_blocks:
            current_pos = tuple(block['pos'])
            all_block_positions.remove(current_pos)
            
            new_pos = list(current_pos)
            while True:
                next_pos = (new_pos[0] + dx, new_pos[1] + dy)
                if not (0 <= next_pos[0] < self.GRID_SIZE and 0 <= next_pos[1] < self.GRID_SIZE):
                    break # Hit a wall
                if next_pos in all_block_positions:
                    break # Hit another block
                
                new_pos[0], new_pos[1] = next_pos
            
            block['pos'] = tuple(new_pos)
            all_block_positions.add(tuple(new_pos))

    def _check_termination(self):
        """Check for win/loss conditions."""
        win = self._count_blocks_on_goals() == self.NUM_BLOCKS
        if win:
            return True, True
        if self.moves_left <= 0:
            return True, False
        if self.time_left <= 0:
            return True, False
        if self.steps >= self.MAX_EPISODE_STEPS:
            return True, False
        return False, False

    def _count_blocks_on_goals(self):
        """Counts how many blocks are on their correct goal zone."""
        count = 0
        for i in range(self.NUM_BLOCKS):
            if self.blocks[i]['pos'] == self.goals[i]['pos']:
                count += 1
        return count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_goals()
        self._render_blocks()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_AREA_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_AREA_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_goals(self):
        for goal in self.goals:
            gx, gy = goal['pos']
            color = goal['color']
            
            center_x = self.GRID_OFFSET_X + int((gx + 0.5) * self.CELL_SIZE)
            center_y = self.GRID_OFFSET_Y + int((gy + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.35)
            
            # Use gfxdraw for anti-aliased, transparent circles
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, (*color, 50))
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (*color, 100))

    def _render_blocks(self):
        for i in range(self.NUM_BLOCKS):
            block = self.blocks[i]
            goal = self.goals[i]
            bx, by = block['pos']
            color = block['color']
            
            rect_x = self.GRID_OFFSET_X + bx * self.CELL_SIZE + self.BLOCK_OFFSET
            rect_y = self.GRID_OFFSET_Y + by * self.CELL_SIZE + self.BLOCK_OFFSET
            
            # Draw a border effect
            border_color = tuple(min(255, c + 40) for c in color)
            pygame.draw.rect(self.screen, border_color, (rect_x, rect_y, self.BLOCK_SIZE, self.BLOCK_SIZE), 0, 4)
            
            inner_size = self.BLOCK_SIZE - 6
            inner_offset = (self.BLOCK_SIZE - inner_size) // 2
            pygame.draw.rect(self.screen, color, (rect_x + inner_offset, rect_y + inner_offset, inner_size, inner_size), 0, 3)

            # Visual feedback for solved blocks
            if block['pos'] == goal['pos']:
                center_x = rect_x + self.BLOCK_SIZE // 2
                center_y = rect_y + self.BLOCK_SIZE // 2
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 5, (255, 255, 255, 200))
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 5, (255, 255, 255))

    def _render_ui(self):
        # Draw UI text
        moves_text = self.font_small.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 15))

        time_text = self.font_small.render(f"Time: {self.time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 15))

        # Game over overlay
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.win_state:
                msg_text = self.font_large.render("SOLVED!", True, self.COLOR_WIN)
            else:
                msg_text = self.font_large.render("GAME OVER", True, self.COLOR_LOSE)
            
            text_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "time_left": self.time_left,
            "blocks_on_goal": self._count_blocks_on_goals()
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
        # Temporarily generate a puzzle to get an observation
        self.reset()
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Use a dummy window to display the game
    pygame.display.set_caption("Push Block Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    terminated = False
    
    print(env.user_guide)
    
    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not terminated:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                    print("--- Game Reset ---")
                
                if action[0] != 0:
                    obs, reward, terminated, _, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
                    if terminated:
                        print("--- Episode Finished --- (Press 'r' to reset)")

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for manual play
        
    env.close()