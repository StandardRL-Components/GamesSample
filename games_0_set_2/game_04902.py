
# Generated: 2025-08-28T03:22:05.785633
# Source Brief: brief_04902.md
# Brief Index: 4902

        
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
    A block-pushing puzzle game where the player applies a force to the entire grid,
    moving all blocks in a chosen direction until they hit a wall or another block.
    The goal is to move colored blocks onto their matching targets within a move limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys to push all blocks in a direction."

    # Must be a short, user-facing description of the game:
    game_description = "Push colored blocks onto matching targets. Solve the puzzle before you run out of moves."

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("monospace", 22, bold=True)
            self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_msg = pygame.font.Font(None, 54)


        # Game constants
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 8
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (640 - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (400 - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.MAX_EPISODE_STEPS = 1000
        self.MOVE_LIMIT = 50
        self.NUM_BLOCKS = 5

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (50, 50, 70)
        self.COLOR_TEXT = (220, 220, 230)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
        ]
        self.TARGET_ALPHA = 100

        # Game state variables
        self.blocks = []
        self.targets = []
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def _generate_puzzle(self):
        """Generates a solvable puzzle by starting with a solved state and scrambling it."""
        self.blocks.clear()
        self.targets.clear()

        all_positions = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_positions)
        target_positions = all_positions[:self.NUM_BLOCKS]

        for i in range(self.NUM_BLOCKS):
            pos = target_positions[i]
            color = self.BLOCK_COLORS[i]
            self.targets.append({'pos': pos, 'color': color})
            self.blocks.append({'pos': pos, 'color': color})

        num_scramble_moves = self.np_random.integers(10, 21)
        for _ in range(num_scramble_moves):
            direction = self.np_random.integers(1, 5)
            self._apply_push(direction)

        if self._check_win_condition():
             self._generate_puzzle()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_puzzle()
        self.moves_left = self.MOVE_LIMIT
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.score = self._count_blocks_on_target()

        return self._get_observation(), self._get_info()

    def _apply_push(self, direction):
        """
        Applies a push to all blocks in the given direction.
        The iteration order is crucial to handle chain reactions correctly.
        """
        # direction: 1=up, 2=down, 3=left, 4=right
        moved = False
        
        grid = [[None for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)]
        for i, block in enumerate(self.blocks):
            grid[block['pos'][0]][block['pos'][1]] = i

        if direction == 1: # UP
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    if grid[x][y] is not None:
                        block_idx = grid[x][y]
                        ny = y
                        while ny > 0 and grid[x][ny - 1] is None:
                            ny -= 1
                        if ny != y:
                            self.blocks[block_idx]['pos'] = (x, ny)
                            grid[x][y] = None
                            grid[x][ny] = block_idx
                            moved = True
        elif direction == 2: # DOWN
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                for x in range(self.GRID_WIDTH):
                    if grid[x][y] is not None:
                        block_idx = grid[x][y]
                        ny = y
                        while ny < self.GRID_HEIGHT - 1 and grid[x][ny + 1] is None:
                            ny += 1
                        if ny != y:
                            self.blocks[block_idx]['pos'] = (x, ny)
                            grid[x][y] = None
                            grid[x][ny] = block_idx
                            moved = True
        elif direction == 3: # LEFT
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT):
                    if grid[x][y] is not None:
                        block_idx = grid[x][y]
                        nx = x
                        while nx > 0 and grid[nx - 1][y] is None:
                            nx -= 1
                        if nx != x:
                            self.blocks[block_idx]['pos'] = (nx, y)
                            grid[x][y] = None
                            grid[nx][y] = block_idx
                            moved = True
        elif direction == 4: # RIGHT
            for x in range(self.GRID_WIDTH - 1, -1, -1):
                for y in range(self.GRID_HEIGHT):
                    if grid[x][y] is not None:
                        block_idx = grid[x][y]
                        nx = x
                        while nx < self.GRID_WIDTH - 1 and grid[nx + 1][y] is None:
                            nx += 1
                        if nx != x:
                            self.blocks[block_idx]['pos'] = (nx, y)
                            grid[x][y] = None
                            grid[nx][y] = block_idx
                            moved = True
        return moved

    def _count_blocks_on_target(self):
        count = 0
        block_positions = {b['pos']: b['color'] for b in self.blocks}
        for target in self.targets:
            if target['pos'] in block_positions and block_positions[target['pos']] == target['color']:
                count += 1
        return count

    def _check_win_condition(self):
        return self._count_blocks_on_target() == len(self.targets)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = 0
        
        if movement in [1, 2, 3, 4]:
            self._apply_push(movement)
            self.moves_left -= 1
            reward += -0.1  # Per-move penalty

        # State-based reward for blocks on target
        self.score = self._count_blocks_on_target()
        reward += self.score
        
        terminated = False
        if self._check_win_condition():
            self.game_won = True
            self.game_over = True
            terminated = True
            reward += 100
        elif self.moves_left <= 0:
            self.game_over = True
            terminated = True
            reward += -10
        elif self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw targets
        for target in self.targets:
            tx, ty = target['pos']
            rect = pygame.Rect(
                self.GRID_OFFSET_X + tx * self.CELL_SIZE,
                self.GRID_OFFSET_Y + ty * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            target_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            target_surf.fill((*target['color'], self.TARGET_ALPHA))
            self.screen.blit(target_surf, rect.topleft)

        # Draw blocks
        for block in self.blocks:
            bx, by = block['pos']
            rect = pygame.Rect(
                self.GRID_OFFSET_X + bx * self.CELL_SIZE + 3,
                self.GRID_OFFSET_Y + by * self.CELL_SIZE + 3,
                self.CELL_SIZE - 6, self.CELL_SIZE - 6
            )
            pygame.draw.rect(self.screen, block['color'], rect, border_radius=5)
            highlight_color = tuple(min(255, c + 50) for c in block['color'])
            pygame.draw.rect(self.screen, highlight_color, (rect.x + 2, rect.y + 2, rect.width - 4, rect.height - 4), 2, border_radius=4)


    def _render_ui(self):
        moves_text = f"Moves: {max(0, self.moves_left)}"
        text_surface = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (20, 10))

        score_text = f"Score: {self.score}/{self.NUM_BLOCKS}"
        score_surface = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topright=(620, 10))
        self.screen.blit(score_surface, score_rect)

        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "PUZZLE SOLVED!" if self.game_won else "OUT OF MOVES"
            color = (150, 255, 150) if self.game_won else (255, 150, 150)
            msg_surface = self.font_msg.render(msg, True, color)
            msg_rect = msg_surface.get_rect(center=(320, 200))
            self.screen.blit(msg_surface, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "blocks_on_target": self.score,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame Interactive Loop ---
    # This is for human play and visualization, not part of the core Gym env
    
    # Set up the display window
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    running = True
    
    while running:
        action = np.array([0, 0, 0]) # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

                # Map keys to actions for one step
                if not env.game_over:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        # If a move key was pressed, step the environment
        if action[0] != 0 or env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Render the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate
        
    env.close()