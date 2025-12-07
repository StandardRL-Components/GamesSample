
# Generated: 2025-08-27T20:07:18.552665
# Source Brief: brief_02354.md
# Brief Index: 2354

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character (the white square). "
        "Push colored blocks onto the matching goal circles. You have a limited number of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist block-pushing puzzle game. Strategically move all blocks to their "
        "designated goals within the move limit to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    NUM_BLOCKS = 10
    MAX_MOVES = 50
    MAX_STEPS = 1000 # Fallback termination

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_GOAL_OFF = (50, 50, 60)
    COLOR_GOAL_ON = (255, 220, 100)
    BLOCK_COLORS = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (255, 150, 50),  (50, 255, 150),  (150, 50, 255),
        (200, 200, 200)
    ]

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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.blocks_pos = None
        self.goals_pos = None
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_moved_block_info = None # (index, old_pos, new_pos)

        # Ensure implementation is valid
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.last_moved_block_info = None

        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        move_made = False
        self.last_moved_block_info = None

        old_blocks_on_goals = self._count_blocks_on_goals()

        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            
            player_x, player_y = self.player_pos
            target_x, target_y = player_x + dx, player_y + dy

            # Check if target is within bounds
            if 0 <= target_x < self.GRID_WIDTH and 0 <= target_y < self.GRID_HEIGHT:
                # Check if target is a block
                block_idx = self._get_block_at((target_x, target_y))

                if block_idx is not None:
                    # Attempt to push the block
                    block_target_x, block_target_y = target_x + dx, target_y + dy
                    
                    # Check if block target is valid (in bounds and not another block)
                    if (0 <= block_target_x < self.GRID_WIDTH and
                        0 <= block_target_y < self.GRID_HEIGHT and
                        self._get_block_at((block_target_x, block_target_y)) is None):
                        
                        old_block_pos = self.blocks_pos[block_idx]
                        self.blocks_pos[block_idx] = (block_target_x, block_target_y)
                        self.player_pos = (target_x, target_y)
                        self.last_moved_block_info = {
                            'index': block_idx, 
                            'pos': (block_target_x, block_target_y)
                        }
                        move_made = True
                        # SFX: Block push sound
                else:
                    # Move player into empty space
                    self.player_pos = (target_x, target_y)
                    move_made = True
                    # SFX: Player step sound

        if move_made:
            self.moves_left -= 1
            reward -= 0.1 # Cost of moving

            new_blocks_on_goals = self._count_blocks_on_goals()
            if new_blocks_on_goals > old_blocks_on_goals:
                reward += 1.0 # Reward for placing a block on a goal
                # SFX: Goal success chime

        self.steps += 1
        self.score += reward
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self._count_blocks_on_goals() == self.NUM_BLOCKS:
                terminal_reward = 100.0 # Big reward for winning
                self.score += terminal_reward
                reward += terminal_reward
                # SFX: Level complete fanfare
            else:
                terminal_reward = -50.0 # Penalty for losing
                self.score += terminal_reward
                reward += terminal_reward
                # SFX: Game over failure sound
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        """Generates a solvable level by starting from a solved state and working backwards."""
        all_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_cells)

        # Place goals
        self.goals_pos = all_cells[:self.NUM_BLOCKS]
        
        # Place blocks on goals (solved state)
        self.blocks_pos = list(self.goals_pos)

        # Scramble the puzzle with reverse moves
        scramble_moves = self.np_random.integers(20, 40)
        for _ in range(scramble_moves):
            block_idx = self.np_random.integers(0, self.NUM_BLOCKS)
            direction = self.np_random.integers(0, 4)
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]

            block_pos = self.blocks_pos[block_idx]
            pull_from_pos = (block_pos[0] + dx, block_pos[1] + dy)
            pull_to_pos = (block_pos[0] - dx, block_pos[1] - dy)

            # Check if reverse move is valid
            if (0 <= pull_to_pos[0] < self.GRID_WIDTH and
                0 <= pull_to_pos[1] < self.GRID_HEIGHT and
                pull_to_pos not in self.blocks_pos and
                0 <= pull_from_pos[0] < self.GRID_WIDTH and
                0 <= pull_from_pos[1] < self.GRID_HEIGHT and
                pull_from_pos not in self.blocks_pos):
                
                self.blocks_pos[block_idx] = pull_to_pos

        # Place player in a random empty spot
        occupied_pos = set(self.blocks_pos + self.goals_pos)
        empty_cells = [cell for cell in all_cells if cell not in occupied_pos]
        if not empty_cells: # Should be rare, but handle it
             empty_cells = [cell for cell in all_cells if cell not in self.blocks_pos]

        self.player_pos = self.np_random.choice(empty_cells) if empty_cells else (0,0)
        
        # Ensure player is a tuple of ints
        self.player_pos = (int(self.player_pos[0]), int(self.player_pos[1]))


    def _get_block_at(self, pos):
        try:
            return self.blocks_pos.index(pos)
        except ValueError:
            return None

    def _check_termination(self):
        win_condition = self._count_blocks_on_goals() == self.NUM_BLOCKS
        lose_condition = self.moves_left <= 0
        step_limit_reached = self.steps >= self.MAX_STEPS
        return win_condition or lose_condition or step_limit_reached

    def _count_blocks_on_goals(self):
        on_goal_count = 0
        goal_set = set(self.goals_pos)
        for block_pos in self.blocks_pos:
            if block_pos in goal_set:
                on_goal_count += 1
        return on_goal_count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py), 1)

        # Draw goals
        goal_set = set(self.goals_pos)
        block_set = set(self.blocks_pos)
        for i, pos in enumerate(self.goals_pos):
            px, py = int((pos[0] + 0.5) * self.CELL_SIZE), int((pos[1] + 0.5) * self.CELL_SIZE)
            radius = self.CELL_SIZE // 3
            
            is_occupied = pos in block_set
            color = self.COLOR_GOAL_ON if is_occupied else self.COLOR_GOAL_OFF
            
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)

        # Draw blocks
        for i, pos in enumerate(self.blocks_pos):
            px, py = pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE
            color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
            
            block_rect = pygame.Rect(px + 4, py + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            pygame.draw.rect(self.screen, color, block_rect, border_radius=4)
            
            # Highlight effect for recently moved block
            if self.last_moved_block_info and self.last_moved_block_info['index'] == i:
                glow_color = (255, 255, 255, 100) # White with alpha
                glow_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, glow_color, (self.CELL_SIZE//2, self.CELL_SIZE//2), self.CELL_SIZE//2 - 2)
                self.screen.blit(glow_surface, (px, py))

        # Draw player
        px, py = self.player_pos[0] * self.CELL_SIZE, self.player_pos[1] * self.CELL_SIZE
        player_rect = pygame.Rect(px + 6, py + 6, self.CELL_SIZE - 12, self.CELL_SIZE - 12)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_BG, player_rect, width=2, border_radius=3)

    def _render_ui(self):
        # Render moves left
        moves_text = f"Moves: {max(0, self.moves_left)}"
        text_surface = self.font.render(moves_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 5))
        
        # Render score
        score_text = f"Score: {int(self.score)}"
        text_surface_score = self.font.render(score_text, True, (255, 255, 255))
        score_rect = text_surface_score.get_rect(topright=(self.SCREEN_WIDTH - 10, 5))
        self.screen.blit(text_surface_score, score_rect)

        # Render win/loss message
        if self.game_over:
            is_win = self._count_blocks_on_goals() == self.NUM_BLOCKS
            message = "LEVEL COMPLETE!" if is_win else "OUT OF MOVES"
            color = self.COLOR_GOAL_ON if is_win else (255, 80, 80)
            
            msg_font = pygame.font.SysFont("monospace", 48, bold=True)
            msg_surface = msg_font.render(message, True, color)
            msg_rect = msg_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Add a background for readability
            bg_rect = msg_rect.inflate(40, 20)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surface.fill((20, 20, 30, 200))
            self.screen.blit(bg_surface, bg_rect.topleft)
            
            self.screen.blit(msg_surface, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "blocks_on_goals": self._count_blocks_on_goals()
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
        
        # Test reset to initialize state for observation
        self.reset()
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset return values
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

# This block allows running the environment directly for testing
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless

    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:")
    print(f"Info: {info}")

    # Test a few random steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n--- Step {i+1} ---")
        print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}")
        print(f"Info: {info}")
        if terminated:
            print("Episode finished.")
            break
    
    # Save a sample frame
    try:
        from PIL import Image
        img = Image.fromarray(obs)
        img.save("game_frame.png")
        print("\nSaved a sample frame to game_frame.png")
    except ImportError:
        print("\nPIL/Pillow not installed, skipping frame save.")

    env.close()