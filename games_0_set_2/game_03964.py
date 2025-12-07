
# Generated: 2025-08-28T00:59:16.410559
# Source Brief: brief_03964.md
# Brief Index: 3964

        
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


class Block:
    """Helper class to store block state."""
    def __init__(self, x, y, color, target_x, target_y):
        self.x = x
        self.y = y
        self.color = color
        self.target_x = target_x
        self.target_y = target_y

    def is_on_target(self):
        return self.x == self.target_x and self.y == self.target_y

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to push all movable blocks in a direction. "
        "Solve the puzzle by moving each block to its matching target."
    )

    game_description = (
        "A minimalist block-pushing puzzle. Plan your moves carefully to slide all "
        "colored blocks onto their matching targets before you run out of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 8
        self.CELL_SIZE = 40
        self.NUM_BLOCKS = 4
        self.SCRAMBLE_MOVES = 20
        self.MAX_MOVES = 30

        # Colors
        self.COLOR_BG = (35, 40, 50)
        self.COLOR_GRID = (50, 55, 65)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_WIN_TEXT = (163, 190, 140)
        self.COLOR_LOSE_TEXT = (191, 97, 106)
        self.BLOCK_COLORS = {
            "red": ((216, 126, 132), (191, 97, 106)),
            "green": ((180, 210, 155), (163, 190, 140)),
            "blue": ((145, 185, 221), (129, 161, 193)),
            "yellow": ((235, 203, 139), (216, 185, 119)),
        }

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # Grid positioning
        self.grid_pixel_width = self.GRID_WIDTH * self.CELL_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.CELL_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2
        
        # Game State
        self.blocks = []
        self.moves_left = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        
        # Initialize state variables
        self.reset()
        
        # Self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        """Generates a new, solvable puzzle by starting from a solved state and scrambling it."""
        while True:
            all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
            self.np_random.shuffle(all_coords)
            
            target_coords = all_coords[:self.NUM_BLOCKS]
            
            self.blocks = []
            colors = list(self.BLOCK_COLORS.keys())
            for i in range(self.NUM_BLOCKS):
                tx, ty = target_coords[i]
                self.blocks.append(Block(tx, ty, colors[i], tx, ty))

            # Scramble the puzzle with random pushes
            for _ in range(self.np_random.integers(self.SCRAMBLE_MOVES // 2, self.SCRAMBLE_MOVES)):
                # 1=up, 2=down, 3=left, 4=right
                random_push = self.np_random.integers(1, 5)
                self._apply_push(random_push)

            # Ensure the puzzle is not already solved
            if not all(b.is_on_target() for b in self.blocks):
                break
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        movement = action[0]
        reward = 0
        
        blocks_on_target_before = sum(1 for b in self.blocks if b.is_on_target())

        if movement in [1, 2, 3, 4]:  # A push was made
            self.moves_left -= 1
            reward -= 0.1  # Cost for making a move

            # Sound effect placeholder
            # play_sound('push')
            
            self._apply_push(movement)
            
            blocks_on_target_after = sum(1 for b in self.blocks if b.is_on_target())
            
            newly_on_target = blocks_on_target_after - blocks_on_target_before
            if newly_on_target > 0:
                reward += newly_on_target * 1.0
                # Sound effect placeholder
                # play_sound('target_lock')

        terminated = self._check_termination()

        if terminated:
            if self.win:
                reward += 100
                # Sound effect placeholder
                # play_sound('win_jingle')
            else:  # Lost by running out of moves
                reward += -10
                # Sound effect placeholder
                # play_sound('lose_buzzer')
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_push(self, direction):
        """Applies a push to all blocks on the grid in the given direction."""
        # 1=up, 2=down, 3=left, 4=right
        if direction == 1: dx, dy, x_range, y_range = 0, -1, range(self.GRID_WIDTH), range(self.GRID_HEIGHT)
        elif direction == 2: dx, dy, x_range, y_range = 0, 1, range(self.GRID_WIDTH), reversed(range(self.GRID_HEIGHT))
        elif direction == 3: dx, dy, x_range, y_range = -1, 0, range(self.GRID_WIDTH), range(self.GRID_HEIGHT)
        else: dx, dy, x_range, y_range = 1, 0, reversed(range(self.GRID_WIDTH)), range(self.GRID_HEIGHT)
        
        moved_blocks_in_step = set()
        
        block_map = {(b.x, b.y): b for b in self.blocks}

        for y in y_range:
            for x in x_range:
                if (x, y) in block_map and (x, y) not in moved_blocks_in_step:
                    chain = []
                    curr_x, curr_y = x, y
                    
                    # Find all blocks in the push chain
                    while (curr_x, curr_y) in block_map:
                        chain.append(block_map[(curr_x, curr_y)])
                        curr_x += dx
                        curr_y += dy
                    
                    # Check if the chain can move
                    can_move = (0 <= curr_x < self.GRID_WIDTH and 0 <= curr_y < self.GRID_HEIGHT)
                    
                    if can_move:
                        # Move all blocks in the chain
                        for block in reversed(chain):
                            block.x += dx
                            block.y += dy
                            moved_blocks_in_step.add((block.x - dx, block.y - dy))

    def _check_termination(self):
        all_on_target = all(b.is_on_target() for b in self.blocks)
        if all_on_target:
            self.game_over = True
            self.win = True
            return True
        
        if self.moves_left <= 0:
            self.game_over = True
            self.win = False
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.grid_offset_x + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.grid_offset_y), (px, self.grid_offset_y + self.grid_pixel_height))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.grid_offset_y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, py), (self.grid_offset_x + self.grid_pixel_width, py))

        # Draw targets
        for block in self.blocks:
            px = self.grid_offset_x + block.target_x * self.CELL_SIZE + self.CELL_SIZE // 2
            py = self.grid_offset_y + block.target_y * self.CELL_SIZE + self.CELL_SIZE // 2
            radius = self.CELL_SIZE // 4
            target_color = self.BLOCK_COLORS[block.color][1]
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(radius), target_color)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), int(radius), target_color)

        # Draw blocks
        for block in self.blocks:
            px = self.grid_offset_x + block.x * self.CELL_SIZE
            py = self.grid_offset_y + block.y * self.CELL_SIZE
            
            inset = 3
            main_color, shadow_color = self.BLOCK_COLORS[block.color]
            
            block_rect = pygame.Rect(px + inset, py + inset, self.CELL_SIZE - 2 * inset, self.CELL_SIZE - 2 * inset)
            
            # Draw a subtle shadow/border
            shadow_rect = block_rect.copy()
            shadow_rect.width += 2
            shadow_rect.height += 2
            shadow_rect.center = block_rect.center
            pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_radius=4)

            # Draw the main block face
            pygame.draw.rect(self.screen, main_color, block_rect, border_radius=4)

    def _render_ui(self):
        # Render moves left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (15, 15))
        
        # Render score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 15))
        self.screen.blit(score_text, score_rect)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                message = "PUZZLE SOLVED!"
                color = self.COLOR_WIN_TEXT
            else:
                message = "OUT OF MOVES"
                color = self.COLOR_LOSE_TEXT
            
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "is_win": self.win if self.game_over else False
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Block Pusher")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
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
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_q: # Quit on 'q' key
                    running = False
        
        # Only step if an action was taken
        if action[0] != 0 and not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")

        # Draw the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate

    env.close()