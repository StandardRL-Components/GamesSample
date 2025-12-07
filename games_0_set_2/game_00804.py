
# Generated: 2025-08-27T14:49:38.235115
# Source Brief: brief_00804.md
# Brief Index: 804

        
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


# Set a dummy video driver to run pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character and push the boxes. "
        "The goal is to move both boxes onto the green target squares."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic block-pushing puzzle. Strategically move the player to push both brown boxes "
        "onto the green targets within the move limit. Each move costs points, but solving the puzzle gives a large bonus."
    )

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

        # Visuals
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (45, 50, 56)
        self.COLOR_WALL = (68, 78, 89)
        self.COLOR_PLAYER = (230, 80, 80)
        self.COLOR_BOX = (166, 124, 82)
        self.COLOR_BOX_ON_TARGET = (186, 144, 102)
        self.COLOR_TARGET = (80, 180, 120)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_WIN = (100, 220, 150)
        self.COLOR_LOSE = (220, 100, 100)

        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Game Grid
        self.GRID_COLS = 16
        self.GRID_ROWS = 10
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (640 - self.GRID_COLS * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (400 - self.GRID_ROWS * self.CELL_SIZE) // 2

        # Level layout
        self.level_layout = [
            "WWWWWWWWWWWWWWWW",
            "W              W",
            "W P  B         W",
            "W      WWWW    W",
            "W   B  W  T W  W",
            "W      W  T W  W",
            "W      WWWW    W",
            "W              W",
            "W              W",
            "WWWWWWWWWWWWWWWW",
        ]
        
        # State variables (will be initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.moves_left = 0
        self.player_pos = (0, 0)
        self.box_positions = []
        self.target_positions = []
        self.wall_positions = set()
        self.boxes_on_target_indices = set()

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.moves_left = 100 # Generous move limit for this level
        
        self.box_positions = []
        self.target_positions = []
        self.wall_positions = set()

        for r, row_str in enumerate(self.level_layout):
            for c, char in enumerate(row_str):
                pos = (c, r)
                if char == 'P':
                    self.player_pos = pos
                elif char == 'B':
                    self.box_positions.append(pos)
                elif char == 'T':
                    self.target_positions.append(pos)
                elif char == 'W':
                    self.wall_positions.add(pos)

        self._update_boxes_on_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        # If game is over, do not process actions
        if self.game_over:
            return (
                self._get_observation(),
                0.0,
                self.game_over,
                False,
                self._get_info(),
            )

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.0
        move_executed = False

        if movement != 0: # 0 is no-op
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            
            next_player_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

            # Check for wall collision
            if next_player_pos in self.wall_positions:
                pass # Blocked move
            # Check for box collision
            elif next_player_pos in self.box_positions:
                box_idx = self.box_positions.index(next_player_pos)
                next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                
                # Check if space behind box is clear
                if next_box_pos not in self.wall_positions and next_box_pos not in self.box_positions:
                    # Push box and move player
                    self.box_positions[box_idx] = next_box_pos
                    self.player_pos = next_player_pos
                    move_executed = True
                    # sfx: Push box
            # Move into empty space
            else:
                self.player_pos = next_player_pos
                move_executed = True
                # sfx: Player step

        # Calculate rewards and update state if a move was made
        if move_executed:
            self.moves_left -= 1
            reward += self._calculate_reward()

        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            # Add terminal rewards
            if len(self.boxes_on_target_indices) == len(self.box_positions): # Win
                reward += 50.0
                self.score += 50.0
                # sfx: Win jingle
            else: # Lose
                reward -= 50.0
                self.score -= 50.0
                # sfx: Lose sound

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _calculate_reward(self):
        """Calculates reward for the current state after a move."""
        reward = -0.1  # Cost for making a move
        
        prev_on_target_count = len(self.boxes_on_target_indices)
        self._update_boxes_on_target()
        new_on_target_count = len(self.boxes_on_target_indices)

        if new_on_target_count > prev_on_target_count:
            reward += 5.0 * (new_on_target_count - prev_on_target_count)
            # sfx: Box on target
            
        return reward

    def _update_boxes_on_target(self):
        self.boxes_on_target_indices = {
            i for i, pos in enumerate(self.box_positions) if pos in self.target_positions
        }

    def _check_termination(self):
        # Win condition: all boxes are on targets
        if len(self.boxes_on_target_indices) == len(self.box_positions):
            return True
        # Loss condition: out of moves
        if self.moves_left <= 0:
            return True
        # Max steps as a fallback
        if self.steps >= 1000:
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "boxes_on_target": len(self.boxes_on_target_indices),
        }
        
    def _grid_to_pixels(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for the center of the cell."""
        gx, gy = grid_pos
        px = self.GRID_OFFSET_X + gx * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + gy * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_COLS * self.CELL_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_ROWS * self.CELL_SIZE))

        # Draw walls
        for pos in self.wall_positions:
            px, py = self._grid_to_pixels(pos)
            rect = pygame.Rect(px - self.CELL_SIZE//2, py - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            
        # Draw targets
        for pos in self.target_positions:
            px, py = self._grid_to_pixels(pos)
            pygame.gfxdraw.box(self.screen, (px - 10, py - 10, 20, 20), (*self.COLOR_TARGET, 150))

        # Draw boxes
        for i, pos in enumerate(self.box_positions):
            px, py = self._grid_to_pixels(pos)
            is_on_target = i in self.boxes_on_target_indices
            color = self.COLOR_BOX_ON_TARGET if is_on_target else self.COLOR_BOX
            
            rect = (px - 16, py - 16, 32, 32)
            pygame.gfxdraw.box(self.screen, rect, color)
            pygame.gfxdraw.rectangle(self.screen, rect, tuple(c-20 for c in color))
            if is_on_target:
                 pygame.gfxdraw.rectangle(self.screen, (px - 18, py - 18, 36, 36), (*self.COLOR_TARGET, 200))


        # Draw player
        px, py = self._grid_to_pixels(self.player_pos)
        pygame.gfxdraw.filled_circle(self.screen, px, py, 14, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, 14, self.COLOR_PLAYER)


    def _render_ui(self):
        # Render Moves Left
        moves_text = self.font_ui.render(f"Moves Left: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Render Score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(620, 20))
        self.screen.blit(score_text, score_rect)
        
        # Render Game Over Message
        if self.game_over:
            is_win = len(self.boxes_on_target_indices) == len(self.box_positions)
            message = "You Win!" if is_win else "Out of Moves!"
            color = self.COLOR_WIN if is_win else self.COLOR_LOSE
            
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(320, 200))
            self.screen.blit(text_surf, text_rect)
            
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a real screen for human play
    real_screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Sokoban Puzzle")
    
    done = False
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")
    
    while not done:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
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
                elif event.key == pygame.K_q: # Quit
                    done = True

        # Only step if an action was taken (not a no-op from key release)
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'R' to reset or 'Q' to quit.")

        # Draw the observation to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    pygame.quit()