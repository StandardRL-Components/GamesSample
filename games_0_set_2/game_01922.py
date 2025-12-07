
# Generated: 2025-08-27T18:43:25.440579
# Source Brief: brief_01922.md
# Brief Index: 1922

        
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
    """
    A minimalist, grid-based puzzle game where the player must push boxes onto target locations.

    The game is inspired by Sokoban. The player receives a large positive reward for solving
    the puzzle and a large negative reward for running out of moves. Small negative rewards
    are given for each move to encourage efficiency.

    The visual design is clean and geometric, prioritizing clarity of the game state.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move and push boxes."
    )

    # Short, user-facing description of the game
    game_description = (
        "Push boxes onto target locations in a minimalist grid-based puzzle game."
    )

    # Frames only advance when an action is received
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    TILE_SIZE = 40
    MAX_MOVES = 30
    MAX_STEPS = 1000 # Gym step limit

    # Colors
    COLOR_BG = (50, 50, 60)
    COLOR_GRID = (70, 70, 80)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_BORDER = (200, 200, 200)
    COLOR_WALL = (100, 100, 110)
    COLOR_BOX = (180, 120, 80)
    COLOR_BOX_BORDER = (140, 80, 40)
    COLOR_TARGET = (80, 180, 80)
    COLOR_BOX_ON_TARGET = (120, 220, 120)
    COLOR_TEXT = (240, 240, 240)

    # Level layout
    LEVEL_MAP = [
        "################",
        "#              #",
        "#  P B T       #",
        "#    #         #",
        "#  B # T       #",
        "#    #         #",
        "#  T B         #",
        "#              #",
        "#              #",
        "################",
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables are initialized in reset()
        self.player_pos = [0, 0]
        self.box_positions = []
        self.target_positions = []
        self.wall_positions = []
        self.steps = 0
        self.moves_made = 0
        self.score = 0
        self.game_over = False
        self.win_state = False

        self.reset()
        
        # Self-check to ensure implementation meets specs
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state from the level map
        self.player_pos = [0, 0]
        self.box_positions = []
        self.target_positions = []
        self.wall_positions = []

        for y, row in enumerate(self.LEVEL_MAP):
            for x, char in enumerate(row):
                pos = [x, y]
                if char == '#':
                    self.wall_positions.append(pos)
                elif char == 'P':
                    self.player_pos = pos
                elif char == 'B':
                    self.box_positions.append(pos)
                elif char == 'T':
                    self.target_positions.append(pos)

        # Ensure lists of lists are used for positions for mutability
        self.box_positions = [list(pos) for pos in self.box_positions]

        self.steps = 0
        self.moves_made = 0
        self.score = 0
        self.game_over = False
        self.win_state = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        self.steps += 1
        
        # Only process non-noop movements
        if movement > 0:
            self.moves_made += 1
            reward -= 0.1  # Cost for making a move

            # Map movement action to a delta vector
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            
            player_x, player_y = self.player_pos
            next_player_x, next_player_y = player_x + dx, player_y + dy

            # Check for collisions with walls
            if [next_player_x, next_player_y] in self.wall_positions:
                pass # Player hits a wall, no movement
            
            # Check for collisions with boxes
            elif [next_player_x, next_player_y] in self.box_positions:
                box_index = self.box_positions.index([next_player_x, next_player_y])
                next_box_x, next_box_y = next_player_x + dx, next_player_y + dy

                # Check if space behind the box is clear
                if [next_box_x, next_box_y] not in self.wall_positions and \
                   [next_box_x, next_box_y] not in self.box_positions:
                    
                    # Check if box was on a target before the push
                    was_on_target = self.box_positions[box_index] in self.target_positions
                    
                    # Move box
                    self.box_positions[box_index] = [next_box_x, next_box_y]
                    # # Sound effect placeholder
                    # print("SFX: box_slide.wav")
                    
                    # Move player
                    self.player_pos = [next_player_x, next_player_y]

                    # Check if box is on a target after the push
                    is_on_target = self.box_positions[box_index] in self.target_positions
                    
                    if not was_on_target and is_on_target:
                        reward += 1.0 # Reward for moving a box onto a target
                    elif was_on_target and not is_on_target:
                        reward -= 1.0 # Penalty for moving a box off a target

            # No collision, move player to empty space
            else:
                self.player_pos = [next_player_x, next_player_y]

        # Update score
        self.score += reward

        # Check for termination conditions
        terminated = self._check_termination()

        if terminated:
            if self.win_state:
                reward += 100.0 # Large reward for winning
            else:
                reward -= 100.0 # Large penalty for losing
            self.score += reward
            self.game_over = True
        
        # Gym step limit
        if self.steps >= self.MAX_STEPS:
            terminated = True


        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _check_termination(self):
        # Win condition: all boxes are on targets
        on_target_count = sum(1 for box_pos in self.box_positions if box_pos in self.target_positions)
        if on_target_count == len(self.target_positions):
            self.win_state = True
            return True

        # Lose condition: ran out of moves
        if self.moves_made >= self.MAX_MOVES:
            self.win_state = False
            return True

        return False

    def _get_observation(self):
        # Clear screen with background color
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert pygame surface to numpy array in the required format
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw targets
        for tx, ty in self.target_positions:
            pygame.gfxdraw.filled_circle(
                self.screen,
                tx * self.TILE_SIZE + self.TILE_SIZE // 2,
                ty * self.TILE_SIZE + self.TILE_SIZE // 2,
                self.TILE_SIZE // 3,
                self.COLOR_TARGET
            )

        # Draw walls
        for wx, wy in self.wall_positions:
            pygame.draw.rect(
                self.screen,
                self.COLOR_WALL,
                (wx * self.TILE_SIZE, wy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            )

        # Draw boxes
        for bx, by in self.box_positions:
            rect = pygame.Rect(
                bx * self.TILE_SIZE + 2, by * self.TILE_SIZE + 2,
                self.TILE_SIZE - 4, self.TILE_SIZE - 4
            )
            border_rect = pygame.Rect(
                bx * self.TILE_SIZE, by * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )

            is_on_target = [bx, by] in self.target_positions
            box_color = self.COLOR_BOX_ON_TARGET if is_on_target else self.COLOR_BOX
            
            pygame.draw.rect(self.screen, self.COLOR_BOX_BORDER, border_rect, 0, 4)
            pygame.draw.rect(self.screen, box_color, rect, 0, 4)

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(
            px * self.TILE_SIZE + 4, py * self.TILE_SIZE + 4,
            self.TILE_SIZE - 8, self.TILE_SIZE - 8
        )
        player_border_rect = pygame.Rect(
            px * self.TILE_SIZE, py * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_BORDER, player_border_rect, 0, 6)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, 0, 6)

    def _render_ui(self):
        # Display moves remaining
        moves_text = f"Moves: {self.moves_made}/{self.MAX_MOVES}"
        text_surface = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Display score
        score_text = f"Score: {self.score:.1f}"
        score_surface = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_surface, score_rect)
        
        # Display game over message
        if self.game_over:
            message = "SOLVED!" if self.win_state else "OUT OF MOVES"
            color = self.COLOR_BOX_ON_TARGET if self.win_state else self.COLOR_BOX
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            game_over_surface = self.font_game_over.render(message, True, color)
            game_over_rect = game_over_surface.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(game_over_surface, game_over_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_made": self.moves_made,
            "moves_remaining": self.MAX_MOVES - self.moves_made,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to use a visible display for testing
    import os
    os.environ.pop('SDL_VIDEODRIVER', None)

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This block allows a human to play the game.
    # It requires a visible Pygame window.
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sokoban Puzzle")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # Action defaults to no-op
        action = [0, 0, 0] 

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
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                    action = [0,0,0] # No move on reset frame
                elif event.key == pygame.K_q: # Quit on 'q' key
                    done = True

        if done:
            break

        # If an action was taken, step the environment
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Move: {info['moves_made']}, Reward: {reward:.1f}, Score: {info['score']:.1f}, Terminated: {terminated}")
            if terminated or truncated:
                print("--- Episode Finished ---")
                print("Press 'r' to reset or 'q' to quit.")


        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()