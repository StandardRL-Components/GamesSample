
# Generated: 2025-08-27T19:45:54.373081
# Source Brief: brief_02250.md
# Brief Index: 2250

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use ↑↓←→ to move. Walk into boxes to push them into the green target zones."
    )

    # User-facing description of the game
    game_description = (
        "A top-down puzzle game. Push all the boxes onto the target locations within the move limit to win."
    )

    # Frames wait for user input
    auto_advance = False
    
    # --- Game Constants ---
    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    TILE_SIZE = 32
    SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE  # 640
    SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE # 384
    
    # The observation space is larger, so we'll pad the rendering
    OBS_WIDTH = 640
    OBS_HEIGHT = 400

    MAX_MOVES_PER_LEVEL = 100
    
    # --- Colors ---
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (30, 35, 40)
    COLOR_WALL = (70, 80, 90)
    COLOR_WALL_ACCENT = (90, 100, 110)
    COLOR_PLAYER = (230, 50, 50)
    COLOR_PLAYER_ACCENT = (255, 100, 100)
    COLOR_BOX = (180, 120, 80)
    COLOR_BOX_ACCENT = (200, 140, 100)
    COLOR_TARGET = (50, 150, 50)
    COLOR_BOX_ON_TARGET = (150, 180, 80)
    COLOR_BOX_ON_TARGET_ACCENT = (170, 200, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (40, 45, 50, 180)

    # --- Level Data ---
    LEVELS = [
        """
####################
#                  #
# . @.$            #
#                  #
# .   $            #
#                  #
# .   $            #
#                  #
# .   $            #
#                  #
# .   $            #
####################
""",
        """
####################
#..  #             #
#..  # @  $ $      #
#..  #      $      #
#..  #####  $      #
#..    #    $      #
#      #           #
#      #############
#                  #
#                  #
#                  #
####################
""",
        """
####################
#..  @ ########### #
#..$ $ #         # #
#..    # $       # #
#..  $ ########### #
#..$   #           #
#      #           #
#  #####           #
#                  #
#                  #
#                  #
####################
"""
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Exact spaces required by the brief
        self.observation_space = Box(
            low=0, high=255, shape=(self.OBS_HEIGHT, self.OBS_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.OBS_WIDTH, self.OBS_HEIGHT))
        self.game_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Initialize state variables
        self.player_pos = (0, 0)
        self.box_positions = []
        self.target_positions = []
        self.wall_positions = set()
        self.total_steps = 0
        self.moves = 0
        self.total_score = 0
        self.level = 0
        self.game_over = False
        self.win_game = False

        self.reset()
        self.validate_implementation()
    
    def _load_level(self, level_index):
        self.moves = 0
        self.player_pos = (0, 0)
        self.box_positions = []
        self.target_positions = []
        self.wall_positions = set()
        
        level_data = self.LEVELS[level_index].strip().split('\n')
        for y, row in enumerate(level_data):
            for x, char in enumerate(row):
                pos = (x, y)
                if char == '#':
                    self.wall_positions.add(pos)
                elif char == '@':
                    self.player_pos = pos
                elif char == '$':
                    self.box_positions.append(pos)
                elif char == '.':
                    self.target_positions.append(pos)
                elif char == '*': # Box on a target
                    self.box_positions.append(pos)
                    self.target_positions.append(pos)
                elif char == '+': # Player on a target
                    self.player_pos = pos
                    self.target_positions.append(pos)
        
        self.target_positions_set = frozenset(self.target_positions)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.total_steps = 0
        self.total_score = 0
        self.level = 1
        self.game_over = False
        self.win_game = False
        
        self._load_level(0)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]
        self.total_steps += 1
        reward = 0
        
        if self.game_over or self.win_game:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # A no-op action still consumes a step, but not a "move"
        if movement == 0:
            return self._get_observation(), 0, False, False, self._get_info()

        self.moves += 1
        
        # --- Game Logic ---
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
        
        next_player_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
        
        old_box_positions = list(self.box_positions)
        
        if next_player_pos in self.wall_positions:
            # Player walks into a wall, no move.
            pass
        elif next_player_pos in self.box_positions:
            # Player tries to push a box
            box_idx = self.box_positions.index(next_player_pos)
            next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
            
            if next_box_pos not in self.wall_positions and next_box_pos not in self.box_positions:
                # Push is successful
                # SFX: push_box.wav
                self.box_positions[box_idx] = next_box_pos
                self.player_pos = next_player_pos
        else:
            # Player moves into empty space
            self.player_pos = next_player_pos

        # --- Reward Calculation ---
        reward += self._calculate_movement_reward(old_box_positions, self.box_positions)
        reward += self._calculate_event_reward(old_box_positions, self.box_positions)
        
        # --- Termination and Level Progression ---
        terminated = False
        level_complete = all(box in self.target_positions_set for box in self.box_positions)

        if level_complete:
            # SFX: level_win.wav
            reward += 100
            self.total_score += reward
            if self.level < len(self.LEVELS):
                self.level += 1
                self._load_level(self.level - 1)
            else:
                self.win_game = True
                terminated = True
        elif self.moves >= self.MAX_MOVES_PER_LEVEL:
            # SFX: level_lose.wav
            reward -= 100
            self.game_over = True
            terminated = True
        
        if not level_complete:
            self.total_score += reward
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _sum_dist_to_targets(self, box_positions):
        total_dist = 0
        if not self.target_positions:
            return 0
        for box_pos in box_positions:
            min_dist = min(self._manhattan_distance(box_pos, target_pos) for target_pos in self.target_positions)
            total_dist += min_dist
        return total_dist

    def _calculate_movement_reward(self, old_positions, new_positions):
        old_dist = self._sum_dist_to_targets(old_positions)
        new_dist = self._sum_dist_to_targets(new_positions)
        return old_dist - new_dist  # +1 for moving closer, -1 for moving away

    def _calculate_event_reward(self, old_positions, new_positions):
        old_on_target = sum(1 for pos in old_positions if pos in self.target_positions_set)
        new_on_target = sum(1 for pos in new_positions if pos in self.target_positions_set)
        if new_on_target > old_on_target:
            # SFX: box_in_place.wav
            return 10  # Reward for placing a box on a target
        return 0

    def _get_observation(self):
        # Clear screen with background color
        self.screen.fill(self.COLOR_BG)
        self.game_surface.fill(self.COLOR_BG)
        
        # Render all game elements onto the game surface
        self._render_game()
        
        # Blit the game surface to the main screen, centered
        x_offset = (self.OBS_WIDTH - self.SCREEN_WIDTH) // 2
        y_offset = (self.OBS_HEIGHT - self.SCREEN_HEIGHT) // 2
        self.screen.blit(self.game_surface, (x_offset, y_offset))
        
        # Render UI overlay on the main screen
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.game_surface, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.game_surface, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw targets
        for x, y in self.target_positions:
            rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.game_surface, self.COLOR_TARGET, rect)

        # Draw walls
        for x, y in self.wall_positions:
            rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.game_surface, self.COLOR_WALL, rect)
            pygame.draw.rect(self.game_surface, self.COLOR_WALL_ACCENT, rect.inflate(-6, -6))

        # Draw boxes
        for x, y in self.box_positions:
            rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            is_on_target = (x, y) in self.target_positions_set
            
            main_color = self.COLOR_BOX_ON_TARGET if is_on_target else self.COLOR_BOX
            accent_color = self.COLOR_BOX_ON_TARGET_ACCENT if is_on_target else self.COLOR_BOX_ACCENT
            
            pygame.draw.rect(self.game_surface, main_color, rect)
            pygame.draw.rect(self.game_surface, accent_color, rect.inflate(-8, -8))

        # Draw player
        px, py = self.player_pos
        rect = pygame.Rect(px * self.TILE_SIZE, py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.game_surface, self.COLOR_PLAYER, rect)
        pygame.draw.rect(self.game_surface, self.COLOR_PLAYER_ACCENT, rect.inflate(-8, -8))
        
    def _render_ui(self):
        # UI Background
        ui_panel = pygame.Surface((self.OBS_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Moves text
        moves_text = self.font_medium.render(f"Moves: {self.moves}/{self.MAX_MOVES_PER_LEVEL}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 8))

        # Level text
        level_text = self.font_medium.render(f"Level: {self.level}/{len(self.LEVELS)}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.OBS_WIDTH / 2 - level_text.get_width() / 2, 8))
        
        # Score text
        score_text = self.font_medium.render(f"Score: {self.total_score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.OBS_WIDTH - score_text.get_width() - 10, 8))

        # Game Over / Win Message
        if self.game_over or self.win_game:
            message = "YOU WIN!" if self.win_game else "GAME OVER"
            color = (100, 255, 100) if self.win_game else (255, 100, 100)
            
            overlay = pygame.Surface((self.OBS_WIDTH, self.OBS_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            msg_render = self.font_large.render(message, True, color)
            msg_rect = msg_render.get_rect(center=(self.OBS_WIDTH / 2, self.OBS_HEIGHT / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(msg_render, msg_rect)


    def _get_info(self):
        return {
            "score": self.total_score,
            "steps": self.total_steps,
            "moves": self.moves,
            "level": self.level,
            "boxes_on_target": sum(1 for pos in self.box_positions if pos in self.target_positions_set),
            "total_boxes": len(self.box_positions)
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
        assert test_obs.shape == (self.OBS_HEIGHT, self.OBS_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.OBS_HEIGHT, self.OBS_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.OBS_HEIGHT, self.OBS_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Set the video driver to dummy to run headless
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # --- Manual Play Example ---
    # This requires a display. Comment out the os.environ line above to run this.
    # pygame.display.set_caption("Sokoban")
    # screen = pygame.display.set_mode((env.OBS_WIDTH, env.OBS_HEIGHT))
    # obs, info = env.reset()
    # done = False
    # clock = pygame.time.Clock()
    
    # while not done:
    #     action = 0 # no-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_UP: action = 1
    #             elif event.key == pygame.K_DOWN: action = 2
    #             elif event.key == pygame.K_LEFT: action = 3
    #             elif event.key == pygame.K_RIGHT: action = 4
    #             elif event.key == pygame.K_r: # Reset
    #                 obs, info = env.reset()
    #                 action = 0

    #     if action > 0:
    #         full_action = [action, 0, 0]
    #         obs, reward, terminated, truncated, info = env.step(full_action)
    #         print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {terminated}")
    #         if terminated:
    #             print("Game Over!")

    #     # Draw the observation to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     clock.tick(30) # Limit frame rate
        
    # env.close()

    # --- Agent Interaction Example ---
    obs, info = env.reset()
    print("Initial State:")
    print(f"  Score: {info['score']}, Level: {info['level']}")
    
    terminated = False
    total_reward = 0
    for i in range(500):
        if terminated:
            print(f"Episode finished after {i+1} steps.")
            break
        
        action = env.action_space.sample() # Random agent
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if action[0] != 0: # Only print for actual moves
             print(f"Step {i+1:3d} | Action: {action[0]} | Reward: {reward:5.1f} | Total Reward: {total_reward:6.1f} | "
                   f"Score: {info['score']:4d} | Level: {info['level']} | Moves: {info['moves']:3d} | Done: {terminated}")

    print("\nFinal State:")
    print(f"  Score: {info['score']}, Level: {info['level']}, Moves: {info['moves']}")
    env.close()