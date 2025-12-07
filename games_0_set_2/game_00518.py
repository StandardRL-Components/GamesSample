
# Generated: 2025-08-27T13:53:22.773992
# Source Brief: brief_00518.md
# Brief Index: 518

        
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
        "Controls: Use arrow keys to move. Push all brown crates onto green goals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A series of Sokoban-style puzzles. Solve each stage in under 50 moves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MOVES_PER_STAGE = 50

    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_WALL = (100, 100, 120)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_CRATE = (160, 82, 45)
    COLOR_GOAL = (50, 205, 50)
    COLOR_CRATE_ON_GOAL = (255, 215, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_VICTORY = (100, 255, 100)
    COLOR_DEFEAT = (255, 100, 100)
    
    LEVELS = [
        [
            "##########",
            "#        #",
            "# P  C G #",
            "#        #",
            "##########",
        ],
        [
            "##########",
            "#        #",
            "# P C G  #",
            "#   C G  #",
            "#        #",
            "##########",
        ],
        [
            "############",
            "#   G      #",
            "# C # C    #",
            "# G #  P   #",
            "# C # C    #",
            "#   G      #",
            "############",
        ]
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
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)
        
        # State variables (initialized in reset)
        self.current_stage_index = 0
        self.player_pos = None
        self.crate_positions = None
        self.goal_positions = None
        self.wall_positions = None
        self.grid_dims = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.victory = False
        self.last_move_was_push = False
        self.random_generator = None

        self.validate_implementation()
        self.reset()
    
    def _load_stage(self, stage_index):
        """Parses a level string and sets up the game state for that stage."""
        self.current_stage_index = stage_index
        self.moves_left = self.MOVES_PER_STAGE
        
        self.player_pos = (0, 0)
        self.crate_positions = []
        self.goal_positions = []
        self.wall_positions = set()
        
        level_data = self.LEVELS[stage_index]
        height = len(level_data)
        width = len(level_data[0])
        self.grid_dims = (width, height)

        for y, row in enumerate(level_data):
            for x, char in enumerate(row):
                pos = (x, y)
                if char == '#':
                    self.wall_positions.add(pos)
                elif char == 'P':
                    self.player_pos = pos
                elif char == 'C':
                    self.crate_positions.append(pos)
                elif char == 'G':
                    self.goal_positions.append(pos)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.random_generator = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self._load_stage(0)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        if movement == 0: # No-op
            return self._get_observation(), reward, self.game_over, False, self._get_info()

        # A move is made, apply costs and update state
        self.steps += 1
        self.moves_left -= 1
        reward -= 0.1 # Cost for making a move
        
        # --- Process Movement ---
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
        
        player_x, player_y = self.player_pos
        target_pos = (player_x + dx, player_y + dy)

        # Check for collisions
        if target_pos in self.wall_positions:
            # Move is invalid, do nothing
            pass
        elif target_pos in self.crate_positions:
            # Attempting to push a crate
            push_target_pos = (target_pos[0] + dx, target_pos[1] + dy)
            if push_target_pos not in self.wall_positions and push_target_pos not in self.crate_positions:
                # Valid push
                crate_index = self.crate_positions.index(target_pos)
                
                was_on_goal = self.crate_positions[crate_index] in self.goal_positions
                self.crate_positions[crate_index] = push_target_pos
                is_on_goal = self.crate_positions[crate_index] in self.goal_positions
                
                # Grant rewards for crate movement relative to goals
                if not was_on_goal and is_on_goal:
                    reward += 1.0
                    self.score += 1.0
                elif was_on_goal and not is_on_goal:
                    reward -= 1.0
                    self.score -= 1.0

                self.player_pos = target_pos
                # Sound: crate_push.wav
            else:
                # Invalid push (blocked)
                # Sound: thud.wav
                pass
        else:
            # Standard move into an empty space
            self.player_pos = target_pos
            # Sound: step.wav

        # --- Check Game State ---
        stage_complete = all(crate in self.goal_positions for crate in self.crate_positions)

        if stage_complete:
            # Sound: stage_clear.wav
            reward += 10.0
            self.score += 10.0
            next_stage = self.current_stage_index + 1
            if next_stage < len(self.LEVELS):
                self._load_stage(next_stage)
            else:
                # All stages complete, VICTORY!
                self.game_over = True
                self.victory = True
                reward += 100.0
                self.score += 100.0
                # Sound: victory_fanfare.wav
        
        elif self.moves_left <= 0:
            # Out of moves, GAME OVER
            self.game_over = True
            self.victory = False
            # Sound: game_over.wav

        terminated = self.game_over
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_text(self, text, font, color, position, shadow_color=None, shadow_offset=(2, 2)):
        """Helper to render text with a shadow for better readability."""
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (position[0] + shadow_offset[0], position[1] + shadow_offset[1]))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, position)
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # --- Calculate grid rendering parameters ---
        grid_w, grid_h = self.grid_dims
        available_w, available_h = self.SCREEN_WIDTH - 40, self.SCREEN_HEIGHT - 80
        cell_size = min(available_w // grid_w, available_h // grid_h)
        
        grid_pixel_w = grid_w * cell_size
        grid_pixel_h = grid_h * cell_size
        offset_x = (self.SCREEN_WIDTH - grid_pixel_w) // 2
        offset_y = (self.SCREEN_HEIGHT - grid_pixel_h) // 2 + 40 # Push grid down for UI

        # --- Draw Grid Elements ---
        # Draw goals first so they are underneath crates
        for gx, gy in self.goal_positions:
            rect = pygame.Rect(offset_x + gx * cell_size, offset_y + gy * cell_size, cell_size, cell_size)
            pygame.draw.rect(self.screen, self.COLOR_GOAL, rect)

        # Draw walls
        for wx, wy in self.wall_positions:
            rect = pygame.Rect(offset_x + wx * cell_size, offset_y + wy * cell_size, cell_size, cell_size)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw crates
        for i, (cx, cy) in enumerate(self.crate_positions):
            rect = pygame.Rect(offset_x + cx * cell_size, offset_y + cy * cell_size, cell_size, cell_size)
            color = self.COLOR_CRATE_ON_GOAL if (cx, cy) in self.goal_positions else self.COLOR_CRATE
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, tuple(max(0, c-40) for c in color), rect, 3) # Border

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(offset_x + px * cell_size, offset_y + py * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.gfxdraw.rectangle(self.screen, player_rect, (255, 255, 255)) # White outline

        # --- Render UI Overlay ---
        self._render_text(
            f"Stage: {self.current_stage_index + 1}/{len(self.LEVELS)}",
            self.font_medium, self.COLOR_TEXT, (20, 10), self.COLOR_TEXT_SHADOW
        )
        self._render_text(
            f"Moves: {self.moves_left}",
            self.font_medium, self.COLOR_TEXT, (20, 40), self.COLOR_TEXT_SHADOW
        )
        self._render_text(
            f"Score: {int(self.score)}",
            self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH - 150, 10), self.COLOR_TEXT_SHADOW
        )

        # --- Render Game Over/Victory Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.victory:
                self._render_text("YOU WIN!", self.font_large, self.COLOR_VICTORY, (190, 170), self.COLOR_TEXT_SHADOW)
            else:
                self._render_text("GAME OVER", self.font_large, self.COLOR_DEFEAT, (170, 170), self.COLOR_TEXT_SHADOW)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage_index + 1,
            "moves_left": self.moves_left,
            "victory": self.victory
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
        
        # Temporarily load a stage to get a valid observation
        self._load_stage(0)
        
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
        
        # Reset to clean state
        self.reset()
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for interactive play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sokoban Solver")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(GameEnv.user_guide)
    print(GameEnv.game_description)

    while not terminated:
        action = [0, 0, 0] # Default action: no-op, buttons released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Press 'r' to reset
                    obs, info = env.reset()
                    print("--- Environment Reset ---")
                    action = [0, 0, 0] # Reset action to no-op
        
        # Only step if a move key was pressed
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
        
        # --- Draw the observation to the screen ---
        # The observation is (H, W, C), but pygame blit needs (W, H) surface
        # And the env's observation is transposed. So we need to transpose it back.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    print("Game Over!")
    env.close()
    pygame.quit()