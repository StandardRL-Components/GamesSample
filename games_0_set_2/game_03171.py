
# Generated: 2025-08-28T07:12:45.600545
# Source Brief: brief_03171.md
# Brief Index: 3171

        
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
        "Controls: ↑↓←→ to move. Collect 20 gems in 15 moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Collect 20 gems in 15 moves by navigating the grid efficiently."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    GAME_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
    GAME_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
    
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 45, 50)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
    ]

    # Game parameters
    TOTAL_GEMS = 20
    MAX_MOVES = 15

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
        
        # Fonts
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Calculate offsets to center the grid
        self.grid_offset_x = (self.SCREEN_WIDTH - self.GAME_AREA_WIDTH) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.GAME_AREA_HEIGHT) // 2

        # Initialize state variables
        self.player_pos = [0, 0]
        self.gems = []
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.collection_anims = []
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.collection_anims = []

        # Place player in the center
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        # Generate gem positions
        self.gems = []
        possible_coords = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) != tuple(self.player_pos):
                    possible_coords.append([x, y])
        
        gem_indices = self.np_random.choice(len(possible_coords), self.TOTAL_GEMS, replace=False)
        for i in gem_indices:
            color = random.choice(self.GEM_COLORS)
            self.gems.append({"pos": possible_coords[i], "color": color})
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        terminated = False
        
        # Every action, including no-op, consumes a move
        self.moves_left -= 1
        
        old_pos = list(self.player_pos)
        
        if movement == 1:  # Up
            self.player_pos[1] -= 1
        elif movement == 2:  # Down
            self.player_pos[1] += 1
        elif movement == 3:  # Left
            self.player_pos[0] -= 1
        elif movement == 4:  # Right
            self.player_pos[0] += 1
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)
        
        # Check for gem collection
        gem_collected = False
        for i, gem in enumerate(self.gems):
            if gem["pos"] == self.player_pos:
                # Gem collected
                # Sfx: Gem collect sound
                self.score += 1
                reward = 1.0
                gem_collected = True
                self._start_collection_animation(gem["pos"], gem["color"])
                self.gems.pop(i)
                break
        
        if not gem_collected:
            # -0.1 for any move (including no-op or hitting a wall) that doesn't collect a gem
            reward = -0.1

        # Check for termination
        win = self.score >= self.TOTAL_GEMS
        lose = self.moves_left <= 0
        
        if win or lose:
            terminated = True
            self.game_over = True
            if win:
                reward += 100 # Goal-oriented win reward
            elif lose:
                reward -= 10 # Goal-oriented lose penalty

        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
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
            "moves_left": self.moves_left
        }

    def _grid_to_pixel(self, grid_pos):
        x = self.grid_offset_x + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.grid_offset_y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            start = (self.grid_offset_x + x * self.CELL_SIZE, self.grid_offset_y)
            end = (self.grid_offset_x + x * self.CELL_SIZE, self.grid_offset_y + self.GAME_AREA_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.grid_offset_x, self.grid_offset_y + y * self.CELL_SIZE)
            end = (self.grid_offset_x + self.GAME_AREA_WIDTH, self.grid_offset_y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw gems
        gem_radius = self.CELL_SIZE // 4
        for gem in self.gems:
            px, py = self._grid_to_pixel(gem["pos"])
            pygame.gfxdraw.filled_circle(self.screen, px, py, gem_radius, gem["color"])
            pygame.gfxdraw.aacircle(self.screen, px, py, gem_radius, gem["color"])

        # Draw player
        player_px, player_py = self._grid_to_pixel(self.player_pos)
        player_size = self.CELL_SIZE // 3
        player_rect = pygame.Rect(player_px - player_size // 2, player_py - player_size // 2, player_size, player_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Draw collection animations
        self._update_and_draw_animations()

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"Gems: {self.score}/{self.TOTAL_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Moves display
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)

        # Game Over display
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.score >= self.TOTAL_GEMS:
                end_text = self.font_large.render("YOU WIN!", True, self.GEM_COLORS[1])
            else:
                end_text = self.font_large.render("GAME OVER", True, self.GEM_COLORS[0])
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _start_collection_animation(self, pos, color):
        px, py = self._grid_to_pixel(pos)
        self.collection_anims.append({
            'pos': (px, py),
            'radius': self.CELL_SIZE // 4,
            'alpha': 255,
            'color': color,
        })

    def _update_and_draw_animations(self):
        for i in range(len(self.collection_anims) - 1, -1, -1):
            anim = self.collection_anims[i]
            
            # Update
            anim['radius'] += 2
            anim['alpha'] -= 15
            
            # Draw
            if anim['alpha'] > 0:
                temp_surface = pygame.Surface((anim['radius'] * 2, anim['radius'] * 2), pygame.SRCALPHA)
                color_with_alpha = (*anim['color'], anim['alpha'])
                pygame.gfxdraw.aacircle(temp_surface, anim['radius'], anim['radius'], anim['radius'] - 1, color_with_alpha)
                self.screen.blit(temp_surface, (anim['pos'][0] - anim['radius'], anim['pos'][1] - anim['radius']))
            
            # Remove if finished
            if anim['alpha'] <= 0:
                self.collection_anims.pop(i)

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