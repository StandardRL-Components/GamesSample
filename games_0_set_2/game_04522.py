
# Generated: 2025-08-28T02:39:23.953089
# Source Brief: brief_04522.md
# Brief Index: 4522

        
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
    GameEnv: A strategic grid-based puzzle game.

    The player navigates a grid to collect gems within a limited number of moves.
    The goal is to collect all gems before running out of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character one square at a time."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Collect all the gems on the grid before you run out of moves. Plan your path carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment.
        """
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_COLS = 16
        self.GRID_ROWS = 10
        self.CELL_SIZE = self.SCREEN_WIDTH // self.GRID_COLS
        self.MAX_MOVES = 25
        self.NUM_GEMS = 10
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_VISITED = (50, 60, 70)
        self.COLOR_PLAYER = (0, 255, 127) # Spring Green
        self.COLOR_PLAYER_GLOW = (*self.COLOR_PLAYER, 100)
        self.COLOR_GEM_LOW = (255, 223, 0) # Gold
        self.COLOR_GEM_MID = (255, 165, 0) # Orange
        self.COLOR_GEM_HIGH = (255, 69, 0)  # OrangeRed
        self.COLOR_GEM_SHINE = (255, 255, 255, 200)
        self.COLOR_UI_BG = (10, 10, 20, 180)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_WIN_TEXT = (144, 238, 144) # Light Green
        self.COLOR_LOSE_TEXT = (255, 105, 97) # Light Red

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_end = pygame.font.SysFont("Arial", 48, bold=True)

        # --- Game State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.player_pos = (0, 0)
        self.gems = []
        self.gems_collected = 0
        self.visited_cells = set()
        self.last_distance_to_nearest_gem = float('inf')
        self.end_message = ""

        # Initialize state for the first time
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.gems_collected = 0
        self.end_message = ""

        # Generate unique positions for player and gems
        all_positions = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.np_random.shuffle(all_positions)

        self.player_pos = all_positions.pop()
        self.visited_cells = {self.player_pos}

        self.gems = []
        gem_positions = all_positions[:self.NUM_GEMS]
        
        # Gem values: 6 low, 3 mid, 1 high
        gem_values = [10] * 6 + [20] * 3 + [50] * 1
        self.np_random.shuffle(gem_values)

        for i in range(self.NUM_GEMS):
            pos = gem_positions[i]
            value = gem_values[i]
            if value == 10: color = self.COLOR_GEM_LOW
            elif value == 20: color = self.COLOR_GEM_MID
            else: color = self.COLOR_GEM_HIGH
            self.gems.append({"pos": pos, "value": value, "color": color})
            
        self.last_distance_to_nearest_gem = self._get_distance_to_nearest_gem()

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Advances the game by one step based on the given action.
        """
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are unused in this game
        
        reward = 0
        moved = False
        
        # --- Process Movement ---
        if movement > 0: # Any move action consumes a move
            px, py = self.player_pos
            nx, ny = px, py

            if movement == 1: ny -= 1  # Up
            elif movement == 2: ny += 1  # Down
            elif movement == 3: nx -= 1  # Left
            elif movement == 4: nx += 1  # Right

            # Check boundaries
            if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                self.player_pos = (nx, ny)
                self.visited_cells.add(self.player_pos)
                moved = True
            # else: Wall bump, no position change
            
            self.moves_remaining -= 1
        # No-op (movement == 0) does nothing and costs nothing

        # --- Calculate Rewards ---
        if moved:
            # Reward for moving closer to a gem
            current_dist = self._get_distance_to_nearest_gem()
            if current_dist < self.last_distance_to_nearest_gem:
                reward += 1.0
            else:
                reward -= 0.1
            self.last_distance_to_nearest_gem = current_dist

        # Check for gem collection
        gem_to_remove = None
        for gem in self.gems:
            if self.player_pos == gem["pos"]:
                # Event-based reward for collecting a gem
                reward += gem["value"]
                self.score += gem["value"]
                # Risky play bonus
                if self.moves_remaining < 5:
                    reward += 5
                    self.score += 5

                self.gems_collected += 1
                gem_to_remove = gem
                self.last_distance_to_nearest_gem = self._get_distance_to_nearest_gem()
                # Placeholder: # play_sound("collect_gem.wav")
                break
        
        if gem_to_remove:
            self.gems.remove(gem_to_remove)

        # --- Check Termination Conditions ---
        terminated = False
        if self.gems_collected == self.NUM_GEMS:
            reward += 100  # Goal-oriented reward
            self.score += 100
            terminated = True
            self.game_over = True
            self.end_message = "YOU WIN!"
            # Placeholder: # play_sound("win.wav")
        elif self.moves_remaining <= 0:
            terminated = True
            self.game_over = True
            self.end_message = "GAME OVER"
            # Placeholder: # play_sound("lose.wav")
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_distance_to_nearest_gem(self):
        """Helper to calculate Manhattan distance to the nearest gem."""
        if not self.gems:
            return 0
        px, py = self.player_pos
        min_dist = float('inf')
        for gem in self.gems:
            gx, gy = gem["pos"]
            dist = abs(px - gx) + abs(py - gy)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_observation(self):
        """
        Renders the current game state to a numpy array.
        """
        # --- Clear screen ---
        self.screen.fill(self.COLOR_BG)

        # --- Render Game Elements ---
        self._render_grid()
        self._render_gems()
        self._render_player()

        # --- Render UI Overlay ---
        self._render_ui()

        # --- Render End Message ---
        if self.game_over:
            self._render_end_message()

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        """Renders the grid lines and visited cells."""
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if (c, r) in self.visited_cells:
                    pygame.draw.rect(self.screen, self.COLOR_VISITED, rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_gems(self):
        """Renders the gems with a shine effect."""
        for gem in self.gems:
            gx, gy = gem["pos"]
            center_x = int((gx + 0.5) * self.CELL_SIZE)
            center_y = int((gy + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.3)
            
            # Draw main gem body
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, gem["color"])
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, gem["color"])
            
            # Draw shine/highlight
            shine_x = center_x + radius // 3
            shine_y = center_y - radius // 3
            shine_radius = radius // 3
            pygame.gfxdraw.aacircle(self.screen, shine_x, shine_y, shine_radius, self.COLOR_GEM_SHINE)
            pygame.gfxdraw.filled_circle(self.screen, shine_x, shine_y, shine_radius, self.COLOR_GEM_SHINE)

    def _render_player(self):
        """Renders the player with a glow effect."""
        px, py = self.player_pos
        player_rect = pygame.Rect(
            px * self.CELL_SIZE,
            py * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        center_x, center_y = player_rect.center
        glow_radius = int(self.CELL_SIZE * 0.7)

        # Draw glow
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, self.COLOR_PLAYER_GLOW)

        # Draw player square on top
        inset = int(self.CELL_SIZE * 0.1)
        player_square = player_rect.inflate(-inset, -inset)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_square, border_radius=3)

    def _render_ui(self):
        """Renders the top UI bar with game information."""
        ui_height = 35
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, ui_height), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))
        
        # Moves remaining
        moves_text = f"Moves: {self.moves_remaining}/{self.MAX_MOVES}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_surf, (10, 7))

        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(centerx=self.SCREEN_WIDTH // 2, top=7)
        self.screen.blit(score_surf, score_rect)

        # Gems collected
        gems_text = f"Gems: {self.gems_collected}/{self.NUM_GEMS}"
        gems_surf = self.font_ui.render(gems_text, True, self.COLOR_UI_TEXT)
        gems_rect = gems_surf.get_rect(right=self.SCREEN_WIDTH - 10, top=7)
        self.screen.blit(gems_surf, gems_rect)

    def _render_end_message(self):
        """Renders the 'YOU WIN' or 'GAME OVER' message."""
        color = self.COLOR_WIN_TEXT if "WIN" in self.end_message else self.COLOR_LOSE_TEXT
        text_surf = self.font_end.render(self.end_message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        
        # Draw a semi-transparent background for the text for readability
        bg_rect = text_rect.inflate(20, 20)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 150))
        self.screen.blit(bg_surf, bg_rect)
        
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        """
        Returns a dictionary with auxiliary diagnostic information.
        """
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "gems_collected": self.gems_collected,
            "player_pos": self.player_pos,
        }

    def close(self):
        """
        Cleans up the environment's resources.
        """
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


if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display window
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    done = False
    total_reward = 0
    
    print("\n" + "="*30)
    print("      MANUAL PLAYING MODE")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("Press ESC or close the window to quit.")
    
    while not done:
        # --- Action Mapping for Human ---
        action = np.array([0, 0, 0]) # Default no-op
        
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
                elif event.key == pygame.K_ESCAPE:
                    done = True
                
                # Only step if a movement key was pressed
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
                    if terminated or truncated:
                        done = True

        # --- Render to screen ---
        # The observation is already a rendered frame
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS
    
    print("\nGame Over!")
    print(f"Final Score: {info.get('score', 0)}")
    print(f"Total Reward: {total_reward:.2f}")

    env.close()