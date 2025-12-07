
# Generated: 2025-08-27T18:55:32.601292
# Source Brief: brief_01992.md
# Brief Index: 1992

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a grid-based monster munching game.
    The player controls a monster that must eat all food items on a grid
    within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move the monster one square at a time."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a monster to devour all the food on the grid before you run out of moves. Plan your path carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment.
        """
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.UI_HEIGHT = 40
        self.GRID_COLS, self.GRID_ROWS = 32, 18
        self.CELL_SIZE = 20
        self.GAME_AREA_HEIGHT = self.GRID_ROWS * self.CELL_SIZE

        self.MAX_MOVES = 50
        self.FOOD_COUNT = 25

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_FOOD = [
            (255, 80, 80),   # Red
            (255, 255, 80),  # Yellow
            (80, 255, 255),  # Cyan
            (255, 80, 255),  # Magenta
        ]
        self.COLOR_CONSUMED = (60, 60, 70)
        self.COLOR_UI_BG = (10, 10, 10)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_marker = pygame.font.SysFont("sans-serif", int(self.CELL_SIZE * 0.9), bold=True)
        self.font_end_game = pygame.font.SysFont("monospace", 60, bold=True)

        # --- State Variables ---
        self.np_random = None
        self.monster_pos = (0, 0)
        self.food_positions = set()
        self.consumed_food_pos = set()
        self.moves_left = 0
        self.score = 0
        self.game_over = False
        self.steps = 0

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES

        # Place monster in the center
        self.monster_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)

        # Generate all possible grid positions
        all_positions = [
            (x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)
        ]
        all_positions.remove(self.monster_pos) # Player can't start on food

        # Randomly select food positions
        chosen_indices = self.np_random.choice(
            len(all_positions), self.FOOD_COUNT, replace=False
        )
        self.food_positions = {all_positions[i] for i in chosen_indices}
        self.consumed_food_pos = set()

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Advances the game state by one step based on the given action.
        """
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        terminated = False
        
        self.steps += 1 # Internal step counter for animations

        # --- Handle Movement Action ---
        if movement != 0:  # 0 is no-op
            self.moves_left -= 1
            
            old_pos = self.monster_pos
            px, py = self.monster_pos
            if movement == 1: py -= 1  # Up
            elif movement == 2: py += 1  # Down
            elif movement == 3: px -= 1  # Left
            elif movement == 4: px += 1  # Right

            # Update position if within bounds
            if 0 <= px < self.GRID_COLS and 0 <= py < self.GRID_ROWS:
                self.monster_pos = (px, py)

            # --- Calculate Reward ---
            if self.monster_pos in self.food_positions:
                # Eaten food
                # Sound effect placeholder: # Chomp!
                self.food_positions.remove(self.monster_pos)
                self.consumed_food_pos.add(self.monster_pos)
                self.score += 1
                reward = 1.0
                
                if not self.food_positions: # Ate the last food
                    reward += 10.0 # Last food bonus
            else:
                # Moved to an empty square
                reward = -0.2
        else:
            # No-op action
            reward = 0.0

        # --- Check Termination Conditions ---
        if not self.food_positions:
            reward += 100.0  # Win bonus
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0:
            reward -= 100.0  # Lose penalty
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        """
        Renders the current game state to an RGB array.
        """
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_end_screen()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the grid, food, and monster."""
        # Draw grid lines
        for x in range(self.GRID_COLS + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.UI_HEIGHT), (px, self.HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = self.UI_HEIGHT + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.WIDTH, py))

        # Draw consumed food markers ('x')
        marker_surf = self.font_marker.render("x", True, self.COLOR_CONSUMED)
        for pos in self.consumed_food_pos:
            px = pos[0] * self.CELL_SIZE
            py = self.UI_HEIGHT + pos[1] * self.CELL_SIZE
            self.screen.blit(marker_surf, (px + (self.CELL_SIZE - marker_surf.get_width()) / 2, 
                                            py + (self.CELL_SIZE - marker_surf.get_height()) / 2 - 2))

        # Draw food items
        food_size = int(self.CELL_SIZE * 0.5)
        offset = (self.CELL_SIZE - food_size) // 2
        food_list = sorted(list(self.food_positions)) # Sort for consistent colors
        for i, pos in enumerate(food_list):
            color = self.COLOR_FOOD[i % len(self.COLOR_FOOD)]
            px = pos[0] * self.CELL_SIZE + offset
            py = self.UI_HEIGHT + pos[1] * self.CELL_SIZE + offset
            pygame.draw.rect(self.screen, color, (px, py, food_size, food_size), border_radius=3)
            
        # Draw monster
        monster_size = int(self.CELL_SIZE * 0.8)
        offset = (self.CELL_SIZE - monster_size) // 2
        px = self.monster_pos[0] * self.CELL_SIZE + offset
        py = self.UI_HEIGHT + self.monster_pos[1] * self.CELL_SIZE + offset
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px, py, monster_size, monster_size), border_radius=4)
        
        # Animated mouth
        if not self.game_over and self.steps % 10 < 5:
            mouth_height = int(monster_size * 0.25)
            mouth_y = py + monster_size // 2 - mouth_height // 2
            pygame.draw.rect(self.screen, self.COLOR_BG, (px, mouth_y, monster_size, mouth_height))

    def _render_ui(self):
        """Renders the top UI bar with score and moves."""
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.WIDTH, self.UI_HEIGHT))
        
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (15, (self.UI_HEIGHT - score_surf.get_height()) // 2))

        moves_text = f"MOVES: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_surf, (self.WIDTH - moves_surf.get_width() - 15, (self.UI_HEIGHT - moves_surf.get_height()) // 2))

    def _render_end_screen(self):
        """Renders the win/loss message."""
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        if not self.food_positions: # Win
            end_text = "YOU WIN!"
            end_color = self.COLOR_WIN
        else: # Lose
            end_text = "GAME OVER"
            end_color = self.COLOR_LOSE
        
        end_surf = self.font_end_game.render(end_text, True, end_color)
        text_rect = end_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(end_surf, text_rect)

    def _get_info(self):
        """
        Returns a dictionary with auxiliary diagnostic information.
        """
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "food_remaining": len(self.food_positions),
        }

    def close(self):
        """
        Cleans up Pygame resources.
        """
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a Pygame window to display the game
    pygame.display.set_caption("Monster Munch")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")

    while running:
        action = np.array([0, 0, 0])  # Default action is no-op

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
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False
                
                # Take a step only when a key is pressed
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
                if terminated:
                    print("Game Over! Press 'r' to restart.")

        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS

    env.close()