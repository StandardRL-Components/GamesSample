
# Generated: 2025-08-28T01:51:14.461430
# Source Brief: brief_04258.md
# Brief Index: 4258

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a number. Match two identical numbers to clear them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic number-matching puzzle. Clear the grid by finding all the pairs before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 5
        self.CELL_SIZE = 70
        self.GRID_MARGIN_X = (self.SCREEN_WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_MARGIN_Y = (self.SCREEN_HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.MAX_STEPS = 1000
        self.INITIAL_MOVES = 25

        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (50, 55, 60)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_SELECTED = (0, 150, 255)
        self.COLOR_FLASH = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)
        self.NUMBER_PALETTE = [
            (255, 87, 87), (255, 170, 87), (230, 255, 87), (87, 255, 95),
            (87, 255, 240), (87, 160, 255), (130, 87, 255), (255, 87, 247),
            (255, 87, 150), (192, 192, 192), (255, 215, 0), (0, 255, 255), (255, 0, 0)
        ]

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
        self.font_main = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_remaining = 0
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_tile = None
        self.prev_space_held = False
        self.flash_animation = []
        self.total_pairs = 0
        self.pairs_cleared = 0
        self.np_random = None

        # Initialize state
        self.reset()

        # Run self-check
        self.validate_implementation()
    
    def _generate_grid(self):
        self.total_pairs = (self.GRID_SIZE * self.GRID_SIZE) // 2
        numbers = list(range(1, self.total_pairs + 1)) * 2
        if len(numbers) < self.GRID_SIZE * self.GRID_SIZE:
            numbers.append(self.total_pairs + 1)
        
        self.np_random.shuffle(numbers)
        
        self.grid = []
        for r in range(self.GRID_SIZE):
            row = []
            for c in range(self.GRID_SIZE):
                value = numbers.pop()
                row.append({'value': value, 'state': 'visible'})
            self.grid.append(row)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random.default_rng()

        self._generate_grid()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_remaining = self.INITIAL_MOVES
        self.cursor_pos = [0, 0]
        self.selected_tile = None
        self.prev_space_held = False
        self.flash_animation = []
        self.pairs_cleared = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        self.flash_animation = [] # Clear flash from previous step

        # Unpack factorized action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if not self.game_over:
            # Handle cursor movement
            if movement == 1: # Up
                self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
            elif movement == 2: # Down
                self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE
            elif movement == 3: # Left
                self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
            elif movement == 4: # Right
                self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE

            # Handle selection on space press (rising edge)
            space_press = space_held and not self.prev_space_held
            if space_press:
                r, c = self.cursor_pos
                tile = self.grid[r][c]

                if tile['state'] == 'visible':
                    if self.selected_tile is None:
                        # First selection
                        self.selected_tile = {'pos': (r, c), 'value': tile['value']}
                        # sfx: select_tile
                    else:
                        # Second selection, this constitutes a "move"
                        self.moves_remaining -= 1

                        if self.selected_tile['pos'] == (r, c):
                            # Deselected the same tile
                            self.selected_tile = None
                            # sfx: deselect_tile
                        elif self.selected_tile['value'] == tile['value']:
                            # Successful match
                            reward += 1
                            self.score += 1
                            
                            # Mark both tiles as cleared
                            self.grid[r][c]['state'] = 'cleared'
                            sel_r, sel_c = self.selected_tile['pos']
                            self.grid[sel_r][sel_c]['state'] = 'cleared'
                            
                            # Trigger flash animation
                            self.flash_animation.extend([(r, c), (sel_r, sel_c)])
                            
                            self.selected_tile = None
                            self.pairs_cleared += 1
                            # sfx: match_success
                        else:
                            # Mismatch
                            self.selected_tile = None
                            # sfx: match_fail

        self.prev_space_held = space_held
        self.steps += 1
        
        # Check termination conditions
        if self.pairs_cleared == self.total_pairs:
            self.game_over = True
            self.win_state = True
            terminated = True
            reward += 100 # Win bonus
        elif self.moves_remaining <= 0 and not self.game_over:
            self.game_over = True
            self.win_state = False
            terminated = True
            reward -= 100 # Loss penalty
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _draw_rounded_rect(self, surface, rect, color, radius):
        pygame.draw.rect(surface, color, rect, border_radius=radius)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render grid and numbers
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile = self.grid[r][c]
                rect = pygame.Rect(
                    self.GRID_MARGIN_X + c * self.CELL_SIZE,
                    self.GRID_MARGIN_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                inner_rect = rect.inflate(-8, -8)

                # Draw grid cell background
                self._draw_rounded_rect(self.screen, inner_rect, self.COLOR_GRID, 8)

                if (r, c) in self.flash_animation:
                    self._draw_rounded_rect(self.screen, inner_rect, self.COLOR_FLASH, 8)
                elif tile['state'] == 'visible':
                    # Draw number
                    num_color = self.NUMBER_PALETTE[(tile['value'] - 1) % len(self.NUMBER_PALETTE)]
                    text_surf = self.font_main.render(str(tile['value']), True, num_color)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)

        # Draw selected tile highlight
        if self.selected_tile:
            r, c = self.selected_tile['pos']
            rect = pygame.Rect(
                self.GRID_MARGIN_X + c * self.CELL_SIZE,
                self.GRID_MARGIN_Y + r * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            ).inflate(-2, -2)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 4, border_radius=10)

        # Draw cursor
        r, c = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_MARGIN_X + c * self.CELL_SIZE,
            self.GRID_MARGIN_Y + r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        ).inflate(-2, -2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 4, border_radius=10)
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Moves remaining
        moves_text = f"Moves: {self.moves_remaining}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (self.SCREEN_WIDTH - moves_surf.get_width() - 20, 20))
        
        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 20))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win_state:
                msg_text = "YOU WIN!"
                msg_color = self.COLOR_WIN
            else:
                msg_text = "GAME OVER"
                msg_color = self.COLOR_LOSE
            
            msg_surf = self.font_game_over.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "pairs_cleared": self.pairs_cleared,
            "total_pairs": self.total_pairs,
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

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a different screen for display if running manually
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Number Match Puzzle")
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op
    
    print(env.user_guide)
    
    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Space
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward}, Score: {info['score']}, Moves: {info['moves_remaining']}")
            
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human playability
        
    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)
    env.close()