
# Generated: 2025-08-28T03:39:01.364719
# Source Brief: brief_04996.md
# Brief Index: 4996

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to change the worm's direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a growing worm to devour all the food while avoiding self-collision and wall impacts."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 40
    GRID_WIDTH = 50
    GRID_HEIGHT = 50
    
    # Calculate grid dimensions to fit the screen
    CELL_SIZE = SCREEN_HEIGHT // GRID_HEIGHT
    GRID_PIXEL_WIDTH = GRID_WIDTH * CELL_SIZE
    GRID_PIXEL_HEIGHT = GRID_HEIGHT * CELL_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_PIXEL_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_PIXEL_HEIGHT) // 2
    
    MAX_STEPS = 1000
    INITIAL_FOOD_COUNT = 10
    INITIAL_WORM_LENGTH = 3

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_WORM_HEAD = (100, 255, 100)
    COLOR_WORM_BODY = (50, 205, 50)
    COLOR_FOOD = (255, 80, 80)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SCORE = (255, 223, 0)
    
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
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_score = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables
        self.worm_body = deque()
        self.food_positions = []
        self.direction = (0, 0)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.reset()
        
        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Initialize worm
        center_x, center_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.worm_body = deque()
        for i in range(self.INITIAL_WORM_LENGTH):
            self.worm_body.append((center_x, center_y + i))
        
        self.direction = (0, -1) # Start moving up
        
        # Initialize food
        self._generate_food()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        reward = 0
        terminated = False
        
        old_head = self.worm_body[0]
        dist_before = self._get_dist_to_closest_food(old_head)

        # Update direction based on action
        self._update_direction(movement)
        
        # Calculate new head position
        new_head = (old_head[0] + self.direction[0], old_head[1] + self.direction[1])
        
        # Check for termination conditions
        if self._is_collision(new_head):
            # sfx: player_die
            reward = -100
            terminated = True
            self.game_over = True
        else:
            self.worm_body.appendleft(new_head)
            
            # Check for food consumption
            if new_head in self.food_positions:
                # sfx: eat_food
                self.score += 1
                reward += 10
                self.food_positions.remove(new_head)
                
                # Check for win condition
                if not self.food_positions:
                    # sfx: win_game
                    self.win = True
                    reward += 100
                    terminated = True
                    self.game_over = True
                # No pop, worm grows
            else:
                self.worm_body.pop()

            # Calculate distance-based reward if not terminated
            if not terminated:
                dist_after = self._get_dist_to_closest_food(new_head)
                if dist_after < dist_before:
                    reward += 0.1
                elif dist_after > dist_before:
                    reward -= 0.2
        
        # Check for max steps termination
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _update_direction(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1 and self.direction != (0, 1):  # Up
            self.direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1): # Down
            self.direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # Left
            self.direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0): # Right
            self.direction = (1, 0)
        # If movement is 0 (none) or an invalid move (reversing), continue in the same direction.

    def _is_collision(self, pos):
        # Wall collision
        if not (0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT):
            return True
        # Self collision
        if pos in self.worm_body:
            return True
        return False

    def _generate_food(self):
        self.food_positions = []
        occupied_spaces = set(self.worm_body)
        while len(self.food_positions) < self.INITIAL_FOOD_COUNT:
            pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if pos not in occupied_spaces:
                self.food_positions.append(pos)
                occupied_spaces.add(pos)

    def _get_dist_to_closest_food(self, head_pos):
        if not self.food_positions:
            return 0
        
        min_dist = float('inf')
        for food_pos in self.food_positions:
            dist = abs(head_pos[0] - food_pos[0]) + abs(head_pos[1] - food_pos[1]) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_food()
        self._render_worm()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "worm_length": len(self.worm_body),
            "food_remaining": len(self.food_positions),
        }

    def _grid_to_pixel(self, grid_pos):
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        return px, py

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_PIXEL_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_PIXEL_WIDTH, py))

    def _render_worm(self):
        if not self.worm_body:
            return
        
        # Draw body
        for segment in list(self.worm_body)[1:]:
            px, py = self._grid_to_pixel(segment)
            pygame.draw.rect(self.screen, self.COLOR_WORM_BODY, (px, py, self.CELL_SIZE, self.CELL_SIZE))

        # Draw head
        head_px, head_py = self._grid_to_pixel(self.worm_body[0])
        pygame.draw.rect(self.screen, self.COLOR_WORM_HEAD, (head_px, head_py, self.CELL_SIZE, self.CELL_SIZE))

    def _render_food(self):
        radius = self.CELL_SIZE // 2
        for pos in self.food_positions:
            px, py = self._grid_to_pixel(pos)
            center_x, center_y = px + radius, py + radius
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)

    def _render_ui(self):
        # Score display
        score_text = f"SCORE: {self.score}"
        score_surface = self.font_score.render(score_text, True, self.COLOR_SCORE)
        self.screen.blit(score_surface, (15, 10))

        # Steps display
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        steps_surface = self.font_main.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(steps_surface, (15, 40))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_SCORE if self.win else self.COLOR_FOOD
            
            # Create a semi-transparent overlay
            overlay = pygame.Surface((self.GRID_PIXEL_WIDTH, 100), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
            
            # Render text on the overlay
            end_font = pygame.font.SysFont("monospace", 48, bold=True)
            end_surface = end_font.render(message, True, color)
            text_rect = end_surface.get_rect(center=(self.GRID_PIXEL_WIDTH // 2, 50))
            overlay.blit(end_surface, text_rect)
            
            # Blit the overlay onto the main screen
            screen_pos_x = self.GRID_OFFSET_X
            screen_pos_y = self.GRID_OFFSET_Y + (self.GRID_PIXEL_HEIGHT - 100) // 2
            self.screen.blit(overlay, (screen_pos_x, screen_pos_y))

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Worm Grid")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    # Game loop runs at a fixed FPS for human playability
    while running:
        action = [0, 0, 0] # Default action: no-op (continue direction)
        
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
                elif event.key == pygame.K_r and terminated:
                    obs, info = env.reset()
                    terminated = False
                elif event.key == pygame.K_q:
                    running = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control game speed for human players
        
    env.close()