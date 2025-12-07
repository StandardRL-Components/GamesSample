
# Generated: 2025-08-27T18:48:57.739404
# Source Brief: brief_01958.md
# Brief Index: 1958

        
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
    A Gymnasium environment for a classic Snake game.

    The player controls a snake on a grid, aiming to eat food pellets.
    Each pellet consumed increases the snake's length and the player's score.
    The game ends if the snake collides with the walls or its own body.
    The goal is to reach a score of 100.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to change the snake's direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a growing snake to eat food. Avoid walls and your own tail. Reach a score of 100 to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = 30
        self.GRID_HEIGHT = 18
        self.PLAY_AREA_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.PLAY_AREA_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.X_OFFSET = (self.SCREEN_WIDTH - self.PLAY_AREA_WIDTH) // 2
        self.Y_OFFSET = (self.SCREEN_HEIGHT - self.PLAY_AREA_HEIGHT) // 2
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100

        # --- Color Palette ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_WALL = (60, 60, 70)
        self.COLOR_SNAKE_BODY = (40, 200, 120)
        self.COLOR_SNAKE_HEAD = (100, 255, 180)
        self.COLOR_FOOD = (255, 80, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

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
        try:
            self.font_score = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_game_over = pygame.font.Font(pygame.font.get_default_font(), 48)
        except IOError:
            self.font_score = pygame.font.SysFont("sans", 24)
            self.font_game_over = pygame.font.SysFont("sans", 48)

        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_pos = []
        self.snake_dir = (0, 0)
        self.pending_growth = 0
        self.food_pos = (0, 0)
        self.last_dist_to_food = 0.0

        # --- Initialize state ---
        self.reset()
        
        # --- Validate implementation ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initialize snake in the center
        start_x = self.GRID_WIDTH // 2
        start_y = self.GRID_HEIGHT // 2
        self.snake_pos = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ]
        self.snake_dir = (1, 0)  # Start moving right
        self.pending_growth = 0

        self._place_food()
        
        head_pos_pixels = self._grid_to_pixels(self.snake_pos[0])
        food_pos_pixels = self._grid_to_pixels(self.food_pos)
        self.last_dist_to_food = math.dist(head_pos_pixels, food_pos_pixels)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Determine new direction ---
        new_dir = self.snake_dir
        if movement == 1 and self.snake_dir != (0, 1):    # Up
            new_dir = (0, -1)
        elif movement == 2 and self.snake_dir != (0, -1):  # Down
            new_dir = (0, 1)
        elif movement == 3 and self.snake_dir != (1, 0):   # Left
            new_dir = (-1, 0)
        elif movement == 4 and self.snake_dir != (-1, 0):  # Right
            new_dir = (1, 0)
        self.snake_dir = new_dir

        # --- Move snake ---
        head = self.snake_pos[0]
        new_head = (head[0] + self.snake_dir[0], head[1] + self.snake_dir[1])

        # --- Check for collisions ---
        reward = 0
        collision = False
        # Wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            collision = True
        # Self collision
        if new_head in self.snake_pos:
            collision = True
        
        if collision:
            self.game_over = True
            reward = -100.0 # Punishment for losing
            # Sound: game_over_sound.play()
        else:
            self.snake_pos.insert(0, new_head) # Add new head

            # --- Check for food ---
            if new_head == self.food_pos:
                self.score += 1
                self.pending_growth += 1
                reward = 1.0  # Reward for eating food
                # Sound: eat_food_sound.play()
                if self.score >= self.WIN_SCORE:
                    self.game_over = True
                    reward = 100.0 # Big reward for winning
                    # Sound: win_game_sound.play()
                else:
                    self._place_food()
            else:
                # Remove tail if not growing
                if self.pending_growth > 0:
                    self.pending_growth -= 1
                else:
                    self.snake_pos.pop()

                # Distance-based reward
                head_pos_pixels = self._grid_to_pixels(new_head)
                food_pos_pixels = self._grid_to_pixels(self.food_pos)
                current_dist = math.dist(head_pos_pixels, food_pos_pixels)
                
                if current_dist < self.last_dist_to_food:
                    reward = 0.1
                else:
                    reward = -0.1
                self.last_dist_to_food = current_dist

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _place_food(self):
        while True:
            pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if pos not in self.snake_pos:
                self.food_pos = pos
                break

    def _grid_to_pixels(self, grid_pos):
        x = self.X_OFFSET + grid_pos[0] * self.CELL_SIZE
        y = self.Y_OFFSET + grid_pos[1] * self.CELL_SIZE
        return (x, y)

    def _render_text(self, text, font, color, pos, shadow_color=None, shadow_offset=(2, 2)):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = self.X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.Y_OFFSET), (px, self.Y_OFFSET + self.PLAY_AREA_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.X_OFFSET, py), (self.X_OFFSET + self.PLAY_AREA_WIDTH, py))
        
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.X_OFFSET, self.Y_OFFSET, self.PLAY_AREA_WIDTH, self.PLAY_AREA_HEIGHT), 3)

        # Draw snake
        for i, pos in enumerate(self.snake_pos):
            px, py = self._grid_to_pixels(pos)
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            pygame.draw.rect(self.screen, color, (px, py, self.CELL_SIZE, self.CELL_SIZE))
            # Add a slight border to segments for definition
            pygame.draw.rect(self.screen, self.COLOR_BG, (px, py, self.CELL_SIZE, self.CELL_SIZE), 1)

        # Draw food
        food_px, food_py = self._grid_to_pixels(self.food_pos)
        center = (food_px + self.CELL_SIZE // 2, food_py + self.CELL_SIZE // 2)
        radius = self.CELL_SIZE // 2 - 2
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_FOOD)
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_FOOD)
    
    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        self._render_text(score_text, self.font_score, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        
        if self.game_over:
            win_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            text_surf = self.font_game_over.render(win_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            shadow_surf = self.font_game_over.render(win_text, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surf.get_rect(center=(self.SCREEN_WIDTH / 2 + 3, self.SCREEN_HEIGHT / 2 + 3))

            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_pos)
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run in a headless environment
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    pygame.display.set_caption(env.game_description)
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop for manual play
    while not done:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
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
                
                # We need to step the environment on key press for a turn-based game
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if done:
                    print(f"Game Over! Final Score: {info['score']}")

        # Render the environment's surface to the real screen
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        real_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        # If the game is over, wait for a key press to reset or quit
        if done:
            wait_for_input = True
            while wait_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_input = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            obs, info = env.reset()
                            done = False
                            wait_for_input = False
                        elif event.key == pygame.K_q:
                             wait_for_input = False

    env.close()