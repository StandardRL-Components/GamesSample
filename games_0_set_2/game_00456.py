
# Generated: 2025-08-27T13:42:17.651458
# Source Brief: brief_00456.md
# Brief Index: 456

        
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
        "Controls: Use arrow keys to direct the snake. The goal is to eat the red food pellets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Classic arcade snake. Grow your snake by eating food, but don't crash into the walls or your own tail!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 60
    GRID_HEIGHT = 40
    CELL_SIZE = 10
    GAME_AREA_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GAME_AREA_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2
    
    MAX_STEPS = 1000
    WIN_LENGTH = 20

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 40)
    COLOR_WALL = (50, 50, 70)
    COLOR_FOOD = (255, 50, 50)
    COLOR_SNAKE_HEAD = (50, 255, 50)
    COLOR_SNAKE_TAIL = (150, 255, 150)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SCORE = (255, 215, 0)
    
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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables
        self.snake_body = []
        self.direction = (0, 0)
        self.food_pos = (0, 0)
        self.score = 0
        self.steps = 0
        self.game_over_reason = ""
        self.game_over = False
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_reason = ""
        
        # Initial snake position and direction
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ]
        self.direction = (1, 0)  # Start moving right

        self._spawn_food()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Update game logic ---
        # 1. Determine new direction
        new_direction = self.direction
        if movement == 1 and self.direction != (0, 1):  # Up
            new_direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # Down
            new_direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            new_direction = (1, 0)
        
        self.direction = new_direction
        
        # 2. Calculate new head position
        head_pos = self.snake_body[0]
        
        # No-op action does not move the snake
        if movement == 0:
            new_head_pos = head_pos
        else:
            new_head_pos = (head_pos[0] + self.direction[0], head_pos[1] + self.direction[1])

        # 3. Calculate distance-based reward
        dist_before = self._manhattan_distance(head_pos, self.food_pos)
        dist_after = self._manhattan_distance(new_head_pos, self.food_pos)
        reward += (dist_before - dist_after) # +1 if closer, -1 if further

        # 4. Check for collisions
        terminated = self._check_collision(new_head_pos)

        # 5. Check for risky moves (if not a terminal state)
        if not terminated and movement != 0:
            is_risky = False
            # Check if new head is adjacent to any body part except the neck
            for segment in self.snake_body[2:]:
                if self._manhattan_distance(new_head_pos, segment) == 1:
                    is_risky = True
                    break
            reward += -5 if is_risky else 2

        # 6. Update snake position
        ate_food = False
        if not terminated:
            # Check for food consumption
            if new_head_pos == self.food_pos:
                # sound_eat.play()
                ate_food = True
                self.score += 1
                reward += 10
                self._spawn_food()
            
            # Move snake: add new head, remove tail unless food was eaten
            if movement != 0:
                self.snake_body.insert(0, new_head_pos)
                if not ate_food:
                    self.snake_body.pop()

        # 7. Check for win/termination conditions
        if len(self.snake_body) >= self.WIN_LENGTH:
            terminated = True
            self.game_over = True
            self.game_over_reason = "YOU WIN! TARGET LENGTH REACHED."
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_over_reason = "GAME OVER: TIME LIMIT REACHED"
            reward -= 100 # Penalize for not winning
        elif terminated: # Collision happened
            self.game_over = True
            reward = -100 # Override other rewards on death

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _spawn_food(self):
        possible_locations = set(
            (x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)
        )
        snake_locations = set(self.snake_body)
        available_locations = list(possible_locations - snake_locations)
        if not available_locations:
            # No space left, this is a de-facto win/draw
            self.food_pos = (-1, -1) 
        else:
            self.food_pos = random.choice(available_locations)

    def _check_collision(self, new_head_pos):
        x, y = new_head_pos
        # Wall collision
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            # sound_crash.play()
            self.game_over_reason = "GAME OVER: HIT A WALL"
            return True
        # Self collision
        if new_head_pos in self.snake_body[1:]: # Check against body, not head
            # sound_crash.play()
            self.game_over_reason = "GAME OVER: HIT YOUR TAIL"
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
    
    def _render_game(self):
        # Draw grid background
        game_area_rect = pygame.Rect(
            self.GAME_AREA_X_OFFSET, self.GAME_AREA_Y_OFFSET,
            self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_WALL, game_area_rect)

        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.GAME_AREA_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GAME_AREA_Y_OFFSET), (px, self.GAME_AREA_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GAME_AREA_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GAME_AREA_X_OFFSET, py), (self.GAME_AREA_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, py))

        # Draw food
        if self.food_pos != (-1, -1):
            food_rect = pygame.Rect(
                self.GAME_AREA_X_OFFSET + self.food_pos[0] * self.CELL_SIZE,
                self.GAME_AREA_Y_OFFSET + self.food_pos[1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_FOOD, food_rect)
            pygame.gfxdraw.aacircle(self.screen, food_rect.centerx, food_rect.centery, self.CELL_SIZE // 2 - 1, self.COLOR_FOOD)
            pygame.gfxdraw.filled_circle(self.screen, food_rect.centerx, food_rect.centery, self.CELL_SIZE // 2 - 1, self.COLOR_FOOD)

        # Draw snake
        num_segments = len(self.snake_body)
        for i, segment in enumerate(self.snake_body):
            seg_rect = pygame.Rect(
                self.GAME_AREA_X_OFFSET + segment[0] * self.CELL_SIZE,
                self.GAME_AREA_Y_OFFSET + segment[1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            
            # Interpolate color from head to tail
            if num_segments > 1:
                fraction = i / (num_segments - 1)
                color = (
                    int(self.COLOR_SNAKE_HEAD[0] + fraction * (self.COLOR_SNAKE_TAIL[0] - self.COLOR_SNAKE_HEAD[0])),
                    int(self.COLOR_SNAKE_HEAD[1] + fraction * (self.COLOR_SNAKE_TAIL[1] - self.COLOR_SNAKE_HEAD[1])),
                    int(self.COLOR_SNAKE_HEAD[2] + fraction * (self.COLOR_SNAKE_TAIL[2] - self.COLOR_SNAKE_HEAD[2])),
                )
            else:
                color = self.COLOR_SNAKE_HEAD
            
            pygame.draw.rect(self.screen, color, seg_rect.inflate(-1, -1))

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 5))
        
        # Length display
        length_text = self.font_large.render(f"LENGTH: {len(self.snake_body)} / {self.WIN_LENGTH}", True, self.COLOR_TEXT)
        text_rect = length_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 5))
        self.screen.blit(length_text, text_rect)
        
        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            reason_text = self.font_large.render(self.game_over_reason, True, self.COLOR_TEXT)
            reason_rect = reason_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(reason_text, reason_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "length": len(self.snake_body),
            "food_pos": self.food_pos,
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
        # We need to reset first to initialize game state for rendering
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human interaction and visualization, not part of the core environment.
    # It requires a display to be available.
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Gym Environment")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        terminated = False
        
        print("--- Manual Play ---")
        print(env.user_guide)
        
        while True:
            action = [0, 0, 0] # Default action: no-op, no buttons
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                    if event.key == pygame.K_q:
                        pygame.quit()
                        quit()

            # Get key presses for continuous actions
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if terminated:
                action[0] = 0 # No movement if game is over

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Limit frame rate
            clock.tick(10) # Slower for turn-based feel

    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("Manual play requires a display. The environment itself is headless and should work.")
    finally:
        env.close()