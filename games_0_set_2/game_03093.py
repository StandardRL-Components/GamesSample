
# Generated: 2025-08-28T07:02:05.680837
# Source Brief: brief_03093.md
# Brief Index: 3093

        
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
        "Controls: Use arrow keys to change the snake's direction. "
        "Try to eat the food to grow longer."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic arcade game. Guide the snake to eat food and grow, "
        "but avoid hitting the walls or your own tail."
    )

    # Should frames auto-advance or wait for user input?
    # This is a turn-based game, so we only advance on action.
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20  # 640/32=20, 400/20=20

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_SNAKE_HEAD = (0, 255, 128)
    COLOR_SNAKE_BODY = (0, 200, 100)
    COLOR_SNAKE_OUTLINE = (0, 150, 75)
    COLOR_FOOD = (255, 50, 50)
    COLOR_FOOD_OUTLINE = (200, 0, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_WIN_TEXT = (100, 255, 100)
    COLOR_LOSE_TEXT = (255, 100, 100)

    # Game parameters
    MAX_STEPS = 1000
    WIN_LENGTH = 20
    INITIAL_SNAKE_LENGTH = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
            self.font_large = pygame.font.SysFont("monospace", 50, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 72)

        self.game_over_reason = ""
        self.snake_body = []
        self.snake_direction = (0, 0)
        self.food_pos = (0, 0)
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_reason = ""
        
        # Initialize snake
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = [(start_x - i, start_y) for i in range(self.INITIAL_SNAKE_LENGTH)]
        self.snake_direction = (1, 0)  # Moving right
        
        self._place_food()
        
        return self._get_observation(), self._get_info()
    
    def _place_food(self):
        """Places food in a random location not occupied by the snake."""
        snake_positions = set(map(tuple, self.snake_body))
        possible_positions = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in snake_positions:
                    possible_positions.append((x, y))
        
        if not possible_positions:
            self.game_over = True
            self.game_over_reason = "PERFECT GAME!"
            self.food_pos = (-1, -1) # No valid position
        else:
            idx = self.np_random.integers(0, len(possible_positions))
            self.food_pos = possible_positions[idx]


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        # Determine new direction, preventing reversal
        current_dx, current_dy = self.snake_direction
        new_direction = self.snake_direction
        
        if movement == 1 and current_dy == 0: new_direction = (0, -1)  # Up
        elif movement == 2 and current_dy == 0: new_direction = (0, 1)   # Down
        elif movement == 3 and current_dx == 0: new_direction = (-1, 0)  # Left
        elif movement == 4 and current_dx == 0: new_direction = (1, 0)   # Right
        # movement == 0 is a no-op, direction remains the same

        self.snake_direction = new_direction
        
        # Calculate old distance to food for reward
        head_x, head_y = self.snake_body[0]
        old_dist = abs(head_x - self.food_pos[0]) + abs(head_y - self.food_pos[1])

        # Move snake
        new_head = (head_x + self.snake_direction[0], head_y + self.snake_direction[1])
        self.snake_body.insert(0, new_head)
        
        # Calculate new distance to food
        new_dist = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])
        
        # Check for events
        food_eaten = (new_head == self.food_pos)
        
        if food_eaten:
            self.score += 10
            # Don't pop tail, snake grows
            if len(self.snake_body) < self.GRID_WIDTH * self.GRID_HEIGHT:
                 self._place_food()
        else:
            self.snake_body.pop()

        # Check for termination
        terminated, reason = self._check_termination(new_head)
        self.game_over = terminated
        if terminated:
            self.game_over_reason = reason

        # Calculate reward
        reward = 0
        if food_eaten:
            reward += 10
        elif new_dist < old_dist:
            reward += 1
        elif new_dist > old_dist:
            reward -= 1
            
        if self.game_over and self.game_over_reason == "YOU WIN!":
            reward += 100
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated
            self._get_info()
        )

    def _check_termination(self, head):
        """Checks for wall collision, self-collision, win condition, or step limit."""
        head_x, head_y = head
        
        # Wall collision
        if not (0 <= head_x < self.GRID_WIDTH and 0 <= head_y < self.GRID_HEIGHT):
            return True, "WALL COLLISION"
            
        # Self-collision
        if head in self.snake_body[1:]:
            return True, "SELF COLLISION"
            
        # Win condition
        if len(self.snake_body) >= self.WIN_LENGTH:
            return True, "YOU WIN!"
            
        # Max steps
        if self.steps >= self.MAX_STEPS - 1:
            return True, "TIME LIMIT"
            
        return False, ""

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw food
        if self.food_pos != (-1, -1):
            food_rect = pygame.Rect(
                self.food_pos[0] * self.CELL_SIZE,
                self.food_pos[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            # Use gfxdraw for anti-aliasing
            center_x = food_rect.left + self.CELL_SIZE // 2
            center_y = food_rect.top + self.CELL_SIZE // 2
            radius = self.CELL_SIZE // 2 - 2
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_FOOD_OUTLINE)

        # Draw snake
        for i, segment in enumerate(self.snake_body):
            seg_rect = pygame.Rect(
                int(segment[0] * self.CELL_SIZE),
                int(segment[1] * self.CELL_SIZE),
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            
            if i == 0: # Head
                color = self.COLOR_SNAKE_HEAD
            else: # Body
                color = self.COLOR_SNAKE_BODY
            
            pygame.draw.rect(self.screen, color, seg_rect.inflate(-2, -2), border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_OUTLINE, seg_rect.inflate(-2, -2), 1, border_radius=4)

    def _render_ui(self):
        score_text = f"SCORE: {self.score}  LENGTH: {len(self.snake_body)}/{self.WIN_LENGTH}"
        text_surface = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        if self.game_over:
            if "WIN" in self.game_over_reason:
                color = self.COLOR_WIN_TEXT
            else:
                color = self.COLOR_LOSE_TEXT
            
            end_text_surface = self.font_large.render(self.game_over_reason, True, color)
            text_rect = end_text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
            "food_pos": self.food_pos,
            "snake_head_pos": self.snake_body[0] if self.snake_body else (0,0)
        }

    def close(self):
        pygame.font.quit()
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
    import time

    # Set up Pygame window for human play
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Gym Environment")
    
    env = GameEnv()
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    # Game loop
    running = True
    action = np.array([0, 0, 0]) # Start with no-op

    # Set a timer to trigger game steps
    GAME_UPDATE = pygame.USEREVENT + 1
    pygame.time.set_timer(GAME_UPDATE, 120) # 120ms per step

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Keydown events for changing direction
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    done = False
                    total_reward = 0
                    action = np.array([0, 0, 0])
            
            # Step the game on timer event
            if event.type == GAME_UPDATE and not done:
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                # Reset action to no-op after it has been processed for one step
                action[0] = 0

        # Rendering
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()
    print(f"Game over! Final Score: {info['score']}, Total Reward: {round(total_reward, 2)}")