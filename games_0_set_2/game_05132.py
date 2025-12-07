
# Generated: 2025-08-28T04:03:49.335805
# Source Brief: brief_05132.md
# Brief Index: 5132

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to change the snake's direction. The snake moves one step per action."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a growing snake to eat glowing orbs. Reach a length of 10 to win, but don't collide with yourself!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    GRID_WIDTH = 20
    GRID_HEIGHT = 15
    CELL_SIZE = 24
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors (Neon-inspired)
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_SNAKE_BODY = (0, 255, 127) # Spring Green
    COLOR_SNAKE_HEAD = (127, 255, 212) # Aquamarine
    COLOR_FOOD = (255, 255, 0) # Yellow
    COLOR_FOOD_GLOW = (255, 255, 150)
    COLOR_TEXT = (255, 255, 255)

    MAX_STEPS = 1000
    WIN_LENGTH = 10

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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.game_area_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2,
            (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2,
            self.GRID_WIDTH * self.CELL_SIZE,
            self.GRID_HEIGHT * self.CELL_SIZE
        )

        # Initialize state variables to be defined in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_pos = []
        self.snake_direction = (0, 0)
        self.food_pos = (0, 0)
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize snake in the center
        start_x = self.GRID_WIDTH // 2
        start_y = self.GRID_HEIGHT // 2
        self.snake_pos = [(start_x, start_y)]
        self.snake_direction = (1, 0) # Start moving right
        
        self._spawn_food()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are ignored as per the brief
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Determine new direction ---
        new_direction = self.snake_direction
        # Prevent the snake from reversing on itself
        if movement == 1 and self.snake_direction != (0, 1): # Up
            new_direction = (0, -1)
        elif movement == 2 and self.snake_direction != (0, -1): # Down
            new_direction = (0, 1)
        elif movement == 3 and self.snake_direction != (1, 0): # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.snake_direction != (-1, 0): # Right
            new_direction = (1, 0)
        # If movement is 0 (no-op), keep current direction
        
        self.snake_direction = new_direction

        # --- Calculate pre-move state for reward ---
        head_pos = self.snake_pos[0]
        dist_before = abs(head_pos[0] - self.food_pos[0]) + abs(head_pos[1] - self.food_pos[1])

        # --- Update snake position with wraparound ---
        new_head_pos = (
            (head_pos[0] + self.snake_direction[0]) % self.GRID_WIDTH,
            (head_pos[1] + self.snake_direction[1]) % self.GRID_HEIGHT
        )
        
        dist_after = abs(new_head_pos[0] - self.food_pos[0]) + abs(new_head_pos[1] - self.food_pos[1])
        
        self.snake_pos.insert(0, new_head_pos)
        
        # --- Check for events and calculate reward ---
        reward = 0
        
        # 1. Self-collision
        if new_head_pos in self.snake_pos[1:]:
            self.game_over = True
            reward = -100 # Punish collision
            # sfx: game_over_sound

        # 2. Food consumption
        elif new_head_pos == self.food_pos:
            self.score += 5
            reward = 10 # Reward for eating
            self._spawn_food()
            # sfx: eat_food_sound
            # Snake grows, so we don't pop the tail
        
        # 3. Normal movement
        else:
            self.snake_pos.pop()
            # Distance-based reward
            if dist_after < dist_before:
                reward = 1 # Reward for moving closer
            else:
                reward = -1 # Punish for moving away or parallel
        
        # --- Check for win condition ---
        if len(self.snake_pos) >= self.WIN_LENGTH and not self.game_over:
            self.game_over = True
            reward = 100 # Big reward for winning
            # sfx: win_sound

        # --- Check for step limit ---
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _spawn_food(self):
        """Spawns food in a random location not occupied by the snake."""
        while True:
            x = self.np_random.integers(0, self.GRID_WIDTH)
            y = self.np_random.integers(0, self.GRID_HEIGHT)
            if (x, y) not in self.snake_pos:
                self.food_pos = (x, y)
                break

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
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = self.game_area_rect.left + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.game_area_rect.top), (px, self.game_area_rect.bottom))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.game_area_rect.top + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.game_area_rect.left, py), (self.game_area_rect.right, py))

        # Draw food with glow effect using gfxdraw for smooth circles
        food_center_x = self.game_area_rect.left + int((self.food_pos[0] + 0.5) * self.CELL_SIZE)
        food_center_y = self.game_area_rect.top + int((self.food_pos[1] + 0.5) * self.CELL_SIZE)
        glow_radius = int(self.CELL_SIZE * 0.6)
        food_radius = int(self.CELL_SIZE * 0.4)
        
        pygame.gfxdraw.filled_circle(self.screen, food_center_x, food_center_y, glow_radius, self.COLOR_FOOD_GLOW)
        pygame.gfxdraw.aacircle(self.screen, food_center_x, food_center_y, glow_radius, self.COLOR_FOOD_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, food_center_x, food_center_y, food_radius, self.COLOR_FOOD)
        pygame.gfxdraw.aacircle(self.screen, food_center_x, food_center_y, food_radius, self.COLOR_FOOD)

        # Draw snake
        for i, segment in enumerate(self.snake_pos):
            rect = pygame.Rect(
                self.game_area_rect.left + segment[0] * self.CELL_SIZE,
                self.game_area_rect.top + segment[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            
            if i == 0: # Head
                # Pulsing effect for the head
                pulse = (math.sin(self.steps * 0.5) + 1) / 2 # Varies between 0 and 1
                head_color = (
                    int(np.clip(self.COLOR_SNAKE_BODY[0] + (self.COLOR_SNAKE_HEAD[0] - self.COLOR_SNAKE_BODY[0]) * pulse, 0, 255)),
                    int(np.clip(self.COLOR_SNAKE_BODY[1] + (self.COLOR_SNAKE_HEAD[1] - self.COLOR_SNAKE_BODY[1]) * pulse, 0, 255)),
                    int(np.clip(self.COLOR_SNAKE_BODY[2] + (self.COLOR_SNAKE_HEAD[2] - self.COLOR_SNAKE_BODY[2]) * pulse, 0, 255))
                )
                pygame.draw.rect(self.screen, head_color, rect.inflate(-4, -4), border_radius=5)
            else: # Body
                pygame.draw.rect(self.screen, self.COLOR_SNAKE_BODY, rect.inflate(-4, -4), border_radius=5)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        length_text = self.font.render(f"LENGTH: {len(self.snake_pos)}/{self.WIN_LENGTH}", True, self.COLOR_TEXT)
        length_rect = length_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(length_text, length_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_pos),
            "food_pos": self.food_pos,
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

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # Set Pygame to run in a window for manual testing
    import os
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Pygame setup for display
    pygame.display.set_caption("Snake RL Environment")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    total_reward = 0
    
    print(env.user_guide)
    print(env.game_description)

    # Game loop for manual play
    while running:
        action = [0, 0, 0] # Default to no-op (continue direction)
        
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
                elif event.key == pygame.K_r: # Reset key
                    terminated = True
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            terminated = False
            total_reward = 0
            continue
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(10) # Control speed for manual play

    env.close()