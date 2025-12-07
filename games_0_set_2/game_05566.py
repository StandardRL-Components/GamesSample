
# Generated: 2025-08-28T05:24:33.265285
# Source Brief: brief_05566.md
# Brief Index: 5566

        
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
        "Controls: ↑↓←→ to change direction. Guide the snake to the yellow orbs."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Classic arcade snake. Grow longer by eating orbs, but don't crash into yourself or the walls!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100

        # Visual style
        self.COLOR_BG = (10, 10, 25)
        self.COLOR_GRID = (25, 25, 60)
        self.COLOR_SNAKE_HEAD = (80, 255, 80)
        self.COLOR_SNAKE_BODY = (60, 200, 60)
        self.COLOR_FOOD = (255, 255, 0)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_GAMEOVER = (255, 50, 50)
        self.COLOR_WIN = (50, 255, 150)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables
        self.snake_pos = []
        self.direction = (0, 0)
        self.next_direction = (0, 0)
        self.food_pos = (0, 0)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_food_dist = 0
        self.speed = 1.0 # Per brief, though unused in turn-based logic

        # This call will set up the initial game state
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.speed = 1.0

        # Initialize snake
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_pos = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
        self.direction = (1, 0)  # Moving right
        self.next_direction = (1, 0)

        # Place initial food
        self._place_food()
        
        head = self.snake_pos[0]
        self.last_food_dist = abs(head[0] - self.food_pos[0]) + abs(head[1] - self.food_pos[1])

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are ignored as per the brief's mechanics
        
        # If the game is over, do nothing
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update direction based on action, preventing 180-degree turns
        new_dir = self.next_direction
        if movement == 1 and self.direction != (0, 1): new_dir = (0, -1)  # Up
        elif movement == 2 and self.direction != (0, -1): new_dir = (0, 1)   # Down
        elif movement == 3 and self.direction != (1, 0): new_dir = (-1, 0)  # Left
        elif movement == 4 and self.direction != (-1, 0): new_dir = (1, 0)  # Right
        self.next_direction = new_dir
        self.direction = self.next_direction

        # Update game logic
        self.steps += 1

        # Move snake
        head = self.snake_pos[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Check for collisions
        # Wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            self.game_over = True
            reward = -100.0
        # Self collision
        elif new_head in self.snake_pos:
            self.game_over = True
            reward = -100.0
        else:
            # No collision, proceed with game logic
            ate_food = new_head == self.food_pos
            
            # Insert new head
            self.snake_pos.insert(0, new_head)

            if ate_food:
                # SFX: Play "eat" sound
                self.score += 10
                reward = 10.0
                
                # Update speed variable as per brief (though it doesn't affect turn-based gameplay)
                self.speed = min(2.0, 1.0 + (self.score // 20) * 0.05)

                if self.score >= self.WIN_SCORE:
                    self.win = True
                    reward = 100.0
                else:
                    self._place_food()
            else:
                # Remove tail if no food was eaten
                self.snake_pos.pop()

                # Calculate distance-based reward shaping
                dist_to_food = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])
                if dist_to_food > self.last_food_dist:
                    reward = -5.0  # Penalty for moving away from food
                else:
                    reward = 0.1  # Small reward for surviving and moving closer/same dist
                self.last_food_dist = dist_to_food

        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _place_food(self):
        possible_locations = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        snake_locations = set(self.snake_pos)
        valid_locations = list(possible_locations - snake_locations)

        if not valid_locations:
            # No space left, player wins
            self.win = True
            self.food_pos = (-1, -1) # Place food off-screen
        else:
            self.food_pos = self.np_random.choice(valid_locations)
            self.food_pos = (self.food_pos[0], self.food_pos[1])


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
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw food with a pulsating effect
        if not self.win:
            pulse = (math.sin(self.steps * 0.25) + 1) / 2.0  # Varies between 0 and 1
            min_radius = self.CELL_SIZE // 3
            max_radius = self.CELL_SIZE // 2 - 2
            radius = int(min_radius + (max_radius - min_radius) * pulse)
            fx, fy = self.food_pos
            center_x = int(fx * self.CELL_SIZE + self.CELL_SIZE / 2)
            center_y = int(fy * self.CELL_SIZE + self.CELL_SIZE / 2)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)

        # Draw snake
        for i, pos in enumerate(self.snake_pos):
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            # Use smaller rect and border radius for a more polished look
            inflated_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, inflated_rect, border_radius=4)


    def _render_ui(self):
        # Display score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Display steps
        steps_text = self.font_ui.render(f"STEPS: {self.steps}", True, self.COLOR_UI)
        self.screen.blit(steps_text, (10, 10))

        # Display game over/win message
        if self.game_over:
            msg_text = self.font_msg.render("GAME OVER", True, self.COLOR_GAMEOVER)
            text_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, text_rect)
        elif self.win:
            msg_text = self.font_msg.render("YOU WIN!", True, self.COLOR_WIN)
            text_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_pos),
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

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a persistent screen for manual play
    manual_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Manual Play - Snake")
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op
    
    print(env.user_guide)
    
    running = True
    while running:
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
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                
                if not terminated:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
                
                action[0] = 0 # Reset movement to no-op after one step
                
        # Draw the observation to the manual play screen
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(manual_screen, frame)
        pygame.display.flip()

        if terminated and running:
            print("Game Over! Press 'r' to restart or close the window.")
            
    env.close()