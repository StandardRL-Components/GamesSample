
# Generated: 2025-08-27T19:00:23.599333
# Source Brief: brief_02018.md
# Brief Index: 2018

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to direct the snake. "
        "Try to eat the green food to grow longer."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic arcade game. Guide the snake to eat food and grow. "
        "Avoid colliding with the walls or your own tail. Reach a length of 20 to win!"
    )

    # Should frames auto-advance or wait for user input?
    # This is a turn-based game, so we wait for an action.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.WIN_LENGTH = 20
        
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_SNAKE_HEAD = (0, 150, 255)
        self.COLOR_SNAKE_BODY_START = (0, 255, 255)
        self.COLOR_SNAKE_BODY_END = (0, 100, 100)
        self.COLOR_FOOD = (0, 255, 0)
        self.COLOR_FOOD_RISKY = (255, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)
        
        # Initialize state variables (will be properly set in reset)
        self.snake_body = []
        self.direction = (0, 0)
        self.food_pos = (0, 0)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.snake_length = 0
        self.prev_dist_to_food = 0
        
        # Initialize state
        self.reset()

        # Validate implementation after setup
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize snake
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
        self.snake_length = len(self.snake_body)
        self.direction = (1, 0)  # Start moving right
        
        # Place initial food
        self._place_food()
        self.prev_dist_to_food = self._manhattan_distance(self.snake_body[0], self.food_pos)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self.steps += 1
        
        # 1. Update direction based on action
        self._update_direction(movement)
        
        # 2. Move snake
        head = self.snake_body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # 3. Check for collisions (termination condition)
        if self._is_collision(new_head):
            self.game_over = True
            reward = -10  # Collision penalty
            return self._get_observation(), reward, True, False, self._get_info()

        # 4. Check for food consumption
        ate_food = new_head == self.food_pos
        
        # 5. Update snake body
        self.snake_body.insert(0, new_head)
        if not ate_food:
            self.snake_body.pop()
        else:
            self.snake_length += 1
            # Sound placeholder: pygame.mixer.Sound("eat.wav").play()
        
        # 6. Calculate reward
        reward = self._calculate_reward(ate_food)

        # 7. Handle food respawn if eaten
        if ate_food:
            self._place_food()
            self.score += 10 # Base score for food
        
        # 8. Update distance for next step's reward calculation
        self.prev_dist_to_food = self._manhattan_distance(new_head, self.food_pos)
        
        # 9. Check for win/max_steps termination
        terminated = False
        if self.snake_length >= self.WIN_LENGTH:
            reward += 100  # Win bonus
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            
        # MUST return exactly this 5-tuple
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
        elif movement == 2 and self.direction != (0, -1):  # Down
            self.direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # Left
            self.direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            self.direction = (1, 0)
        # movement == 0 (no-op) maintains current direction

    def _is_collision(self, head_pos):
        # Wall collision
        if not (0 <= head_pos[0] < self.GRID_WIDTH and 0 <= head_pos[1] < self.GRID_HEIGHT):
            # Sound placeholder: pygame.mixer.Sound("crash.wav").play()
            return True
        # Self collision
        if head_pos in self.snake_body[1:]:
            # Sound placeholder: pygame.mixer.Sound("crash.wav").play()
            return True
        return False

    def _place_food(self):
        while True:
            pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if pos not in self.snake_body:
                self.food_pos = pos
                break

    def _calculate_reward(self, ate_food):
        reward = 0
        # Event-based rewards
        if ate_food:
            reward += 10
            # Risky food bonus
            if self._manhattan_distance(self.snake_body[1], self.food_pos) <= 3:
                reward += 2
        
        # Continuous feedback rewards
        reward -= 0.01  # Penalty for each step to encourage efficiency
        
        # Distance penalty
        current_dist = self._manhattan_distance(self.snake_body[0], self.food_pos)
        if current_dist > self.prev_dist_to_food:
            reward -= 0.2 # Penalty for moving away from food
        
        return reward
        
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
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw food
        food_rect = pygame.Rect(self.food_pos[0] * self.GRID_SIZE, self.food_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        is_risky = self._manhattan_distance(self.snake_body[0], self.food_pos) <= 3
        food_color = self.COLOR_FOOD
        if is_risky and (self.steps // 5) % 2 == 0: # Flicker every 5 steps
            food_color = self.COLOR_FOOD_RISKY
        pygame.draw.rect(self.screen, food_color, food_rect)
        pygame.gfxdraw.rectangle(self.screen, food_rect, tuple(c*0.8 for c in food_color)) # Border

        # Draw snake
        for i, segment in enumerate(self.snake_body):
            rect = pygame.Rect(segment[0] * self.GRID_SIZE, segment[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            if i == 0:
                # Head
                pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, rect, border_radius=4)
            else:
                # Body with gradient
                lerp_factor = min(1.0, i / (self.snake_length * 1.5))
                color = tuple(
                    int(s + (e - s) * lerp_factor)
                    for s, e in zip(self.COLOR_SNAKE_BODY_START, self.COLOR_SNAKE_BODY_END)
                )
                pygame.draw.rect(self.screen, color, rect, border_radius=2)
                
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Length
        length_text = self.font_large.render(f"Length: {self.snake_length} / {self.WIN_LENGTH}", True, self.COLOR_TEXT)
        self.screen.blit(length_text, (self.WIDTH - length_text.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "You Won!" if self.snake_length >= self.WIN_LENGTH else "Game Over"
            text_surf = self.font_large.render(status_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "length": self.snake_length,
            "food_pos": self.food_pos,
            "head_pos": self.snake_body[0],
        }

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

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

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You can also use it to test the environment's behavior
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Must be set for pygame to run headlessly
    
    env = GameEnv()
    
    # --- Manual Play ---
    # To play manually, you need a window.
    # Comment out the `os.environ` line above and run this script.
    # Note: The gym wrapper for human play is a better way to do this.
    # This is just a simple demo.
    
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Snake Gym Environment")
    except pygame.error:
        print("Could not create display. Running in headless mode.")
        screen = None

    obs, info = env.reset()
    done = False
    
    # Game loop for manual play
    if screen:
        running = True
        action = np.array([0, 0, 0]) # Start with no-op
        
        print("\n" + "="*30)
        print("MANUAL PLAY MODE")
        print(env.user_guide)
        print("Press ESC or close window to quit.")
        print("="*30 + "\n")

        while running:
            # Convert observation to a surface and draw it
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if done:
                # Wait for a moment on the game over screen, then reset
                pygame.time.wait(2000)
                obs, info = env.reset()
                done = False
                action = np.array([0, 0, 0])

            # Get player input
            current_movement = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_UP:
                        current_movement = 1
                    elif event.key == pygame.K_DOWN:
                        current_movement = 2
                    elif event.key == pygame.K_LEFT:
                        current_movement = 3
                    elif event.key == pygame.K_RIGHT:
                        current_movement = 4
            
            # Since auto_advance is False, we only step when a key is pressed
            # or to continue in the current direction.
            # For a better play experience, we step every frame.
            if current_movement != 0:
                action[0] = current_movement
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Maintain current direction if no new key is pressed
            if current_movement == 0:
                action[0] = 0 # Next frame is a no-op unless a key is pressed
            
            env.clock.tick(10) # Control game speed for manual play
            
        env.close()

    # --- Agent Test ---
    else:
        print("\n" + "="*30)
        print("AGENT TEST MODE")
        print("="*30 + "\n")
        
        obs, info = env.reset()
        total_reward = 0
        for i in range(200):
            action = env.action_space.sample() # Random agent
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if i % 20 == 0:
                print(f"Step {i}: Action={action}, Reward={reward:.2f}, Info={info}")
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
        env.close()