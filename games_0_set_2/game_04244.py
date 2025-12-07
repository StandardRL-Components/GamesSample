import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to change direction. Survive and eat food to grow."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade snake game. Grow your snake to length 20 before the time runs out, but don't hit yourself or the walls!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.GRID_SIZE

        self.WIN_LENGTH = 20
        self.MAX_STEPS = 60 * 30  # 60 seconds at 30 "frames" per action

        # --- Colors ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_SNAKE_BODY = (50, 200, 50)
        self.COLOR_SNAKE_HEAD = (100, 255, 100)
        self.COLOR_FOOD_FILL = (220, 50, 50)
        self.COLOR_FOOD_OUTLINE = (255, 100, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_WALL = (10, 10, 15)

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_body = deque()
        self.snake_direction = (0, 0)
        self.snake_length = 0
        self.food_pos = (0, 0)
        self.particles = []

        # Initialize state variables
        # self.reset() is called here to set up the initial state,
        # but we need to initialize self.np_random first.
        # super().reset() does this.
        super().reset(seed=None)
        
        # --- Self-Validation ---
        # self.validate_implementation() # This will be run after reset sets up the state
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        # Initialize snake
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = deque([(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)])
        self.snake_length = 3
        self.snake_direction = (1, 0)  # Start moving right

        # Place initial food
        self._place_food()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        # Update direction based on action, preventing reversal
        new_direction = self.snake_direction
        if movement == 1 and self.snake_direction != (0, 1):    # Up
            new_direction = (0, -1)
        elif movement == 2 and self.snake_direction != (0, -1):  # Down
            new_direction = (0, 1)
        elif movement == 3 and self.snake_direction != (1, 0):   # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.snake_direction != (-1, 0):  # Right
            new_direction = (1, 0)
        self.snake_direction = new_direction

        # Update game logic
        self.steps += 1

        # Move snake
        head = self.snake_body[0]
        new_head = (head[0] + self.snake_direction[0], head[1] + self.snake_direction[1])

        # --- Check for Termination Conditions ---
        terminated = False
        reward = 0.0

        # 1. Wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            terminated = True
            reward = -100.0
            # sfx: wall_thud
        # 2. Self collision
        elif new_head in self.snake_body:
            terminated = True
            reward = -100.0
            # sfx: self_bite

        if terminated:
            self.game_over = True
            return self._get_observation(), reward, True, False, self._get_info()

        # Update snake body
        self.snake_body.appendleft(new_head)

        # --- Check for Events ---
        # 1. Food consumption
        # The comparison must be between two tuples. self.food_pos can be a numpy array.
        if np.array_equal(new_head, self.food_pos):
            self.snake_length += 1
            self.score += 10
            reward += 1.0
            self._place_food()
            self._create_particles(new_head)
            # sfx: eat_food
        else:
            self.snake_body.pop()
            reward += 0.01  # Small survival reward

        # 2. Win condition
        if self.snake_length >= self.WIN_LENGTH:
            terminated = True
            reward += 100.0
            # sfx: win_jingle
        # 3. Time out
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -50.0
            # sfx: timeout_buzz

        self.game_over = terminated

        # Update particles
        self._update_particles()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_food(self):
        """Finds an empty spot on the grid to place the food."""
        possible_positions = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        snake_positions = set(self.snake_body)
        available_positions = list(possible_positions - snake_positions)

        if not available_positions:
            # No space left, this is a very rare win/draw condition
            self.game_over = True
        else:
            # self.np_random.choice on a list of tuples returns a numpy array,
            # which causes comparison issues. Instead, we pick an index and
            # select the tuple, ensuring self.food_pos remains a tuple.
            idx = self.np_random.integers(0, len(available_positions))
            self.food_pos = available_positions[idx]

    def _create_particles(self, pos):
        """Create a burst of particles at the food's location."""
        grid_x, grid_y = pos
        screen_x = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
        screen_y = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            lifetime = random.randint(10, 20)
            size = random.randint(3, 6)
            self.particles.append([screen_x, screen_y, dx, dy, lifetime, size])

    def _update_particles(self):
        """Update position and lifetime of particles."""
        active_particles = []
        for p in self.particles:
            p[0] += p[2]  # x += dx
            p[1] += p[3]  # y += dy
            p[4] -= 1     # lifetime -= 1
            if p[4] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), self.GRID_SIZE//4)

        # Draw food
        food_x = int(self.food_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2)
        food_y = int(self.food_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2)
        radius = int(self.GRID_SIZE / 2.5)
        pygame.gfxdraw.filled_circle(self.screen, food_x, food_y, radius, self.COLOR_FOOD_FILL)
        pygame.gfxdraw.aacircle(self.screen, food_x, food_y, radius, self.COLOR_FOOD_OUTLINE)

        # Draw snake
        for i, segment in enumerate(self.snake_body):
            rect = pygame.Rect(
                segment[0] * self.GRID_SIZE,
                segment[1] * self.GRID_SIZE,
                self.GRID_SIZE,
                self.GRID_SIZE
            )
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            # Use a smaller inner rect to create a border effect
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, inner_rect, border_radius=4)

        # Draw snake eyes
        if self.snake_body:
            head = self.snake_body[0]
            eye_dir = pygame.Vector2(self.snake_direction).rotate(90)
            eye_center = pygame.Vector2(head[0] * self.GRID_SIZE + self.GRID_SIZE/2, head[1] * self.GRID_SIZE + self.GRID_SIZE/2)
            eye_offset = 4
            eye1_pos = eye_center + eye_dir * eye_offset
            eye2_pos = eye_center - eye_dir * eye_offset
            pygame.draw.circle(self.screen, self.COLOR_BG, (int(eye1_pos.x), int(eye1_pos.y)), 2)
            pygame.draw.circle(self.screen, self.COLOR_BG, (int(eye2_pos.x), int(eye2_pos.y)), 2)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p[4] / 20))))  # Fade out
            color = (*self.COLOR_FOOD_OUTLINE, alpha)
            temp_surf = pygame.Surface((p[5], p[5]), pygame.SRCALPHA)
            temp_surf.fill(color)
            self.screen.blit(temp_surf, (int(p[0] - p[5]/2), int(p[1] - p[5]/2)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) // 30)
        time_text = self.font_large.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

        # Length
        length_text = self.font_small.render(f"LENGTH: {self.snake_length} / {self.WIN_LENGTH}", True, self.COLOR_TEXT)
        length_rect = length_text.get_rect(bottomleft=(20, self.SCREEN_HEIGHT - 10))
        self.screen.blit(length_text, length_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg = "GAME OVER"
            if self.snake_length >= self.WIN_LENGTH:
                msg = "YOU WIN!"

            game_over_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "length": self.snake_length,
            "food_pos": self.food_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to play yourself
    # Note: Human rendering is not part of the core requirement but useful for testing
    render_mode = "human" 
    
    env = GameEnv()
    # Run validation after the first reset in __init__ is complete.
    try:
        env.validate_implementation()
    except Exception as e:
        print(f"Validation failed: {e}")
        env.close()
        exit()

    if render_mode == "human":
        # Special setup for human play, not using the Gymnasium render loop
        os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS", etc.
        pygame.display.init()
        pygame.display.set_caption(env.game_description)
        human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        
        obs, info = env.reset()
        terminated = False
        
        # Game loop for human play
        while not terminated:
            action = env.action_space.sample()  # Default to random action
            action[0] = 0  # Default to no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    elif event.key == pygame.K_r:  # Reset
                        obs, info = env.reset()
                    elif event.key == pygame.K_q:  # Quit
                        terminated = True

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Draw the observation to the human-facing screen
            draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(draw_surface, (0, 0))
            pygame.display.flip()

            env.clock.tick(15)  # Limit human play speed
            
        env.close()

    else:  # Default headless run for verification
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0
        
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            if terminated or truncated:
                print(f"Episode finished in {step_count} steps. Final score: {info['score']}, Total reward: {total_reward:.2f}")
                break
        env.close()