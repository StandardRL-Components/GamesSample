
# Generated: 2025-08-28T04:55:48.955020
# Source Brief: brief_02457.md
# Brief Index: 2457

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to change direction. Grow the snake by eating apples. Avoid hitting your own tail!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade classic. Guide a growing snake to eat apples and achieve a high score. The game ends if you collide with yourself or run out of steps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100

        # Colors
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_GRID = (40, 40, 40)
        self.COLOR_SNAKE_BODY = (50, 205, 50)
        self.COLOR_SNAKE_HEAD = (124, 252, 0)
        self.COLOR_APPLE = (255, 69, 0)
        self.COLOR_APPLE_OUTLINE = (255, 99, 71)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # State variables (will be initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake = None
        self.direction = None
        self.apple_pos = None
        self.previous_distance_to_apple = 0

        # Initialize state variables
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initialize snake
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake = deque([(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)])
        self.direction = (1, 0)  # Start moving right

        # Spawn apple
        self._spawn_apple()
        
        # Initial distance for reward calculation
        head_x, head_y = self.snake[0]
        self.previous_distance_to_apple = abs(head_x - self.apple_pos[0]) + abs(head_y - self.apple_pos[1])

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- 1. Update Direction based on Action ---
        new_direction = self.direction
        if movement == 1 and self.direction != (0, 1):  # Up
            new_direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # Down
            new_direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            new_direction = (1, 0)
        # movement == 0 is a no-op, direction remains the same
        self.direction = new_direction

        # --- 2. Update Game Logic ---
        self.steps += 1
        
        # Move snake
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = ((head_x + dx) % self.GRID_WIDTH, (head_y + dy) % self.GRID_HEIGHT)

        # Check for self-collision (before adding new head)
        if new_head in self.snake:
            self.game_over = True
            
        self.snake.appendleft(new_head)

        # Check for apple consumption
        apple_eaten = False
        if new_head == self.apple_pos:
            apple_eaten = True
            self.score += 10
            # Don't pop tail, snake grows
            self._spawn_apple()
            # Sound placeholder: pygame.mixer.Sound("eat.wav").play()
        else:
            # Pop tail if no apple was eaten
            self.snake.pop()

        # --- 3. Calculate Reward ---
        reward = self._calculate_reward(apple_eaten, self.game_over)
        
        # Update distance for next step's reward
        current_distance = abs(new_head[0] - self.apple_pos[0]) + abs(new_head[1] - self.apple_pos[1])
        self.previous_distance_to_apple = current_distance

        # --- 4. Check Termination ---
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _calculate_reward(self, apple_eaten, self_collided):
        if self.score >= self.WIN_SCORE:
            return 100.0  # Win condition
        if self_collided:
            return -100.0 # Self-collision penalty
        if self.steps >= self.MAX_STEPS:
            return -100.0 # Timeout penalty
        if apple_eaten:
            return 10.0   # Apple consumption reward
        
        # Proximity reward
        head_x, head_y = self.snake[0]
        current_distance = abs(head_x - self.apple_pos[0]) + abs(head_y - self.apple_pos[1])
        
        if current_distance < self.previous_distance_to_apple:
            return 0.1 # Moved closer to apple
        else:
            return -0.1 # Moved further or same distance
            
    def _check_termination(self):
        if self.game_over: # self.game_over is set on self-collision
            return True
        if self.score >= self.WIN_SCORE:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _spawn_apple(self):
        while True:
            x = self.np_random.integers(0, self.GRID_WIDTH)
            y = self.np_random.integers(0, self.GRID_HEIGHT)
            if (x, y) not in self.snake:
                self.apple_pos = (x, y)
                break

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
        # Draw grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw apple
        apple_x = self.apple_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2
        apple_y = self.apple_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2
        radius = self.GRID_SIZE // 2 - 2
        pygame.gfxdraw.filled_circle(self.screen, apple_x, apple_y, radius, self.COLOR_APPLE)
        pygame.gfxdraw.aacircle(self.screen, apple_x, apple_y, radius, self.COLOR_APPLE_OUTLINE)
        
        # Draw snake
        # Body
        for i, segment in enumerate(list(self.snake)[1:]):
            rect = pygame.Rect(segment[0] * self.GRID_SIZE, segment[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_BODY, rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, rect.inflate(-4, -4), border_radius=3) # Inner highlight
        # Head
        head = self.snake[0]
        head_rect = pygame.Rect(head[0] * self.GRID_SIZE, head[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect, border_radius=4)
        
        # Eyes
        eye_size = 3
        dx, dy = self.direction
        if dx == 1: # Right
            eye1 = (head_rect.centerx + 4, head_rect.centery - 4)
            eye2 = (head_rect.centerx + 4, head_rect.centery + 4)
        elif dx == -1: # Left
            eye1 = (head_rect.centerx - 4, head_rect.centery - 4)
            eye2 = (head_rect.centerx - 4, head_rect.centery + 4)
        elif dy == 1: # Down
            eye1 = (head_rect.centerx - 4, head_rect.centery + 4)
            eye2 = (head_rect.centerx + 4, head_rect.centery + 4)
        else: # Up
            eye1 = (head_rect.centerx - 4, head_rect.centery - 4)
            eye2 = (head_rect.centerx + 4, head_rect.centery - 4)
            
        pygame.draw.circle(self.screen, self.COLOR_BG, eye1, eye_size)
        pygame.draw.circle(self.screen, self.COLOR_BG, eye2, eye_size)


    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, x, y, align="topleft"):
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surf = font.render(text, True, color)
            
            shadow_rect = shadow_surf.get_rect()
            text_rect = text_surf.get_rect()

            if align == "topleft":
                shadow_rect.topleft = (x + 2, y + 2)
                text_rect.topleft = (x, y)
            elif align == "topright":
                shadow_rect.topright = (x + 2, y + 2)
                text_rect.topright = (x, y)
            elif align == "center":
                shadow_rect.center = (x + 2, y + 2)
                text_rect.center = (x, y)

            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

        # Score display
        score_text = f"SCORE: {self.score}"
        draw_text(score_text, self.font_large, self.COLOR_TEXT, 20, 10)

        # Steps display
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        draw_text(steps_text, self.font_large, self.COLOR_TEXT, self.WIDTH - 20, 10, align="topright")
        
        # Game Over message
        if self._check_termination():
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = self.COLOR_SNAKE_HEAD
            else:
                msg = "GAME OVER"
                color = self.COLOR_APPLE
            
            draw_text(msg, self.font_large, color, self.WIDTH // 2, self.HEIGHT // 2 - 20, align="center")
            final_score_text = f"Final Score: {self.score}"
            draw_text(final_score_text, self.font_small, self.COLOR_TEXT, self.WIDTH // 2, self.HEIGHT // 2 + 20, align="center")


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake),
            "apple_pos": self.apple_pos,
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
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
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Render the observation to the screen
        game_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Create a display if one doesn't exist
        try:
            display_surface = pygame.display.get_surface()
            if display_surface is None:
                raise Exception
        except Exception:
            display_surface = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            pygame.display.set_caption("Snake Gym Environment")

        display_surface.blit(game_surface, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we need a small delay for human playability
        pygame.time.wait(100) # ~10 FPS for human play

    env.close()