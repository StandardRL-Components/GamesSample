
# Generated: 2025-08-28T01:11:02.326404
# Source Brief: brief_04024.md
# Brief Index: 4024

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to direct the snake. "
        "Try to eat the red pellets to grow and score points. Avoid hitting walls or your own tail."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a growing snake to devour glowing food pellets in a fast-paced race against "
        "time and your own ever-increasing length. A classic arcade game with a neon twist."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 16
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_SNAKE_BODY = (50, 255, 50)
        self.COLOR_SNAKE_GLOW = (50, 255, 50, 40)
        self.COLOR_FOOD = (255, 50, 50)
        self.COLOR_FOOD_GLOW = (255, 50, 50, 60)
        self.COLOR_TEXT = (240, 240, 240)
        
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
        try:
            self.font = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            self.font = pygame.font.SysFont("arial", 24)
            self.font_small = pygame.font.SysFont("arial", 16)
        
        # Initialize state variables
        self.snake_body = None
        self.direction = None
        self.food_pos = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initial snake state
        start_x, start_y = self.GRID_W // 2, self.GRID_H // 2
        self.snake_body = deque([
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y)
        ])
        self.direction = (1, 0)  # Start moving right

        self._place_food()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self._handle_input(movement)
        
        old_dist = self._distance_to_food()
        
        self._move_snake()
        
        new_dist = self._distance_to_food()
        
        reward = 0
        
        # Check for food consumption
        if self.snake_body[0] == self.food_pos:
            # SFX: EAT
            self.score += 1
            reward += 10.0  # Big reward for eating
            self._place_food()
        else:
            self.snake_body.pop() # Remove tail if no food was eaten

        # Reward for getting closer to food
        if new_dist < old_dist:
            reward += 0.1
        else:
            reward -= 0.15 # Penalize moving away more
        
        # Check for termination conditions
        terminated = self._check_collisions() or self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE
        self.game_over = terminated
        
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                # SFX: WIN
                reward += 100.0 # Huge reward for winning
            elif self.steps < self.MAX_STEPS: # Only penalize for collision, not timeout
                # SFX: LOSE
                reward -= 100.0 # Huge penalty for dying
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        if movement == 1 and self.direction != (0, 1):  # UP
            self.direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # DOWN
            self.direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # LEFT
            self.direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # RIGHT
            self.direction = (1, 0)
        # if movement is 0 (no-op), direction remains unchanged

    def _move_snake(self):
        head_x, head_y = self.snake_body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        self.snake_body.appendleft(new_head)
    
    def _check_collisions(self):
        head_x, head_y = self.snake_body[0]
        
        # Wall collision
        if not (0 <= head_x < self.GRID_W and 0 <= head_y < self.GRID_H):
            return True
        
        # Self collision
        for i in range(1, len(self.snake_body)):
            if self.snake_body[i] == self.snake_body[0]:
                return True
        
        return False

    def _place_food(self):
        while True:
            x = self.np_random.integers(0, self.GRID_W)
            y = self.np_random.integers(0, self.GRID_H)
            if (x, y) not in self.snake_body:
                self.food_pos = (x, y)
                return

    def _distance_to_food(self):
        if self.food_pos is None or not self.snake_body:
            return 0
        head_x, head_y = self.snake_body[0]
        food_x, food_y = self.food_pos
        return abs(head_x - food_x) + abs(head_y - food_y)

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
        # Draw Food with glow
        if self.food_pos:
            fx, fy = self.food_pos
            center_x = int((fx + 0.5) * self.GRID_SIZE)
            center_y = int((fy + 0.5) * self.GRID_SIZE)
            
            # Pulsating glow effect
            glow_radius = int(self.GRID_SIZE * (0.8 + 0.2 * math.sin(self.steps * 0.3)))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, self.COLOR_FOOD_GLOW)
            
            # Main food circle
            radius = int(self.GRID_SIZE * 0.4)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)

        # Draw Snake with glow
        for i, segment in enumerate(self.snake_body):
            sx, sy = segment
            center_x = int((sx + 0.5) * self.GRID_SIZE)
            center_y = int((sy + 0.5) * self.GRID_SIZE)
            
            # Glow effect for the whole body
            glow_radius = int(self.GRID_SIZE * 0.6)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, self.COLOR_SNAKE_GLOW)
            
            # Main body segment
            radius = int(self.GRID_SIZE * 0.45)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_SNAKE_BODY)

            # Draw eyes on the head
            if i == 0:
                eye_radius = 2
                offset_x = self.direction[1] * self.GRID_SIZE * 0.2
                offset_y = self.direction[0] * self.GRID_SIZE * 0.2
                eye1_x, eye1_y = int(center_x - offset_x), int(center_y + offset_y)
                eye2_x, eye2_y = int(center_x + offset_x), int(center_y - offset_y)
                pygame.draw.circle(self.screen, (0,0,0), (eye1_x, eye1_y), eye_radius)
                pygame.draw.circle(self.screen, (0,0,0), (eye2_x, eye2_y), eye_radius)

    def _render_ui(self):
        # Render score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render step count
        step_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        step_rect = step_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(step_text, step_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            status = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            status_text = self.font.render(status, True, self.COLOR_TEXT)
            status_rect = status_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(status_text, status_rect)

            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
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
    
    running = True
    terminated = False
    
    # --- Manual Control Setup ---
    # Map Pygame keys to MultiDiscrete actions
    # action = [movement, space, shift]
    action = [0, 0, 0] 
    
    # Pygame display setup for manual play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Snake Neon")
    clock = pygame.time.Clock()

    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                    action = [0, 0, 0]

        if not terminated:
            # --- Get Player Input ---
            keys = pygame.key.get_pressed()
            
            # Movement (only one can be active, priority order)
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            else:
                action[0] = 0 # No-op
            
            # Space and Shift (not used in this game)
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            # --- Step the Environment ---
            # In a turn-based game, we only step when an action is taken.
            # For a smoother manual play experience, we can step every few frames.
            # Here we step on every frame for simplicity.
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it.
        # Pygame uses (width, height), but our obs is (height, width, channels)
        # So we need to transpose it back for display
        frame_to_show = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_to_show)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control the frame rate for manual play
        clock.tick(10) # Snake moves 10 times per second

    env.close()