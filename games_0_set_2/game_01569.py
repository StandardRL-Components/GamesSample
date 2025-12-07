
# Generated: 2025-08-28T02:00:46.196936
# Source Brief: brief_01569.md
# Brief Index: 1569

        
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
        "Controls: ↑↓←→ to change the snake's direction. The snake moves one step per action."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade Snake game. Guide the snake to eat food and grow longer. "
        "Earn points for eating and for moving towards food, but be careful! "
        "Colliding with walls or your own body ends the game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100
        self.INITIAL_SNAKE_LENGTH = 5

        # --- Colors ---
        self.COLOR_BG = (15, 15, 15) # Dark gray
        self.COLOR_GRID = (40, 40, 40)
        self.COLOR_SNAKE = (0, 200, 0)
        self.COLOR_SNAKE_HEAD = (100, 255, 100)
        self.COLOR_FOOD = (220, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_DANGER = (100, 0, 0, 150) # Semi-transparent red for flashing

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Consolas", 30)
        except pygame.error:
            self.font = pygame.font.SysFont(None, 40)
            
        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_pos = deque()
        self.snake_direction = (1, 0) # (dx, dy)
        self.food_pos = (0, 0)

        self.reset()
        
        # This is a good practice to ensure the implementation is correct
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # --- Initialize Snake ---
        start_x = self.GRID_WIDTH // 2
        start_y = self.GRID_HEIGHT // 2
        self.snake_pos = deque()
        for i in range(self.INITIAL_SNAKE_LENGTH):
            self.snake_pos.append((start_x - i, start_y))
        
        self.snake_direction = (1, 0) # Start moving right

        # --- Initialize Food ---
        self._spawn_food()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action and Update Direction ---
        movement = action[0]
        self._update_direction(movement)

        # --- Store pre-move state for reward calculation ---
        old_head = self.snake_pos[0]
        old_dist_food = self._manhattan_distance(old_head, self.food_pos)
        old_min_dist_body = self._get_min_dist_to_body(self.snake_pos)

        # --- Move Snake ---
        new_head = (old_head[0] + self.snake_direction[0], old_head[1] + self.snake_direction[1])
        self.snake_pos.appendleft(new_head)

        # --- Check for Events and Update State ---
        ate_food = False
        if new_head == self.food_pos:
            ate_food = True
            self.score += 1
            # Don't pop tail, snake grows
            self._spawn_food()
            # Placeholder for sound effect
            # sfx.play('eat')
        else:
            self.snake_pos.pop() # Pop tail to move

        # --- Termination Checks ---
        collided = self._check_collisions(new_head)
        won = self.score >= self.WIN_SCORE
        max_steps_reached = self.steps >= self.MAX_STEPS -1
        
        terminated = collided or won or max_steps_reached
        self.game_over = terminated

        # --- Calculate Reward ---
        reward = 0
        
        # Event-based rewards
        if ate_food:
            reward += 10
        if collided:
            reward -= 50
        if won:
            reward += 100

        # Continuous feedback rewards
        new_dist_food = self._manhattan_distance(new_head, self.food_pos)
        if new_dist_food < old_dist_food:
            reward += 0.1
        else:
            reward -= 0.2

        new_min_dist_body = self._get_min_dist_to_body(self.snake_pos)
        if new_min_dist_body > old_min_dist_body:
            reward += 2.0
        elif new_min_dist_body < old_min_dist_body:
            reward -= 2.0

        self.steps += 1

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
        if movement == 1 and self.snake_direction != (0, 1): # Up
            self.snake_direction = (0, -1)
        elif movement == 2 and self.snake_direction != (0, -1): # Down
            self.snake_direction = (0, 1)
        elif movement == 3 and self.snake_direction != (1, 0): # Left
            self.snake_direction = (-1, 0)
        elif movement == 4 and self.snake_direction != (-1, 0): # Right
            self.snake_direction = (1, 0)
        # If movement is 0 (no-op) or an invalid turn, direction remains the same.

    def _check_collisions(self, head):
        # Wall collision
        if not (0 <= head[0] < self.GRID_WIDTH and 0 <= head[1] < self.GRID_HEIGHT):
            # Placeholder for sound effect
            # sfx.play('hit_wall')
            return True
        # Self collision
        if head in list(self.snake_pos)[1:]:
            # Placeholder for sound effect
            # sfx.play('hit_self')
            return True
        return False

    def _spawn_food(self):
        possible_spawns = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        snake_body_set = set(self.snake_pos)
        valid_spawns = list(possible_spawns - snake_body_set)
        
        if not valid_spawns:
             # No space left, should trigger termination, but handle defensively
             self.game_over = True
             self.food_pos = (-1, -1) # Place off-screen
             return

        # Ensure food spawns reasonably far from the head initially
        if self.steps == 0:
            far_spawns = [p for p in valid_spawns if self._manhattan_distance(self.snake_pos[0], p) > 10]
            if far_spawns:
                valid_spawns = far_spawns

        self.food_pos = self.np_random.choice(valid_spawns)
        self.food_pos = (int(self.food_pos[0]), int(self.food_pos[1]))


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # --- Draw Flashing Risk Zones ---
        # Flashing effect based on steps
        if (self.steps // 5) % 2 == 0:
            danger_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            danger_surface.fill(self.COLOR_DANGER)

            # Wall risk
            for i in range(self.GRID_WIDTH):
                self.screen.blit(danger_surface, (i * self.CELL_SIZE, 0))
                self.screen.blit(danger_surface, (i * self.CELL_SIZE, (self.GRID_HEIGHT - 1) * self.CELL_SIZE))
            for i in range(1, self.GRID_HEIGHT - 1):
                self.screen.blit(danger_surface, (0, i * self.CELL_SIZE))
                self.screen.blit(danger_surface, ((self.GRID_WIDTH - 1) * self.CELL_SIZE, i * self.CELL_SIZE))

            # Snake body risk
            if len(self.snake_pos) > 0:
                head = self.snake_pos[0]
                body_set = set(list(self.snake_pos)[1:])
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    check_pos = (head[0] + dx, head[1] + dy)
                    if check_pos in body_set:
                        px, py = check_pos[0] * self.CELL_SIZE, check_pos[1] * self.CELL_SIZE
                        self.screen.blit(danger_surface, (px, py))


        # Draw snake
        for i, pos in enumerate(self.snake_pos):
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1) # Border

        # Draw food
        food_px = int((self.food_pos[0] + 0.5) * self.CELL_SIZE)
        food_py = int((self.food_pos[1] + 0.5) * self.CELL_SIZE)
        radius = int(self.CELL_SIZE / 2 * 0.8)
        pygame.gfxdraw.aacircle(self.screen, food_px, food_py, radius, self.COLOR_FOOD)
        pygame.gfxdraw.filled_circle(self.screen, food_px, food_py, radius, self.COLOR_FOOD)


    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_pos),
            "food_pos": self.food_pos,
        }

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_min_dist_to_body(self, snake_deque):
        if len(snake_deque) < 4:
            return self.GRID_WIDTH + self.GRID_HEIGHT # A large number
        
        head = snake_deque[0]
        body = list(snake_deque)[3:] # Check against parts of body that are not the neck
        
        if not body:
            return self.GRID_WIDTH + self.GRID_HEIGHT
            
        return min(self._manhattan_distance(head, part) for part in body)

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Interactive Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Override screen for display
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Snake")

    action = [0, 0, 0] # no-op, no space, no shift
    
    print(env.user_guide)

    while not terminated:
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
                
                # Since auto_advance is False, we step on key press
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
                
                # Reset action to no-op for next frame unless another key is pressed
                action[0] = 0

        # Since this is a turn-based game, we only need to redraw
        # after a step has occurred. The main loop can be slow.
        env.screen.fill(env.COLOR_BG)
        env._render_game()
        env._render_ui()
        pygame.display.flip()
        
        env.clock.tick(10) # Limit frame rate for interactive play

    print("Game Over!")
    env.close()