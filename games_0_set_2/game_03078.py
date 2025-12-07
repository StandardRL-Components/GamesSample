
# Generated: 2025-08-27T22:18:39.216567
# Source Brief: brief_03078.md
# Brief Index: 3078

        
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
        "Controls: Use arrow keys (↑↓←→) to change the snake's direction. "
        "Try to eat the yellow orbs to grow and score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a rapidly accelerating snake to devour glowing orbs. Reach a score of 100 to win, "
        "but avoid crashing into yourself or the walls as you get faster and longer."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    GRID_SIZE = 20
    GRID_W, GRID_H = SCREEN_W // GRID_SIZE, SCREEN_H // GRID_SIZE
    MAX_STEPS = 2000
    WIN_SCORE = 100

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_SNAKE = (0, 200, 50)
    COLOR_SNAKE_HEAD = (100, 255, 150)
    COLOR_FOOD = (255, 255, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_DANGER = (255, 0, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Etc...        
        self.render_mode = render_mode
        
        # Initialize state variables
        self.validate_implementation() # Run validation before first reset
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        start_x, start_y = self.GRID_W // 4, self.GRID_H // 2
        self.snake = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ]
        
        self.direction = (1, 0)  # Moving right
        self.pending_direction = (1, 0)
        
        self.move_interval = 6 # Initial speed: moves every 6 frames
        self.frames_since_move = 0
        self.collision_flash_alpha = 0
        
        self._place_food()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0
        
        # --- Handle Input ---
        new_pending_direction = self.pending_direction
        if movement == 1: new_pending_direction = (0, -1)  # Up
        elif movement == 2: new_pending_direction = (0, 1)   # Down
        elif movement == 3: new_pending_direction = (-1, 0)  # Left
        elif movement == 4: new_pending_direction = (1, 0)   # Right
        
        # Prevent reversing direction
        if len(self.snake) > 1 and self.direction[0] != 0 and new_pending_direction[0] == -self.direction[0]:
            pass # Ignore horizontal reversal
        elif len(self.snake) > 1 and self.direction[1] != 0 and new_pending_direction[1] == -self.direction[1]:
            pass # Ignore vertical reversal
        else:
            self.pending_direction = new_pending_direction

        # --- Game Logic Update ---
        self.frames_since_move += 1
        
        if self.game_over:
            self.collision_flash_alpha = max(0, self.collision_flash_alpha - 15)
        
        # Only move the snake every `move_interval` frames
        if not self.game_over and self.frames_since_move >= self.move_interval:
            self.frames_since_move = 0
            self.direction = self.pending_direction
            
            reward = self._perform_move()

        self.steps += 1
        
        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _perform_move(self):
        reward = 0
        head = self.snake[0]
        
        dist_before = abs(head[0] - self.food_pos[0]) + abs(head[1] - self.food_pos[1])
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        dist_after = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])

        if dist_after < dist_before: reward += 0.1
        else: reward -= 0.1
            
        is_near_miss = False
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            check_pos = (new_head[0] + dx, new_head[1] + dy)
            if not (0 <= check_pos[0] < self.GRID_W and 0 <= check_pos[1] < self.GRID_H):
                is_near_miss = True; break
            if check_pos in self.snake[1:]:
                is_near_miss = True; break
        if is_near_miss: reward -= 2.0

        if not (0 <= new_head[0] < self.GRID_W and 0 <= new_head[1] < self.GRID_H):
            # sfx: crash_wall
            self.game_over = True
            self.collision_flash_alpha = 255
            return -100.0

        if new_head in self.snake:
            # sfx: crash_self
            self.game_over = True
            self.collision_flash_alpha = 255
            return -100.0
            
        self.snake.insert(0, new_head)

        if new_head == self.food_pos:
            # sfx: eat_food
            self.score += 1
            reward += 1.0
            self._place_food()
            
            if self.score > 0 and self.score % 10 == 0:
                self.move_interval = max(1, self.move_interval - 1)
                
            if self.score >= self.WIN_SCORE:
                # sfx: win_game
                self.win = True
                reward += 100.0
        else:
            self.snake.pop()
            
        return reward

    def _place_food(self):
        available_pos = []
        snake_set = set(self.snake)
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if (x, y) not in snake_set:
                    available_pos.append((x, y))
        
        if available_pos:
            idx = self.np_random.integers(0, len(available_pos))
            self.food_pos = available_pos[idx]
        else:
            self.win = True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.SCREEN_W, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_H))
        for y in range(0, self.SCREEN_H, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_W, y))

        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        food_radius = int(self.GRID_SIZE * 0.3 + pulse * self.GRID_SIZE * 0.15)
        food_center = (
            int(self.food_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2),
            int(self.food_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2)
        )
        pygame.gfxdraw.filled_circle(self.screen, food_center[0], food_center[1], food_radius, self.COLOR_FOOD)
        pygame.gfxdraw.aacircle(self.screen, food_center[0], food_center[1], food_radius, self.COLOR_FOOD)

        for i, segment in enumerate(self.snake):
            rect = pygame.Rect(
                segment[0] * self.GRID_SIZE, segment[1] * self.GRID_SIZE,
                self.GRID_SIZE, self.GRID_SIZE
            )
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE
            pygame.draw.rect(self.screen, color, rect.inflate(-4, -4), border_radius=4)
        
        if self.collision_flash_alpha > 0:
            flash_surface = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_DANGER, self.collision_flash_alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_W - score_text.get_width() - 10, 10))

        if self.game_over:
            end_text = self.font_large.render("GAME OVER", True, self.COLOR_DANGER)
            text_rect = end_text.get_rect(center=(self.SCREEN_W / 2, self.SCREEN_H / 2))
            self.screen.blit(end_text, text_rect)
        elif self.win:
            win_text = self.font_large.render("YOU WIN!", True, self.COLOR_FOOD)
            text_rect = win_text.get_rect(center=(self.SCREEN_W / 2, self.SCREEN_H / 2))
            self.screen.blit(win_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Temporarily initialize state for validation before the first reset
        self.steps = 0
        self.score = 0
        self.snake = [(0, 0)]
        self.food_pos = (1, 1)
        self.game_over = False
        self.win = False
        self.collision_flash_alpha = 0
        super().reset(seed=0) # Seed the RNG for deterministic validation

        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    import os
    # Set the display driver. "x11", "directfb", "fbcon" for linux; "windows" for windows.
    # Use "dummy" for headless execution, but you won't see the game window.
    if os.name == "posix":
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
    pygame.display.set_caption("Neon Snake")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            move_action = 0  # no-op
            if keys[pygame.K_UP]: move_action = 1
            elif keys[pygame.K_DOWN]: move_action = 2
            elif keys[pygame.K_LEFT]: move_action = 3
            elif keys[pygame.K_RIGHT]: move_action = 4
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [move_action, space_action, shift_action]
            obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()