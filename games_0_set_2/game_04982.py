
# Generated: 2025-08-28T03:38:38.913930
# Source Brief: brief_04982.md
# Brief Index: 4982

        
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
        "Controls: ↑↓←→ to change direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a growing snake to eat food and power-ups, avoiding collisions in a race against time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 50

        # --- Colors ---
        self.COLOR_BG = (25, 25, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_SNAKE_HEAD = (0, 255, 100)
        self.COLOR_SNAKE_BODY = (0, 200, 80)
        self.COLOR_SNAKE_OUTLINE = (150, 255, 150)
        self.COLOR_FOOD = (255, 50, 50)
        self.COLOR_POWERUP = (50, 150, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

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
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            self.font_large = pygame.font.SysFont("monospace", 24)
            self.font_small = pygame.font.SysFont("monospace", 16)

        # --- Game State ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_body = deque()
        self.snake_direction = (0, 0)
        self.food_pos = (0, 0)
        self.powerup_pos = None
        self.powerup_active = False
        self.powerup_spawn_timer = 0
        self.powerup_duration_timer = 0
        self.last_dist_to_food = float('inf')
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        # --- Snake ---
        start_x, start_y = self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2
        self.snake_body = deque([(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)])
        self.snake_direction = (1, 0)  # Start moving right

        # --- Food & Power-ups ---
        self._spawn_food()
        self.powerup_pos = None
        self.powerup_active = False
        self.powerup_spawn_timer = self.np_random.integers(100, 200)
        self.powerup_duration_timer = 0

        # --- Reward State ---
        head_pos = self.snake_body[0]
        self.last_dist_to_food = self._manhattan_distance(head_pos, self.food_pos)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean (unused)
        shift_held = action[2] == 1  # Boolean (unused)
        
        reward = 0
        terminated = False
        
        # --- Update Direction ---
        new_direction = self.snake_direction
        if movement == 1 and self.snake_direction != (0, 1):  # Up
            new_direction = (0, -1)
        elif movement == 2 and self.snake_direction != (0, -1):  # Down
            new_direction = (0, 1)
        elif movement == 3 and self.snake_direction != (1, 0):  # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.snake_direction != (-1, 0):  # Right
            new_direction = (1, 0)
        # movement == 0 is a no-op for direction change, snake continues in current direction
        self.snake_direction = new_direction

        # --- Move Snake ---
        head_x, head_y = self.snake_body[0]
        dir_x, dir_y = self.snake_direction
        new_head = (head_x + dir_x, head_y + dir_y)

        # --- Check Collisions ---
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            self.game_over = True
            reward = -100  # Terminal penalty for wall collision
            # sfx: wall_thud
        elif new_head in list(self.snake_body)[:-1]:
            self.game_over = True
            reward = -100  # Terminal penalty for self collision
            # sfx: self_bite
        
        if self.game_over:
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Update Snake Body ---
        self.snake_body.appendleft(new_head)

        # --- Check Food Consumption ---
        ate_food = False
        if new_head == self.food_pos:
            ate_food = True
            if self.powerup_active:
                self.score += 20
                reward += 20
                self.powerup_active = False
                # sfx: powerup_eat
            else:
                self.score += 10
                reward += 10
                # sfx: eat_food
            self._spawn_food()
        else:
            self.snake_body.pop() # Remove tail if no food eaten

        # --- Check Power-up Consumption ---
        if self.powerup_pos and new_head == self.powerup_pos:
            self.powerup_active = True
            self.powerup_pos = None
            self.powerup_duration_timer = 0
            # sfx: powerup_get

        # --- Update Power-up Timers ---
        if self.powerup_pos:
            self.powerup_duration_timer -= 1
            if self.powerup_duration_timer <= 0:
                self.powerup_pos = None
                self.powerup_spawn_timer = self.np_random.integers(100, 200)
        else:
            self.powerup_spawn_timer -= 1
            if self.powerup_spawn_timer <= 0 and not self.powerup_pos:
                self._spawn_powerup()
                self.powerup_duration_timer = self.np_random.integers(50, 80)

        # --- Calculate Distance-based Reward ---
        current_dist_to_food = self._manhattan_distance(new_head, self.food_pos)
        if not ate_food:
            if current_dist_to_food < self.last_dist_to_food:
                reward += 1
            else:
                reward -= 1
        self.last_dist_to_food = current_dist_to_food

        # --- Update Step and Check Termination ---
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            terminated = True
            reward += 100 # Win bonus
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
            "powerup_active": self.powerup_active,
        }

    def _spawn_food(self):
        while True:
            pos = (self.np_random.integers(0, self.GRID_WIDTH),
                   self.np_random.integers(0, self.GRID_HEIGHT))
            if pos not in self.snake_body and pos != self.powerup_pos:
                self.food_pos = pos
                break

    def _spawn_powerup(self):
        for _ in range(50):
            pos = (self.np_random.integers(0, self.GRID_WIDTH),
                   self.np_random.integers(0, self.GRID_HEIGHT))
            if pos not in self.snake_body and pos != self.food_pos:
                self.powerup_pos = pos
                return
        self.powerup_pos = None

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _render_game(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        food_rect = pygame.Rect(self.food_pos[0] * self.GRID_SIZE, self.food_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_FOOD, food_rect.inflate(-4, -4))

        if self.powerup_pos:
            powerup_rect = pygame.Rect(self.powerup_pos[0] * self.GRID_SIZE, self.powerup_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_POWERUP, powerup_rect.inflate(-4, -4))
            px, py = powerup_rect.center
            pygame.draw.line(self.screen, self.COLOR_TEXT, (px - 4, py - 4), (px, py), 2)
            pygame.draw.line(self.screen, self.COLOR_TEXT, (px - 4, py), (px, py - 4), 2)
            pygame.draw.line(self.screen, self.COLOR_TEXT, (px + 1, py + 4), (px + 5, py), 2)
            pygame.draw.line(self.screen, self.COLOR_TEXT, (px + 1, py), (px + 5, py + 4), 2)

        for i, segment in enumerate(self.snake_body):
            rect = pygame.Rect(segment[0] * self.GRID_SIZE, segment[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            if i == 0:
                pygame.draw.rect(self.screen, self.COLOR_SNAKE_OUTLINE, rect)
                pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, rect.inflate(-4, -4))
            else:
                pygame.draw.rect(self.screen, self.COLOR_SNAKE_BODY, rect.inflate(-2, -2))
                
        if self.powerup_active:
            head_rect = pygame.Rect(self.snake_body[0][0] * self.GRID_SIZE, self.snake_body[0][1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            center = head_rect.center
            radius = self.GRID_SIZE
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            for i in range(5):
                alpha = 80 - i * 15
                color = (*self.COLOR_POWERUP, alpha)
                pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius - i*2, color)
            self.screen.blit(temp_surf, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        def draw_text(text, font, color, shadow_color, x, y, align="left"):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            text_rect = text_surf.get_rect()
            if align == "left": text_rect.topleft = (x, y)
            elif align == "right": text_rect.topright = (x, y)
            elif align == "center": text_rect.center = (x, y)
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)

        draw_text(f"SCORE: {self.score}", self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, 10, 10)
        steps_left = max(0, self.MAX_STEPS - self.steps)
        draw_text(f"STEPS: {steps_left}", self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, self.WIDTH - 10, 10, align="right")

        if self.game_over:
            message = "GAME OVER"
            if self.score >= self.WIN_SCORE: message = "YOU WIN!"
            draw_text(message, self.font_large, self.COLOR_FOOD, self.COLOR_TEXT_SHADOW, self.WIDTH // 2, self.HEIGHT // 2 - 20, align="center")
            draw_text(f"Final Score: {self.score}", self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, self.WIDTH // 2, self.HEIGHT // 2 + 20, align="center")

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gymnasium Snake")
    running = True
    terminated = False
    
    GAME_TICK = pygame.USEREVENT + 1
    pygame.time.set_timer(GAME_TICK, 120)
    
    current_movement = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if not terminated:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: current_movement = 1
                    elif event.key == pygame.K_DOWN: current_movement = 2
                    elif event.key == pygame.K_LEFT: current_movement = 3
                    elif event.key == pygame.K_RIGHT: current_movement = 4
                
                if event.type == GAME_TICK:
                    action = [current_movement, 0, 0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated:
                        print(f"Game Over! Final Info: {info}")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            current_movement = 0

    env.close()