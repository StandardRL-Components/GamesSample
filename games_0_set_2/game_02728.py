
# Generated: 2025-08-27T21:16:14.015403
# Source Brief: brief_02728.md
# Brief Index: 2728

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to change the snake's direction. "
        "Try to eat the red food to grow and score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A neon-retro Snake game. Guide the snake to eat food, but avoid "
        "colliding with the walls or its own tail. Risky moves near your "
        "own body are rewarded, but be careful!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_SCORE = 100
        self.MAX_STEPS = 1000

        # --- Colors (Neon-Retro Theme) ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_SNAKE_HEAD = (0, 255, 255)  # Bright Cyan
        self.COLOR_SNAKE_TAIL = (0, 100, 255)  # Deep Blue
        self.COLOR_FOOD = (255, 0, 80)
        self.COLOR_PARTICLE = (255, 255, 0) # Yellow
        self.COLOR_SCORE = (50, 255, 50) # Neon Green
        self.COLOR_GAMEOVER = (255, 50, 50)
        self.COLOR_WIN = (255, 255, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_score = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 60, bold=True)
        
        # --- State Variables ---
        self.snake_body = None
        self.snake_direction = None
        self.food_pos = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []

        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles.clear()

        # Initialize Snake
        start_pos = pygame.Vector2(self.GRID_W // 4, self.GRID_H // 2)
        self.snake_body = deque([
            start_pos - pygame.Vector2(2, 0),
            start_pos - pygame.Vector2(1, 0),
            start_pos
        ])
        self.snake_direction = pygame.Vector2(1, 0)  # Move right

        # Place initial food
        self._place_food()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        
        # --- Determine New Direction ---
        new_direction = self.snake_direction
        if movement == 1 and self.snake_direction.y == 0:  # Up
            new_direction = pygame.Vector2(0, -1)
        elif movement == 2 and self.snake_direction.y == 0:  # Down
            new_direction = pygame.Vector2(0, 1)
        elif movement == 3 and self.snake_direction.x == 0:  # Left
            new_direction = pygame.Vector2(-1, 0)
        elif movement == 4 and self.snake_direction.x == 0:  # Right
            new_direction = pygame.Vector2(1, 0)
        
        self.snake_direction = new_direction
        
        # --- Pre-move Reward Calculation ---
        reward = -0.1  # Step penalty
        
        head = self.snake_body[-1]
        old_dist_to_food = head.distance_to(self.food_pos)
        
        # Risky/safe move reward
        next_head_pos = head + self.snake_direction
        is_risky = False
        # Check adjacency to any body part except the neck
        for i in range(len(self.snake_body) - 2):
            if next_head_pos.distance_to(self.snake_body[i]) == 1:
                is_risky = True
                break
        reward += 5.0 if is_risky else -2.0

        # --- Update Game Logic ---
        self.steps += 1
        
        # Move snake
        self.snake_body.append(next_head_pos)
        
        # Check for food collision
        if next_head_pos == self.food_pos:
            # Sfx: food_eat.wav
            self.score += 10
            reward += 10.0
            self._create_particles(self.food_pos)
            if self.score >= self.MAX_SCORE:
                self.win = True
                reward += 100.0
            else:
                self._place_food()
        else:
            self.snake_body.popleft() # Remove tail if no food eaten

        # --- Post-move Reward & Termination Check ---
        # Distance to food reward
        new_dist_to_food = next_head_pos.distance_to(self.food_pos)
        if new_dist_to_food < old_dist_to_food:
            reward += 0.5
        else:
            reward -= 0.2

        # Check for collisions
        if (
            not (0 <= next_head_pos.x < self.GRID_W) or
            not (0 <= next_head_pos.y < self.GRID_H) or
            list(self.snake_body).count(next_head_pos) > 1
        ):
            # Sfx: game_over.wav
            self.game_over = True
            reward = -100.0

        # Update particles
        self._update_particles()
        
        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _place_food(self):
        while True:
            x = self.np_random.integers(0, self.GRID_W)
            y = self.np_random.integers(0, self.GRID_H)
            pos = pygame.Vector2(x, y)
            if pos not in self.snake_body:
                self.food_pos = pos
                break

    def _create_particles(self, pos):
        pixel_pos = (pos.x * self.GRID_SIZE + self.GRID_SIZE / 2, 
                     pos.y * self.GRID_SIZE + self.GRID_SIZE / 2)
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pixel_pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw food (pulsating)
        pulse = abs(math.sin(self.steps * 0.2))
        radius = int(self.GRID_SIZE * 0.4 + pulse * 3)
        food_px = int(self.food_pos.x * self.GRID_SIZE + self.GRID_SIZE / 2)
        food_py = int(self.food_pos.y * self.GRID_SIZE + self.GRID_SIZE / 2)
        pygame.gfxdraw.filled_circle(self.screen, food_px, food_py, radius, self.COLOR_FOOD)
        pygame.gfxdraw.aacircle(self.screen, food_px, food_py, radius, self.COLOR_FOOD)

        # Draw snake (gradient)
        num_segments = len(self.snake_body)
        for i, segment in enumerate(self.snake_body):
            # Interpolate color from head to tail
            ratio = i / max(1, num_segments - 1)
            color = (
                int(self.COLOR_SNAKE_TAIL[0] + ratio * (self.COLOR_SNAKE_HEAD[0] - self.COLOR_SNAKE_TAIL[0])),
                int(self.COLOR_SNAKE_TAIL[1] + ratio * (self.COLOR_SNAKE_HEAD[1] - self.COLOR_SNAKE_TAIL[1])),
                int(self.COLOR_SNAKE_TAIL[2] + ratio * (self.COLOR_SNAKE_HEAD[2] - self.COLOR_SNAKE_TAIL[2])),
            )
            
            rect = pygame.Rect(
                segment.x * self.GRID_SIZE,
                segment.y * self.GRID_SIZE,
                self.GRID_SIZE,
                self.GRID_SIZE
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            # Add a darker inner rect for 3D effect
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), rect.inflate(-4,-4), border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'].x), int(p['pos'].y), 
                int(p['radius'] * (p['life'] / 30.0)), color
            )

    def _render_ui(self):
        # Draw score
        score_text = self.font_score.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(score_text, score_rect)

        # Draw Game Over/Win message
        if self.game_over:
            msg_text = self.font_msg.render("GAME OVER", True, self.COLOR_GAMEOVER)
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)
        elif self.win:
            msg_text = self.font_msg.render("YOU WIN!", True, self.COLOR_WIN)
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
            "food_pos": (self.food_pos.x, self.food_pos.y)
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human interaction and visualization, not part of the core environment.
    # It requires a display to be available.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Neon Snake")
        
        obs, info = env.reset()
        terminated = False
        
        # Game loop for manual play
        while not terminated:
            movement = 0 # No-op by default
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        movement = 1
                    elif event.key == pygame.K_DOWN:
                        movement = 2
                    elif event.key == pygame.K_LEFT:
                        movement = 3
                    elif event.key == pygame.K_RIGHT:
                        movement = 4
                    elif event.key == pygame.K_r: # Reset key
                        obs, info = env.reset()
            
            # Since auto_advance is False, we must step
            action = [movement, 0, 0] # Space/Shift are not used
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation from the environment to the display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(10) # Control game speed for human play
            
    finally:
        env.close()