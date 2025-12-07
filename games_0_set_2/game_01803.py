
# Generated: 2025-08-28T02:44:56.442429
# Source Brief: brief_01803.md
# Brief Index: 1803

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to change the snake's direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A modern, neon-style Snake game. Grow your snake by eating apples, but avoid colliding with yourself or the walls. Score points for eating apples and for getting closer to them."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100

        # Colors
        self.COLOR_BG = (10, 10, 42)  # Dark Blue
        self.COLOR_GRID = (42, 42, 72) # Lighter Blue
        self.COLOR_SNAKE = (0, 255, 128) # Bright Green
        self.COLOR_SNAKE_HEAD = (128, 255, 200) # Lighter Green
        self.COLOR_APPLE = (255, 0, 100) # Bright Red/Pink
        self.COLOR_TEXT = (255, 255, 255)

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
            self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font = pygame.font.Font(None, 30)

        # Initialize state variables (will be properly set in reset)
        self.snake = []
        self.direction = (0, 0)
        self.apple_pos = (0, 0)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.last_dist_to_apple = 0

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles.clear()

        # Initialize snake
        start_x, start_y = self.GRID_W // 2, self.GRID_H // 2
        self.snake = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
        self.direction = (1, 0)  # Start moving right

        # Place apple
        self._place_apple()
        
        # Calculate initial distance for reward
        head_pos = self.snake[0]
        self.last_dist_to_apple = abs(head_pos[0] - self.apple_pos[0]) + abs(head_pos[1] - self.apple_pos[1])

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update direction based on action, preventing reversal
        new_direction = self.direction
        if movement == 1 and self.direction != (0, 1):   # Up
            new_direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1): # Down
            new_direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0): # Right
            new_direction = (1, 0)
        self.direction = new_direction

        # Update game logic
        self.steps += 1
        
        # Move snake
        head_pos = self.snake[0]
        new_head = (head_pos[0] + self.direction[0], head_pos[1] + self.direction[1])
        
        # Calculate distance-based reward
        current_dist = abs(new_head[0] - self.apple_pos[0]) + abs(new_head[1] - self.apple_pos[1])
        reward = 0
        if current_dist < self.last_dist_to_apple:
            reward = 1.0  # Moved closer
        else:
            reward = -1.0 # Moved away or same
        self.last_dist_to_apple = current_dist

        # Check for apple consumption
        ate_apple = new_head == self.apple_pos
        if ate_apple:
            self.score += 10
            reward = 10.0  # Override distance reward
            self._create_particles(self.apple_pos)
            # Apple will be replaced, snake grows by not popping tail
        else:
            self.snake.pop()

        self.snake.insert(0, new_head)
        
        # Check termination conditions
        terminated = False
        head_x, head_y = new_head
        
        # 1. Wall collision
        if not (0 <= head_x < self.GRID_W and 0 <= head_y < self.GRID_H):
            terminated = True
            reward = -50.0
        # 2. Self collision
        elif new_head in self.snake[1:]:
            terminated = True
            reward = -50.0
        # 3. Win condition
        elif self.score >= self.WIN_SCORE:
            terminated = True
            reward = 100.0
        # 4. Max steps
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated:
            self.game_over = True
        
        if ate_apple and not self.game_over:
            self._place_apple()
            # Update last_dist_to_apple for the new apple
            self.last_dist_to_apple = abs(self.snake[0][0] - self.apple_pos[0]) + abs(self.snake[0][1] - self.apple_pos[1])

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _place_apple(self):
        while True:
            self.apple_pos = (
                self.np_random.integers(0, self.GRID_W),
                self.np_random.integers(0, self.GRID_H)
            )
            if self.apple_pos not in self.snake:
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
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Update and draw particles
        for p in self.particles[:]:
            p.update()
            if p.is_dead():
                self.particles.remove(p)
            else:
                p.draw(self.screen)

        # Draw apple with pulsating glow
        apple_center_x = int(self.apple_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2)
        apple_center_y = int(self.apple_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2)
        pulse = (math.sin(self.steps * 0.3) + 1) / 2  # 0 to 1
        glow_radius = int(self.GRID_SIZE * (0.6 + pulse * 0.4))
        glow_alpha = int(50 + pulse * 50)
        
        # Use gfxdraw for smooth circles
        pygame.gfxdraw.filled_circle(self.screen, apple_center_x, apple_center_y, glow_radius, (*self.COLOR_APPLE, glow_alpha))
        pygame.gfxdraw.filled_circle(self.screen, apple_center_x, apple_center_y, int(self.GRID_SIZE * 0.4), self.COLOR_APPLE)
        pygame.gfxdraw.aacircle(self.screen, apple_center_x, apple_center_y, int(self.GRID_SIZE * 0.4), self.COLOR_APPLE)

        # Draw snake
        for i, segment in enumerate(self.snake):
            rect = pygame.Rect(segment[0] * self.GRID_SIZE, segment[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE
            
            # Create a subtle glow effect for the snake
            glow_rect = rect.inflate(4, 4)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*color, 30), glow_surface.get_rect(), border_radius=6)
            self.screen.blit(glow_surface, glow_rect.topleft)

            pygame.draw.rect(self.screen, color, rect.inflate(-2, -2), border_radius=4)


    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake),
            "apple_pos": self.apple_pos,
        }

    def _create_particles(self, pos):
        # sound effect placeholder: # play_apple_eat_sound()
        pixel_pos = (pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2)
        for _ in range(20):
            self.particles.append(Particle(pixel_pos, self.COLOR_APPLE, self.np_random))

    def validate_implementation(self):
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

class Particle:
    def __init__(self, pos, color, rng):
        self.x, self.y = pos
        self.color = color
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = rng.integers(15, 30)
        self.radius = rng.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.radius -= 0.1

    def is_dead(self):
        return self.lifetime <= 0 or self.radius <= 0

    def draw(self, surface):
        if self.is_dead():
            return
        alpha = int(255 * (self.lifetime / 30))
        alpha = max(0, min(255, alpha))
        
        # Use gfxdraw for antialiased circles
        pygame.gfxdraw.filled_circle(
            surface, int(self.x), int(self.y), int(self.radius), (*self.color, alpha)
        )

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run in a headless environment
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we'll use pygame directly
    pygame.display.set_caption("Neon Snake")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
        
        if not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Since auto_advance is False, we need to step on every frame
            # to see the game progress with human input.
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Draw the observation to the screen
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        # The game logic is step-based, but we add a small delay for human playability
        pygame.time.wait(100) 
        
    pygame.quit()