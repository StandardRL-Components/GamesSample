
# Generated: 2025-08-27T20:31:24.992705
# Source Brief: brief_02490.md
# Brief Index: 2490

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a classic Snake game with enhanced visuals.

    The agent controls a snake, aiming to eat food to grow longer. The episode
    ends if the snake collides with the walls or its own body, reaches a
    target length, or the step limit is exceeded. The game is designed with
    a retro-arcade aesthetic, featuring a grid, particle effects, and clear UI.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys (↑↓←→) to change the snake's direction."

    # Must be a short, user-facing description of the game:
    game_description = "Guide a growing snake to consume food and reach a target length in a vibrant, top-down arcade environment."

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
        self.WIN_LENGTH = 20

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_SNAKE_HEAD = (0, 200, 0)
        self.COLOR_SNAKE_BODY = (0, 255, 0)
        self.COLOR_FOOD = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (255, 150, 50)

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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 18)
        
        # Game state variables (will be initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_body = None
        self.snake_direction = None
        self.food_pos = None
        self.particles = []
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize snake in the center
        start_x, start_y = self.GRID_W // 2, self.GRID_H // 2
        self.snake_body = deque([
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y)
        ])
        self.snake_direction = (1, 0)  # Start moving right
        
        self.particles = []
        self._place_food()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # 1. Update direction based on action
        new_direction = self.snake_direction
        if movement == 1 and self.snake_direction != (0, 1):  # Up
            new_direction = (0, -1)
        elif movement == 2 and self.snake_direction != (0, -1):  # Down
            new_direction = (0, 1)
        elif movement == 3 and self.snake_direction != (1, 0):  # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.snake_direction != (-1, 0):  # Right
            new_direction = (1, 0)
        # movement == 0 (no-op) continues in the same direction
        self.snake_direction = new_direction

        # 2. Calculate rewards and new state
        reward = self._calculate_reward()
        
        head_pos = self.snake_body[0]
        new_head_pos = (head_pos[0] + self.snake_direction[0], head_pos[1] + self.snake_direction[1])

        # 3. Check for collisions (wall or self)
        if (new_head_pos[0] < 0 or new_head_pos[0] >= self.GRID_W or
            new_head_pos[1] < 0 or new_head_pos[1] >= self.GRID_H or
            new_head_pos in set(self.snake_body)):
            reward = -100.0
            self.game_over = True
        else:
            # 4. Move snake
            self.snake_body.appendleft(new_head_pos)
            
            # 5. Check for food consumption
            if new_head_pos == self.food_pos:
                # Sound: Nom nom
                reward += 10.0
                self.score += 10
                self._create_particles(self.food_pos)
                self._place_food()
                
                # Check for win condition
                if len(self.snake_body) >= self.WIN_LENGTH:
                    reward += 100.0
                    self.game_over = True
            else:
                self.snake_body.pop()

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
        
        terminated = self._check_termination()

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_reward(self):
        # Calculate reward based on current state before the move
        reward = -0.1  # Per-step penalty to encourage efficiency

        head_pos = self.snake_body[0]
        dist_before = self._distance_to_food(head_pos)
        
        next_head_pos = (head_pos[0] + self.snake_direction[0], head_pos[1] + self.snake_direction[1])
        dist_after = self._distance_to_food(next_head_pos)
        
        # Proximity reward
        if dist_after < dist_before:
            reward += 0.5
        else:
            reward -= 0.5
            
        return reward

    def _check_termination(self):
        return self.game_over

    def _place_food(self):
        while True:
            x = self.np_random.integers(0, self.GRID_W)
            y = self.np_random.integers(0, self.GRID_H)
            self.food_pos = (x, y)
            if self.food_pos not in self.snake_body:
                break
    
    def _distance_to_food(self, pos):
        return math.hypot(self.food_pos[0] - pos[0], self.food_pos[1] - pos[1])

    def _create_particles(self, pos):
        # Sound: Pop!
        px, py = (pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, 
                  pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2)
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifetime = self.np_random.integers(10, 20)
            self.particles.append([
                px, py, # position
                math.cos(angle) * speed, math.sin(angle) * speed, # velocity
                self.np_random.integers(2, 5), # radius
                lifetime
            ])

    def _update_and_draw_particles(self):
        remaining_particles = []
        for p in self.particles:
            p[0] += p[2]  # Update x
            p[1] += p[3]  # Update y
            p[5] -= 1     # Decrease lifetime
            if p[5] > 0:
                alpha = max(0, min(255, int(255 * (p[5] / 20.0))))
                color = (*self.COLOR_PARTICLE, alpha)
                try:
                    # Use gfxdraw for antialiased shapes
                    pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), int(p[4]), color)
                    pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), int(p[4]), color)
                except OverflowError: # Catches potential errors if particles fly too far
                    pass
                remaining_particles.append(p)
        self.particles = remaining_particles

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
        food_rect = pygame.Rect(
            self.food_pos[0] * self.GRID_SIZE,
            self.food_pos[1] * self.GRID_SIZE,
            self.GRID_SIZE,
            self.GRID_SIZE
        )
        pygame.draw.ellipse(self.screen, self.COLOR_FOOD, food_rect)
        
        # Draw snake
        # Body
        for i, segment in enumerate(list(self.snake_body)[1:]):
            seg_rect = pygame.Rect(
                segment[0] * self.GRID_SIZE,
                segment[1] * self.GRID_SIZE,
                self.GRID_SIZE,
                self.GRID_SIZE
            )
            # Fade effect for the tail
            fade_factor = (i + 1) / len(self.snake_body)
            color = (
                self.COLOR_SNAKE_BODY[0],
                int(self.COLOR_SNAKE_HEAD[1] + (self.COLOR_SNAKE_BODY[1] - self.COLOR_SNAKE_HEAD[1]) * fade_factor),
                self.COLOR_SNAKE_BODY[2]
            )
            pygame.draw.rect(self.screen, color, seg_rect.inflate(-2, -2), border_radius=4)
        
        # Head
        head = self.snake_body[0]
        head_rect = pygame.Rect(
            head[0] * self.GRID_SIZE,
            head[1] * self.GRID_SIZE,
            self.GRID_SIZE,
            self.GRID_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect.inflate(-2, -2), border_radius=6)

        # Draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        length_text = self.small_font.render(f"LENGTH: {len(self.snake_body)} / {self.WIN_LENGTH}", True, self.COLOR_TEXT)
        self.screen.blit(length_text, (10, 40))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "VICTORY!" if len(self.snake_body) >= self.WIN_LENGTH else "GAME OVER"
            end_text = self.font.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually and serves as a visual test.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Snake Gym Environment")
    
    terminated = False
    running = True
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0])  # Start with no-op

    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("Press 'R' to reset the game.")
    print("Press 'ESC' to quit.")
    print("="*30 + "\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    action = np.array([0, 0, 0])
                
                if not terminated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            # After a directional key press, reset to no-op.
            # The snake continues in the chosen direction until another key is pressed.
            action[0] = 0

        # The observation is a rendered frame; we just need to display it.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10)  # Control game speed for manual play
        
    env.close()