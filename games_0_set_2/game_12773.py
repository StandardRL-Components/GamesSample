import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:43:44.790687
# Source Brief: brief_02773.md
# Brief Index: 2773
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls two mirrored rectangles.
    The goal is to survive for 45 seconds by dodging 20 moving obstacles.
    Colliding with an obstacle or going out of bounds ends the game.
    The two player-controlled rectangles transform into circles for a short
    duration upon colliding with each other.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) - No effect
    - actions[2]: Shift button (0=released, 1=held) - No effect

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +0.01 per step for survival.
    - -1.0 for player-player collision (transformation).
    - +100 for winning (surviving 45 seconds).
    - -100 for losing (obstacle collision or out of bounds).
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Control two mirrored rectangles and survive for 45 seconds by dodging moving obstacles. "
        "Colliding with your mirrored self temporarily transforms you into circles."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to move the rectangles. Dodge all the red obstacles to win."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_TIME_SECONDS = 45.0

    # Colors
    COLOR_BG = pygame.Color("#1a1a2e")
    COLOR_BOUNDARY = pygame.Color("#f1f2f6")
    COLOR_PLAYER_1 = pygame.Color("#00a8ff") # Blue
    COLOR_PLAYER_2 = pygame.Color("#4caf50") # Green
    COLOR_OBSTACLE = pygame.Color("#e94560") # Red
    COLOR_TEXT = pygame.Color("#f1f2f6")

    # Player settings
    PLAYER_WIDTH, PLAYER_HEIGHT = 20, 60
    PLAYER_SPEED = 6.0
    PLAYER_Y_POS = HEIGHT - PLAYER_HEIGHT - 10

    # Obstacle settings
    NUM_OBSTACLES = 20
    OBSTACLE_RADIUS = 8
    OBSTACLE_MIN_SPEED = 1.0
    OBSTACLE_MAX_SPEED = 2.5

    # Transformation settings
    TRANSFORMATION_DURATION = 0.5 * FPS # 0.5 seconds in steps
    TRANSFORMATION_MAX_RADIUS = (PLAYER_WIDTH + PLAYER_HEIGHT) / 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.terminated = False
        self.remaining_time = 0.0

        self.player1_rect = None
        self.player2_rect = None
        self.obstacles = []

        self.is_transformed = False
        self.transformation_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0.0
        self.terminated = False
        self.remaining_time = self.MAX_TIME_SECONDS

        # Player state
        self.player1_rect = pygame.Rect(
            self.WIDTH / 4 - self.PLAYER_WIDTH / 2, self.PLAYER_Y_POS,
            self.PLAYER_WIDTH, self.PLAYER_HEIGHT
        )
        self.player2_rect = pygame.Rect(
            3 * self.WIDTH / 4 - self.PLAYER_WIDTH / 2, self.PLAYER_Y_POS,
            self.PLAYER_WIDTH, self.PLAYER_HEIGHT
        )

        # Transformation state
        self.is_transformed = False
        self.transformation_timer = 0

        # Obstacle state
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            self._spawn_obstacle()
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.01  # Survival reward

        # --- Update Game Logic ---
        self._handle_input(action)
        self._update_obstacles()
        self._update_transformation()
        
        self.remaining_time -= 1.0 / self.FPS
        
        # --- Check Collisions and Termination ---
        collision_reward, terminated_by_collision = self._check_collisions()
        reward += collision_reward

        out_of_bounds = (
            self.player1_rect.left < 1 or self.player1_rect.right > self.WIDTH - 1 or
            self.player2_rect.left < 1 or self.player2_rect.right > self.WIDTH - 1
        )
        
        victory = self.remaining_time <= 0
        
        self.terminated = terminated_by_collision or out_of_bounds or victory
        
        if terminated_by_collision or out_of_bounds:
            reward = -100.0 # Loss penalty
        elif victory:
            reward = 100.0 # Victory bonus
            
        self.score += reward

        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _spawn_obstacle(self):
        # Spawn away from the initial player area
        while True:
            pos = pygame.math.Vector2(
                self.np_random.uniform(self.OBSTACLE_RADIUS, self.WIDTH - self.OBSTACLE_RADIUS),
                self.np_random.uniform(self.OBSTACLE_RADIUS, self.HEIGHT - self.PLAYER_HEIGHT - 50)
            )
            # Ensure it's not too close to other obstacles on spawn
            if not any(pos.distance_to(o['pos']) < self.OBSTACLE_RADIUS * 4 for o in self.obstacles):
                break
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(self.OBSTACLE_MIN_SPEED, self.OBSTACLE_MAX_SPEED)
        vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        
        self.obstacles.append({'pos': pos, 'vel': vel})

    def _handle_input(self, action):
        movement = action[0]

        # Mirrored horizontal movement
        if movement == 3:  # Left
            self.player1_rect.x -= self.PLAYER_SPEED
            self.player2_rect.x += self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player1_rect.x += self.PLAYER_SPEED
            self.player2_rect.x -= self.PLAYER_SPEED
        
        # Clamp player positions to stay within bounds
        self.player1_rect.left = max(1, self.player1_rect.left)
        self.player1_rect.right = min(self.WIDTH - 1, self.player1_rect.right)
        self.player2_rect.left = max(1, self.player2_rect.left)
        self.player2_rect.right = min(self.WIDTH - 1, self.player2_rect.right)

    def _update_obstacles(self):
        for o in self.obstacles:
            o['pos'] += o['vel']
            # Boundary bouncing
            if o['pos'].x <= self.OBSTACLE_RADIUS or o['pos'].x >= self.WIDTH - self.OBSTACLE_RADIUS:
                o['vel'].x *= -1
            if o['pos'].y <= self.OBSTACLE_RADIUS or o['pos'].y >= self.HEIGHT - self.OBSTACLE_RADIUS:
                o['vel'].y *= -1
            o['pos'].x = np.clip(o['pos'].x, self.OBSTACLE_RADIUS, self.WIDTH - self.OBSTACLE_RADIUS)
            o['pos'].y = np.clip(o['pos'].y, self.OBSTACLE_RADIUS, self.HEIGHT - self.OBSTACLE_RADIUS)

    def _update_transformation(self):
        if self.is_transformed:
            self.transformation_timer -= 1
            if self.transformation_timer <= 0:
                self.is_transformed = False

    def _check_collisions(self):
        reward = 0.0
        terminated = False

        # Player-Player collision
        if not self.is_transformed and self.player1_rect.colliderect(self.player2_rect):
            self.is_transformed = True
            self.transformation_timer = self.TRANSFORMATION_DURATION
            reward = -1.0
        
        # Player-Obstacle collision
        players = [self.player1_rect, self.player2_rect]
        for player_rect in players:
            for o in self.obstacles:
                if self._rect_circle_collide(player_rect, o['pos'], self.OBSTACLE_RADIUS):
                    terminated = True
                    break
            if terminated:
                break
        
        return reward, terminated
    
    def _rect_circle_collide(self, rect, circle_pos, circle_radius):
        # Find the closest point on the rect to the circle's center
        closest_x = np.clip(circle_pos.x, rect.left, rect.right)
        closest_y = np.clip(circle_pos.y, rect.top, rect.bottom)
        
        distance_x = circle_pos.x - closest_x
        distance_y = circle_pos.y - closest_y
        
        # Check if the distance is less than the radius
        return (distance_x**2 + distance_y**2) < (circle_radius**2)

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
        # Draw boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Draw obstacles
        for o in self.obstacles:
            pos = (int(o['pos'].x), int(o['pos'].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.OBSTACLE_RADIUS, self.COLOR_OBSTACLE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.OBSTACLE_RADIUS, self.COLOR_OBSTACLE)

        # Draw players
        if self.is_transformed:
            self._render_transformed_players()
        else:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_1, self.player1_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_2, self.player2_rect, border_radius=3)

    def _render_transformed_players(self):
        # Animation: expand then shrink
        progress = self.transformation_timer / self.TRANSFORMATION_DURATION
        # Use a parabolic curve for smooth expansion and contraction
        radius_progress = -4 * (progress - 0.5)**2 + 1
        radius = int(self.TRANSFORMATION_MAX_RADIUS * radius_progress)
        
        # Player 1
        p1_center = (int(self.player1_rect.centerx), int(self.player1_rect.centery))
        pygame.gfxdraw.aacircle(self.screen, p1_center[0], p1_center[1], max(0, radius), self.COLOR_PLAYER_1)
        pygame.gfxdraw.filled_circle(self.screen, p1_center[0], p1_center[1], max(0, radius), self.COLOR_PLAYER_1)
        
        # Player 2
        p2_center = (int(self.player2_rect.centerx), int(self.player2_rect.centery))
        pygame.gfxdraw.aacircle(self.screen, p2_center[0], p2_center[1], max(0, radius), self.COLOR_PLAYER_2)
        pygame.gfxdraw.filled_circle(self.screen, p2_center[0], p2_center[1], max(0, radius), self.COLOR_PLAYER_2)


    def _render_ui(self):
        # Timer display
        time_text = f"Time: {max(0, self.remaining_time):.1f}"
        text_surface = self.font.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_time": self.remaining_time,
            "is_transformed": self.is_transformed,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    
    # For human play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.init()
    display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Mirrored Evasion")

    running = True
    total_reward = 0
    
    # Use a dictionary to track key presses for smooth human control
    keys_pressed = {
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
    }
    
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_pressed:
                    keys_pressed[event.key] = True
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
            if event.type == pygame.KEYUP:
                if event.key in keys_pressed:
                    keys_pressed[event.key] = False

        # Map pressed keys to action
        if keys_pressed[pygame.K_LEFT]:
            action[0] = 3
        elif keys_pressed[pygame.K_RIGHT]:
            action[0] = 4
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        # Pygame uses a different axis order than numpy
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(env.FPS)
        
    env.close()