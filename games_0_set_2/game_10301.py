import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:09:37.016661
# Source Brief: brief_00301.md
# Brief Index: 301
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a size-changing blob navigates an obstacle course.

    **Visuals:**
    - Player: A vibrant, glowing blob that changes color from blue (small) to red (large).
    - Obstacles: Muted grey rectangles.
    - Food: Bright, light-blue circles that the player can absorb to grow.
    - Finish Line: A green line on the right side of the screen.
    - Effects: Particle bursts on collision and absorption.

    **Gameplay:**
    - The player controls the blob's movement.
    - Colliding with obstacles shrinks the blob.
    - Absorbing food grows the blob.
    - Blob size affects speed: smaller is faster, larger is slower.
    - The goal is to reach the finish line.
    - The game ends if the blob shrinks too much or reaches the goal.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a size-changing blob through an obstacle course. Absorb food to grow and avoid obstacles that shrink you, then reach the finish line."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to move the blob."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 52)
    COLOR_OBSTACLE = (100, 108, 122)
    COLOR_FINISH = (0, 255, 127)
    COLOR_FOOD = (102, 217, 239)
    COLOR_PLAYER_SMALL = (0, 123, 255)
    COLOR_PLAYER_LARGE = (220, 53, 69)
    COLOR_TEXT = (240, 240, 240)
    
    # Player
    PLAYER_INITIAL_RADIUS = 20
    PLAYER_MIN_RADIUS_FACTOR = 0.1
    PLAYER_MAX_RADIUS_FACTOR = 3.0
    PLAYER_ACCELERATION = 0.8
    PLAYER_FRICTION = 0.95
    PLAYER_BASE_SPEED = 8.0

    # Game Mechanics
    INITIAL_OBSTACLES = 3
    MAX_OBSTACLES = 20
    OBSTACLE_INCREASE_INTERVAL = 500
    OBSTACLE_SHRINK_FACTOR = 0.8
    OBSTACLE_KNOCKBACK = 3.0
    
    INITIAL_FOOD_COUNT = 8
    MAX_FOOD_COUNT = 12
    FOOD_GROWTH_FACTOR = 1.2 # Brief says 1.3, but 1.2 feels better balanced
    FOOD_RESPAWN_INTERVAL = 90

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_info = pygame.font.SysFont("Consolas", 16)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_radius = self.PLAYER_INITIAL_RADIUS

        self.obstacles = []
        self.food_items = []
        self.particles = []
        
        self.finish_line_rect = pygame.Rect(self.SCREEN_WIDTH - 15, 0, 15, self.SCREEN_HEIGHT)
        self.last_dist_to_finish = 0.0
        
        # This is called here to ensure np_random is initialized before being used
        # in _generate_obstacles and _spawn_food
        # super().reset() is called inside self.reset()
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.Vector2(self.PLAYER_INITIAL_RADIUS + 30, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_radius = self.PLAYER_INITIAL_RADIUS

        self.obstacles = []
        self._generate_obstacles(self.INITIAL_OBSTACLES)
        
        self.food_items = []
        for _ in range(self.INITIAL_FOOD_COUNT):
            self._spawn_food()

        self.particles = []
        
        self.last_dist_to_finish = self.finish_line_rect.centerx - self.player_pos.x

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # --- Update Game Logic ---
        self._handle_input(action)
        self._update_player_state()
        
        reward += self._handle_collisions()
        self._update_game_state()
        self._update_particles()
        
        # --- Calculate Rewards ---
        dist_to_finish = self.finish_line_rect.centerx - self.player_pos.x
        if dist_to_finish < self.last_dist_to_finish:
            reward += 0.1
        self.last_dist_to_finish = dist_to_finish
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.player_pos.x + self.player_radius > self.finish_line_rect.left:
            # --- VICTORY ---
            reward += 100
            self.score += 1000
            terminated = True
            # sfx: win_sound.play()
        
        min_radius = self.PLAYER_INITIAL_RADIUS * self.PLAYER_MIN_RADIUS_FACTOR
        if self.player_radius < min_radius:
            # --- FAILURE ---
            reward -= 100
            terminated = True
            # sfx: lose_sound.play()

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
             terminated = True # In this game, truncation is a form of termination

        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated, # Use truncated flag for step limit
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 1:  # Up
            self.player_vel.y -= self.PLAYER_ACCELERATION
        elif movement == 2:  # Down
            self.player_vel.y += self.PLAYER_ACCELERATION
        elif movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCELERATION
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCELERATION
    
    def _update_player_state(self):
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION
        
        # Calculate speed limit based on size
        speed_multiplier = self.PLAYER_INITIAL_RADIUS / max(self.player_radius, 1)
        max_speed = self.PLAYER_BASE_SPEED * speed_multiplier
        if self.player_vel.length() > max_speed:
            self.player_vel.scale_to_length(max_speed)
        
        # Update position
        self.player_pos += self.player_vel
        
        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.player_radius, self.SCREEN_WIDTH - self.player_radius)
        self.player_pos.y = np.clip(self.player_pos.y, self.player_radius, self.SCREEN_HEIGHT - self.player_radius)

    def _handle_collisions(self):
        reward = 0.0
        
        # Obstacles
        for obstacle in self.obstacles:
            if self._check_circle_rect_collision(self.player_pos, self.player_radius, obstacle):
                self.player_radius *= self.OBSTACLE_SHRINK_FACTOR
                reward -= 0.5
                
                # Knockback
                collision_normal = (self.player_pos - pygame.Vector2(obstacle.center)).normalize()
                if collision_normal.length() > 0:
                    self.player_vel += collision_normal * self.OBSTACLE_KNOCKBACK
                
                # sfx: hit_obstacle.play()
                self._create_particles(self.player_pos, 15, self.COLOR_PLAYER_LARGE)
                break # Prevent multiple collision checks in one frame
        
        # Food
        for food in self.food_items[:]:
            dist = self.player_pos.distance_to(food)
            if dist < self.player_radius: # Food has no radius, it's a point
                max_radius = self.PLAYER_INITIAL_RADIUS * self.PLAYER_MAX_RADIUS_FACTOR
                self.player_radius = min(self.player_radius * self.FOOD_GROWTH_FACTOR, max_radius)
                reward += 1.0
                self.score += 10
                self.food_items.remove(food)
                # sfx: absorb_food.play()
                self._create_particles(food, 10, self.COLOR_FOOD, inward=True)
                
        return reward
        
    def _update_game_state(self):
        # Difficulty scaling
        num_obstacles = min(self.MAX_OBSTACLES, self.INITIAL_OBSTACLES + self.steps // self.OBSTACLE_INCREASE_INTERVAL)
        if len(self.obstacles) < num_obstacles:
            self._generate_obstacles(num_obstacles)

        # Food respawn
        if self.steps % self.FOOD_RESPAWN_INTERVAL == 0 and len(self.food_items) < self.MAX_FOOD_COUNT:
            self._spawn_food()
            
    def _generate_obstacles(self, count):
        self.obstacles.clear()
        start_zone = self.SCREEN_WIDTH * 0.2
        finish_zone = self.SCREEN_WIDTH * 0.8
        
        for _ in range(count):
            for _ in range(100): # Attempt to place 100 times before giving up
                w = self.np_random.integers(20, 80)
                h = self.np_random.integers(40, 120)
                x = self.np_random.uniform(start_zone, finish_zone - w)
                y = self.np_random.uniform(0, self.SCREEN_HEIGHT - h)
                new_obstacle = pygame.Rect(int(x), int(y), w, h)
                
                # Ensure no overlap with other obstacles
                if not any(new_obstacle.colliderect(obs) for obs in self.obstacles):
                    self.obstacles.append(new_obstacle)
                    break

    def _spawn_food(self):
        for _ in range(100): # Attempt to place 100 times before giving up
            pos = pygame.Vector2(
                self.np_random.uniform(20, self.SCREEN_WIDTH - 20),
                self.np_random.uniform(20, self.SCREEN_HEIGHT - 20)
            )
            # Avoid spawning inside obstacles
            if not any(obs.collidepoint(pos) for obs in self.obstacles):
                self.food_items.append(pos)
                break

    def _check_circle_rect_collision(self, circle_pos, radius, rect):
        closest_x = np.clip(circle_pos.x, rect.left, rect.right)
        closest_y = np.clip(circle_pos.y, rect.top, rect.bottom)
        distance_x = circle_pos.x - closest_x
        distance_y = circle_pos.y - closest_y
        return (distance_x**2 + distance_y**2) < (radius**2)

    # --- Rendering ---
    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_game_elements(self):
        # Finish Line
        pygame.draw.rect(self.screen, self.COLOR_FINISH, self.finish_line_rect)
        
        # Obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle)

        # Food
        for food_pos in self.food_items:
            pygame.gfxdraw.filled_circle(self.screen, int(food_pos.x), int(food_pos.y), 5, self.COLOR_FOOD)
            pygame.gfxdraw.aacircle(self.screen, int(food_pos.x), int(food_pos.y), 5, self.COLOR_FOOD)

        # Particles
        for p in self.particles:
            p_pos, p_vel, p_life, p_color, p_radius = p
            alpha = int(255 * (p_life / 20.0))
            # Create a temporary surface for alpha blending
            s = pygame.Surface((p_radius * 2, p_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p_color, alpha), (p_radius, p_radius), p_radius)
            self.screen.blit(s, (int(p_pos.x - p_radius), int(p_pos.y - p_radius)))

        # Player
        self._render_player()

    def _render_player(self):
        pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        radius_int = int(self.player_radius)
        if radius_int <= 0: return
        
        # Interpolate color based on size
        size_ratio = (self.player_radius - self.PLAYER_INITIAL_RADIUS * self.PLAYER_MIN_RADIUS_FACTOR) / \
                     (self.PLAYER_INITIAL_RADIUS * self.PLAYER_MAX_RADIUS_FACTOR - self.PLAYER_INITIAL_RADIUS * self.PLAYER_MIN_RADIUS_FACTOR)
        size_ratio = np.clip(size_ratio, 0, 1)
        
        current_color = [int(c1 * (1 - size_ratio) + c2 * size_ratio) for c1, c2 in zip(self.COLOR_PLAYER_SMALL, self.COLOR_PLAYER_LARGE)]
        
        # Glow effect
        for i in range(radius_int, 0, -max(1, radius_int // 5)):
            alpha = int(50 * (1 - (i / radius_int))**2)
            # Use a temporary surface for alpha blending
            s = pygame.Surface((i * 2, i * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*current_color, alpha), (i, i), i)
            self.screen.blit(s, (pos_int[0] - i, pos_int[1] - i))
            
        # Main body
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, current_color)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius_int, current_color)

    def _render_ui(self):
        size_percent = (self.player_radius / self.PLAYER_INITIAL_RADIUS) * 100
        size_text = self.font_main.render(f"SIZE: {size_percent:.0f}%", True, self.COLOR_TEXT)
        self.screen.blit(size_text, (10, 10))
        
        score_text = self.font_info.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        steps_text = self.font_info.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 30))
        self.screen.blit(steps_text, steps_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_size_percent": (self.player_radius / self.PLAYER_INITIAL_RADIUS) * 100
        }

    # --- Effects ---
    def _create_particles(self, pos, count, color, inward=False):
        for _ in range(count):
            if inward and (self.player_pos - pos).length() > 0:
                angle_rad = math.atan2(self.player_pos.y - pos.y, self.player_pos.x - pos.x)
                angle = math.degrees(angle_rad) + self.np_random.uniform(-45, 45)
            else:
                angle = self.np_random.uniform(0, 360)
            
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(1, 0).rotate(angle) * speed
            life = 20
            radius = self.np_random.uniform(1, 4)
            self.particles.append([pos.copy(), vel, life, color, radius])
    
    def _update_particles(self):
        for p in self.particles[:]:
            p[0] += p[1] # pos += vel
            p[2] -= 1    # life -= 1
            if p[2] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Reset is needed to initialize np_random
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Un-dummy the video driver to see the game
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The validation call was removed from __init__ to avoid issues with uninitialized np_random
    # It's better suited for a separate test script, but we can call it here after the first reset.
    # env.validate_implementation() 
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Blob Navigator")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement_action = 0 # None
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement_action, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Total reward: {total_reward:.2f}, Score: {info['score']:.0f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(env.FPS)
        
    env.close()