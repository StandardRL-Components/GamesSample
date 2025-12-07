
# Generated: 2025-08-28T06:44:56.785006
# Source Brief: brief_03023.md
# Brief Index: 3023

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper class for game entities
class Entity:
    def __init__(self, pos, size, color):
        self.pos = pygame.Vector2(pos)  # Cartesian coordinates
        self.size = size
        self.color = color
        self.iso_pos = pygame.Vector2(0, 0)
        self.depth = 0

    def update_iso_pos(self, origin):
        self.iso_pos.x = origin.x + (self.pos.x - self.pos.y) * 20
        self.iso_pos.y = origin.y + (self.pos.x + self.pos.y) * 10
        self.depth = self.pos.y

    def draw_shadow(self, surface):
        shadow_pos = (int(self.iso_pos.x), int(self.iso_pos.y + self.size * 1.5))
        pygame.gfxdraw.filled_ellipse(surface, shadow_pos[0], shadow_pos[1], int(self.size), int(self.size / 2), (0, 0, 0, 50))

    def draw(self, surface):
        # Default draw method, can be overridden
        pygame.draw.circle(surface, self.color, (int(self.iso_pos.x), int(self.iso_pos.y)), self.size)


class Player(Entity):
    def __init__(self, pos, size, color):
        super().__init__(pos, size, color)
        self.speed = 4.0
        self.bob_angle = 0

    def update(self, movement_action, bounds):
        vel = pygame.Vector2(0, 0)
        if movement_action == 1: # Up (Iso Up-Left)
            vel.y -= 1
        elif movement_action == 2: # Down (Iso Down-Right)
            vel.y += 1
        elif movement_action == 3: # Left (Iso Down-Left)
            vel.x -= 1
        elif movement_action == 4: # Right (Iso Up-Right)
            vel.x += 1
        
        if vel.length() > 0:
            vel = vel.normalize() * self.speed
        
        self.pos += vel
        
        # Clamp position to Cartesian bounds
        self.pos.x = np.clip(self.pos.x, bounds[0], bounds[1])
        self.pos.y = np.clip(self.pos.y, bounds[2], bounds[3])
        
        self.bob_angle = (self.bob_angle + 0.2) % (2 * math.pi)

    def draw(self, surface):
        bob_offset = math.sin(self.bob_angle) * 3
        draw_pos = (int(self.iso_pos.x), int(self.iso_pos.y - bob_offset))

        # Glow effect
        for i in range(self.size // 2, 0, -2):
            alpha = 80 - (i * 10)
            pygame.gfxdraw.filled_circle(surface, draw_pos[0], draw_pos[1], self.size + i, (*self.color, alpha))
        
        # Main body (isometric cube)
        points = [
            (draw_pos[0], draw_pos[1] - self.size),  # Top
            (draw_pos[0] + self.size, draw_pos[1]), # Right
            (draw_pos[0], draw_pos[1] + self.size), # Bottom
            (draw_pos[0] - self.size, draw_pos[1]), # Left
        ]
        top_face = [points[0], points[1], (draw_pos[0], draw_pos[1]), points[3]]
        left_face = [points[3], (draw_pos[0], draw_pos[1]), points[2], (draw_pos[0] - self.size, draw_pos[1] + self.size)]
        right_face = [points[1], (draw_pos[0], draw_pos[1]), points[2], (draw_pos[0] + self.size, draw_pos[1] + self.size)]

        pygame.draw.polygon(surface, (200, 255, 200), top_face)
        pygame.draw.polygon(surface, (100, 200, 100), left_face)
        pygame.draw.polygon(surface, (50, 150, 50), right_face)


class Fruit(Entity):
    def __init__(self, pos, size, color, fall_speed):
        super().__init__(pos, size, color)
        self.fall_speed = fall_speed

    def update(self):
        self.pos.y += self.fall_speed

    def draw(self, surface):
        draw_pos = (int(self.iso_pos.x), int(self.iso_pos.y))
        pygame.gfxdraw.filled_circle(surface, draw_pos[0], draw_pos[1], self.size, self.color)
        pygame.gfxdraw.aacircle(surface, draw_pos[0], draw_pos[1], self.size, self.color)


class Obstacle(Entity):
    def __init__(self, pos, size, color, velocity):
        super().__init__(pos, size, color)
        self.velocity = pygame.Vector2(velocity)
    
    def update(self):
        self.pos += self.velocity

    def draw(self, surface):
        # Draw as a long isometric bar
        p1 = self.iso_pos
        p2 = pygame.Vector2(self.iso_pos.x - self.size * 20, self.iso_pos.y - self.size * 10) # Using cartesian size to project length
        
        height = 10
        points = [
            (p1.x, p1.y),
            (p2.x, p2.y),
            (p2.x, p2.y + height),
            (p1.x, p1.y + height),
        ]
        
        pygame.draw.polygon(surface, self.color, points)
        pygame.draw.polygon(surface, (50, 0, 0), [(p1.x, p1.y+height), (p2.x, p2.y+height), (p2.x, p2.y+height+5), (p1.x, p1.y+height+5)])


class Particle:
    def __init__(self, pos, vel, size, color, lifetime):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.size = size
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95 # friction
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            color_with_alpha = (*self.color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.size), color_with_alpha)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your character on the isometric grid. "
        "Collect falling fruit and avoid the red obstacles."
    )

    game_description = (
        "Fast-paced arcade game. Race against the clock to collect 50 falling fruits. "
        "Dodge the red obstacles that slide across the floor. Good luck!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.origin = pygame.Vector2(self.width // 2, 100)
        self.cartesian_bounds = (-10, 10, -5, 15) # x_min, x_max, y_min, y_max

        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Colors and Fonts ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PLAYER = (100, 255, 100)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.FRUIT_COLORS = [(255, 200, 0), (255, 0, 255), (0, 200, 255)]
        self.UI_FONT = pygame.font.SysFont("Consolas", 24)
        self.MSG_FONT = pygame.font.SysFont("Impact", 50)

        # --- Game State ---
        self.player = None
        self.fruits = []
        self.obstacles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.win = False
        self.initial_obstacle_speed = 0.05
        self.obstacle_speed = self.initial_obstacle_speed
        self.fruits_for_difficulty_increase = 0

        self.max_fruits = 10
        self.max_obstacles = 5
        self.win_score = 50
        self.max_time = 60 * 30 # 60 seconds at 30 fps

        # This will be initialized in reset()
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player = Player(pos=(0, 0), size=10, color=self.COLOR_PLAYER)
        self.fruits = []
        self.obstacles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.max_time
        self.game_over = False
        self.win = False
        self.obstacle_speed = self.initial_obstacle_speed
        self.fruits_for_difficulty_increase = 0
        
        for _ in range(5):
            self._spawn_fruit()
        for _ in range(2):
            self._spawn_obstacle()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0

        # --- RL Reward Calculation (Part 1: Proximity) ---
        dist_before = self._get_dist_to_nearest_fruit()
        
        # --- Update Game Logic ---
        self.player.update(movement, self.cartesian_bounds)
        
        # Update fruits
        for fruit in self.fruits[:]:
            fruit.update()
            if fruit.pos.y > self.cartesian_bounds[3] + 2:
                self.fruits.remove(fruit)

        # Update obstacles
        for obstacle in self.obstacles[:]:
            obstacle.update()
            if not (-15 < obstacle.pos.x < 15 and -10 < obstacle.pos.y < 20):
                 self.obstacles.remove(obstacle)
        
        # Update particles
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

        # Spawn new entities
        if self.np_random.random() < 0.1 and len(self.fruits) < self.max_fruits:
            self._spawn_fruit()
        if self.np_random.random() < 0.05 and len(self.obstacles) < self.max_obstacles:
            self._spawn_obstacle()
            
        # --- Collision Detection & Event Rewards ---
        # Fruit collection
        for fruit in self.fruits[:]:
            if self.player.pos.distance_to(fruit.pos) < (self.player.size + fruit.size) / 4: # smaller collision radius
                self.fruits.remove(fruit)
                self.score += 1
                self.fruits_for_difficulty_increase += 1
                reward += 10
                self._spawn_particles(self.player.iso_pos, fruit.color) # Sound: fruit collect pop
                
                if self.fruits_for_difficulty_increase >= 10:
                    self.fruits_for_difficulty_increase = 0
                    self.obstacle_speed += 0.005 # Difficulty scaling

        # Obstacle collision
        for obstacle in self.obstacles:
            # Simple AABB in cartesian space
            if (abs(self.player.pos.x - obstacle.pos.x) < (self.player.size/4 + obstacle.size/2) and
                abs(self.player.pos.y - obstacle.pos.y) < (self.player.size/4 + 0.5)):
                self.game_over = True
                reward -= 50 # Sound: crash
                break
        
        # --- RL Reward Calculation (Part 2: Proximity change) ---
        dist_after = self._get_dist_to_nearest_fruit()
        if dist_after is not None and dist_before is not None:
            if dist_after < dist_before:
                reward += 0.1 # Small reward for moving closer
            else:
                reward -= 0.01 # Tiny penalty for moving away
        
        # --- Update Timers and Steps ---
        self.steps += 1
        self.time_remaining -= 1
        
        # --- Check Termination Conditions ---
        terminated = self.game_over
        if self.score >= self.win_score:
            self.win = True
            terminated = True
            reward += 100 # Sound: win jingle
        elif self.time_remaining <= 0:
            terminated = True # Sound: lose buzzer

        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_dist_to_nearest_fruit(self):
        if not self.fruits:
            return None
        return min(self.player.pos.distance_to(f.pos) for f in self.fruits)

    def _spawn_fruit(self):
        x = self.np_random.uniform(self.cartesian_bounds[0], self.cartesian_bounds[1])
        y = self.cartesian_bounds[2] - 2
        pos = (x, y)
        size = self.np_random.integers(5, 8)
        color = random.choice(self.FRUIT_COLORS)
        fall_speed = self.np_random.uniform(0.03, 0.06)
        self.fruits.append(Fruit(pos, size, color, fall_speed))

    def _spawn_obstacle(self):
        side = self.np_random.choice([-1, 1])
        x = side * 14
        y = self.np_random.uniform(self.cartesian_bounds[2] + 2, self.cartesian_bounds[3] - 2)
        pos = (x, y)
        size = self.np_random.uniform(2, 4)
        vel_x = -side * self.obstacle_speed
        vel_y = 0
        self.obstacles.append(Obstacle(pos, size, self.COLOR_OBSTACLE, (vel_x, vel_y)))

    def _spawn_particles(self, pos, color):
        # Sound: particle burst
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            size = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(20, 40)
            self.particles.append(Particle(pos, vel, size, color, lifetime))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for i in range(-15, 16):
            # Lines along one axis
            p1_cart = (i, self.cartesian_bounds[2] - 5)
            p2_cart = (i, self.cartesian_bounds[3] + 5)
            p1_iso = (self.origin.x + (p1_cart[0] - p1_cart[1]) * 20, self.origin.y + (p1_cart[0] + p1_cart[1]) * 10)
            p2_iso = (self.origin.x + (p2_cart[0] - p2_cart[1]) * 20, self.origin.y + (p2_cart[0] + p2_cart[1]) * 10)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1_iso, p2_iso, 1)

            # Lines along other axis
            p1_cart = (self.cartesian_bounds[0] - 5, i)
            p2_cart = (self.cartesian_bounds[1] + 5, i)
            p1_iso = (self.origin.x + (p1_cart[0] - p1_cart[1]) * 20, self.origin.y + (p1_cart[0] + p1_cart[1]) * 10)
            p2_iso = (self.origin.x + (p2_cart[0] - p2_cart[1]) * 20, self.origin.y + (p2_cart[0] + p2_cart[1]) * 10)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1_iso, p2_iso, 1)

        # Create a list of all entities to be rendered for depth sorting
        render_list = [self.player] + self.fruits + self.obstacles
        
        for entity in render_list:
            entity.update_iso_pos(self.origin)
            
        render_list.sort(key=lambda e: e.depth)

        # Draw shadows first, then entities
        for entity in render_list:
            entity.draw_shadow(self.screen)
        for entity in render_list:
            entity.draw(self.screen)

        # Draw particles on top
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.UI_FONT.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_sec = max(0, self.time_remaining // 30)
        time_text = self.UI_FONT.render(f"TIME: {time_sec}", True, (255, 255, 255))
        time_rect = time_text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(time_text, time_rect)

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            msg_surf = self.MSG_FONT.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.width // 2, self.height // 2))
            
            # Draw a semi-transparent background for the message
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0,0))
            
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Isometric Fruit Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Space and Shift not used
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()