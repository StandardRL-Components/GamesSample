import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:35:57.318848
# Source Brief: brief_00514.md
# Brief Index: 514
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, min_vel=-1.5, max_vel=1.5, gravity=0.05, lifespan=30):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(random.uniform(min_vel, max_vel), random.uniform(min_vel, max_vel))
        self.color = color
        self.gravity = gravity
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.radius = random.randint(2, 5)

    def update(self):
        self.pos += self.vel
        self.vel.y += self.gravity
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color = (*self.color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a fast character to reach a moving target while avoiding obstacles. "
        "Use a second, slower character to deploy a protective shield."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the fast character. "
        "Hold Shift + arrow keys to move the slow character. Press Space to activate the shield."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_FAST_CHAR = (0, 150, 255)
    COLOR_SLOW_CHAR = (255, 150, 0)
    COLOR_SHIELD = (255, 170, 50)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_TARGET = (50, 255, 50)
    COLOR_WALL = (40, 50, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Game Parameters
    MAX_STEPS = 2700  # 90 seconds @ 30 FPS
    FPS = 30
    FAST_CHAR_SPEED = 5.0
    SLOW_CHAR_SPEED = 2.5
    FAST_CHAR_SIZE = 12
    SLOW_CHAR_SIZE = 15
    SHIELD_BASE_RADIUS = 45
    SHIELD_DURATION = 150  # 5 seconds @ 30 FPS
    SHIELD_COOLDOWN = 150  # 5 seconds
    TARGET_SIZE = 15
    NUM_OBSTACLES = 8
    OBSTACLE_SPEED_INCREASE_INTERVAL = 500
    OBSTACLE_SPEED_INCREMENT = 0.05

    # Reward Parameters
    REWARD_REACH_TARGET = 100.0
    REWARD_COLLISION = -100.0
    REWARD_TIMEOUT = -50.0
    REWARD_DISTANCE_FACTOR = 0.1
    REWARD_STRATEGIC_SHIELD = 5.0
    REWARD_WASTED_SHIELD = -1.0
    STRATEGIC_RADIUS = 150

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Initialize state variables (will be properly set in reset)
        self.fast_char_pos = pygame.Vector2(0, 0)
        self.slow_char_pos = pygame.Vector2(0, 0)
        self.target_pos = pygame.Vector2(0, 0)
        self.obstacles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        
        self.obstacle_speed_multiplier = 1.0
        self.last_dist_to_target = 0.0

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.fast_char_pos = pygame.Vector2(self.WIDTH * 0.25, self.HEIGHT / 2)
        self.slow_char_pos = pygame.Vector2(self.WIDTH * 0.1, self.HEIGHT / 2)
        
        self.target_angle = self.np_random.uniform(0, 2 * math.pi)
        self._update_target()
        
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        
        self.obstacle_speed_multiplier = 1.0
        
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            self._spawn_obstacle()
            
        self.particles = []
        
        self.last_dist_to_target = self.fast_char_pos.distance_to(self.target_pos)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.0
        
        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- 1. Handle Actions ---
        self._handle_movement(movement, shift_held)
        reward += self._handle_shield(space_pressed)
        
        # --- 2. Update Game State ---
        self._update_shield_timers()
        self._update_target()
        self._update_obstacles()
        self._update_particles()
        self._update_difficulty()
        
        # --- 3. Calculate Rewards & Check Termination ---
        dist_reward, dist_change = self._calculate_distance_reward()
        reward += dist_reward
        self.last_dist_to_target -= dist_change

        term_reward, terminated = self._handle_collisions()
        reward += term_reward
        
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
            reward += self.REWARD_TIMEOUT
            
        self.score += reward
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Internal Logic Methods ---
    
    def _handle_movement(self, movement, shift_held):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()

        if shift_held: # Move slow character
            self.slow_char_pos += move_vec * self.SLOW_CHAR_SPEED
        else: # Move fast character
            self.fast_char_pos += move_vec * self.FAST_CHAR_SPEED
            
        # Boundary checks
        self.fast_char_pos.x = np.clip(self.fast_char_pos.x, 0, self.WIDTH)
        self.fast_char_pos.y = np.clip(self.fast_char_pos.y, 0, self.HEIGHT)
        self.slow_char_pos.x = np.clip(self.slow_char_pos.x, 0, self.WIDTH)
        self.slow_char_pos.y = np.clip(self.slow_char_pos.y, 0, self.HEIGHT)
        
    def _handle_shield(self, space_pressed):
        reward = 0.0
        if space_pressed and not self.shield_active and self.shield_cooldown_timer <= 0:
            self.shield_active = True
            self.shield_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN + self.SHIELD_DURATION
            # sfx: shield_activate.wav
            
            # Strategic shield reward
            is_strategic = False
            if self.fast_char_pos.distance_to(self.slow_char_pos) < self.STRATEGIC_RADIUS:
                for obs in self.obstacles:
                    if self.slow_char_pos.distance_to(obs['pos']) < self.STRATEGIC_RADIUS:
                        is_strategic = True
                        break
            
            if is_strategic:
                reward += self.REWARD_STRATEGIC_SHIELD
            else:
                reward += self.REWARD_WASTED_SHIELD
        return reward
        
    def _update_shield_timers(self):
        if self.shield_active:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.shield_active = False
                # sfx: shield_deactivate.wav
        if self.shield_cooldown_timer > 0:
            self.shield_cooldown_timer -= 1

    def _update_target(self):
        self.target_angle += 0.01
        center_x, center_y = self.WIDTH * 0.8, self.HEIGHT / 2
        radius = self.HEIGHT * 0.3
        self.target_pos.x = center_x + math.cos(self.target_angle) * radius
        self.target_pos.y = center_y + math.sin(self.target_angle) * radius

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['pos'] += obs['vel'] * self.obstacle_speed_multiplier
            if obs['axis'] == 'x' and (obs['pos'].x < obs['range'][0] or obs['pos'].x > obs['range'][1]):
                obs['vel'].x *= -1
            if obs['axis'] == 'y' and (obs['pos'].y < obs['range'][0] or obs['pos'].y > obs['range'][1]):
                obs['vel'].y *= -1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
            
    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.OBSTACLE_SPEED_INCREASE_INTERVAL == 0:
            self.obstacle_speed_multiplier += self.OBSTACLE_SPEED_INCREMENT

    def _calculate_distance_reward(self):
        current_dist = self.fast_char_pos.distance_to(self.target_pos)
        dist_change = self.last_dist_to_target - current_dist
        return dist_change * self.REWARD_DISTANCE_FACTOR, dist_change
        
    def _handle_collisions(self):
        fast_char_rect = pygame.Rect(self.fast_char_pos.x - self.FAST_CHAR_SIZE / 2, self.fast_char_pos.y - self.FAST_CHAR_SIZE / 2, self.FAST_CHAR_SIZE, self.FAST_CHAR_SIZE)
        target_rect = pygame.Rect(self.target_pos.x - self.TARGET_SIZE, self.target_pos.y - self.TARGET_SIZE, self.TARGET_SIZE * 2, self.TARGET_SIZE * 2)

        # Win condition
        if fast_char_rect.colliderect(target_rect):
            # sfx: win.wav
            self._create_explosion(self.target_pos.x, self.target_pos.y, self.COLOR_TARGET, 30)
            return self.REWARD_REACH_TARGET, True

        shield_center = self.slow_char_pos
        
        obstacles_to_remove = []
        for i, obs in enumerate(self.obstacles):
            obs_rect = pygame.Rect(obs['pos'].x, obs['pos'].y, obs['size'][0], obs['size'][1])
            
            # Loss condition
            if fast_char_rect.colliderect(obs_rect):
                # sfx: player_hit.wav
                self._create_explosion(self.fast_char_pos.x, self.fast_char_pos.y, self.COLOR_FAST_CHAR, 50)
                return self.REWARD_COLLISION, True
            
            # Shield interaction
            if self.shield_active:
                dist_to_shield = shield_center.distance_to(obs_rect.center)
                if dist_to_shield < self.SHIELD_BASE_RADIUS + max(obs['size']) / 2:
                    obstacles_to_remove.append(i)
                    # sfx: shield_block.wav
                    self._create_explosion(obs_rect.centerx, obs_rect.centery, self.COLOR_OBSTACLE, 20)
        
        if obstacles_to_remove:
            self.obstacles = [obs for i, obs in enumerate(self.obstacles) if i not in obstacles_to_remove]
            for _ in obstacles_to_remove:
                self._spawn_obstacle() # Replenish
        
        return 0.0, False
        
    def _spawn_obstacle(self):
        axis = self.np_random.choice(['x', 'y'])
        size = (self.np_random.integers(15, 40), self.np_random.integers(15, 40))
        if axis == 'x': # Horizontal movement
            pos = pygame.Vector2(self.np_random.uniform(self.WIDTH * 0.4, self.WIDTH * 0.9), self.np_random.uniform(0, self.HEIGHT - size[1]))
            vel = pygame.Vector2(self.np_random.choice([-1, 1]), 0)
            travel_range = (self.WIDTH * 0.3, self.WIDTH)
        else: # Vertical movement
            pos = pygame.Vector2(self.np_random.uniform(self.WIDTH * 0.4, self.WIDTH - size[0]), self.np_random.uniform(0, self.HEIGHT * 0.8))
            vel = pygame.Vector2(0, self.np_random.choice([-1, 1]))
            travel_range = (0, self.HEIGHT)
        
        self.obstacles.append({'pos': pos, 'size': size, 'vel': vel, 'axis': axis, 'range': travel_range})

    def _create_explosion(self, x, y, color, num_particles):
        for _ in range(num_particles):
            self.particles.append(Particle(x, y, color))
            
    # --- Rendering Methods ---
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles first (background layer)
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw target with glow
        self._draw_glowing_circle(self.target_pos, self.TARGET_SIZE, self.COLOR_TARGET, 1.8)
        
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (int(obs['pos'].x), int(obs['pos'].y), obs['size'][0], obs['size'][1]))

        # Draw slow character
        self._draw_glowing_circle(self.slow_char_pos, self.SLOW_CHAR_SIZE, self.COLOR_SLOW_CHAR, 1.5)
        
        # Draw shield
        if self.shield_active:
            pulse = math.sin(self.steps * 0.3) * 3
            radius = int(self.SHIELD_BASE_RADIUS + pulse)
            alpha = int(100 + 50 * (self.shield_timer / self.SHIELD_DURATION))
            pygame.gfxdraw.filled_circle(self.screen, int(self.slow_char_pos.x), int(self.slow_char_pos.y), radius, (*self.COLOR_SHIELD, alpha))
            pygame.gfxdraw.aacircle(self.screen, int(self.slow_char_pos.x), int(self.slow_char_pos.y), radius, (*self.COLOR_SHIELD, alpha + 50))
            
        # Draw fast character
        fast_char_rect = pygame.Rect(0, 0, self.FAST_CHAR_SIZE * 2, self.FAST_CHAR_SIZE * 2)
        fast_char_rect.center = (int(self.fast_char_pos.x), int(self.fast_char_pos.y))
        glow_surf = pygame.Surface(fast_char_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_FAST_CHAR, 50), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, fast_char_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_FAST_CHAR, (int(self.fast_char_pos.x - self.FAST_CHAR_SIZE / 2), int(self.fast_char_pos.y - self.FAST_CHAR_SIZE / 2), self.FAST_CHAR_SIZE, self.FAST_CHAR_SIZE), border_radius=3)
        
    def _draw_glowing_circle(self, pos, radius, color, glow_factor):
        x, y = int(pos.x), int(pos.y)
        glow_radius = int(radius * glow_factor)
        # Draw multiple transparent circles for a soft glow
        for i in range(glow_radius - radius, 0, -2):
            alpha = 40 * (1 - i / (glow_radius - radius))
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius + i, (*color, int(alpha)))
        # Draw main circle
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _render_ui(self):
        # Time remaining
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"TIME: {time_left:.1f}"
        self._draw_text(time_text, (self.WIDTH - 10, 10), self.font_main, self.COLOR_TEXT, align="topright")
        
        # Distance to target
        dist_text = f"DIST: {self.last_dist_to_target:.0f}"
        self._draw_text(dist_text, (10, 10), self.font_main, self.COLOR_TEXT, align="topleft")
        
        # Score
        score_text = f"SCORE: {self.score:.1f}"
        self._draw_text(score_text, (self.WIDTH / 2, self.HEIGHT - 10), self.font_main, self.COLOR_TEXT, align="bottommid")
        
    def _draw_text(self, text, pos, font, color, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "topright":
            text_rect.topright = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "bottommid":
            text_rect.midbottom = pos
        
        # Simple shadow
        shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surface, text_rect.move(2, 2))
        self.screen.blit(text_surface, text_rect)

    # --- Gymnasium Interface Compliance ---
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_to_target": self.last_dist_to_target,
            "shield_active": self.shield_active
        }
    
    def render(self):
        # This is not part of the standard API for rgb_array mode,
        # but can be useful for human play testing.
        if 'render_modes' in self.metadata and 'human' in self.metadata['render_modes']:
            if not hasattr(self, 'render_screen'):
                pygame.display.init()
                self.render_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                pygame.display.set_caption("Dual Guard")
            
            obs = self._get_observation()
            # The observation is (H, W, C), but pygame surfaces are (W, H).
            # We need to transpose back before blitting.
            surf_to_render = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            self.render_screen.blit(surf_to_render, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.FPS)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # Example of how to run the environment for human play
    # Add 'human' to render_modes to allow for display
    GameEnv.metadata['render_modes'].append('human')
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    # Mapping keyboard keys to actions
    # Action: [movement, space, shift]
    # Movement: 0=none, 1=up, 2=down, 3=left, 4=right
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("  - Reset: R")

    while not terminated:
        env.render() # Use the custom render method for display
        
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0
        
        # This is a basic event loop for human play.
        # It does not perfectly match the auto_advance=True logic
        # but is sufficient for testing.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Environment Reset ---")
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
            
        if keys[pygame.K_SPACE]:
            space_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
    print(f"Episode Finished. Total Reward: {total_reward:.2f}, Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    env.close()