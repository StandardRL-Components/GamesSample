import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:53:37.384891
# Source Brief: brief_02028.md
# Brief Index: 2028
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Navigate a shifting square through a timed obstacle course by transforming 
    into a slower circle to bypass narrow passages and synchronize with moving 
    platforms for bonus points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a shifting square through a timed obstacle course. Transform into a circle to bypass "
        "narrow passages and land on moving platforms for bonus points."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move. Press space to transform between a fast square and a small, slow circle."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30  # Assumed frame rate for smooth interpolation
    MAX_STEPS = 1200 # Approx 40 seconds at 30 FPS

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_BG_GRID = (30, 35, 50)
    COLOR_PLAYER = (255, 200, 0) # Bright Yellow
    COLOR_PLAYER_GLOW = (255, 200, 0, 40)
    COLOR_OBSTACLE = (220, 50, 50) # Bright Red
    COLOR_PLATFORM = (50, 150, 255) # Bright Blue
    COLOR_GOAL = (50, 220, 50) # Bright Green
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_pos = None
        self.player_is_square = None
        self.player_rect = None
        self.prev_space_held = None
        self.prev_dist_to_goal = None
        self.particles = []
        self.platforms = []

        # --- Level Design ---
        self.start_pos = pygame.Vector2(50, self.SCREEN_HEIGHT - 50)
        self.end_zone = pygame.Rect(self.SCREEN_WIDTH - 80, 20, 60, 60)
        self.obstacles = [
            pygame.Rect(150, 100, 20, 300),
            pygame.Rect(150, 100, 340, 20),
            pygame.Rect(470, 100, 20, 200),
            pygame.Rect(250, 250, 240, 20)
        ]
        self._setup_platforms()
        
        # Initialize state variables
        # self.reset() # reset() is called by the wrapper, no need to call it here
        
        # Run self-check
        # self.validate_implementation() # This is a helper and not part of the standard API

    def _setup_platforms(self):
        """Initializes the moving platforms."""
        self.platforms_config = [
            {
                'start': (200, 200), 'end': (420, 200), 'size': (50, 15), 
                'speed': 2.0
            },
            {
                'start': (200, 320), 'end': (200, 150), 'size': (15, 50), 
                'speed': 1.5
            }
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = self.start_pos.copy()
        self.player_is_square = True
        self.player_rect = pygame.Rect(0, 0, 30, 30)
        self.player_rect.center = self.player_pos
        
        self.prev_space_held = False
        self.prev_dist_to_goal = self._get_dist_to_goal()
        
        self.particles.clear()
        self.platforms.clear()
        for config in self.platforms_config:
            rect = pygame.Rect(config['start'], config['size'])
            self.platforms.append({
                'rect': rect,
                'start': pygame.Vector2(config['start']),
                'end': pygame.Vector2(config['end']),
                'speed': config['speed'],
                'direction': 1
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Handle Input and State Updates ---
        movement, space_held, _ = self._unpack_action(action)
        
        # Transformation logic
        transform_reward = self._handle_transformation(space_held)
        reward += transform_reward

        # Movement logic
        self._update_player_position(movement)
        
        # Update platforms
        self._update_platforms()

        # Update particles
        self._update_particles()

        # --- Collision and Termination Checks ---
        collision_penalty, termination_reason = self._check_collisions()
        reward += collision_penalty
        
        if termination_reason:
            self.game_over = True
            if termination_reason == "win":
                self.win = True
                reward += 100.0
            else:
                reward -= 10.0 # Terminal penalty for loss

        # Time-based termination
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            reward -= 10.0 # Timeout penalty

        # --- Distance-based Reward ---
        current_dist = self._get_dist_to_goal()
        reward += (self.prev_dist_to_goal - current_dist) * 0.1 # Reward for getting closer
        self.prev_dist_to_goal = current_dist
        
        self.score += reward
        self.steps += 1
        
        terminated = self.game_over
        truncated = False # Not using truncation based on time limit, but termination

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _unpack_action(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1 # Unused as per brief
        return movement, space_held, shift_held

    def _handle_transformation(self, space_held):
        reward = 0
        just_pressed_space = space_held and not self.prev_space_held
        if just_pressed_space:
            self.player_is_square = not self.player_is_square
            # SFX: Play transform sound
            
            # Check for platform synchronization bonus
            for platform in self.platforms:
                if self.player_rect.colliderect(platform['rect']):
                    reward += 10.0
                    # SFX: Play bonus sound
                    break # Only one bonus per transform

            # Spawn particles for visual feedback
            for _ in range(20):
                self.particles.append(Particle(self.player_pos.copy(), self.np_random))
        
        self.prev_space_held = space_held
        return reward

    def _update_player_position(self, movement):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right

        speed = 4.0 if self.player_is_square else 2.0
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * speed
        
        # Clamp to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, 15, self.SCREEN_WIDTH - 15)
        self.player_pos.y = np.clip(self.player_pos.y, 15, self.SCREEN_HEIGHT - 15)
        self.player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))

    def _update_platforms(self):
        for p in self.platforms:
            start_vec = p['start']
            end_vec = p['end']
            path_vec = end_vec - start_vec
            
            if path_vec.length_squared() == 0: continue

            current_pos_vec = pygame.Vector2(p['rect'].topleft)
            
            # Project current position onto path to find progress
            progress_vec = current_pos_vec - start_vec
            dist_along_path = progress_vec.dot(path_vec.normalize())
            
            # Move
            dist_along_path += p['speed'] * p['direction']
            
            # Check for reversal
            if dist_along_path >= path_vec.length() or dist_along_path <= 0:
                p['direction'] *= -1
                dist_along_path = np.clip(dist_along_path, 0, path_vec.length())

            new_pos = start_vec + path_vec.normalize() * dist_along_path
            p['rect'].topleft = (int(new_pos.x), int(new_pos.y))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update()

    def _check_collisions(self):
        # Win condition
        if self.player_rect.colliderect(self.end_zone):
            return 0, "win"

        # Platform collision (both shapes)
        for platform in self.platforms:
            if self.player_rect.colliderect(platform['rect']):
                # SFX: Play collision/fail sound
                return 0, "platform_collision"

        # Obstacle collision (square only)
        if self.player_is_square:
            for obstacle in self.obstacles:
                if self.player_rect.colliderect(obstacle):
                    # SFX: Play collision/fail sound
                    return 0, "obstacle_collision"
        
        return 0, None

    def _get_dist_to_goal(self):
        return self.player_pos.distance_to(pygame.Vector2(self.end_zone.center))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_BG_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_BG_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game_elements(self):
        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.end_zone)
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle)
        for platform in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform['rect'])

    def _render_player(self):
        center_pos = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Glow effect
        glow_radius = 25 if self.player_is_square else 20
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (center_pos[0] - glow_radius, center_pos[1] - glow_radius))

        if self.player_is_square:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)
        else:
            pygame.gfxdraw.aacircle(self.screen, center_pos[0], center_pos[1], 15, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, center_pos[0], center_pos[1], 15, self.COLOR_PLAYER)
    
    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_ui.render(f"TIME: {time_left / self.FPS:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        message = "LEVEL COMPLETE!" if self.win else "GAME OVER"
        text_surface = self.font_game_over.render(message, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win
        }

    def close(self):
        pygame.quit()


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, np_random):
        self.pos = pos.copy()
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.radius = np_random.uniform(2, 4)
        self.lifetime = np_random.integers(15, 30) # frames
        self.color = GameEnv.COLOR_PARTICLE
        self.np_random = np_random

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95 # friction
        self.lifetime -= 1
        self.radius -= 0.1

    def is_alive(self):
        return self.lifetime > 0 and self.radius > 0

    def draw(self, surface):
        if self.is_alive():
            pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), max(0, int(self.radius)))


if __name__ == '__main__':
    # --- Example Usage ---
    # This part allows a human to play the game to test the mechanics and feel.
    # It will not run when the file is imported.
    # To run, ensure you have a display driver available, e.g., by commenting out
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    # For local testing with display, uncomment the following line
    # os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Pygame setup for rendering to screen
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gymnasium Environment Test")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # --- Map keyboard to MultiDiscrete action space ---
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0 # No-op

        # Space button
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift button
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    
    # Wait a bit before closing
    pygame.time.wait(2000)
    env.close()