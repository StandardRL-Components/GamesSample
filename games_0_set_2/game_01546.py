
# Generated: 2025-08-28T00:03:09.692562
# Source Brief: brief_01546.md
# Brief Index: 1546

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper class for Fish
class Fish:
    """Represents a single fish with position, velocity, and drawing logic."""
    def __init__(self, np_random, bounds):
        self.np_random = np_random
        self.bounds = bounds
        # Start fish somewhat centrally to make initial catches possible
        self.pos = np.array([
            self.np_random.uniform(bounds[0] * 0.2, bounds[0] * 0.8),
            self.np_random.uniform(bounds[1] * 0.2, bounds[1] * 0.8)
        ], dtype=float)
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1.5, 2.5)
        self.vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=float)
        
        self.size = (20, 10)
        self.color = (255, 120, 0) # Bright orange

    def update(self):
        """Updates the fish's position and handles wall collisions."""
        self.pos += self.vel

        # Bounce off walls
        if self.pos[0] - self.size[0] / 2 < 0 or self.pos[0] + self.size[0] / 2 > self.bounds[0]:
            self.vel[0] *= -1
            self.pos[0] = np.clip(self.pos[0], self.size[0] / 2, self.bounds[0] - self.size[0] / 2)
        if self.pos[1] - self.size[1] / 2 < 0 or self.pos[1] + self.size[1] / 2 > self.bounds[1]:
            self.vel[1] *= -1
            self.pos[1] = np.clip(self.pos[1], self.size[1] / 2, self.bounds[1] - self.size[1] / 2)

    def get_rect(self):
        """Returns the pygame.Rect for the fish."""
        return pygame.Rect(
            self.pos[0] - self.size[0] / 2,
            self.pos[1] - self.size[1] / 2,
            self.size[0],
            self.size[1]
        )

    def draw(self, surface):
        """Draws the fish as an ellipse on the given surface."""
        rect = self.get_rect()
        pygame.draw.ellipse(surface, self.color, rect)
        
# Helper class for Particles (bubbles on catch)
class Particle:
    """Represents a single bubble particle for visual effects."""
    def __init__(self, pos, np_random):
        self.pos = list(pos)
        self.vel = [np_random.uniform(-0.5, 0.5), np_random.uniform(-1, -0.2)]
        self.radius = np_random.uniform(2, 6)
        self.max_life = 30 # 1 second at 30 FPS
        self.life = self.max_life
        self.color = (220, 240, 255)

    def update(self):
        """Updates the particle's position, size, and lifespan."""
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1
        self.radius *= 0.98

    def draw(self, surface):
        """Draws the particle with alpha blending."""
        if self.life > 0:
            alpha = int(200 * (self.life / self.max_life))
            pos_int = (int(self.pos[0]), int(self.pos[1]))
            radius_int = int(self.radius)
            if radius_int > 0:
                pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], radius_int, (*self.color, alpha))
                pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], radius_int, (*self.color, alpha))

# Helper class for background waves
class Wave:
    """Represents a single decorative wave in the background."""
    def __init__(self, np_random, width, height):
        self.np_random = np_random
        self.width = width
        self.y = self.np_random.uniform(0, height)
        self.speed = self.np_random.uniform(0.1, 0.3)
        self.amplitude = self.np_random.uniform(2, 5)
        self.frequency = self.np_random.uniform(0.01, 0.03)
        self.offset = self.np_random.uniform(0, 2 * math.pi)
        self.color = (30, 80, 160, 50) # Dark, transparent blue

    def update(self):
        """Updates the wave's vertical position."""
        self.y += self.speed
        if self.y > 400:
            self.y = -10

    def draw(self, surface):
        """Draws the wave as a series of lines."""
        points = []
        for x in range(0, self.width + 1, 10):
            y_offset = math.sin(x * self.frequency + self.offset) * self.amplitude
            points.append((x, self.y + y_offset))
        if len(points) > 1:
            pygame.draw.lines(surface, self.color, False, points, 3)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your net. Catch all the fish before time runs out!"
    )

    game_description = (
        "A top-down arcade fishing game. Maneuver your net to catch as many fish as possible within the 60-second time limit."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.WIN_CONDITION_FISH = 25
        self.NUM_FISH = 25

        # Colors
        self.COLOR_WATER = (40, 100, 180)
        self.COLOR_NET = (50, 220, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.fish_caught = 0
        self.time_limit_steps = self.TIME_LIMIT_SECONDS * self.FPS
        self.net_pos = None
        self.net_size = 50
        self.net_speed = 6
        self.fish_list = []
        self.particles = []
        self.waves = []
        self.np_random = None

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.fish_caught = 0
        
        self.net_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        
        self.fish_list = [
            Fish(self.np_random, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)) 
            for _ in range(self.NUM_FISH)
        ]
        
        self.particles = []
        
        if not self.waves: # Only create waves once
            self.waves = [
                Wave(self.np_random, self.SCREEN_WIDTH, self.SCREEN_HEIGHT) 
                for _ in range(10)
            ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        # 1. Unpack action
        movement = action[0]
        
        # 2. Calculate reward
        reward = 0
        
        # -- Shaping reward for distance to nearest fish
        old_dist = self._get_min_fish_dist()
        
        # 3. Update game logic
        self._move_net(movement)
        
        new_dist = self._get_min_fish_dist()
        
        if old_dist is not None and new_dist is not None:
            if new_dist < old_dist:
                reward += 0.1
            else:
                reward -= 0.01
        
        # -- Update fish and particles
        for fish in self.fish_list:
            fish.update()
        for particle in self.particles:
            particle.update()
        self.particles = [p for p in self.particles if p.life > 0]
        for wave in self.waves:
            wave.update()

        # -- Handle collisions (catching fish)
        net_rect = pygame.Rect(
            self.net_pos[0] - self.net_size / 2,
            self.net_pos[1] - self.net_size / 2,
            self.net_size,
            self.net_size
        )
        
        caught_fish = []
        for fish in self.fish_list:
            if net_rect.colliderect(fish.get_rect()):
                caught_fish.append(fish)
                self.fish_caught += 1
                self.score = self.fish_caught
                reward += 1.0 # Event-based reward for catching a fish
                # Spawn particles
                for _ in range(15):
                    self.particles.append(Particle(fish.pos, self.np_random))
                # Sound effect placeholder: # pygame.mixer.Sound('catch.wav').play()

        self.fish_list = [f for f in self.fish_list if f not in caught_fish]
        
        self.steps += 1
        
        # 4. Check for termination
        terminated = False
        if self.fish_caught >= self.WIN_CONDITION_FISH:
            terminated = True
            reward += 100.0 # Goal-oriented reward for winning
        elif self.steps >= self.time_limit_steps:
            terminated = True
            reward -= 100.0 # Penalty for running out of time
        
        # 5. Return observation, reward, termination, truncation, info
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _move_net(self, movement):
        if movement == 1:  # Up
            self.net_pos[1] -= self.net_speed
        elif movement == 2:  # Down
            self.net_pos[1] += self.net_speed
        elif movement == 3:  # Left
            self.net_pos[0] -= self.net_speed
        elif movement == 4:  # Right
            self.net_pos[0] += self.net_speed
        
        # Clamp position to screen boundaries
        self.net_pos[0] = np.clip(self.net_pos[0], self.net_size / 2, self.SCREEN_WIDTH - self.net_size / 2)
        self.net_pos[1] = np.clip(self.net_pos[1], self.net_size / 2, self.SCREEN_HEIGHT - self.net_size / 2)

    def _get_min_fish_dist(self):
        if not self.fish_list:
            return None
        
        net_center = self.net_pos
        min_dist = float('inf')
        for fish in self.fish_list:
            dist = np.linalg.norm(net_center - fish.pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_observation(self):
        # Clear screen with background color
        self.screen.fill(self.COLOR_WATER)
        
        # Render background effects
        self._render_background()
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for wave in self.waves:
            wave.draw(self.screen)

    def _render_game(self):
        # Render fish
        for fish in self.fish_list:
            fish.draw(self.screen)

        # Render particles
        for particle in self.particles:
            particle.draw(self.screen)
            
        # Render net
        net_rect = pygame.Rect(
            int(self.net_pos[0] - self.net_size / 2),
            int(self.net_pos[1] - self.net_size / 2),
            int(self.net_size),
            int(self.net_size)
        )
        # Semi-transparent fill
        shape_surf = pygame.Surface(net_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (*self.COLOR_NET, 100), shape_surf.get_rect())
        self.screen.blit(shape_surf, net_rect.topleft)
        # Solid border
        pygame.draw.rect(self.screen, self.COLOR_NET, net_rect, 3)

    def _render_text(self, text, font, position, color, shadow_color=None):
        if shadow_color:
            text_surface_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surface_shadow, (position[0] + 2, position[1] + 2))
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def _render_ui(self):
        # Timer
        time_left = max(0, (self.time_limit_steps - self.steps) / self.FPS)
        timer_text = f"Time: {time_left:.1f}"
        self._render_text(timer_text, self.font_small, (self.SCREEN_WIDTH - 150, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Score (Fish Caught)
        score_text = f"Caught: {self.fish_caught} / {self.WIN_CONDITION_FISH}"
        self._render_text(score_text, self.font_small, (10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fish_caught": self.fish_caught,
            "time_remaining": (self.time_limit_steps - self.steps) / self.FPS
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this to verify the environment's implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for visualization
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # This part is for demonstration and debugging, not part of the core environment
    try:
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Arcade Fisher")
        
        obs, info = env.reset()
        terminated = False
        
        # To play as a human, uncomment this section
        key_map = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }
        human_action = np.array([0, 0, 0])
        use_human_player = True
        
        while True:
            if use_human_player:
                human_action.fill(0)
                keys = pygame.key.get_pressed()
                for key, move_action in key_map.items():
                    if keys[key]:
                        human_action[0] = move_action
                action = human_action
            else:
                # For random agent
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            
            # Draw the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Allow quitting by closing the window
                    pygame.quit()
                    exit()
            
            if terminated:
                print(f"Game Over! Final Info: {info}")
                # Wait a bit then reset
                pygame.time.wait(2000)
                obs, info = env.reset()

    finally:
        env.close()