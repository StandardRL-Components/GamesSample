import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import math
import random
import os
import os
import pygame


# Set the SDL video driver to dummy to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to pilot the rocket. Avoid asteroids and collect fuel."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a retro rocket through a dense asteroid field, collecting fuel to reach the finish line before you run out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 20, 40)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_THRUSTER = (255, 180, 80)
    COLOR_ASTEROID = [(100, 110, 120), (120, 130, 140), (80, 90, 100)]
    COLOR_FUEL = (255, 255, 100)
    COLOR_FINISH = (80, 255, 80)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_FUEL_BAR_BG = (60, 60, 60)
    
    # Player
    PLAYER_SIZE = 12
    PLAYER_SPEED = 4.0
    
    # Fuel
    MAX_FUEL = 100.0
    FUEL_PER_PICKUP = 25.0
    FUEL_CONSUMPTION = 0.1
    
    # Game
    MAX_STEPS = 1000
    FINISH_LINE_X = SCREEN_WIDTH - 20
    
    # Asteroids
    INITIAL_ASTEROID_COUNT = 20
    ASTEROID_SPEED_INCREASE_INTERVAL = 500
    ASTEROID_SPEED_INCREASE_AMOUNT = 0.2
    
    # Pickups
    FUEL_PICKUP_COUNT = 5
    FUEL_PICKUP_SIZE = 12

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        
        # Game state variables
        self.player_pos = None
        self.fuel = None
        self.asteroids = []
        self.fuel_pickups = []
        self.stars = []
        self.thruster_particles = []
        self.asteroid_speed_bonus = 0.0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.asteroid_speed_bonus = 0.0
        
        self.player_pos = pygame.math.Vector2(50, self.SCREEN_HEIGHT / 2)
        self.fuel = self.MAX_FUEL
        
        self._generate_stars()
        self._generate_asteroids()
        self._generate_fuel_pickups()
        
        self.thruster_particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        
        reward = -0.01  # Cost of existing per step
        terminated = False
        
        if not self.game_over:
            self._handle_player_movement(movement)
            self._update_asteroids()
            self._update_thruster_particles()
            
            self.fuel -= self.FUEL_CONSUMPTION
            
            # Check collisions and events
            pickup_reward = self._check_fuel_pickup_collisions()
            reward += pickup_reward
            self.score += pickup_reward

            if self._check_asteroid_collision():
                reward = -50.0
                self.score -= 50
                terminated = True
                self.game_over = True
            elif self.player_pos.x >= self.FINISH_LINE_X:
                reward = 100.0
                self.score += 100
                terminated = True
                self.game_over = True
            elif self.fuel <= 0:
                reward = -100.0
                self.score -= 100
                self.fuel = 0
                terminated = True
                self.game_over = True

            self.steps += 1
            if self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        moved = False
        if movement == 1: # Up
            self.player_pos.y -= self.PLAYER_SPEED
            moved = True
        elif movement == 2: # Down
            self.player_pos.y += self.PLAYER_SPEED
            moved = True
        elif movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
            moved = True
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
            moved = True

        # Clamp player position to screen
        self.player_pos.x = max(self.PLAYER_SIZE, min(self.player_pos.x, self.SCREEN_WIDTH - self.PLAYER_SIZE))
        self.player_pos.y = max(self.PLAYER_SIZE, min(self.player_pos.y, self.SCREEN_HEIGHT - self.PLAYER_SIZE))
        
        if moved:
            self._emit_thruster_particles()

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            speed = asteroid['base_speed'] + self.asteroid_speed_bonus
            asteroid['pos'] += asteroid['vel'] * speed
            
            # Screen wrapping
            if asteroid['pos'].x < -asteroid['radius']: asteroid['pos'].x = self.SCREEN_WIDTH + asteroid['radius']
            if asteroid['pos'].x > self.SCREEN_WIDTH + asteroid['radius']: asteroid['pos'].x = -asteroid['radius']
            if asteroid['pos'].y < -asteroid['radius']: asteroid['pos'].y = self.SCREEN_HEIGHT + asteroid['radius']
            if asteroid['pos'].y > self.SCREEN_HEIGHT + asteroid['radius']: asteroid['pos'].y = -asteroid['radius']

    def _check_asteroid_collision(self):
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < self.PLAYER_SIZE * 0.8 + asteroid['radius']:
                # sfx: explosion
                return True
        return False

    def _check_fuel_pickup_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for i, pickup_rect in enumerate(self.fuel_pickups):
            if player_rect.colliderect(pickup_rect):
                # sfx: fuel pickup
                self.fuel = min(self.MAX_FUEL, self.fuel + self.FUEL_PER_PICKUP)
                self.fuel_pickups.pop(i)
                self._spawn_single_fuel_pickup()
                return 5.0
        return 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_stars()
        self._render_finish_line()
        self._render_fuel_pickups()
        self._render_asteroids()
        self._render_thruster()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
        }

    # --- Generation Methods ---
    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            brightness = self.np_random.uniform(50, 150)
            self.stars.append(((x, y), int(brightness)))

    def _generate_asteroids(self):
        self.asteroids = []
        for _ in range(self.INITIAL_ASTEROID_COUNT):
            self._spawn_single_asteroid()

    def _spawn_single_asteroid(self):
        # Spawn away from the player's initial area
        side = self.np_random.integers(0, 4)
        if side == 0: # top
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -30)
        elif side == 1: # bottom
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 30)
        elif side == 2: # left
            pos = pygame.math.Vector2(-30, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else: # right
            pos = pygame.math.Vector2(self.SCREEN_WIDTH + 30, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = pygame.math.Vector2(math.cos(angle), math.sin(angle))
        
        self.asteroids.append({
            'pos': pos,
            'vel': vel,
            'radius': self.np_random.integers(10, 30),
            'base_speed': self.np_random.uniform(0.5, 1.5),
            'color': random.choice(self.COLOR_ASTEROID)
        })

    def _generate_fuel_pickups(self):
        self.fuel_pickups = []
        for _ in range(self.FUEL_PICKUP_COUNT):
            self._spawn_single_fuel_pickup()

    def _spawn_single_fuel_pickup(self):
        # Avoid spawning near the edges
        x = self.np_random.integers(50, self.SCREEN_WIDTH - 50)
        y = self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
        self.fuel_pickups.append(pygame.Rect(x, y, self.FUEL_PICKUP_SIZE, self.FUEL_PICKUP_SIZE))
        
    # --- Thruster Particle System ---
    def _emit_thruster_particles(self):
        for _ in range(3):
            particle = {
                'pos': pygame.math.Vector2(self.player_pos) + pygame.math.Vector2(-self.PLAYER_SIZE * 0.6, self.np_random.uniform(-3, 3)),
                'vel': pygame.math.Vector2(self.np_random.uniform(-3, -1), self.np_random.uniform(-1, 1)),
                'life': self.np_random.integers(10, 20),
                'radius': self.np_random.uniform(2, 5)
            }
            self.thruster_particles.append(particle)

    def _update_thruster_particles(self):
        for p in self.thruster_particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] -= 0.2
        self.thruster_particles = [p for p in self.thruster_particles if p['life'] > 0 and p['radius'] > 0]

    # --- Rendering Methods ---
    def _render_stars(self):
        for pos, brightness in self.stars:
            color = (brightness, brightness, brightness)
            size = 1 if brightness < 100 else 2
            pygame.draw.circle(self.screen, color, pos, size)

    def _render_finish_line(self):
        pygame.draw.rect(self.screen, self.COLOR_FINISH, (self.FINISH_LINE_X, 0, 10, self.SCREEN_HEIGHT))

    def _render_fuel_pickups(self):
        for pickup_rect in self.fuel_pickups:
            pygame.draw.rect(self.screen, self.COLOR_FUEL, pickup_rect)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_FUEL), pickup_rect, 2)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            x, y = int(asteroid['pos'].x), int(asteroid['pos'].y)
            radius = int(asteroid['radius'])
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, asteroid['color'])
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, tuple(c*0.8 for c in asteroid['color']))

    def _render_player(self):
        x, y = self.player_pos.x, self.player_pos.y
        s = self.PLAYER_SIZE
        points = [
            (x + s, y),
            (x - s / 2, y - s * 0.8),
            (x - s / 2, y + s * 0.8)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_thruster(self):
        for p in self.thruster_particles:
            alpha = p['life'] / 20
            color = (
                self.COLOR_THRUSTER[0],
                self.COLOR_THRUSTER[1],
                self.COLOR_THRUSTER[2],
                int(255 * alpha)
            )
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color
            )

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Fuel Bar
        fuel_ratio = self.fuel / self.MAX_FUEL
        bar_width = 150
        bar_height = 20
        
        # Animate color from green to red
        fuel_color = (
            int(255 * (1 - fuel_ratio)),
            int(200 * fuel_ratio),
            40
        )
        
        pygame.draw.rect(self.screen, self.COLOR_FUEL_BAR_BG, (10, 10, bar_width, bar_height))
        if fuel_ratio > 0:
            pygame.draw.rect(self.screen, fuel_color, (10, 10, int(bar_width * fuel_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, bar_height), 2)
        
    def close(self):
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

if __name__ == "__main__":
    # To run and play the game manually
    # The environment variable should be set before importing pygame
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "dummy" for headless, "x11" or "windows" for visible
    
    # We need to re-import pygame if the video driver has changed
    import pygame
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Rocket Fuel")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and Shift are unused

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Limit to 30 FPS

    env.close()