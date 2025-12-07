
# Generated: 2025-08-28T01:29:43.826862
# Source Brief: brief_04120.md
# Brief Index: 4120

        
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
        "Controls: ←→ to turn your rocket. Collect yellow fuel cells and avoid grey asteroids to reach the green finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race your rocket through an asteroid field to the finish line, collecting fuel and dodging obstacles for the fastest time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_STAR = (200, 200, 220)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_GLOW = (255, 100, 100, 100)
    COLOR_ASTEROID = (120, 130, 140)
    COLOR_FUEL = (255, 220, 50)
    COLOR_FINISH_LINE = (50, 255, 50)
    COLOR_TEXT = (255, 255, 255)
    COLOR_SPARK = (255, 180, 50)
    COLOR_THRUST = (255, 200, 150)

    # Game parameters
    MAX_STEPS = 5000
    MAX_COLLISIONS = 5
    INITIAL_FUEL = 1000
    
    ROCKET_THRUST = 2.5
    ROCKET_ROTATION_SPEED = 4.5
    ROCKET_SIZE = 12
    
    INITIAL_ASTEROIDS = 3
    MAX_ASTEROIDS = 20
    ASTEROID_DIFFICULTY_INTERVAL = 2000
    
    NUM_FUEL_CELLS = 15
    NUM_STARS = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.render_mode = render_mode
        self.game_over = False
        
        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.fuel = 0
        self.collisions = 0
        self.rocket = {}
        self.asteroids = []
        self.fuel_cells = []
        self.stars = []
        self.particles = []
        self.finish_line_rect = pygame.Rect(self.SCREEN_WIDTH - 30, 0, 30, self.SCREEN_HEIGHT)
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.collisions = 0
        self.fuel = self.INITIAL_FUEL

        self.rocket = {
            "pos": np.array([50.0, self.SCREEN_HEIGHT / 2.0]),
            "angle": 0.0,
            "speed": self.ROCKET_THRUST,
        }

        self.asteroids = self._generate_asteroids()
        self.fuel_cells = self._generate_fuel_cells()
        self.stars = self._generate_stars()
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = -0.01  # Time penalty
        self.steps += 1
        
        self._update_rocket(movement)
        self._update_particles()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        self.fuel -= 1

        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            if self.rocket['pos'][0] >= self.finish_line_rect.left:
                reward += 100.0 # Reached finish line
                # sfx: win_sound
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_rocket(self, movement):
        # --- Rotation ---
        if movement == 3:  # Left
            self.rocket['angle'] += self.ROCKET_ROTATION_SPEED
        elif movement == 4:  # Right
            self.rocket['angle'] -= self.ROCKET_ROTATION_SPEED
        self.rocket['angle'] %= 360

        # --- Movement ---
        angle_rad = math.radians(self.rocket['angle'])
        velocity = np.array([math.cos(angle_rad), -math.sin(angle_rad)]) * self.rocket['speed']
        self.rocket['pos'] += velocity

        # --- Screen Wrap ---
        self.rocket['pos'][0] %= self.SCREEN_WIDTH
        self.rocket['pos'][1] %= self.SCREEN_HEIGHT
        
        # --- Thrust Particles ---
        if self.steps % 2 == 0:
            self._create_thrust_particles()

    def _handle_collisions(self):
        reward = 0
        
        # Rocket vs Asteroids
        for asteroid in self.asteroids[:]:
            dist = np.linalg.norm(self.rocket['pos'] - asteroid['pos'])
            if dist < self.ROCKET_SIZE + asteroid['radius']:
                self.collisions += 1
                reward -= 10.0
                self.asteroids.remove(asteroid)
                self._create_explosion(self.rocket['pos'])
                # sfx: explosion_sound
                # Add new asteroid to maintain count
                num_asteroids = min(self.MAX_ASTEROIDS, self.INITIAL_ASTEROIDS + (self.steps // self.ASTEROID_DIFFICULTY_INTERVAL))
                if len(self.asteroids) < num_asteroids:
                     self.asteroids.extend(self._generate_asteroids(count=1))
                break

        # Rocket vs Fuel Cells
        rocket_rect = pygame.Rect(self.rocket['pos'][0] - 5, self.rocket['pos'][1] - 5, 10, 10)
        for fuel_cell in self.fuel_cells[:]:
            if rocket_rect.colliderect(fuel_cell['rect']):
                self.fuel = min(self.INITIAL_FUEL, self.fuel + 150)
                reward += 0.1
                self.fuel_cells.remove(fuel_cell)
                self._create_collect_effect(fuel_cell['rect'].center)
                # sfx: collect_fuel_sound
                
        return reward

    def _check_termination(self):
        if self.collisions >= self.MAX_COLLISIONS:
            return True
        if self.fuel <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        if self.rocket['pos'][0] >= self.finish_line_rect.left:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "collisions": self.collisions,
        }

    # --- Generation Methods ---
    def _generate_asteroids(self, count=None):
        if count is None:
            count = min(self.MAX_ASTEROIDS, self.INITIAL_ASTEROIDS + (self.steps // self.ASTEROID_DIFFICULTY_INTERVAL))
        
        asteroids = []
        for _ in range(count):
            while True:
                pos = np.array([self.np_random.uniform(100, self.SCREEN_WIDTH - 50), 
                                self.np_random.uniform(0, self.SCREEN_HEIGHT)])
                # Ensure they don't spawn too close to the start or each other
                if np.linalg.norm(pos - self.rocket['pos']) > 100:
                    if all(np.linalg.norm(pos - other['pos']) > 50 for other in asteroids):
                        break
            radius = self.np_random.uniform(10, 25)
            asteroids.append({"pos": pos, "radius": radius})
        return asteroids

    def _generate_fuel_cells(self):
        fuel_cells = []
        for _ in range(self.NUM_FUEL_CELLS):
            size = 12
            rect = pygame.Rect(self.np_random.uniform(100, self.SCREEN_WIDTH - 50),
                               self.np_random.uniform(20, self.SCREEN_HEIGHT - 20),
                               size, size)
            fuel_cells.append({"rect": rect})
        return fuel_cells

    def _generate_stars(self):
        return [(self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT))
                for _ in range(self.NUM_STARS)]

    # --- Particle Effects ---
    def _create_thrust_particles(self):
        angle_rad = math.radians(self.rocket['angle'])
        # Back of the rocket
        offset = np.array([-math.cos(angle_rad), math.sin(angle_rad)]) * self.ROCKET_SIZE
        pos = self.rocket['pos'] + offset
        # Velocity opposite to rocket direction
        vel = np.array([-math.cos(angle_rad), math.sin(angle_rad)]) * 2 + self.np_random.uniform(-0.5, 0.5, 2)
        self.particles.append({'pos': pos, 'vel': vel, 'life': 15, 'type': 'thrust'})

    def _create_explosion(self, position):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({'pos': position.copy(), 'vel': vel, 'life': self.np_random.integers(20, 40), 'type': 'spark'})
            
    def _create_collect_effect(self, position):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({'pos': np.array(position, dtype=float), 'vel': vel, 'life': self.np_random.integers(10, 20), 'type': 'collect'})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    # --- Rendering Methods ---
    def _render_game(self):
        # Stars
        for star_pos in self.stars:
            pygame.gfxdraw.pixel(self.screen, star_pos[0], star_pos[1], self.COLOR_STAR)
            
        # Finish Line
        pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, self.finish_line_rect)

        # Asteroids
        for asteroid in self.asteroids:
            pos = asteroid['pos'].astype(int)
            radius = int(asteroid['radius'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)

        # Fuel Cells
        for cell in self.fuel_cells:
            pygame.draw.rect(self.screen, self.COLOR_FUEL, cell['rect'])
            pygame.gfxdraw.rectangle(self.screen, cell['rect'], self.COLOR_TEXT)

        # Particles
        for p in self.particles:
            pos = p['pos'].astype(int)
            if p['type'] == 'thrust':
                size = max(0, int(p['life'] / 4))
                pygame.draw.circle(self.screen, self.COLOR_THRUST, pos, size)
            elif p['type'] == 'spark':
                pygame.draw.line(self.screen, self.COLOR_SPARK, pos, pos + p['vel'] * 2, 2)
            elif p['type'] == 'collect':
                alpha = int(255 * (p['life'] / 20))
                color = (*self.COLOR_FUEL, alpha)
                temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (2,2), 2)
                self.screen.blit(temp_surf, pos-np.array([2,2]))


        # Rocket
        self._render_rocket()

    def _render_rocket(self):
        pos = self.rocket['pos']
        angle = self.rocket['angle']
        
        # Glow effect
        glow_radius = int(self.ROCKET_SIZE * 1.8)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (int(pos[0] - glow_radius), int(pos[1] - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Main body
        points_rel = [
            (self.ROCKET_SIZE, 0),
            (-self.ROCKET_SIZE * 0.6, -self.ROCKET_SIZE * 0.7),
            (-self.ROCKET_SIZE * 0.6, self.ROCKET_SIZE * 0.7),
        ]
        
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        points_abs = []
        for x, y in points_rel:
            rx = x * cos_a - y * sin_a + pos[0]
            ry = x * sin_a + y * cos_a + pos[1]
            points_abs.append((int(rx), int(ry)))
            
        pygame.gfxdraw.filled_polygon(self.screen, points_abs, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points_abs, self.COLOR_PLAYER)

    def _render_ui(self):
        # Time / Steps
        time_text = self.font_small.render(f"TIME: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))
        
        # Fuel
        fuel_perc = max(0, self.fuel / self.INITIAL_FUEL)
        fuel_bar_width = 150
        fuel_bar_height = 15
        fuel_bar_x = self.SCREEN_WIDTH - fuel_bar_width - 10
        fuel_bar_y = 10
        
        pygame.draw.rect(self.screen, (80,80,80), (fuel_bar_x, fuel_bar_y, fuel_bar_width, fuel_bar_height))
        current_fuel_width = int(fuel_bar_width * fuel_perc)
        fuel_color = (255, 255, 0) if fuel_perc > 0.5 else (255, 165, 0) if fuel_perc > 0.2 else (255, 0, 0)
        pygame.draw.rect(self.screen, fuel_color, (fuel_bar_x, fuel_bar_y, current_fuel_width, fuel_bar_height))
        fuel_text = self.font_small.render("FUEL", True, self.COLOR_TEXT)
        self.screen.blit(fuel_text, (fuel_bar_x - fuel_text.get_width() - 5, 8))

        # Collisions
        collision_text = self.font_small.render(f"HITS: {self.collisions}/{self.MAX_COLLISIONS}", True, self.COLOR_TEXT)
        text_rect = collision_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
        self.screen.blit(collision_text, text_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = ""
            if self.rocket['pos'][0] >= self.finish_line_rect.left:
                message = "FINISH!"
            elif self.collisions >= self.MAX_COLLISIONS:
                message = "DESTROYED"
            elif self.fuel <= 0:
                message = "OUT OF FUEL"
            else:
                message = "TIME UP"
            
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

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
    # This block allows you to play the game directly
    # Set SDL_VIDEODRIVER to a dummy value to run without a display
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    # To run with a display, comment out the line above and uncomment the block below.
    # This requires a display environment.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Interactive Play Setup ---
    pygame.display.set_caption("Rocket Racer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("="*30 + "\n")
    
    while running:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait for a moment then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(30) # Control the frame rate for human play

    env.close()