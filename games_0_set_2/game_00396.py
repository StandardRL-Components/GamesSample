
# Generated: 2025-08-27T13:31:55.498063
# Source Brief: brief_00396.md
# Brief Index: 396

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to select a target planet. Press space to hop. Avoid the red asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated planets, dodging asteroids, to reach the glowing purple target planet."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_TRAIL = (0, 150, 90)
    COLOR_PLANET_SAFE = (60, 150, 255)
    COLOR_PLANET_RISKY = (255, 200, 0)
    COLOR_PLANET_TARGET = (200, 50, 255)
    COLOR_ASTEROID = (255, 50, 50)
    COLOR_SELECTOR = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    
    # Game parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    NUM_PLANETS = 10
    NUM_ASTEROIDS = 15
    MAX_STEPS = 1500 # Increased for more play time
    INITIAL_LIVES = 5
    
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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.player_pos = np.array([0.0, 0.0])
        self.player_state = "ON_PLANET" # "ON_PLANET", "HOPPING"
        self.player_radius = 8
        self.player_trail = deque(maxlen=15)
        
        self.planets = []
        self.asteroids = []
        self.particles = []
        self.stars = []
        
        self.current_planet_idx = 0
        self.hop_source_pos = np.array([0.0, 0.0])
        self.hop_target_pos = np.array([0.0, 0.0])
        self.hop_duration = 0
        self.hop_progress = 0
        
        self.selector_idx = 0
        self.sorted_planet_indices = []
        
        self.last_movement_action = 0
        self.last_space_held = False
        self.hop_counter = 0
        self.base_asteroid_speed = 1.0
        
        self.reset()

        # Run self-check
        # self.validate_implementation() # Comment out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.hop_counter = 0
        self.base_asteroid_speed = 1.0
        
        self.player_trail.clear()
        self.particles.clear()
        
        self._generate_stars()
        self._generate_planets()
        self._generate_asteroids()

        start_planet_options = [i for i, p in enumerate(self.planets) if p['type'] != 'target']
        self.current_planet_idx = self.np_random.choice(start_planet_options)
        self.player_pos = np.array(self.planets[self.current_planet_idx]['pos'], dtype=float)
        self.player_state = "ON_PLANET"

        self._update_sorted_planet_indices()
        self.selector_idx = 0

        self.last_movement_action = 0
        self.last_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01 # Small penalty per frame to encourage speed
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._update_particles()
        self._update_asteroids()

        if self.player_state == "ON_PLANET":
            reward += self._handle_input(movement, space_held)
        elif self.player_state == "HOPPING":
            reward += self._update_hop()
        
        self.last_movement_action = movement
        self.last_space_held = space_held
        
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # --- Handle planet selection ---
        # Trigger on new key press only
        if movement != 0 and movement != self.last_movement_action:
            if movement in [1, 4]: # Up or Right for clockwise
                self.selector_idx = (self.selector_idx + 1) % len(self.sorted_planet_indices)
            elif movement in [2, 3]: # Down or Left for counter-clockwise
                self.selector_idx = (self.selector_idx - 1 + len(self.sorted_planet_indices)) % len(self.sorted_planet_indices)

        # --- Handle hop initiation ---
        # Trigger on rising edge of space bar
        if space_held and not self.last_space_held and len(self.sorted_planet_indices) > 0:
            target_idx = self.sorted_planet_indices[self.selector_idx]
            
            self.player_state = "HOPPING"
            self.hop_source_pos = np.array(self.planets[self.current_planet_idx]['pos'], dtype=float)
            self.hop_target_pos = np.array(self.planets[target_idx]['pos'], dtype=float)
            
            distance = np.linalg.norm(self.hop_target_pos - self.hop_source_pos)
            self.hop_duration = max(20, int(distance / 6)) # Hop speed, min 20 frames
            self.hop_progress = 0
            
            self.player_trail.clear()
            # sfx: player_hop_start.wav

        return 0

    def _update_hop(self):
        reward = 0
        self.hop_progress += 1
        
        t = self.hop_progress / self.hop_duration
        # Ease-in-out interpolation for smooth feel
        eased_t = t * t * (3.0 - 2.0 * t)
        
        self.player_pos = self.hop_source_pos * (1 - eased_t) + self.hop_target_pos * eased_t
        self.player_trail.append(self.player_pos.copy())

        # --- Collision Check ---
        collision, asteroid_idx = self._check_asteroid_collision()
        if collision:
            reward -= 5
            self.lives -= 1
            self._create_explosion(self.player_pos, self.COLOR_PLAYER)
            # sfx: player_hit.wav
            
            # Reset to source planet
            self.player_state = "ON_PLANET"
            self.player_pos = np.array(self.planets[self.current_planet_idx]['pos'], dtype=float)
            self.player_trail.clear()
            # sfx: hop_fail.wav
            return reward

        # --- Hop Completion ---
        if self.hop_progress >= self.hop_duration:
            target_idx = self.sorted_planet_indices[self.selector_idx]
            self.current_planet_idx = target_idx
            self.player_state = "ON_PLANET"
            self.player_pos = self.hop_target_pos.copy()
            
            reward += 1 # Reward for successful hop
            self.score += 10
            self.hop_counter += 1
            
            # Increase difficulty
            if self.hop_counter > 0 and self.hop_counter % 5 == 0:
                self.base_asteroid_speed = min(3.0, self.base_asteroid_speed + 0.1)

            # Check for win condition
            if self.planets[self.current_planet_idx]['type'] == 'target':
                reward += 100
                self.score += 1000
                self.game_over = True
                self._create_explosion(self.player_pos, self.COLOR_PLANET_TARGET, 100)
                # sfx: victory.wav
            else:
                # sfx: hop_land.wav
                pass
            
            self._update_sorted_planet_indices()
            self.selector_idx = 0
            
        return reward

    def _check_termination(self):
        return self.lives <= 0 or self.steps >= self.MAX_STEPS or self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_stars()
        self._render_planets_and_selector()
        self._render_asteroids()
        self._render_player_and_trail()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "player_state": self.player_state
        }

    # --- Generation Methods ---
    def _generate_stars(self):
        self.stars = []
        for _ in range(200):
            self.stars.append({
                'pos': [self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)],
                'size': self.np_random.uniform(0.5, 1.5)
            })

    def _generate_planets(self):
        self.planets = []
        min_dist = 80
        padding = 50
        attempts = 0
        while len(self.planets) < self.NUM_PLANETS and attempts < 1000:
            attempts += 1
            radius = self.np_random.integers(15, 25)
            pos = (
                self.np_random.integers(padding + radius, self.SCREEN_WIDTH - padding - radius),
                self.np_random.integers(padding + radius, self.SCREEN_HEIGHT - padding - radius)
            )
            
            too_close = False
            for p in self.planets:
                dist = math.hypot(pos[0] - p['pos'][0], pos[1] - p['pos'][1])
                if dist < p['radius'] + radius + min_dist:
                    too_close = True
                    break
            
            if not too_close:
                self.planets.append({'pos': pos, 'radius': radius, 'type': 'safe'})

        if len(self.planets) > 0:
            target_idx = self.np_random.choice(len(self.planets))
            self.planets[target_idx]['type'] = 'target'

    def _generate_asteroids(self):
        self.asteroids = []
        for _ in range(self.NUM_ASTEROIDS):
            pos = np.array([
                self.np_random.uniform(0, self.SCREEN_WIDTH),
                self.np_random.uniform(0, self.SCREEN_HEIGHT)
            ], dtype=float)
            
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.base_asteroid_speed * self.np_random.uniform(0.7, 1.3)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            
            self.asteroids.append({
                'pos': pos,
                'vel': vel,
                'size': self.np_random.integers(8, 15),
                'rot': self.np_random.uniform(0, 360),
                'rot_speed': self.np_random.uniform(-2, 2)
            })
    
    # --- Update Methods ---
    def _update_asteroids(self):
        for a in self.asteroids:
            a['pos'] += a['vel']
            a['rot'] = (a['rot'] + a['rot_speed']) % 360
            
            # Screen wrap
            if a['pos'][0] < -a['size']: a['pos'][0] = self.SCREEN_WIDTH + a['size']
            if a['pos'][0] > self.SCREEN_WIDTH + a['size']: a['pos'][0] = -a['size']
            if a['pos'][1] < -a['size']: a['pos'][1] = self.SCREEN_HEIGHT + a['size']
            if a['pos'][1] > self.SCREEN_HEIGHT + a['size']: a['pos'][1] = -a['size']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _update_sorted_planet_indices(self):
        if not self.planets or self.current_planet_idx >= len(self.planets):
            self.sorted_planet_indices = []
            return

        player_planet_pos = self.planets[self.current_planet_idx]['pos']
        angles = []
        for i, planet in enumerate(self.planets):
            if i == self.current_planet_idx:
                continue
            dx = planet['pos'][0] - player_planet_pos[0]
            dy = planet['pos'][1] - player_planet_pos[1]
            angle = math.atan2(-dy, dx)
            angles.append((angle, i))
        
        angles.sort(key=lambda x: x[0], reverse=True) # Sort clockwise
        self.sorted_planet_indices = [i for angle, i in angles]

    # --- Collision and Effects ---
    def _check_asteroid_collision(self):
        for i, asteroid in enumerate(self.asteroids):
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.player_radius + asteroid['size']:
                return True, i
        return False, -1
        
    def _create_explosion(self, pos, color, count=40):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(1, 3)
            })

    # --- Rendering Methods ---
    def _render_stars(self):
        for star in self.stars:
            # Parallax effect
            star['pos'][0] = (star['pos'][0] - 0.1 * star['size']) % self.SCREEN_WIDTH
            brightness = int(100 * star['size'])
            color = (brightness, brightness, brightness + 20)
            pygame.draw.circle(self.screen, color, star['pos'], star['size'])

    def _render_planets_and_selector(self):
        pulse = math.sin(self.steps * 0.05)
        
        # Determine risky planets dynamically
        for i, p in enumerate(self.planets):
            is_risky = False
            for a in self.asteroids:
                dist = math.hypot(p['pos'][0] - a['pos'][0], p['pos'][1] - a['pos'][1])
                if dist < p['radius'] + a['size'] + 50:
                    is_risky = True
                    break
            p['is_risky'] = is_risky

        # Draw planets
        for i, p in enumerate(self.planets):
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            radius = p['radius']
            
            if p['type'] == 'target':
                color = self.COLOR_PLANET_TARGET
                glow_radius = int(radius + 10 + 3 * pulse)
                glow_color = (*color, 60)
                surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, glow_color, (glow_radius, glow_radius), glow_radius)
                self.screen.blit(surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius))
            elif p['is_risky']:
                color = self.COLOR_PLANET_RISKY
            else:
                color = self.COLOR_PLANET_SAFE
            
            final_radius = int(radius + (1 if i == self.current_planet_idx else 0) * pulse * 2)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], final_radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], final_radius, color)

        # Draw selector
        if self.player_state == "ON_PLANET" and len(self.sorted_planet_indices) > 0:
            target_idx = self.sorted_planet_indices[self.selector_idx]
            p = self.planets[target_idx]
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'] + 8 + 3 * pulse)
            
            angle = (self.steps * 0.1) % (2 * math.pi)
            for i in range(3):
                start_angle = angle + i * (2 * math.pi / 3)
                end_angle = start_angle + math.pi / 3
                pygame.draw.arc(self.screen, self.COLOR_SELECTOR, (pos_int[0] - radius, pos_int[1] - radius, radius*2, radius*2), start_angle, end_angle, 2)

    def _render_asteroids(self):
        for a in self.asteroids:
            size = a['size']
            angle_rad = math.radians(a['rot'])
            points = []
            for i in range(3):
                theta = angle_rad + i * (2 * math.pi / 3)
                x = a['pos'][0] + size * math.cos(theta)
                y = a['pos'][1] + size * math.sin(theta)
                points.append((int(x), int(y)))
            
            if len(points) == 3:
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_player_and_trail(self):
        # Trail
        for i, pos in enumerate(self.player_trail):
            alpha = int(255 * (i / len(self.player_trail)))
            color = (*self.COLOR_TRAIL, alpha)
            radius = int(self.player_radius * 0.5 * (i / len(self.player_trail)))
            if radius > 0:
                surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (radius, radius), radius)
                self.screen.blit(surf, (int(pos[0]) - radius, int(pos[1]) - radius))

        # Player
        pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        glow_radius = int(self.player_radius * 1.8)
        glow_color = (*self.COLOR_PLAYER, 80)
        surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius))

        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.player_radius, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / 30)))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / 30))
            if size > 0:
                surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size, size), size)
                self.screen.blit(surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        # Lives
        for i in range(self.lives):
            pos = (self.SCREEN_WIDTH - 20 - i * 25, 20)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Planet Hopper")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # --- Action mapping for human play ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = [0, 0, 0]

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            # Key Down
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    action = [0, 0, 0]
            
            # Key Up
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    action[0] = 0
                elif event.key == pygame.K_SPACE: action[1] = 0
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: action[2] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()