import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ↑↓←→ to apply thrust. Hold space to mine nearby asteroids."
    )

    # User-facing game description
    game_description = (
        "Pilot a spaceship in a top-down asteroid field, mining ore while dodging collisions. "
        "Collect 100 units of ore before the 60-second timer runs out to win."
    )

    # Frames auto-advance for smooth gameplay
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds
    WIN_SCORE = 100

    # Colors
    COLOR_BG = (13, 13, 30)
    COLOR_PLAYER = (66, 245, 176)
    COLOR_PLAYER_EXHAUST = (255, 100, 0)
    COLOR_ASTEROID_DARK = (90, 50, 30)
    COLOR_ASTEROID_LIGHT = (139, 69, 19)
    COLOR_ORE_PARTICLE = (255, 215, 0)
    COLOR_EXPLOSION = (255, 69, 0)
    COLOR_LASER = (255, 0, 0, 150)
    COLOR_TEXT = (255, 255, 255)
    
    # Player Physics
    PLAYER_THRUST = 0.4
    PLAYER_DRAG = 0.96
    PLAYER_ROTATION_SPEED = 0.1
    PLAYER_SIZE = 12
    PLAYER_COLLISION_RADIUS = 8

    # Asteroid Physics
    MIN_ASTEROIDS = 8
    MAX_ASTEROIDS = 15
    ASTEROID_BASE_SPEED = 0.5
    DIFFICULTY_INTERVAL = 1000 # steps

    # Mining
    MINING_RANGE = 100
    MINING_RATE = 1 # ore per step

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.asteroids = None
        self.particles = None
        self.mining_target = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.asteroid_speed_multiplier = None
        self.starfield = None

        self.reset()
        
        # Run validation check
        try:
            self.validate_implementation()
        except AssertionError as e:
            print(f"Implementation validation failed: {e}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float64)
        self.player_angle = -math.pi / 2  # Pointing up

        self.asteroids = []
        self.particles = []
        self.mining_target = None
        self.asteroid_speed_multiplier = 1.0

        if self.starfield is None:
             self.starfield = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
                for _ in range(150)
            ]

        for _ in range(self.np_random.integers(self.MIN_ASTEROIDS, self.MAX_ASTEROIDS + 1)):
            self._add_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If game is over, do nothing but return the final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement, space_held, _ = action
        space_held = space_held == 1

        reward = -0.01  # Time penalty

        # --- Update Game Logic ---
        self._handle_input(movement, space_held)
        self._update_player()
        self._update_asteroids()
        self._update_particles()
        
        # Mining logic
        if self.mining_target is not None:
            # Sound placeholder: # sfx_mining_laser_loop()
            mined_amount = min(self.mining_target['ore'], self.MINING_RATE)
            self.mining_target['ore'] -= mined_amount
            self.score += mined_amount
            reward += mined_amount * 0.1
            
            # Create ore particles
            for _ in range(2):
                self._create_particle(
                    self.mining_target['pos'], self.COLOR_ORE_PARTICLE, 1.5, 20, 
                    target=self.player_pos, speed_factor=0.1
                )

            if self.mining_target['ore'] <= 0:
                reward += 1.0  # Bonus for fully mining an asteroid
                # Sound placeholder: # sfx_asteroid_depleted()
                depleted_asteroid = self.mining_target
                self.asteroids = [a for a in self.asteroids if a is not depleted_asteroid]
                self.mining_target = None

        # Check for collisions
        if self._check_collisions():
            reward = -100.0
            self.game_over = True
            # Sound placeholder: # sfx_player_explosion()
            for _ in range(50):
                self._create_particle(self.player_pos, self.COLOR_EXPLOSION, 3, 40, speed_factor=0.2)

        # Update difficulty
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.asteroid_speed_multiplier += 0.05

        # Check termination conditions
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward = 100.0
            self.game_over = True
            self.win = True
            terminated = True
            # Sound placeholder: # sfx_win_game()
        elif self.steps >= self.MAX_STEPS - 1:
            if not self.game_over: # Don't overwrite collision penalty
                reward = -10.0
            self.game_over = True
            terminated = True
            # Sound placeholder: # sfx_time_up()
        elif self.game_over: # From collision
            terminated = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # --- Movement ---
        thrust_vec = np.array([0.0, 0.0])
        target_angle = self.player_angle
        moved = False
        
        if movement == 1: # Up
            thrust_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * self.PLAYER_THRUST
        if movement == 2: # Down
            thrust_vec = -np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * self.PLAYER_THRUST * 0.5
        if movement == 3: # Left
            self.player_angle -= self.PLAYER_ROTATION_SPEED
        if movement == 4: # Right
            self.player_angle += self.PLAYER_ROTATION_SPEED

        self.player_vel += thrust_vec

        if np.any(thrust_vec):
             # Sound placeholder: # sfx_player_thrust()
             self._create_exhaust_particles()

        # --- Mining ---
        self.mining_target = None
        if space_held:
            closest_asteroid = None
            min_dist = self.MINING_RANGE
            for asteroid in self.asteroids:
                dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid
            self.mining_target = closest_asteroid

    def _update_player(self):
        self.player_vel *= self.PLAYER_DRAG
        self.player_pos += self.player_vel

        # World wrapping
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT
    
    def _add_asteroid(self, pos=None, size=None):
        if pos is None:
            # Spawn at edges
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = np.array([self.np_random.uniform(0, self.WIDTH), -50.0])
            elif edge == 1: # Bottom
                pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 50.0])
            elif edge == 2: # Left
                pos = np.array([-50.0, self.np_random.uniform(0, self.HEIGHT)])
            else: # Right
                pos = np.array([self.WIDTH + 50.0, self.np_random.uniform(0, self.HEIGHT)])

        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.5, 1.5) * self.ASTEROID_BASE_SPEED * self.asteroid_speed_multiplier
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        
        if size is None:
            size = self.np_random.uniform(15, 45)

        num_vertices = self.np_random.integers(7, 12)
        shape_points = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            radius = size * self.np_random.uniform(0.7, 1.1)
            shape_points.append((math.cos(angle) * radius, math.sin(angle) * radius))

        self.asteroids.append({
            'pos': pos.astype(np.float64),
            'vel': vel.astype(np.float64),
            'size': size,
            'ore': int(size * 2),
            'shape': shape_points,
            'angle': self.np_random.uniform(0, 2 * math.pi),
            'rot_speed': self.np_random.uniform(-0.01, 0.01)
        })

    def _update_asteroids(self):
        asteroids_on_screen = []
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_speed']
            
            # Check if it's still on-screen
            x, y = asteroid['pos']
            s = asteroid['size']
            if not (x < -s*2 or x > self.WIDTH + s*2 or y < -s*2 or y > self.HEIGHT + s*2):
                asteroids_on_screen.append(asteroid)
        
        self.asteroids = asteroids_on_screen
        
        # Respawn if count is too low
        while len(self.asteroids) < self.MIN_ASTEROIDS:
            self._add_asteroid()

    def _create_particle(self, pos, color, size, lifetime, target=None, speed_factor=1.0):
        if target is None:
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_factor
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        else:
            direction = target - pos
            dist = np.linalg.norm(direction)
            if dist > 1:
                vel = (direction / dist) * self.np_random.uniform(2, 4) * speed_factor
            else:
                vel = np.array([0.0, 0.0])

        self.particles.append({
            'pos': pos.copy(),
            'vel': vel,
            'color': color,
            'size': size,
            'lifetime': lifetime,
            'max_lifetime': lifetime
        })
    
    def _create_exhaust_particles(self):
        # Create particles moving away from the ship's nose
        exhaust_angle = self.player_angle + math.pi + self.np_random.uniform(-0.3, 0.3)
        # Position behind the player
        offset = np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * -self.PLAYER_SIZE
        pos = self.player_pos + offset
        
        speed = self.np_random.uniform(1, 2)
        vel = np.array([math.cos(exhaust_angle), math.sin(exhaust_angle)]) * speed + self.player_vel * 0.5
        
        self.particles.append({
            'pos': pos,
            'vel': vel,
            'color': self.COLOR_PLAYER_EXHAUST,
            'size': self.np_random.uniform(2, 4),
            'lifetime': 15,
            'max_lifetime': 15
        })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _check_collisions(self):
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < asteroid['size'] * 0.8 + self.PLAYER_COLLISION_RADIUS:
                return True
        return False

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
        # Render starfield
        for x, y, size in self.starfield:
            c = self.np_random.integers(50, 80)
            pygame.draw.rect(self.screen, (c,c,c), (x, y, size, size))

        # Render particles
        for p in self.particles:
            alpha = p['lifetime'] / p['max_lifetime']
            current_size = int(p['size'] * alpha)
            if current_size > 0:
                # Create a temporary color tuple with alpha for drawing
                color_val = p['color']
                if len(color_val) == 4: # already has alpha
                    color_with_alpha = (color_val[0], color_val[1], color_val[2], color_val[3] * alpha)
                else: # does not have alpha
                    color_with_alpha = (color_val[0], color_val[1], color_val[2])
                
                # Pygame circle does not support alpha, but we can fake it by blending
                # For simplicity, we just use the color and let size represent lifetime
                pygame.draw.circle(self.screen, color_with_alpha[:3], p['pos'].astype(int), current_size)


        # Render asteroids
        for asteroid in self.asteroids:
            points = [
                (
                    asteroid['pos'][0] + p[0] * math.cos(asteroid['angle']) - p[1] * math.sin(asteroid['angle']),
                    asteroid['pos'][1] + p[0] * math.sin(asteroid['angle']) + p[1] * math.cos(asteroid['angle'])
                )
                for p in asteroid['shape']
            ]
            
            # Interpolate color based on ore content
            ore_ratio = max(0, min(1, asteroid['ore'] / (asteroid['size'] * 2)))
            color = tuple(int(c1 * ore_ratio + c2 * (1 - ore_ratio)) for c1, c2 in zip(self.COLOR_ASTEROID_LIGHT, self.COLOR_ASTEROID_DARK))

            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Render mining laser
        if self.mining_target:
            flicker = self.np_random.uniform(0.8, 1.0)
            start_pos = self.player_pos.astype(int)
            end_pos = self.mining_target['pos'].astype(int)
            
            # Draw a thick, semi-transparent line
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, int(3 * flicker))
        
        # Render player
        if not (self.game_over and not self.win): # Don't draw player if destroyed
            s = self.PLAYER_SIZE
            a = self.player_angle
            
            p1 = (self.player_pos[0] + s * math.cos(a), self.player_pos[1] + s * math.sin(a))
            p2 = (self.player_pos[0] + s * math.cos(a + 2.356), self.player_pos[1] + s * math.sin(a + 2.356))
            p3 = (self.player_pos[0] + s * math.cos(a - 2.356), self.player_pos[1] + s * math.sin(a - 2.356))
            
            points = [p1, p2, p3]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Render score (ore)
        score_text = self.font_ui.render(f"ORE: {int(self.score)}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Render game over message
        if self.game_over:
            if self.win:
                msg = "MISSION COMPLETE"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_EXPLOSION
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS),
            "win": self.win
        }

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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen for display
    env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        # Get player input
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame specific loop management ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        env.clock.tick(env.FPS)
        
        # Print info
        if terminated:
            print(f"Episode finished. Final score: {info['score']}, Total reward: {total_reward:.2f}")

    # Keep the window open for a bit after the game ends
    pygame.time.wait(2000)
    env.close()