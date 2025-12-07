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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space for a speed burst."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a spaceship through a dense asteroid field, collecting valuable coins while a 60-second timer counts down. Survive until the end for a massive bonus!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 60 * FPS

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_ASTEROID = (100, 100, 110)
    COLOR_COIN = (255, 223, 0)
    COLOR_UI_TEXT = (200, 200, 220)
    COLOR_TIMER_BAR = (0, 150, 255)
    COLOR_COLLISION_FLASH = (255, 0, 0)
    COLOR_THRUSTER = (255, 150, 50)
    COLOR_SPARK = (220, 220, 255)
    
    # Player settings
    PLAYER_ACCELERATION = 0.4
    PLAYER_DAMPING = 0.95
    PLAYER_MAX_SPEED = 8.0
    PLAYER_BOOST_MULTIPLIER = 2.0
    PLAYER_RADIUS = 10

    # Game settings
    INITIAL_ASTEROIDS = 10
    ASTEROID_ADD_INTERVAL = 10 * FPS  # Add asteroid every 10 seconds
    ASTEROID_SPEED_INCREASE_INTERVAL = 15 * FPS # Increase speed every 15 seconds
    INITIAL_COINS = 5
    COIN_RADIUS = 8
    
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
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.game_over_font = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.asteroids = []
        self.coins = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.asteroid_speed_multiplier = 1.0
        self.collision_flash_timer = 0
        
        # Seed the random number generator
        self.np_random = None
        self.reset(seed=0) # Initialize np_random and other states

        # Ensure the environment is correctly implemented
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_angle = -math.pi / 2 # Pointing up
        
        self.asteroids = []
        self.coins = []
        self.particles = []
        
        self.asteroid_speed_multiplier = 1.0
        num_asteroids = self.INITIAL_ASTEROIDS
        for _ in range(num_asteroids):
            self._spawn_asteroid()
            
        for _ in range(self.INITIAL_COINS):
            self._spawn_coin()
            
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.collision_flash_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If game is over, no state should change. Return last state.
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1 # Survival reward
        
        self._handle_input(action)
        self._update_player()
        self._update_asteroids()
        self._update_particles()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        # Coin distance penalty
        if self.coins:
            distances_to_coins = [np.linalg.norm(self.player_pos - coin['pos']) for coin in self.coins]
            if min(distances_to_coins) > 20:
                reward -= 0.2
        
        self.steps += 1
        self._update_difficulty()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over and self.steps >= self.MAX_STEPS:
            reward += 50 # Survival bonus

        if self.game_over:
            reward = -100 # Collision penalty
            self.collision_flash_timer = 3 # frames
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        accel = np.array([0.0, 0.0], dtype=np.float32)
        
        if movement == 1: # Up
            accel[1] -= self.PLAYER_ACCELERATION
        if movement == 2: # Down
            accel[1] += self.PLAYER_ACCELERATION
        if movement == 3: # Left
            accel[0] -= self.PLAYER_ACCELERATION
        if movement == 4: # Right
            accel[0] += self.PLAYER_ACCELERATION
            
        if space_held:
            accel *= self.PLAYER_BOOST_MULTIPLIER
            # Sound: Boost sound
            
        self.player_vel += accel
        
    def _update_player(self):
        # Limit speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel * (self.PLAYER_MAX_SPEED / speed)
            
        # Apply damping
        self.player_vel *= self.PLAYER_DAMPING
        
        # Update position and wrap around screen
        self.player_pos += self.player_vel
        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] %= self.SCREEN_HEIGHT
        
        # Update angle to face velocity direction
        if np.linalg.norm(self.player_vel) > 0.1:
            self.player_angle = math.atan2(self.player_vel[1], self.player_vel[0])

        # Add thruster particles if moving
        if np.linalg.norm(self.player_vel) > 1.0:
            for _ in range(2):
                self._spawn_particle(
                    self.player_pos, 
                    -self.player_vel, 
                    self.COLOR_THRUSTER, 
                    lifespan=10, 
                    size=3
                )
    
    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['pos'][0] %= self.SCREEN_WIDTH
            asteroid['pos'][1] %= self.SCREEN_HEIGHT
            asteroid['angle'] += asteroid['rot_speed']
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
    
    def _handle_collisions(self):
        reward = 0
        
        # Player-Asteroid
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                self.game_over = True
                # Sound: Explosion
                self._spawn_explosion(self.player_pos, self.COLOR_PLAYER, 30)
                self._spawn_explosion(asteroid['pos'], self.COLOR_ASTEROID, 30)
                return 0 # Terminal reward handled in step()
            elif dist < self.PLAYER_RADIUS + asteroid['radius'] + 20: # Near miss
                if self.np_random.random() < 0.1:
                    self._spawn_particle(self.player_pos, self.player_vel * 0.1, self.COLOR_SPARK, lifespan=15, size=2)

        # Player-Coin
        collected_coins = []
        for i, coin in enumerate(self.coins):
            dist = np.linalg.norm(self.player_pos - coin['pos'])
            if dist < self.PLAYER_RADIUS + self.COIN_RADIUS:
                collected_coins.append(i)
                self.score += 1
                reward += 1.0
                # Sound: Coin collect
                
                # Risky collection bonus
                distances_to_asteroids = [np.linalg.norm(coin['pos'] - ast['pos']) for ast in self.asteroids]
                if distances_to_asteroids:
                    bonus = 0.5 * (100 - min(distances_to_asteroids))
                    reward += min(5.0, max(0, bonus / 20.0))

                self._spawn_explosion(coin['pos'], self.COLOR_COIN, 10, speed_mult=0.5)

        if collected_coins:
            self.coins = [c for i, c in enumerate(self.coins) if i not in collected_coins]
            for _ in collected_coins:
                self._spawn_coin()
        
        return reward
        
    def _update_difficulty(self):
        if self.steps > 0:
            if self.steps % self.ASTEROID_ADD_INTERVAL == 0:
                self._spawn_asteroid()
            if self.steps % self.ASTEROID_SPEED_INCREASE_INTERVAL == 0:
                self.asteroid_speed_multiplier += 0.2

    def _spawn_asteroid(self):
        # Spawn away from the player
        while True:
            pos = self.np_random.uniform([0, 0], [self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
            if np.linalg.norm(pos - self.player_pos) > 100:
                break
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.5, 1.5) * self.asteroid_speed_multiplier
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        radius = self.np_random.integers(15, 35)
        
        # Generate points for a semi-random polygon
        num_vertices = self.np_random.integers(5, 9)
        points = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(0.7, 1.0) * radius
            points.append((math.cos(angle) * dist, math.sin(angle) * dist))
            
        self.asteroids.append({
            'pos': pos,
            'vel': vel,
            'radius': radius,
            'angle': 0,
            'rot_speed': self.np_random.uniform(-0.02, 0.02),
            'shape_points': points
        })
        
    def _spawn_coin(self):
        while True:
            pos = self.np_random.uniform([0, 0], [self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
            # Ensure it doesn't spawn inside an asteroid
            too_close = False
            for ast in self.asteroids:
                if np.linalg.norm(pos - ast['pos']) < ast['radius'] + self.COIN_RADIUS + 5:
                    too_close = True
                    break
            if not too_close:
                break
        self.coins.append({'pos': pos})
        
    def _spawn_particle(self, pos, base_vel, color, lifespan, size):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.5, 2.0)
        vel = base_vel + np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        self.particles.append({
            'pos': pos.copy(),
            'vel': vel,
            'color': color,
            'lifespan': lifespan,
            'size': size
        })
        
    def _spawn_explosion(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, 4.0) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'lifespan': self.np_random.integers(15, 30),
                'size': self.np_random.uniform(1, 4)
            })
            
    def _get_observation(self):
        # Clear screen with background
        if self.collision_flash_timer > 0:
            self.screen.fill(self.COLOR_COLLISION_FLASH)
            self.collision_flash_timer -= 1
        else:
            self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]),
                int(p['size']), p['color']
            )

        # Render asteroids
        for asteroid in self.asteroids:
            points = []
            for x, y in asteroid['shape_points']:
                rotated_x = x * math.cos(asteroid['angle']) - y * math.sin(asteroid['angle'])
                rotated_y = x * math.sin(asteroid['angle']) + y * math.cos(asteroid['angle'])
                points.append((
                    int(asteroid['pos'][0] + rotated_x),
                    int(asteroid['pos'][1] + rotated_y)
                ))
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
        
        # Render coins
        for coin in self.coins:
            x, y = int(coin['pos'][0]), int(coin['pos'][1])
            pulse = abs(math.sin(self.steps * 0.1))
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.COIN_RADIUS, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.COIN_RADIUS, self.COLOR_COIN)
            inner_color = (min(255, self.COLOR_COIN[0]+50), min(255, self.COLOR_COIN[1]+50), self.COLOR_COIN[2])
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(self.COIN_RADIUS * 0.6 * pulse), inner_color)

        # Render player
        if not self.game_over and self.player_pos is not None and self.player_angle is not None:
            p1 = (
                self.player_pos[0] + math.cos(self.player_angle) * self.PLAYER_RADIUS * 1.5,
                self.player_pos[1] + math.sin(self.player_angle) * self.PLAYER_RADIUS * 1.5
            )
            p2 = (
                self.player_pos[0] + math.cos(self.player_angle + 2.5) * self.PLAYER_RADIUS,
                self.player_pos[1] + math.sin(self.player_angle + 2.5) * self.PLAYER_RADIUS
            )
            p3 = (
                self.player_pos[0] + math.cos(self.player_angle - 2.5) * self.PLAYER_RADIUS,
                self.player_pos[1] + math.sin(self.player_angle - 2.5) * self.PLAYER_RADIUS
            )
            points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer bar
        timer_width = self.SCREEN_WIDTH * (1 - self.steps / self.MAX_STEPS)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (0, 0, timer_width, 5))
        
        # Game Over text
        if self.game_over:
            text = "GAME OVER"
            if self.steps < self.MAX_STEPS: # Lost by collision
                text_surface = self.game_over_font.render(text, True, self.COLOR_COLLISION_FLASH)
            else: # Won by survival
                text_surface = self.game_over_font.render("YOU SURVIVED!", True, self.COLOR_TIMER_BAR)
            
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Temporarily set up state to get a valid observation
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        # FIX: self.player_angle was None, causing a TypeError in _render_game
        self.player_angle = -math.pi / 2
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Re-enable display for direct play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Dodger")
    clock = pygame.time.Clock()

    # --- Game Loop ---
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()


        # --- Get Player Input ---
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)

        # Movement
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
        else:
            action[0] = 0

        # Space for boost
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        # Shift (unused)
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()

        # --- Render to Screen ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Control Framerate ---
        clock.tick(GameEnv.FPS)

    env.close()