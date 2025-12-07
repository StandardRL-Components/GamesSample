
# Generated: 2025-08-27T13:02:41.899504
# Source Brief: brief_00240.md
# Brief Index: 240

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold Space to mine nearby asteroids. Dodge the red meteors!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship, dodging meteors and collecting minerals from asteroids in a top-down arcade environment."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.MINERALS_TO_WIN = 50
        
        # Player settings
        self.PLAYER_ACCELERATION = 0.8
        self.PLAYER_FRICTION = 0.92
        self.PLAYER_MAX_SPEED = 8
        self.PLAYER_RADIUS = 12
        self.MINING_RADIUS = 60
        self.MINING_RATE = 0.1 # minerals per step

        # Entity settings
        self.INITIAL_ASTEROIDS = 10
        self.MAX_ASTEROIDS = 15
        self.INITIAL_METEORS = 3
        self.MAX_METEORS = 8
        self.BASE_METEOR_SPEED = 1.5
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_SHIP_EMPTY = (150, 150, 170)
        self.COLOR_SHIP_FULL = (255, 223, 0)
        self.COLOR_SHIP_GLOW = (100, 150, 255)
        self.COLOR_ASTEROID = (139, 69, 19)
        self.COLOR_METEOR = (255, 50, 50)
        self.COLOR_MINERAL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.player_vel = None
        self.asteroids = None
        self.meteors = None
        self.particles = None
        self.stars = None
        self.steps = 0
        self.score = 0
        self.minerals_collected = 0
        self.game_over = False
        self.win = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        
        self.steps = 0
        self.score = 0
        self.minerals_collected = 0
        self.game_over = False
        self.win = False
        
        self.particles = []
        self._spawn_stars()
        self._spawn_initial_asteroids()
        self._spawn_initial_meteors()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.02  # Cost of living

        self._handle_input(action)
        self._update_player()
        self._update_meteors()
        self._update_particles()
        
        mining_reward = self._handle_mining(action[1] == 1)
        reward += mining_reward

        collision_penalty = self._handle_collisions()
        reward += collision_penalty

        self._spawn_entities()

        terminated = self.game_over or self.steps >= self.MAX_STEPS or self.win
        if self.win:
            reward = 100.0
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 1: # Up
            self.player_vel[1] -= self.PLAYER_ACCELERATION
        elif movement == 2: # Down
            self.player_vel[1] += self.PLAYER_ACCELERATION
        elif movement == 3: # Left
            self.player_vel[0] -= self.PLAYER_ACCELERATION
        elif movement == 4: # Right
            self.player_vel[0] += self.PLAYER_ACCELERATION

    def _update_player(self):
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION
        
        # Clamp speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
            
        # Update position
        self.player_pos += self.player_vel
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

        # Exhaust particles
        if np.linalg.norm(self.player_vel) > 0.5:
            for _ in range(2):
                offset = self.np_random.uniform(-5, 5, size=2)
                vel_offset = self.np_random.uniform(-0.5, 0.5, size=2)
                particle_pos = self.player_pos - self.player_vel * 2 + offset
                particle_vel = -self.player_vel * 0.5 + vel_offset
                self.particles.append({
                    'pos': particle_pos, 'vel': particle_vel, 'life': 15, 'color': (100, 150, 255), 'radius': self.np_random.integers(1, 4)
                })


    def _handle_mining(self, space_held):
        if not space_held:
            return 0

        reward = 0
        minable_asteroid = None
        min_dist = self.MINING_RADIUS

        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                minable_asteroid = asteroid
        
        if minable_asteroid:
            # sfx: mining_beam.wav
            minerals_mined = self.MINING_RATE
            minable_asteroid['minerals'] -= minerals_mined
            self.minerals_collected += minerals_mined
            
            reward += 0.1 * minerals_mined
            if minable_asteroid['initial_minerals'] >= 4 and minable_asteroid['reward_given'] == False:
                 reward += 1.0
                 minable_asteroid['reward_given'] = True
            
            # Mining particles
            for _ in range(3):
                angle = self.np_random.uniform(0, 2 * math.pi)
                p_pos = minable_asteroid['pos'] + np.array([math.cos(angle), math.sin(angle)]) * minable_asteroid['radius']
                p_vel = (self.player_pos - p_pos) / 30.0
                self.particles.append({
                    'pos': p_pos, 'vel': p_vel, 'life': 30, 'color': self.COLOR_MINERAL, 'radius': self.np_random.integers(2, 4)
                })

            if minable_asteroid['minerals'] <= 0:
                # sfx: asteroid_break.wav
                self.asteroids.remove(minable_asteroid)
        
        if self.minerals_collected >= self.MINERALS_TO_WIN and not self.win:
            self.win = True
            # sfx: victory.wav

        return reward

    def _handle_collisions(self):
        for meteor in self.meteors:
            dist = np.linalg.norm(self.player_pos - meteor['pos'])
            if dist < self.PLAYER_RADIUS + meteor['radius']:
                self.game_over = True
                # sfx: explosion.wav
                # Explosion particles
                for _ in range(50):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 6)
                    p_vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                    self.particles.append({
                        'pos': self.player_pos.copy(), 'vel': p_vel, 'life': self.np_random.integers(20, 40), 'color': random.choice([self.COLOR_METEOR, (255,165,0), (200,200,200)]), 'radius': self.np_random.integers(2, 5)
                    })
                return -5.0
        return 0.0

    def _update_meteors(self):
        for meteor in self.meteors[:]:
            meteor['pos'] += meteor['vel']
            if not ((-meteor['radius'] < meteor['pos'][0] < self.WIDTH + meteor['radius']) and \
                    (-meteor['radius'] < meteor['pos'][1] < self.HEIGHT + meteor['radius'])):
                self.meteors.remove(meteor)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_entities(self):
        if len(self.asteroids) < self.MAX_ASTEROIDS and self.np_random.random() < 0.02:
            self._spawn_asteroid()
        if len(self.meteors) < self.MAX_METEORS and self.np_random.random() < 0.05:
            self._spawn_meteor()

    def _spawn_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append({
                'pos': (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'brightness': self.np_random.uniform(50, 150)
            })

    def _spawn_initial_asteroids(self):
        self.asteroids = []
        for _ in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid(on_screen=True)

    def _spawn_initial_meteors(self):
        self.meteors = []
        for _ in range(self.INITIAL_METEORS):
            self._spawn_meteor()

    def _spawn_asteroid(self, on_screen=False):
        minerals = self.np_random.integers(1, 6)
        radius = 10 + minerals * 3
        
        pos = np.array([
            self.np_random.uniform(radius, self.WIDTH - radius),
            self.np_random.uniform(radius, self.HEIGHT - radius)
        ])

        # Ensure it doesn't spawn on the player
        while np.linalg.norm(pos - self.player_pos) < self.MINING_RADIUS + radius:
             pos = np.array([
                self.np_random.uniform(radius, self.WIDTH - radius),
                self.np_random.uniform(radius, self.HEIGHT - radius)
            ])

        self.asteroids.append({
            'pos': pos, 'minerals': minerals, 'initial_minerals': minerals, 'radius': radius,
            'shape': self._create_asteroid_shape(radius), 'reward_given': False
        })
    
    def _create_asteroid_shape(self, radius):
        points = []
        num_points = self.np_random.integers(7, 12)
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = self.np_random.uniform(radius * 0.7, radius * 1.0)
            points.append((dist * math.cos(angle), dist * math.sin(angle)))
        return points

    def _spawn_meteor(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -20.0])
            angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75)
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20.0])
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        elif edge == 2: # Left
            pos = np.array([-20.0, self.np_random.uniform(0, self.HEIGHT)])
            angle = self.np_random.uniform(-math.pi * 0.25, math.pi * 0.25)
        else: # Right
            pos = np.array([self.WIDTH + 20.0, self.np_random.uniform(0, self.HEIGHT)])
            angle = self.np_random.uniform(math.pi * 0.75, math.pi * 1.25)

        level = self.minerals_collected // 10
        speed = self.BASE_METEOR_SPEED + level * 0.2 + self.np_random.uniform(-0.2, 0.2)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        radius = self.np_random.integers(5, 15)

        self.meteors.append({'pos': pos, 'vel': vel, 'radius': radius})

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
            "minerals": self.minerals_collected,
        }

    def _render_game(self):
        # Stars
        for star in self.stars:
            brightness = star['brightness'] * (0.8 + 0.2 * math.sin(pygame.time.get_ticks() / 1000.0 + star['pos'][0]))
            color = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, color, star['pos'], 1)

        # Asteroids
        for asteroid in self.asteroids:
            points = [(p[0] + asteroid['pos'][0], p[1] + asteroid['pos'][1]) for p in asteroid['shape']]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

        # Meteors
        for meteor in self.meteors:
            pos = meteor['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], meteor['radius'], self.COLOR_METEOR)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], meteor['radius'], self.COLOR_METEOR)
            # Trail
            for i in range(5):
                trail_pos = pos - meteor['vel'] * (i + 1) * 1.5
                alpha = 150 - i * 30
                color = (*self.COLOR_METEOR, alpha)
                temp_surf = pygame.Surface((meteor['radius']*2, meteor['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (meteor['radius'], meteor['radius']), int(meteor['radius'] * (1 - i/6)))
                self.screen.blit(temp_surf, (int(trail_pos[0]-meteor['radius']), int(trail_pos[1]-meteor['radius'])))

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), int(p['radius'] * (p['life']/30.0)))

        # Player
        if not self.game_over:
            pos = self.player_pos.astype(int)
            
            # Glow effect
            glow_radius = int(self.PLAYER_RADIUS * 2.5)
            temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            glow_color = (*self.COLOR_SHIP_GLOW, 50)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (pos[0]-glow_radius, pos[1]-glow_radius))

            # Ship color based on minerals
            ratio = min(1.0, self.minerals_collected / self.MINERALS_TO_WIN)
            ship_color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(self.COLOR_SHIP_EMPTY, self.COLOR_SHIP_FULL))

            # Ship body (triangle)
            angle = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else -math.pi/2
            p1 = (pos[0] + self.PLAYER_RADIUS * math.cos(angle), pos[1] + self.PLAYER_RADIUS * math.sin(angle))
            p2 = (pos[0] + self.PLAYER_RADIUS * math.cos(angle + 2.2), pos[1] + self.PLAYER_RADIUS * math.sin(angle + 2.2))
            p3 = (pos[0] + self.PLAYER_RADIUS * math.cos(angle - 2.2), pos[1] + self.PLAYER_RADIUS * math.sin(angle - 2.2))
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], ship_color)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], ship_color)

    def _render_ui(self):
        # Mineral count
        mineral_text = self.font.render(f"MINERALS: {int(self.minerals_collected)} / {self.MINERALS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(mineral_text, (10, 10))
        
        # Level (meteor speed)
        level = self.minerals_collected // 10
        level_text = self.font.render(f"DANGER LEVEL: {level + 1}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 35))

        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER" if not self.win else "MISSION COMPLETE!"
            color = self.COLOR_METEOR if not self.win else self.COLOR_MINERAL
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    import time
    
    # Set this to 'human' to see the game window
    render_mode = "human" # "rgb_array" for training, "human" for playing
    
    if render_mode == "human":
        # Pygame needs a display for human rendering
        env = GameEnv(render_mode="rgb_array")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Asteroid Miner")
    else:
        env = GameEnv()

    obs, info = env.reset()
    terminated = False
    
    total_reward = 0
    start_time = time.time()
    
    # --- Human player controls ---
    keys_pressed = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
    }

    while not terminated:
        # For human play
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key in keys_pressed:
                        keys_pressed[event.key] = True
                    if event.key == pygame.K_r: # Reset
                        obs, info = env.reset()
                        total_reward = 0
                if event.type == pygame.KEYUP:
                    if event.key in keys_pressed:
                        keys_pressed[event.key] = False
            
            # Map keys to MultiDiscrete action
            movement = 0 # no-op
            if keys_pressed[pygame.K_UP]: movement = 1
            elif keys_pressed[pygame.K_DOWN]: movement = 2
            elif keys_pressed[pygame.K_LEFT]: movement = 3
            elif keys_pressed[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys_pressed[pygame.K_SPACE] else 0
            shift_held = 1 if keys_pressed[pygame.K_LSHIFT] else 0
            
            action = [movement, space_held, shift_held]
        else:
            # For agent play (random actions)
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Minerals: {int(info['minerals'])}")
            time.sleep(2) # Pause on game over
            obs, info = env.reset()
            total_reward = 0

        # Render for human
        if render_mode == "human":
            # The observation is already a rendered frame
            # We just need to convert it back to a Pygame surface to display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.FPS)

    env.close()