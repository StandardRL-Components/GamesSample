
# Generated: 2025-08-27T16:17:53.535289
# Source Brief: brief_01182.md
# Brief Index: 1182

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move your ship. Hold spacebar near an asteroid to mine it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship in a dangerous asteroid field. Collect 100 minerals to win, but watch out for deadly lasers and a ticking clock!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000 # Approx 100 seconds at 30fps
        self.TIME_LIMIT_SECONDS = 60
        
        # Player settings
        self.PLAYER_SPEED = 6
        self.PLAYER_SIZE = 12
        
        # Asteroid settings
        self.NUM_ASTEROIDS = 10
        self.MINING_RANGE = 50
        self.MINERALS_TO_WIN = 100
        
        # Laser settings
        self.INITIAL_LASER_HZ = 0.2
        self.MAX_LASER_HZ = 1.0
        self.LASER_SPEED = 10
        self.LASER_WARN_FRAMES = 15

        # Colors
        self.COLOR_BG = (10, 10, 26)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 128, 128)
        self.COLOR_ASTEROID = (128, 128, 128)
        self.COLOR_MINERAL = (255, 255, 0)
        self.COLOR_LASER = (255, 0, 0)
        self.COLOR_LASER_CORE = (255, 255, 255)
        self.COLOR_LASER_WARN = (255, 100, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Initialize state variables to None
        self.player_pos = None
        self.asteroids = []
        self.lasers = []
        self.particles = []
        self.starfield = []
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.laser_frequency = 0
        self.laser_spawn_timer = 0
        self.game_over = False
        self.win_status = False
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        
        self.asteroids = [self._create_asteroid() for _ in range(self.NUM_ASTEROIDS)]
        self.lasers = []
        self.particles = []
        self.starfield = self._create_starfield()

        self.steps = 0
        self.score = 0
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.laser_frequency = self.INITIAL_LASER_HZ
        self.laser_spawn_timer = self.FPS / self.laser_frequency
        
        self.game_over = False
        self.win_status = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Update Game Logic ---
        self._handle_player_movement(movement)
        minerals_collected = self._handle_mining(space_held)
        reward += minerals_collected

        self._update_lasers()
        self._update_particles()
        
        self.steps += 1
        self.time_left -= 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.laser_frequency = min(self.MAX_LASER_HZ, self.laser_frequency + 0.01)

        # --- Check for Termination ---
        collision_with_laser = any(self._check_collision(self.player_pos, self.PLAYER_SIZE, l['pos'], self.LASER_SPEED) for l in self.lasers if l['state'] == 'firing')
        
        self.win_status = self.score >= self.MINERALS_TO_WIN
        timed_out = self.time_left <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS

        terminated = self.win_status or collision_with_laser or timed_out or max_steps_reached
        
        if terminated:
            self.game_over = True
            if self.win_status:
                reward += 100
            else: # Lost
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        
        # Clamp position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _handle_mining(self, space_held):
        minerals_collected = 0
        if not space_held:
            return minerals_collected

        # Find closest asteroid
        closest_asteroid = None
        min_dist = float('inf')
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_asteroid = asteroid
        
        # Mine if in range
        if closest_asteroid and min_dist < self.MINING_RANGE + closest_asteroid['size'] and closest_asteroid['minerals'] > 0:
            closest_asteroid['minerals'] -= 1
            closest_asteroid['size'] = closest_asteroid['initial_size'] * (closest_asteroid['minerals'] / closest_asteroid['initial_minerals'])
            self.score += 1
            minerals_collected = 1
            
            # Spawn collection particle
            self._spawn_particle(closest_asteroid['pos'], self.COLOR_MINERAL, self.player_pos)
            
            if closest_asteroid['minerals'] <= 0:
                self.asteroids.remove(closest_asteroid)
                self.asteroids.append(self._create_asteroid()) # Respawn a new one
        
        return minerals_collected

    def _update_lasers(self):
        # Spawn new lasers
        self.laser_spawn_timer -= 1
        if self.laser_spawn_timer <= 0:
            self._spawn_laser()
            self.laser_spawn_timer = self.FPS / self.laser_frequency

        # Update existing lasers
        for laser in self.lasers[:]:
            if laser['state'] == 'warning':
                laser['warn_timer'] -= 1
                if laser['warn_timer'] <= 0:
                    laser['state'] = 'firing'
            elif laser['state'] == 'firing':
                laser['pos'] += laser['vel'] * self.LASER_SPEED
                if not (0 <= laser['pos'][0] <= self.WIDTH and 0 <= laser['pos'][1] <= self.HEIGHT):
                    self.lasers.remove(laser)
    
    def _update_particles(self):
        for p in self.particles[:]:
            # Move towards target
            if p['target'] is not None:
                direction = p['target'] - p['pos']
                dist = np.linalg.norm(direction)
                if dist < p['speed']:
                    self.particles.remove(p)
                    continue
                p['pos'] += (direction / dist) * p['speed']
            
            # Fade out
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render starfield with parallax
        player_offset = (self.player_pos - np.array([self.WIDTH/2, self.HEIGHT/2]))
        for star in self.starfield:
            pos = star['pos'] - player_offset * star['depth']
            pygame.draw.circle(self.screen, star['color'], pos.astype(int), int(star['size']))

        # Render asteroids
        for asteroid in self.asteroids:
            points = [
                (asteroid['pos'][0] + p[0] * asteroid['size'], asteroid['pos'][1] + p[1] * asteroid['size'])
                for p in asteroid['shape_points']
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Render lasers
        for laser in self.lasers:
            if laser['state'] == 'warning':
                radius = int(15 * (1 - laser['warn_timer'] / self.LASER_WARN_FRAMES))
                pygame.draw.circle(self.screen, self.COLOR_LASER_WARN, laser['spawn_pos'].astype(int), radius, 2)
            elif laser['state'] == 'firing':
                start_pos = laser['pos']
                end_pos = laser['pos'] - laser['vel'] * 20 # Tail effect
                pygame.draw.line(self.screen, self.COLOR_LASER, start_pos.astype(int), end_pos.astype(int), 5)
                pygame.draw.line(self.screen, self.COLOR_LASER_CORE, start_pos.astype(int), end_pos.astype(int), 2)

        # Render player
        player_int_pos = self.player_pos.astype(int)
        # Glow
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_GLOW, player_int_pos, self.PLAYER_SIZE + 4, 0)
        # Ship body
        points = [
            (player_int_pos[0], player_int_pos[1] - self.PLAYER_SIZE),
            (player_int_pos[0] - self.PLAYER_SIZE // 2, player_int_pos[1] + self.PLAYER_SIZE // 2),
            (player_int_pos[0] + self.PLAYER_SIZE // 2, player_int_pos[1] + self.PLAYER_SIZE // 2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"Minerals: {self.score}/{self.MINERALS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Time
        time_str = f"Time: {max(0, self.time_left // self.FPS):02d}"
        time_text = self.font.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        # Game Over message
        if self.game_over:
            outcome_text_str = "VICTORY!" if self.win_status else "GAME OVER"
            outcome_color = (0, 255, 0) if self.win_status else (255, 0, 0)
            outcome_text = self.font.render(outcome_text_str, True, outcome_color)
            text_rect = outcome_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(outcome_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "win": self.win_status,
        }
    
    # --- Helper methods for object creation ---
    def _create_asteroid(self):
        minerals = self.np_random.integers(15, 30)
        initial_size = minerals
        pos = self.np_random.uniform(low=20, high=[self.WIDTH-20, self.HEIGHT-20], size=2)
        
        # Create a random blob shape
        shape_points = []
        num_points = self.np_random.integers(6, 10)
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            radius = self.np_random.uniform(0.7, 1.1)
            shape_points.append((math.cos(angle) * radius, math.sin(angle) * radius))
            
        return {'pos': pos, 'size': initial_size, 'minerals': minerals, 'initial_size': initial_size, 'initial_minerals': minerals, 'shape_points': shape_points}

    def _spawn_laser(self):
        edge = self.np_random.integers(4) # 0:top, 1:bottom, 2:left, 3:right
        if edge == 0: # Top
            spawn_pos = np.array([self.np_random.uniform(0, self.WIDTH), -10])
            vel = np.array([0, 1])
        elif edge == 1: # Bottom
            spawn_pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 10])
            vel = np.array([0, -1])
        elif edge == 2: # Left
            spawn_pos = np.array([-10, self.np_random.uniform(0, self.HEIGHT)])
            vel = np.array([1, 0])
        else: # Right
            spawn_pos = np.array([self.WIDTH + 10, self.np_random.uniform(0, self.HEIGHT)])
            vel = np.array([-1, 0])
        
        self.lasers.append({'spawn_pos': spawn_pos, 'pos': spawn_pos.copy(), 'vel': vel, 'state': 'warning', 'warn_timer': self.LASER_WARN_FRAMES})

    def _spawn_particle(self, pos, color, target=None):
        self.particles.append({
            'pos': pos.copy(),
            'vel': self.np_random.uniform(-1, 1, size=2),
            'color': color,
            'lifetime': self.np_random.integers(15, 30),
            'max_lifetime': 30,
            'size': self.np_random.integers(2, 5),
            'target': target,
            'speed': self.np_random.uniform(4, 8)
        })

    def _create_starfield(self):
        stars = []
        for _ in range(150):
            depth = self.np_random.uniform(0.1, 0.6)
            stars.append({
                'pos': self.np_random.uniform(low=0, high=[self.WIDTH, self.HEIGHT], size=2),
                'size': (1 - depth) * 2.5,
                'depth': depth,
                'color': (int(255 * (1-depth*0.5)),) * 3
            })
        return stars

    def _check_collision(self, pos1, r1, pos2, r2):
        return np.linalg.norm(pos1 - pos2) < r1 + r2

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a separate display for human play
    pygame.display.set_caption("Space Miner")
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    done = False
    while not done:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0 # unused
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render for Human ---
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over! Final Score: {info['score']}")
    env.close()