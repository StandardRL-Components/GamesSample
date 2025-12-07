
# Generated: 2025-08-28T05:05:42.052619
# Source Brief: brief_02513.md
# Brief Index: 2513

        
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
        "Controls: Use arrow keys to move your ship. Survive the asteroid field for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship through an asteroid field, dodging incoming rocks for 60 seconds to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ASTEROID = (128, 132, 142)
        self.COLOR_STAR = (200, 200, 200)
        self.COLOR_UI = (255, 255, 255)
        self.COLOR_EXPLOSION = [(255, 69, 0), (255, 165, 0), (255, 215, 0)]

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Game state variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_radius = 10
        self.player_speed = 4
        self.asteroids = []
        self.stars = []
        self.particles = []
        
        # Will be properly initialized in reset()
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        
        self.asteroids = []
        self.particles = []
        
        # Generate a static starfield for the episode
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                'size': self.np_random.integers(1, 3)
            })

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # If the game has already ended, subsequent steps do nothing until reset
        if self.game_over:
            self._update_particles() # Allow explosion to animate
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        
        # Update game logic
        self._handle_player_movement(movement)
        self._update_asteroids()
        self._update_particles()
        
        # Check for termination conditions
        collision = self._check_collisions()
        win = self.steps >= self.MAX_STEPS
        
        terminated = collision or win
        
        # Calculate reward
        if collision:
            reward = -10.0
            self.game_over = True
            self._create_explosion(self.player_pos)
            # Sound effect placeholder: # pygame.mixer.Sound('explosion.wav').play()
        elif win:
            reward = 100.0
            self.game_over = True
            # Sound effect placeholder: # pygame.mixer.Sound('win.wav').play()
        else:
            reward = 0.1  # Reward for surviving a frame

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.player_speed
        elif movement == 2:  # Down
            self.player_pos[1] += self.player_speed
        elif movement == 3:  # Left
            self.player_pos[0] -= self.player_speed
        elif movement == 4:  # Right
            self.player_pos[0] += self.player_speed
            
        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.WIDTH - self.player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_radius, self.HEIGHT - self.player_radius)

    def _update_asteroids(self):
        # Difficulty scaling
        # Max asteroids increases from 3 to 20 over 30 seconds
        max_asteroids = int(min(20, 3 + (self.steps / (self.FPS * 30)) * 17))
        # Base speed increases every 10 seconds
        base_speed = 1.0 + (self.steps // (self.FPS * 10)) * 0.4
        
        # Spawning
        if len(self.asteroids) < max_asteroids and self.np_random.random() < 0.05:
            radius = self.np_random.integers(10, 31)
            x = self.np_random.integers(radius, self.WIDTH - radius)
            y = -radius
            
            angle = self.np_random.uniform(math.pi * 0.4, math.pi * 0.6)  # Downward cone
            speed = base_speed + self.np_random.uniform(-0.5, 1.0)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            
            self.asteroids.append({
                'pos': [x, y],
                'vel': [vx, vy],
                'radius': radius
            })

        # Movement and despawning
        for asteroid in self.asteroids[:]:
            asteroid['pos'][0] += asteroid['vel'][0]
            asteroid['pos'][1] += asteroid['vel'][1]
            if asteroid['pos'][1] > self.HEIGHT + asteroid['radius'] or \
               asteroid['pos'][0] < -asteroid['radius'] or \
               asteroid['pos'][0] > self.WIDTH + asteroid['radius']:
                self.asteroids.remove(asteroid)

    def _check_collisions(self):
        for asteroid in self.asteroids:
            dist = math.hypot(self.player_pos[0] - asteroid['pos'][0], self.player_pos[1] - asteroid['pos'][1])
            if dist < self.player_radius + asteroid['radius']:
                return True
        return False

    def _create_explosion(self, position):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            color = random.choice(self.COLOR_EXPLOSION)
            self.particles.append({
                'pos': list(position),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.98  # Drag
            p['vel'][1] *= 0.98
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render stars
        for star in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, star['pos'], star['size'])
        
        # Render asteroids
        for asteroid in self.asteroids:
            pos = (int(asteroid['pos'][0]), int(asteroid['pos'][1]))
            radius = int(asteroid['radius'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)

        # Render player if not exploded
        if not (self.game_over and self.steps < self.MAX_STEPS):
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            s = self.player_radius
            p1 = (px, py - int(s * 1.2))
            p2 = (px - s, py + s)
            p3 = (px + s, py + s)
            
            # Main ship body
            pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_PLAYER)
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_PLAYER)

        # Render particles
        for p in self.particles:
            size = int(5 * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.draw.rect(self.screen, p['color'], (pos[0] - size // 2, pos[1] - size // 2, size, size))

    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.2f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI)
        self.screen.blit(timer_surf, (10, 10))
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

        # Game over / Win message
        msg = None
        if self.game_over and self.steps < self.MAX_STEPS:
            msg = "GAME OVER"
            color = self.COLOR_EXPLOSION[0]
        elif self.game_over and self.steps >= self.MAX_STEPS:
            msg = "YOU SURVIVED!"
            color = (0, 255, 127)  # Spring Green
        
        if msg:
            msg_surf = self.font_game_over.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")