
# Generated: 2025-08-27T18:04:17.672838
# Source Brief: brief_01722.md
# Brief Index: 1722

        
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
        "Controls: Press ↑ for a small jump. Hold Shift while pressing ↑ for a high jump. Time your jumps to the beat!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling rhythm runner. Jump over obstacles to the beat of the music in a vibrant, neon world."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.BPM = 120
        self.BEAT_INTERVAL_FRAMES = int((60 / self.BPM) * self.FPS)
        self.GROUND_Y = self.SCREEN_HEIGHT - 50
        self.MAX_STEPS = 1500 # Increased for a longer track

        # Physics constants
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH_SMALL = -12
        self.JUMP_STRENGTH_LARGE = -16
        self.PLAYER_X = 100
        self.PLAYER_SIZE = 20

        # Color palette
        self.COLOR_BG_TOP = (10, 5, 20)
        self.COLOR_BG_BOTTOM = (30, 10, 40)
        self.COLOR_PLAYER = (0, 191, 255) # Bright Cyan
        self.COLOR_PLAYER_GLOW = (0, 191, 255, 50)
        self.COLOR_OBSTACLE = (255, 0, 127) # Hot Pink
        self.COLOR_OBSTACLE_GLOW = (255, 0, 127, 50)
        self.COLOR_PARTICLE_JUMP = (0, 255, 127) # Spring Green
        self.COLOR_TRACK = (200, 200, 255, 50)
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_BEAT = (255, 255, 255)

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
        self.font = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.player_y = 0
        self.player_vy = 0
        self.on_ground = True
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacle_speed = 0.0
        self.beat_frame_counter = 0

        self.reset()

        # Run self-check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.beat_frame_counter = 0
        
        # Player state
        self.player_y = self.GROUND_Y - self.PLAYER_SIZE
        self.player_vy = 0
        self.on_ground = True

        # Game entities
        self.obstacles = []
        self.particles = []
        
        # Difficulty
        self.obstacle_speed = 5.0

        # Spawn initial obstacles to fill the screen
        for i in range(5):
            self._spawn_obstacle(initial_spawn=True, offset=i * 250)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Handle Beat and Timing ---
        self.beat_frame_counter = (self.beat_frame_counter + 1) % (self.BEAT_INTERVAL_FRAMES * 100) # Avoid overflow
        is_on_beat = (self.beat_frame_counter % self.BEAT_INTERVAL_FRAMES) == 0
        
        # --- Unpack Action and Handle Input ---
        movement = action[0]
        # space_held = action[1] == 1 # Unused for now
        shift_held = action[2] == 1
        
        jump_triggered = (movement == 1)
        
        if jump_triggered and self.on_ground:
            # sfx: player_jump.wav
            jump_strength = self.JUMP_STRENGTH_LARGE if shift_held else self.JUMP_STRENGTH_SMALL
            self.player_vy = jump_strength
            self.on_ground = False
            self._create_jump_particles()
            
            # Reward for jumping on the beat
            if is_on_beat:
                reward += 0.1

        # --- Update Game State ---
        
        # Update Player Physics
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy
        
        if self.player_y >= self.GROUND_Y - self.PLAYER_SIZE:
            if not self.on_ground:
                # sfx: player_land.wav
                self._create_land_particles()
            self.player_y = self.GROUND_Y - self.PLAYER_SIZE
            self.player_vy = 0
            self.on_ground = True
            
        # Update and Manage Obstacles
        player_rect = pygame.Rect(self.PLAYER_X, self.player_y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        for obs in self.obstacles[:]:
            obs['x'] -= self.obstacle_speed
            
            # Check for clearing obstacle
            if not obs['cleared'] and obs['x'] + obs['size'] < self.PLAYER_X:
                obs['cleared'] = True
                reward += 1.0
                self.score += 1
                # sfx: clear_obstacle.wav

            # Check for collision
            obs_rect = self._get_obstacle_rect(obs)
            if player_rect.colliderect(obs_rect):
                self.game_over = True
                reward = -100.0
                self.score = max(0, self.score - 5)
                # sfx: player_hit.wav
                self._create_death_particles()

            # Remove off-screen obstacles
            if obs['x'] < -obs['size']:
                self.obstacles.remove(obs)
        
        # Update Particles
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # Spawn New Obstacles
        if self.beat_frame_counter % (self.BEAT_INTERVAL_FRAMES * 2) == 0:
            if self.np_random.random() < 0.75:
                self._spawn_obstacle()
        
        # Penalty for not jumping near an obstacle on the beat
        if is_on_beat and not jump_triggered:
            is_obstacle_near = any(
                self.PLAYER_X < obs['x'] < self.PLAYER_X + 200 for obs in self.obstacles
            )
            if is_obstacle_near:
                reward -= 0.05
        
        # --- Update Difficulty and Step Counter ---
        self.steps += 1
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_speed = min(12.0, self.obstacle_speed + 0.05)

        # --- Check Termination ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if not self.game_over and self.steps >= self.MAX_STEPS:
            reward += 100.0 # Victory reward
            # sfx: level_complete.wav

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_obstacle_rect(self, obs):
        if obs['type'] == 'triangle':
            return pygame.Rect(obs['x'], self.GROUND_Y - obs['size'], obs['size'], obs['size'])
        elif obs['type'] == 'rect':
            return pygame.Rect(obs['x'], self.GROUND_Y - obs['size'], obs['size'], obs['size'])
        return pygame.Rect(0,0,0,0)

    def _spawn_obstacle(self, initial_spawn=False, offset=0):
        obs_type = self.np_random.choice(['triangle', 'rect'])
        size = self.np_random.integers(20, 41)
        x_pos = self.SCREEN_WIDTH + offset
        if not initial_spawn:
            # Ensure minimum spacing between obstacles
            if self.obstacles and self.obstacles[-1]['x'] > self.SCREEN_WIDTH - 150:
                return

        self.obstacles.append({
            'x': x_pos,
            'size': size,
            'type': obs_type,
            'cleared': False
        })
    
    def _create_jump_particles(self):
        for _ in range(10):
            self.particles.append({
                'x': self.PLAYER_X + self.PLAYER_SIZE / 2,
                'y': self.GROUND_Y,
                'vx': self.np_random.uniform(-1, 1),
                'vy': self.np_random.uniform(-3, -1),
                'lifespan': self.np_random.integers(15, 25),
                'color': self.COLOR_PARTICLE_JUMP,
                'radius': self.np_random.uniform(2, 4)
            })
            
    def _create_land_particles(self):
        for _ in range(5):
            self.particles.append({
                'x': self.PLAYER_X + self.PLAYER_SIZE / 2,
                'y': self.GROUND_Y,
                'vx': self.np_random.uniform(-2, 2),
                'vy': self.np_random.uniform(-1, 0),
                'lifespan': self.np_random.integers(10, 20),
                'color': self.COLOR_BEAT,
                'radius': self.np_random.uniform(1, 3)
            })

    def _create_death_particles(self):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            self.particles.append({
                'x': self.PLAYER_X + self.PLAYER_SIZE / 2,
                'y': self.player_y + self.PLAYER_SIZE / 2,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'lifespan': self.np_random.integers(30, 60),
                'color': self.COLOR_PLAYER,
                'radius': self.np_random.uniform(2, 5)
            })

    def _render_gradient_bg(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render track
        track_rect = pygame.Rect(0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_BG_BOTTOM, track_rect)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 2)
        
        # Render beat indicator
        pulse = abs(math.sin(self.beat_frame_counter / self.BEAT_INTERVAL_FRAMES * math.pi))
        beat_radius = int(15 + pulse * 10)
        beat_alpha = int(50 + pulse * 100)
        pygame.gfxdraw.aacircle(self.screen, int(self.PLAYER_X + self.PLAYER_SIZE / 2), int(self.GROUND_Y - 40), beat_radius, (*self.COLOR_BEAT, beat_alpha))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 25))
            if alpha > 0:
                color = (*p['color'], max(0, min(255, alpha)))
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), color)

        # Render obstacles
        for obs in self.obstacles:
            x, size = int(obs['x']), int(obs['size'])
            y = self.GROUND_Y - size
            glow_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, size, size, size, self.COLOR_OBSTACLE_GLOW)
            self.screen.blit(glow_surf, (x - size//2, y - size//2), special_flags=pygame.BLEND_RGBA_ADD)

            if obs['type'] == 'triangle':
                points = [(x, self.GROUND_Y), (x + size, self.GROUND_Y), (x + size / 2, y)]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)
            elif obs['type'] == 'rect':
                rect = pygame.Rect(x, y, size, size)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect.inflate(4,4), 1)

        # Render player
        if not self.game_over:
            player_rect = pygame.Rect(int(self.PLAYER_X), int(self.player_y), self.PLAYER_SIZE, self.PLAYER_SIZE)
            glow_size = int(self.PLAYER_SIZE * 2.5)
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_size//2, glow_size//2, glow_size//2, self.COLOR_PLAYER_GLOW)
            self.screen.blit(glow_surf, (player_rect.centerx - glow_size//2, player_rect.centery - glow_size//2), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        progress = self.steps / self.MAX_STEPS
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 5
        pygame.draw.rect(self.screen, (255,255,255,50), (10, self.SCREEN_HEIGHT - 15, bar_width, bar_height), border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, self.SCREEN_HEIGHT - 15, bar_width * progress, bar_height), border_radius=2)

    def _get_observation(self):
        # Clear screen with background
        self._render_gradient_bg()
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "on_ground": self.on_ground,
            "obstacle_speed": self.obstacle_speed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
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

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Must be set for headless pygame

    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # To visualize, you would need a different setup not using dummy driver
    # Example:
    # os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    # env = GameEnv()
    # screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    # pygame.display.set_caption("Rhythm Runner")
    # clock = pygame.time.Clock()
    #
    # obs, info = env.reset()
    # done = False
    # while not done:
    #     # Simple keyboard mapping for human play
    #     keys = pygame.key.get_pressed()
    #     action = [0, 0, 0] # no-op
    #     if keys[pygame.K_UP]:
    #         action[0] = 1 # up
    #     if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
    #         action[2] = 1 # shift
    #
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #
    #     # Render to the display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
    #
    #     clock.tick(env.FPS)
    #
    # env.close()