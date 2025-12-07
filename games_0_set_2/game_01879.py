
# Generated: 2025-08-28T02:59:33.392024
# Source Brief: brief_01879.md
# Brief Index: 1879

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to pilot your ship and dodge the asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship through a dense asteroid field. Survive for 30 seconds to win!"
    )

    # Frames auto-advance at 60fps for smooth, real-time gameplay.
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 30 * FPS  # 30 seconds to win

    # Colors
    COLOR_BG = (10, 15, 25)
    COLOR_GRID = (30, 40, 60)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (150, 200, 255)
    COLOR_ASTEROID = (120, 130, 140)
    COLOR_TEXT = (220, 220, 220)
    
    # Player
    PLAYER_SIZE = 20
    PLAYER_SPEED = 5.0
    PLAYER_COLLISION_RADIUS = PLAYER_SIZE * 0.4

    # Asteroids
    INITIAL_SPAWN_PROB = 1.0 / FPS  # 1 per second
    SPAWN_RATE_INCREASE_INTERVAL = 1 * FPS # Every 1 second
    SPAWN_RATE_INCREASE_AMOUNT = 0.001
    
    MIN_ASTEROID_SPEED = 1.0
    INITIAL_MAX_ASTEROID_SPEED = 5.0
    SPEED_INCREASE_INTERVAL = 5 * FPS # Every 5 seconds
    SPEED_INCREASE_AMOUNT = 0.1
    
    MIN_ASTEROID_RADIUS = 8
    MAX_ASTEROID_RADIUS = 25
    
    # Play Area
    PADDING = 10
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Game state variables
        self.player_pos = None
        self.asteroids = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False

        self.current_spawn_prob = self.INITIAL_SPAWN_PROB
        self.current_max_asteroid_speed = self.INITIAL_MAX_ASTEROID_SPEED
        
        self.play_area = pygame.Rect(self.PADDING, self.PADDING, self.WIDTH - 2 * self.PADDING, self.HEIGHT - 2 * self.PADDING)

        # Initialize state
        self.reset()

        # This check is not part of the standard API but is required by the prompt
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.player_pos = pygame.Vector2(self.play_area.centerx, self.play_area.bottom - self.PLAYER_SIZE)
        self.asteroids = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        # Reset difficulty
        self.current_spawn_prob = self.INITIAL_SPAWN_PROB
        self.current_max_asteroid_speed = self.INITIAL_MAX_ASTEROID_SPEED

        # Pre-spawn a few asteroids to make the start interesting
        for _ in range(5):
            self._spawn_asteroid(random_y=True)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Tick the clock for auto-advancing frames
        self.clock.tick(self.FPS)

        reward = 0
        terminated = self.game_over

        if not terminated:
            self.steps += 1
            
            # Unpack action
            movement = action[0]
            
            # --- Update Game Logic ---
            self._handle_input(movement)
            self._update_asteroids()
            self._update_particles()
            self._update_difficulty()
            
            # --- Check for Termination ---
            collision = self._check_collisions()
            self.win = self.steps >= self.MAX_STEPS
            
            if collision:
                # Sound: player_explosion.wav
                terminated = True
                reward = -100.0
                self._create_explosion(self.player_pos, 50)
            elif self.win:
                # Sound: victory_chime.wav
                terminated = True
                reward = 100.0
            else:
                # Continuous survival reward
                reward = 0.1
            
            self.game_over = terminated
            self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            move_vec.y = -1
        elif movement == 2:  # Down
            move_vec.y = 1
        elif movement == 3:  # Left
            move_vec.x = -1
        elif movement == 4:  # Right
            move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
            # Create thruster particles
            self._create_thruster_particles(move_vec)

        # Clamp player position to the play area
        self.player_pos.x = max(self.play_area.left, min(self.play_area.right, self.player_pos.x))
        self.player_pos.y = max(self.play_area.top, min(self.play_area.bottom, self.player_pos.y))

    def _update_asteroids(self):
        # Move existing asteroids and remove off-screen ones
        for asteroid in self.asteroids[:]:
            asteroid['pos'] += asteroid['vel']
            if not self.screen.get_rect().inflate(50, 50).collidepoint(asteroid['pos']):
                self.asteroids.remove(asteroid)

        # Spawn new asteroids
        if self.np_random.random() < self.current_spawn_prob:
            self._spawn_asteroid()

    def _spawn_asteroid(self, random_y=False):
        radius = self.np_random.integers(self.MIN_ASTEROID_RADIUS, self.MAX_ASTEROID_RADIUS + 1)
        
        # Spawn randomly along any edge
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -radius)
        elif edge == 1: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + radius)
        elif edge == 2: # Left
            pos = pygame.Vector2(-radius, self.np_random.uniform(0, self.HEIGHT))
        else: # Right
            pos = pygame.Vector2(self.WIDTH + radius, self.np_random.uniform(0, self.HEIGHT))
            
        if random_y: # For initial spawn, place them anywhere
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))


        # Aim towards a random point inside the play area
        target = pygame.Vector2(
            self.np_random.uniform(self.play_area.left, self.play_area.right),
            self.np_random.uniform(self.play_area.top, self.play_area.bottom)
        )
        
        vel = (target - pos).normalize()
        speed = self.np_random.uniform(self.MIN_ASTEROID_SPEED, self.current_max_asteroid_speed)
        vel *= speed
        
        self.asteroids.append({'pos': pos, 'radius': radius, 'vel': vel})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_difficulty(self):
        # Increase spawn rate
        if self.steps > 0 and self.steps % self.SPAWN_RATE_INCREASE_INTERVAL == 0:
            self.current_spawn_prob += self.SPAWN_RATE_INCREASE_AMOUNT
        # Increase max speed
        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            self.current_max_asteroid_speed += self.SPEED_INCREASE_AMOUNT

    def _check_collisions(self):
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < self.PLAYER_COLLISION_RADIUS + asteroid['radius']:
                return True
        return False

    def _create_thruster_particles(self, move_vec):
        # Spawn particles opposite to movement direction
        for _ in range(2):
            vel = -move_vec * self.np_random.uniform(1, 3) + pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5))
            pos = self.player_pos - move_vec * (self.PLAYER_SIZE * 0.6)
            life = self.np_random.integers(10, 20)
            color = random.choice([(255, 200, 100), (255, 150, 50), (255, 255, 255)])
            self.particles.append({'pos': pos, 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _create_explosion(self, position, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 7)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(20, 50)
            color = random.choice([(255, 255, 255), (255, 200, 100), (255, 100, 0)])
            self.particles.append({'pos': position.copy(), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _get_observation(self):
        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_particles()
        self._render_asteroids()
        if not (self.game_over and not self.win): # Don't draw player if they lost
             self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_player(self):
        # Main triangle shape
        p1 = self.player_pos + pygame.Vector2(0, -self.PLAYER_SIZE * 0.8)
        p2 = self.player_pos + pygame.Vector2(-self.PLAYER_SIZE * 0.5, self.PLAYER_SIZE * 0.4)
        p3 = self.player_pos + pygame.Vector2(self.PLAYER_SIZE * 0.5, self.PLAYER_SIZE * 0.4)
        points = [p1, p2, p3]

        # Draw a soft glow behind the player
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.player_pos.x), int(self.player_pos.y),
            int(self.PLAYER_SIZE * 0.8), (*self.COLOR_PLAYER_GLOW, 50)
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.player_pos.x), int(self.player_pos.y),
            int(self.PLAYER_SIZE * 0.8), (*self.COLOR_PLAYER_GLOW, 50)
        )
        
        # Draw the anti-aliased polygon
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos = (int(asteroid['pos'].x), int(asteroid['pos'].y))
            radius = int(asteroid['radius'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 4)
            if radius > 0:
                color = tuple(c * life_ratio for c in p['color'])
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'].x), int(p['pos'].y),
                    radius, color
                )

    def _render_ui(self):
        time_survived = self.steps / self.FPS
        time_text = f"TIME: {time_survived:.2f}s"
        time_surf = self.font.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (15, 10))

        if self.game_over:
            if self.win:
                end_text = "VICTORY!"
                end_color = (100, 255, 100)
            else:
                end_text = "GAME OVER"
                end_color = (255, 100, 100)
            
            end_surf = self.font.render(end_text, True, end_color)
            text_rect = end_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_survived": self.steps / self.FPS,
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
        # A reset is needed to initialize all state for _get_observation
        self.reset()
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

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0.0
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Asteroid Survival")
    
    while running:
        # --- Human Input ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        action = [movement, 0, 0] # Space and shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0.0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Time: {info['time_survived']:.2f}s")
            # In a real game loop, you might wait for a reset key press here
            # For this example, we'll just keep running to see the end screen
        
        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()