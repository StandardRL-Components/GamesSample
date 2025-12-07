import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode, which is required for server-side execution.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Press ↑ to jump over the obstacles. Timing is everything."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced, side-scrolling rhythm runner. Jump to the beat to survive the neon course."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 1800  # 30 seconds at 60 FPS

        # Colors
        self.COLOR_BG = (10, 0, 30)
        self.COLOR_GRID_1 = (20, 10, 50)
        self.COLOR_GRID_2 = (40, 20, 80)
        self.COLOR_GROUND = (60, 40, 120)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        self.COLOR_OBSTACLE = (255, 100, 0)
        self.COLOR_OBSTACLE_GLOW = (180, 50, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_BEAT = (255, 255, 255, 150)

        # Player properties
        self.PLAYER_SIZE = 20
        self.PLAYER_X = self.WIDTH // 4
        self.GRAVITY = 0.6
        self.JUMP_STRENGTH = -12

        # Obstacle properties
        self.OBSTACLE_WIDTH = 30
        self.OBSTACLE_MIN_HEIGHT = 20
        self.OBSTACLE_MAX_HEIGHT = 80
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("consola", 24, bold=True)
        self.font_timer = pygame.font.SysFont("consola", 32, bold=True)
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Game state variables are initialized in reset()
        self.player_y = 0
        self.player_vy = 0
        self.ground_y = 0
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_obstacle_speed = 0
        self.obstacle_speed = 0
        self.world_scroll = 0
        self.beat_timer = 0
        self.beat_interval = 0
        self.obstacle_spawn_counter = 0
        self.np_random = None

        # Initialize state
        # The reset call is deferred until the first call to reset() by the user
        # to ensure proper seeding. However, we need to initialize some values
        # for the first observation to be possible.
        self.ground_y = self.HEIGHT - 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.ground_y = self.HEIGHT - 50
        self.player_y = self.ground_y - self.PLAYER_SIZE
        self.player_vy = 0
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_obstacle_speed = 7.0
        self.obstacle_speed = self.base_obstacle_speed
        self.world_scroll = 0
        self.beat_interval = 45  # A beat every 0.75s at 60 FPS
        self.beat_timer = 0
        self.obstacle_spawn_counter = self.beat_interval # Spawn first obstacle immediately

        # Seed the random number generator
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Procedurally generate the initial screen
        for i in range(3):
            self._spawn_obstacle(initial_x=self.WIDTH + i * self.WIDTH / 2.5)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # --- Game Logic ---
        self._handle_input(action)
        self._update_player()
        reward += self._update_obstacles()
        self._update_particles()
        self._update_world()
        
        # Check for collisions
        if self._check_collisions():
            self.game_over = True
            # sfx: player_death_sound
            reward = -50.0 # Strong penalty for dying
            
        # Survival reward
        reward += 0.01

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if not self.game_over and self.steps >= self.MAX_STEPS:
            reward += 50.0 # Bonus for finishing

        # Advance frame rate
        self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        # Only jump if on the ground
        if movement == 1 and self.player_y >= self.ground_y - self.PLAYER_SIZE:
            self.player_vy = self.JUMP_STRENGTH
            # sfx: player_jump_sound
            # Spawn jump particles
            for _ in range(15):
                self.particles.append(self._create_particle(
                    self.PLAYER_X + self.PLAYER_SIZE / 2, 
                    self.ground_y, 
                    self.COLOR_PLAYER,
                    angle_min=math.pi * 1.1, angle_max=math.pi * 1.9,
                    speed_min=1, speed_max=4,
                    gravity=0.1
                ))

    def _update_player(self):
        on_ground_before = self.player_y >= self.ground_y - self.PLAYER_SIZE
        
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy

        if self.player_y >= self.ground_y - self.PLAYER_SIZE:
            self.player_y = self.ground_y - self.PLAYER_SIZE
            self.player_vy = 0
            # Check if we just landed
            if not on_ground_before:
                # sfx: player_land_sound
                # Spawn landing particles
                for _ in range(8):
                    self.particles.append(self._create_particle(
                        self.PLAYER_X + self.PLAYER_SIZE / 2, 
                        self.ground_y, 
                        self.COLOR_GROUND,
                        angle_min=-math.pi/2, angle_max=math.pi/2,
                        speed_min=0.5, speed_max=2.5,
                        life=15,
                        gravity=0.2
                    ))

    def _update_obstacles(self):
        # Difficulty scaling
        self.obstacle_speed = self.base_obstacle_speed + 2.0 * (self.steps / self.MAX_STEPS)
        
        obstacle_reward = 0
        
        for obs in self.obstacles:
            obs['x'] -= self.obstacle_speed
            # Check for passing an obstacle
            if not obs['passed'] and obs['x'] + self.OBSTACLE_WIDTH < self.PLAYER_X:
                obs['passed'] = True
                self.score += 10
                obstacle_reward += 1.0
                # sfx: score_point_sound

        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] > -self.OBSTACLE_WIDTH]

        # Spawn new obstacles based on beat/rhythm
        self.obstacle_spawn_counter += 1
        if self.obstacle_spawn_counter >= self.beat_interval:
            self.obstacle_spawn_counter = 0
            self._spawn_obstacle()
            
        return obstacle_reward

    def _update_particles(self):
        for p in self.particles:
            p['life'] -= 1
            p['vy'] += p['gravity']
            p['x'] += p['vx']
            p['y'] += p['vy']
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _update_world(self):
        self.world_scroll = (self.world_scroll - self.obstacle_speed * 0.3) % self.WIDTH
        self.beat_timer = (self.beat_timer + 1) % self.beat_interval

    def _check_collisions(self):
        player_rect = pygame.Rect(self.PLAYER_X, self.player_y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['x'], obs['y'], self.OBSTACLE_WIDTH, obs['h'])
            if player_rect.colliderect(obs_rect):
                # Spawn collision particles
                for _ in range(30):
                    self.particles.append(self._create_particle(
                        player_rect.centerx, player_rect.centery, 
                        self.COLOR_OBSTACLE,
                        speed_min=2, speed_max=7
                    ))
                return True
        return False

    def _spawn_obstacle(self, initial_x=None):
        height = self.np_random.integers(self.OBSTACLE_MIN_HEIGHT, self.OBSTACLE_MAX_HEIGHT + 1)
        x_pos = initial_x if initial_x is not None else self.WIDTH
        
        self.obstacles.append({
            'x': x_pos,
            'y': self.ground_y - height,
            'h': height,
            'passed': False
        })
        
    def _create_particle(self, x, y, color, angle_min=0, angle_max=2*math.pi, speed_min=1, speed_max=5, life=30, gravity=0.0):
        angle = self.np_random.uniform(angle_min, angle_max)
        speed = self.np_random.uniform(speed_min, speed_max)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        return {'x': x, 'y': y, 'vx': vx, 'vy': vy, 'life': life, 'max_life': life, 'color': color, 'gravity': gravity}

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
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _render_game(self):
        self._render_background()
        self._render_obstacles()
        self._render_particles()
        self._render_player()

    def _render_background(self):
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.ground_y, self.WIDTH, self.HEIGHT - self.ground_y))
        
        # Draw parallax grid lines
        grid_spacing = 50
        # Farther grid
        for i in range(self.WIDTH // grid_spacing + 2):
            x = (i * grid_spacing + self.world_scroll * 0.5) % (self.WIDTH + grid_spacing) - grid_spacing
            pygame.draw.line(self.screen, self.COLOR_GRID_1, (x, 0), (x, self.ground_y), 1)
        # Closer grid
        for i in range(self.WIDTH // grid_spacing + 2):
            x = (i * grid_spacing + self.world_scroll) % (self.WIDTH + grid_spacing) - grid_spacing
            pygame.draw.line(self.screen, self.COLOR_GRID_2, (x, 0), (x, self.ground_y), 2)
            
        # Draw beat indicator
        if self.beat_timer < 5:
            alpha = 150 * (1 - self.beat_timer / 5)
            # FIX: The height of the surface cannot be negative.
            # The height of the ground area is self.HEIGHT - self.ground_y.
            beat_surface = pygame.Surface((self.WIDTH, self.HEIGHT - self.ground_y), pygame.SRCALPHA)
            beat_surface.fill((255, 255, 255, alpha))
            self.screen.blit(beat_surface, (0, self.ground_y))

    def _render_player(self):
        if self.game_over:
            return
        player_rect = pygame.Rect(self.PLAYER_X, int(self.player_y), self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Glow effect
        glow_size = int(self.PLAYER_SIZE * 1.8)
        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PLAYER_GLOW, 80), glow_surface.get_rect(), border_radius=5)
        self.screen.blit(glow_surface, (player_rect.centerx - glow_size/2, player_rect.centery - glow_size/2), special_flags=pygame.BLEND_RGBA_ADD)

        # Player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_BG, player_rect.inflate(-6,-6))

    def _render_obstacles(self):
        for obs in self.obstacles:
            rect = pygame.Rect(int(obs['x']), int(obs['y']), self.OBSTACLE_WIDTH, int(obs['h']))
            
            # Glow effect
            glow_size_w = int(self.OBSTACLE_WIDTH * 1.5)
            glow_size_h = int(obs['h'] * 1.2)
            glow_surface = pygame.Surface((glow_size_w, glow_size_h), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*self.COLOR_OBSTACLE_GLOW, 100), glow_surface.get_rect(), border_radius=5)
            self.screen.blit(glow_surface, (rect.centerx - glow_size_w/2, rect.centery - glow_size_h/2), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Obstacle
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)
            
    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            color = (*p['color'], alpha)
            size = max(1, int(4 * (p['life'] / p['max_life'])))
            
            # Using gfxdraw for antialiased circles
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, size, size, size, color)
            self.screen.blit(temp_surf, (int(p['x']) - size, int(p['y']) - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))
        
        # Timer display
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_timer.render(f"{time_left:.1f}", True, self.COLOR_TEXT)
        text_rect = timer_text.get_rect(center=(self.WIDTH / 2, 30))
        self.screen.blit(timer_text, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # To play the game, you may need to comment out the os.environ line at the top of the file
    # and ensure you have pygame installed (pip install pygame).
    
    # Create the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # Setup for human play
    # This part requires a display.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Neon Jumper")
        human_render = True
    except pygame.error:
        print("Could not create display. Running in headless mode.")
        human_render = False

    terminated = False
    total_reward = 0
    
    # Main game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key in [pygame.K_UP, pygame.K_SPACE]:
                    action[0] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation to the display if available
        if human_render:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset(seed=np.random.randint(0, 10000)) # Use a new random seed
            terminated = False
            total_reward = 0

    env.close()