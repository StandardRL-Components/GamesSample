
# Generated: 2025-08-27T23:45:21.130415
# Source Brief: brief_03568.md
# Brief Index: 3568

        
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


# --- Helper Classes for Game Entities ---

class Rocket:
    """Represents the player's rocket."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 10  # Constant horizontal speed
        self.vy = 0
        self.width = 30
        self.height = 15
        self.color = (255, 255, 255)
        self.outline_color = (200, 200, 255)

    def get_rect(self):
        return pygame.Rect(self.x - self.width / 2, self.y - self.height / 2, self.width, self.height)

    def draw(self, surface, camera_x):
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y)

        # Main body (polygon)
        points = [
            (screen_x + self.width / 2, screen_y),
            (screen_x - self.width / 2, screen_y - self.height / 2),
            (screen_x - self.width / 2, screen_y + self.height / 2),
        ]
        pygame.gfxdraw.aapolygon(surface, points, self.outline_color)
        pygame.gfxdraw.filled_polygon(surface, points, self.color)

        # Cockpit
        cockpit_pos = (int(screen_x + self.width/4), screen_y)
        pygame.gfxdraw.aacircle(surface, cockpit_pos[0], cockpit_pos[1], 4, (100, 150, 255))
        pygame.gfxdraw.filled_circle(surface, cockpit_pos[0], cockpit_pos[1], 4, (150, 200, 255))


class Obstacle:
    """Represents an obstacle in the course."""
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.type = random.choice(["circle", "rect"])
        self.color = (255, 80, 80)
        self.outline_color = (255, 120, 120)

    def get_rect(self):
        if self.type == "rect":
            return pygame.Rect(self.x - self.size / 2, self.y - self.size / 2, self.size, self.size)
        else: # circle
            return pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)

    def draw(self, surface, camera_x):
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y)
        if self.type == "rect":
            rect = pygame.Rect(screen_x - self.size / 2, screen_y - self.size / 2, self.size, self.size)
            pygame.draw.rect(surface, self.color, rect, border_radius=3)
            pygame.draw.rect(surface, self.outline_color, rect, width=2, border_radius=3)
        else: # circle
            pygame.gfxdraw.aacircle(surface, screen_x, screen_y, int(self.size), self.outline_color)
            pygame.gfxdraw.filled_circle(surface, screen_x, screen_y, int(self.size), self.color)

    def collides_with(self, rocket_rect):
        if self.type == "rect":
            return self.get_rect().colliderect(rocket_rect)
        else: # circle
            # More accurate circle-rect collision
            closest_x = max(rocket_rect.left, min(self.x, rocket_rect.right))
            closest_y = max(rocket_rect.top, min(self.y, rocket_rect.bottom))
            distance_x = self.x - closest_x
            distance_y = self.y - closest_y
            return (distance_x**2 + distance_y**2) < (self.size**2)


class Particle:
    """Represents a single particle for effects."""
    def __init__(self, x, y, vx, vy, size, lifespan, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface, camera_x):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color = self.color + (alpha,)
            temp_surf = pygame.Surface((int(self.size*2), int(self.size*2)), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (int(self.size), int(self.size)), int(self.size))
            surface.blit(temp_surf, (int(self.x - camera_x - self.size), int(self.y - self.size)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑ and ↓ to apply vertical thrust to your rocket. Avoid the red obstacles."
    )

    game_description = (
        "Pilot a rocket through a procedurally generated obstacle course. Reach the finish line before the timer runs out, and try not to crash more than twice!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH = 8000
        self.TIME_LIMIT_SECONDS = 30.0
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.MAX_COLLISIONS = 3
        
        # Physics
        self.GRAVITY = 0.2
        self.THRUST_POWER = 0.6
        self.MAX_VEL_Y = 6
        
        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_FINISH = (80, 255, 80)
        self.COLOR_CHECKPOINT = (80, 80, 255)
        self.COLOR_TIMER = (255, 255, 0)
        self.COLOR_COLLISION = (255, 0, 0)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.rocket = None
        self.obstacles = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.time_elapsed = 0.0
        self.collisions = 0
        self.game_over = False
        self.win = False
        self.checkpoints_reached = []
        self.last_rocket_x = 0

        # Run validation
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.time_elapsed = 0.0
        self.collisions = 0
        self.game_over = False
        self.win = False
        
        # Player
        self.rocket = Rocket(self.WIDTH / 4, self.HEIGHT / 2)
        self.last_rocket_x = self.rocket.x
        
        # World
        self._generate_obstacles()
        self._generate_stars()
        self.checkpoints = [self.WORLD_WIDTH / 3, 2 * self.WORLD_WIDTH / 3]
        self.checkpoints_reached = [False, False]

        # Effects
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]
        
        reward = 0
        
        if not self.game_over:
            # --- Update Game Logic ---
            self.steps += 1
            self.time_elapsed += 1.0 / self.FPS
            
            # 1. Apply physics and move rocket
            self._apply_physics_and_move(movement)

            # 2. Update particles
            for p in self.particles[:]:
                p.update()
                if p.lifespan <= 0:
                    self.particles.remove(p)

            # 3. Handle collisions
            collision_reward = self._handle_collisions()
            reward += collision_reward

            # 4. Handle checkpoints
            checkpoint_reward = self._handle_checkpoints()
            reward += checkpoint_reward

            # 5. Continuous reward for progress
            progress = self.rocket.x - self.last_rocket_x
            reward += progress * 0.01  # Small reward for moving forward
            self.last_rocket_x = self.rocket.x

        # --- Check Termination ---
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += reward
        
        if terminated:
            self.game_over = True

        # --- Return 5-tuple ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_physics_and_move(self, movement):
        # Apply gravity
        self.rocket.vy += self.GRAVITY

        # Apply thrust from action
        if movement == 1:  # Up
            self.rocket.vy -= self.THRUST_POWER
            # Thrust particles
            if self.steps % 2 == 0:
                self._create_particles(
                    count=1,
                    x=self.rocket.x - self.rocket.width / 2, y=self.rocket.y,
                    vx_range=(2, 4), vy_range=(-0.5, 0.5), size=4, lifespan=20,
                    color=(255, 200, 100)
                )
        elif movement == 2:  # Down
            self.rocket.vy += self.THRUST_POWER
            # Thrust particles
            if self.steps % 2 == 0:
                 self._create_particles(
                    count=1,
                    x=self.rocket.x - self.rocket.width / 2, y=self.rocket.y,
                    vx_range=(2, 4), vy_range=(-0.5, 0.5), size=4, lifespan=20,
                    color=(100, 200, 255)
                )

        # Clamp vertical velocity
        self.rocket.vy = max(-self.MAX_VEL_Y, min(self.MAX_VEL_Y, self.rocket.vy))

        # Update position
        self.rocket.x += self.rocket.vx
        self.rocket.y += self.rocket.vy

        # Screen boundaries
        self.rocket.y = max(self.rocket.height, min(self.HEIGHT - self.rocket.height, self.rocket.y))

    def _handle_collisions(self):
        rocket_rect = self.rocket.get_rect()
        camera_x = self.rocket.x - self.WIDTH / 4
        
        for obs in self.obstacles:
            # Only check nearby obstacles for performance
            if abs(obs.x - self.rocket.x) < self.WIDTH:
                if obs.collides_with(rocket_rect):
                    self.obstacles.remove(obs)
                    self.collisions += 1
                    # sfx: explosion
                    self._create_particles(
                        count=30, x=self.rocket.x, y=self.rocket.y,
                        vx_range=(-5, 5), vy_range=(-5, 5), size=5, lifespan=40,
                        color=(255, 80, 80)
                    )
                    return -10  # Collision penalty
        return 0

    def _handle_checkpoints(self):
        reward = 0
        for i, cp_x in enumerate(self.checkpoints):
            if not self.checkpoints_reached[i] and self.rocket.x >= cp_x:
                self.checkpoints_reached[i] = True
                # sfx: checkpoint_reached
                self._create_particles(
                    count=50, x=self.rocket.x, y=self.rocket.y,
                    vx_range=(self.rocket.vx-2, self.rocket.vx+2), vy_range=(-8, 8), size=4, lifespan=60,
                    color=self.COLOR_CHECKPOINT
                )
                reward += 50
        return reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0
        
        if self.rocket.x >= self.WORLD_WIDTH:
            terminated = True
            self.win = True
            time_bonus = 100 * max(0, self.TIME_LIMIT_SECONDS - self.time_elapsed) / self.TIME_LIMIT_SECONDS
            terminal_reward = 100 + time_bonus
        elif self.collisions >= self.MAX_COLLISIONS:
            terminated = True
            terminal_reward = -50
        elif self.time_elapsed >= self.TIME_LIMIT_SECONDS:
            terminated = True
            terminal_reward = -50
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            terminal_reward = -20
            
        return terminated, terminal_reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        camera_x = self.rocket.x - self.WIDTH / 4

        # Render stars with parallax
        for s in self.stars:
            x, y, speed, size = s
            screen_x = (x - camera_x * speed) % self.WIDTH
            pygame.draw.circle(self.screen, (200, 200, 255), (screen_x, y), size)

        # Render checkpoints and finish line
        for i, cp_x in enumerate(self.checkpoints):
            if abs(cp_x - self.rocket.x) < self.WIDTH:
                screen_x = int(cp_x - camera_x)
                color = self.COLOR_CHECKPOINT if not self.checkpoints_reached[i] else (50, 50, 100)
                pygame.draw.line(self.screen, color, (screen_x, 0), (screen_x, self.HEIGHT), 3)

        if abs(self.WORLD_WIDTH - self.rocket.x) < self.WIDTH:
            screen_x = int(self.WORLD_WIDTH - camera_x)
            pygame.draw.line(self.screen, self.COLOR_FINISH, (screen_x, 0), (screen_x, self.HEIGHT), 5)
        
        # Render obstacles
        for obs in self.obstacles:
            if abs(obs.x - self.rocket.x) < self.WIDTH:
                obs.draw(self.screen, camera_x)

        # Render particles
        for p in self.particles:
            p.draw(self.screen, camera_x)

        # Render rocket
        self.rocket.draw(self.screen, camera_x)

    def _render_ui(self):
        # Collision counter
        col_text = self.font_small.render(f"HITS: {self.collisions}/{self.MAX_COLLISIONS}", True, self.COLOR_COLLISION)
        self.screen.blit(col_text, (10, 10))

        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - self.time_elapsed)
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TIMER)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_FINISH
            else:
                msg = "GAME OVER"
                color = self.COLOR_COLLISION
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed,
            "collisions": self.collisions,
            "rocket_x": self.rocket.x,
            "rocket_y": self.rocket.y,
        }

    def _generate_obstacles(self):
        self.obstacles = []
        current_x = 800
        while current_x < self.WORLD_WIDTH - 800:
            # Difficulty scaling: spacing decreases as x increases
            progress = current_x / self.WORLD_WIDTH
            spacing = 200 + 200 * (1 - progress)
            current_x += random.uniform(spacing * 0.8, spacing * 1.2)
            
            # Add a cluster of obstacles
            num_in_cluster = random.randint(1, 3)
            cluster_y_center = random.uniform(50, self.HEIGHT - 50)
            for _ in range(num_in_cluster):
                y = cluster_y_center + random.uniform(-80, 80)
                y = max(30, min(self.HEIGHT - 30, y))
                size = random.uniform(15, 35)
                self.obstacles.append(Obstacle(current_x + random.uniform(-50, 50), y, size))

    def _generate_stars(self):
        self.stars = []
        for _ in range(200):
            self.stars.append([
                random.uniform(0, self.WIDTH),  # x
                random.uniform(0, self.HEIGHT), # y
                random.uniform(0.1, 0.5),       # speed
                random.uniform(0.5, 1.5)        # size
            ])

    def _create_particles(self, count, x, y, vx_range, vy_range, size, lifespan, color):
        for _ in range(count):
            vx = random.uniform(*vx_range)
            vy = random.uniform(*vy_range)
            p_size = random.uniform(size * 0.8, size * 1.2)
            p_life = random.uniform(lifespan * 0.8, lifespan * 1.2)
            self.particles.append(Particle(x, y, vx, vy, p_size, p_life, color))

    def validate_implementation(self):
        print("Running implementation validation...")
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
    # This block allows you to play the game directly
    # Note: Gymnasium's 'human' render mode is not used here.
    # We manually handle rendering and user input for direct play.
    
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    # Override pygame screen for display
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Rocket Pilot")

    running = True
    total_reward = 0
    
    # --- Main Game Loop ---
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")

        # --- Action Mapping from Keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already the rendered frame. We just need to display it.
        # We need to transpose it back for pygame's `surfarray`
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(env.screen, frame)
        pygame.display.flip()
        
        # --- Frame Rate Control ---
        env.clock.tick(env.FPS)
        
        if terminated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds on game over
            obs, info = env.reset()
            total_reward = 0

    pygame.quit()