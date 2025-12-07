import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move horizontally. Press 'Up Arrow' for a small jump, "
        "'Space' for a medium jump, and 'Shift' for a high jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race against time to climb a procedurally generated mountain. "
        "Jump between platforms, collect coins, and avoid the moving red blocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.GOAL_HEIGHT = 10000

        # Player physics
        self.GRAVITY = 0.8
        self.PLAYER_HORZ_SPEED = 5
        self.JUMP_SMALL = -12
        self.JUMP_MEDIUM = -16
        self.JUMP_LARGE = -20
        
        # Colors
        self.COLOR_BG_SKY = (135, 206, 235)
        self.COLOR_PLAYER = (0, 128, 255)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_PLATFORM = (100, 100, 100)
        self.COLOR_COIN = (255, 215, 0)
        self.COLOR_OBSTACLE = (255, 0, 0)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BAR = (0, 255, 0)
        self.COLOR_UI_BAR_BG = (50, 50, 50)
        self.MOUNTAIN_COLORS = [(47, 79, 79), (80, 100, 100), (112, 128, 144)]

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # State variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.player = None
        self.player_vel = None
        self.on_ground = None
        self.camera_y = None
        self.platforms = None
        self.coins = None
        self.obstacles = None
        self.particles = None
        self.time_left = None
        self.highest_y = None
        self.obstacle_speed = None
        self.last_platform_y = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # The first platform's top is at self.HEIGHT - 50.
        # Player height is 30. To stand on it, player's y must be (HEIGHT - 50) - 30.
        player_start_y = self.HEIGHT - 80
        self.player = pygame.Rect(self.WIDTH // 2 - 15, player_start_y, 30, 30)
        
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.camera_y = 0
        self.highest_y = self.player.y
        self.time_left = self.MAX_STEPS
        self.obstacle_speed = 1.0

        self.platforms = []
        self.coins = []
        self.obstacles = []
        self.particles = []
        
        # Generate initial world
        self.last_platform_y = self.HEIGHT
        self._generate_initial_platforms()
        self._manage_world_generation()

        # Perform an initial collision check to correctly set on_ground status
        self._handle_collisions()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # The environment has already terminated. Return the final observation.
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # --- 1. Update Player based on Action ---
        self._handle_input(action)
        self._update_player_physics()

        # --- 2. Update World Objects ---
        self._update_obstacles()
        self._update_particles()
        
        # --- 3. Handle Collisions ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- 4. Update Camera and World Generation ---
        self._update_camera()
        self._manage_world_generation()
        
        # --- 5. Calculate Rewards ---
        # Reward for upward movement
        current_height = self.player.y - self.camera_y
        y_change = self.highest_y - current_height
        if y_change > 0: # Moved up
            reward += y_change * 0.1
            self.highest_y = current_height
        elif y_change < 0: # Moved down
            reward += y_change * 0.01 # y_change is negative, so this is a penalty

        # --- 6. Check Termination Conditions ---
        self.steps += 1
        self.time_left -= 1
        
        terminated = False
        if self.player.top > self.HEIGHT + self.player.height: # Fell off screen
            terminated = True
            reward -= 100
        elif self.time_left <= 0: # Time ran out
            terminated = True
            reward -= 100
        elif -self.camera_y >= self.GOAL_HEIGHT: # Reached goal
            terminated = True
            reward += 100
        
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        if movement == 3: # Left
            self.player_vel.x = -self.PLAYER_HORZ_SPEED
        elif movement == 4: # Right
            self.player_vel.x = self.PLAYER_HORZ_SPEED
        else:
            self.player_vel.x = 0
            
        # Jumping (only if on ground)
        if self.on_ground:
            jump_velocity = 0
            if shift_held:
                jump_velocity = self.JUMP_LARGE
            elif space_held:
                jump_velocity = self.JUMP_MEDIUM
            elif movement == 1: # Up
                jump_velocity = self.JUMP_SMALL
            
            if jump_velocity != 0:
                self.player_vel.y = jump_velocity
                self.on_ground = False
                self._create_particles(self.player.midbottom, 10, (150, 150, 150))

    def _update_player_physics(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        if self.player_vel.y > 15: # Terminal velocity
            self.player_vel.y = 15

        # Update position
        self.player.x += self.player_vel.x
        self.player.y += self.player_vel.y

        # Screen bounds
        if self.player.left < 0: self.player.left = 0
        if self.player.right > self.WIDTH: self.player.right = self.WIDTH

    def _handle_collisions(self):
        reward = 0
        self.on_ground = False
        
        # Platform collisions
        player_rect_world = self.player.move(0, -self.camera_y)
        for plat in self.platforms:
            if self.player_vel.y >= 0 and player_rect_world.colliderect(plat):
                # Check if player was above the platform in the last frame
                # This prevents zipping through platforms from below
                previous_player_bottom = player_rect_world.bottom - self.player_vel.y
                if previous_player_bottom <= plat.top + 1: # +1 for floating point tolerance
                    self.player.bottom = plat.top + self.camera_y
                    self.player_vel.y = 0
                    self.on_ground = True
                    break
        
        # Coin collisions
        player_rect_world = self.player.move(0, -self.camera_y) # Re-evaluate after position snap
        for coin in self.coins[:]:
            if player_rect_world.colliderect(coin):
                self.coins.remove(coin)
                self.score += 10
                reward += 1
                self._create_particles(coin.center, 15, self.COLOR_COIN)

        # Obstacle collisions
        for obs_data in self.obstacles:
            obs_rect = obs_data[0]
            if player_rect_world.colliderect(obs_rect):
                reward -= 1
                # Simple knockback
                if self.player.centerx < obs_rect.centerx:
                    self.player_vel.x = -5
                else:
                    self.player_vel.x = 5
                self.player_vel.y = -5 # Bounce up
                self._create_particles(player_rect_world.center, 20, self.COLOR_OBSTACLE)

        return reward

    def _update_camera(self):
        # Camera follows player upwards
        scroll_threshold = self.HEIGHT * 0.4
        if self.player.y < scroll_threshold:
            scroll_amount = scroll_threshold - self.player.y
            self.player.y += scroll_amount
            self.camera_y -= scroll_amount
    
    def _update_obstacles(self):
        # Increase speed over time
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_speed = min(3.0, self.obstacle_speed + 0.05)
            
        for obs_data in self.obstacles:
            obs_rect, direction, bounds = obs_data
            obs_rect.x += self.obstacle_speed * direction
            if obs_rect.left < bounds[0] or obs_rect.right > bounds[1]:
                obs_data[1] *= -1 # Reverse direction

    def _generate_initial_platforms(self):
        # Create a safe starting area
        for i in range(10):
            y = self.HEIGHT - 50 - i * 80
            self.platforms.append(pygame.Rect(self.WIDTH // 2 - 50, y, 100, 20))
        self.last_platform_y = self.platforms[-1].y

    def _manage_world_generation(self):
        # Generate new platforms, coins, obstacles as player climbs
        while self.last_platform_y > self.camera_y - 50:
            plat_width = self.np_random.integers(60, 150)
            plat_x = self.np_random.integers(0, self.WIDTH - plat_width)
            
            # Ensure next platform is reachable
            last_plat = self.platforms[-1]
            max_dy = int(abs(self.JUMP_LARGE) * 10)
            min_dy = 40
            dy = self.np_random.integers(min_dy, max_dy)
            
            plat_y = last_plat.y - dy
            
            new_plat = pygame.Rect(plat_x, plat_y, plat_width, 20)
            self.platforms.append(new_plat)
            self.last_platform_y = plat_y

            # Chance to spawn a coin on the platform
            if self.np_random.random() < 0.4:
                coin_x = plat_x + plat_width // 2 - 10
                coin_y = plat_y - 30
                self.coins.append(pygame.Rect(coin_x, coin_y, 20, 20))
                
            # Chance to spawn an obstacle
            if self.np_random.random() < 0.2 and -self.camera_y > 500:
                obs_y = plat_y - self.np_random.integers(80, 150)
                obs_x = self.np_random.integers(50, self.WIDTH - 150)
                bounds = (obs_x - 50, obs_x + 100)
                direction = 1 if self.np_random.random() < 0.5 else -1
                self.obstacles.append([pygame.Rect(obs_x, obs_y, 50, 25), direction, bounds])

        # Clean up old objects
        min_y = self.camera_y + self.HEIGHT + 100
        self.platforms = [p for p in self.platforms if p.bottom > self.camera_y]
        self.coins = [c for c in self.coins if c.bottom > self.camera_y]
        self.obstacles = [o for o in self.obstacles if o[0].bottom > self.camera_y]

    def _get_observation(self):
        self._render_background()
        self._render_game_objects()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_SKY)
        # Parallax mountains
        for i, color in enumerate(self.MOUNTAIN_COLORS):
            scroll_factor = 0.1 * (i + 1)
            y_offset = (self.camera_y * scroll_factor) % self.HEIGHT
            for j in range(-1, 2):
                # Simple triangle mountains
                points = [
                    (j*self.WIDTH, self.HEIGHT),
                    (j*self.WIDTH + self.WIDTH/2, self.HEIGHT/2 + y_offset),
                    (j*self.WIDTH + self.WIDTH, self.HEIGHT)
                ]
                pygame.draw.polygon(self.screen, color, points)

    def _render_game_objects(self):
        # Draw platforms
        for plat in self.platforms:
            screen_plat = plat.move(0, -self.camera_y)
            if screen_plat.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_plat, border_radius=3)

        # Draw coins
        for coin in self.coins:
            screen_coin = coin.move(0, -self.camera_y)
            if screen_coin.colliderect(self.screen.get_rect()):
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, screen_coin)
                shine_rect = screen_coin.copy()
                shine_rect.width //= 2
                shine_rect.height -= 4
                shine_rect.center = screen_coin.center
                pygame.draw.ellipse(self.screen, (255, 255, 150), shine_rect)

        # Draw obstacles
        for obs_data in self.obstacles:
            screen_obs = obs_data[0].move(0, -self.camera_y)
            if screen_obs.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_obs, border_radius=5)
                glow_alpha = 100 + 50 * math.sin(pygame.time.get_ticks() * 0.005)
                glow_surface = pygame.Surface(screen_obs.size, pygame.SRCALPHA)
                pygame.draw.rect(glow_surface, (*self.COLOR_OBSTACLE, glow_alpha), (0, 0, *screen_obs.size), border_radius=5)
                self.screen.blit(glow_surface, screen_obs.topleft)

        # Draw player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, self.player, 2, border_radius=3)

    def _render_ui(self):
        # Time remaining
        time_text = f"TIME: {self.time_left // self.FPS:02d}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Height bar
        bar_height = self.HEIGHT - 40
        bar_x = 15
        progress = min(1, (-self.camera_y) / self.GOAL_HEIGHT)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, 20, 20, bar_height))
        fill_height = bar_height * progress
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, 20 + bar_height - fill_height, 20, fill_height))
        
        height_text = f"{int(-self.camera_y)}m"
        height_surf = self.font_small.render(height_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(height_surf, (bar_x + 25, 20 + bar_height - fill_height - 8))


    def _create_particles(self, position, count, color):
        for _ in range(count):
            vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 0))
            self.particles.append([pygame.Vector2(position), vel, self.np_random.integers(5, 10), color])

    def _update_particles(self):
        for p in self.particles[:]:
            p[0] += p[1] # position += velocity
            p[1].y += 0.2 # gravity on particles
            p[2] -= 0.3 # shrink
            if p[2] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            pos = p[0]
            size = int(p[2])
            color = p[3]
            screen_pos = pos + pygame.Vector2(0, -self.camera_y)
            if 0 <= screen_pos.x < self.WIDTH and 0 <= screen_pos.y < self.HEIGHT:
                pygame.draw.rect(self.screen, color, (int(screen_pos.x), int(screen_pos.y), size, size))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": -self.camera_y,
            "time_left": self.time_left // self.FPS
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # To play manually, you might need to install pygame with display support
    # and unset the SDL_VIDEODRIVER variable.
    # For example:
    # pip install pygame
    # unset SDL_VIDEODRIVER
    
    # For running the environment as is (headless):
    env = GameEnv()
    obs, info = env.reset()
    
    # Test stability with no-op actions
    terminated = False
    truncated = False
    for i in range(100):
        action = [0, 0, 0] # No-op
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Terminated/Truncated at step {i+1}")
            break
    print(f"After 100 no-op steps: {info}")
    
    # Example of running with random actions
    env.reset()
    done = False
    total_reward = 0
    step_count = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated
    
    print(f"Random agent finished in {step_count} steps. Total reward: {total_reward}. Final info: {info}")
    
    env.close()