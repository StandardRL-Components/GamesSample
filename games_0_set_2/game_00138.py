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


# Helper class for particle effects
class Particle:
    def __init__(self, x, y, color, life, size, gravity, dx, dy):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.initial_life = life
        self.size = size
        self.gravity = gravity
        self.dx = dx
        self.dy = dy

    def update(self):
        self.life -= 1
        self.dy += self.gravity
        self.x += self.dx
        self.y += self.dy
        self.size = max(0, self.size - 0.1)

    def draw(self, surface, camera_offset_y):
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            temp_surf = pygame.Surface((int(self.size) * 2, int(self.size) * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (int(self.size), int(self.size)), int(self.size))
            surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size + camera_offset_y)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Press space to jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer. Jump between platforms, avoid obstacles, and reach the goal as fast as you can."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (15, 25, 40)
    COLOR_BG_BOTTOM = (30, 50, 80)
    COLOR_PLAYER = (255, 200, 0)
    COLOR_PLAYER_GLOW = (255, 200, 0)
    COLOR_PLATFORM = (120, 130, 140)
    COLOR_GOAL = (0, 255, 120)
    COLOR_OBSTACLE = (255, 60, 60)
    COLOR_TEXT = (240, 240, 240)

    # Physics
    GRAVITY = 0.8
    JUMP_STRENGTH = -15
    PLAYER_SPEED = 7
    PLAYER_SIZE = 20
    
    # Game settings
    TIME_LIMIT_SECONDS = 60
    MAX_OBSTACLES = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font = pygame.font.Font(None, 36)
        
        # State variables will be initialized in reset()
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = False
        self.last_space_held = False
        
        self.platforms = []
        self.obstacles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0.0
        self.timer = 0.0
        self.level = 1
        self.highest_platform_index = 0
        
        self.camera_y = 0.0
        self.rng = np.random.default_rng()

        # self.reset() is called at the end of init, but some attributes
        # need to be initialized before the first call to _generate_level
        self._generate_level()
        self.reset()
        

    def _generate_level(self):
        self.platforms.clear()
        self.obstacles.clear()

        # Starting platform
        start_platform = pygame.Rect(50, self.HEIGHT - 50, 100, 20)
        self.platforms.append(start_platform)
        
        # Procedurally generate platforms
        current_x = start_platform.right
        for i in range(15):
            gap_x = self.rng.integers(80, 150)
            gap_y = self.rng.integers(-80, 80)
            width = self.rng.integers(80, 150)
            
            new_x = current_x + gap_x
            new_y = self.platforms[-1].y + gap_y
            new_y = np.clip(new_y, 100, self.HEIGHT - 50)
            
            platform = pygame.Rect(new_x, new_y, width, 20)
            self.platforms.append(platform)
            current_x = platform.right

        # Goal platform
        goal_platform = pygame.Rect(current_x + 100, self.platforms[-1].y - 50, 100, 50)
        self.platforms.append(goal_platform)

        # Generate obstacles
        num_obstacles = min(self.MAX_OBSTACLES, 2 + self.level)
        # Ensure there are platforms to choose from
        if len(self.platforms) > 2:
            for _ in range(num_obstacles):
                # FIX: Use random.choice for lists of objects, as np.random.choice
                # can convert them to arrays, losing object attributes.
                platform_to_haunt = random.choice(self.platforms[1:-1])
                obs_y = platform_to_haunt.top - 30
                obs_x = platform_to_haunt.left + self.rng.integers(0, platform_to_haunt.width - 20)
                
                obstacle = {
                    "rect": pygame.Rect(obs_x, obs_y, 20, 20),
                    "speed": self.rng.uniform(1.5, 2.5) * (1 if self.rng.random() > 0.5 else -1),
                    "bounds": (platform_to_haunt.left, platform_to_haunt.right)
                }
                self.obstacles.append(obstacle)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._generate_level()

        self.player_pos = pygame.Vector2(self.platforms[0].centerx, self.platforms[0].top - self.PLAYER_SIZE)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = True
        self.last_space_held = False

        self.steps = 0
        self.score = 0.0
        self.timer = self.TIME_LIMIT_SECONDS
        self.highest_platform_index = 0
        
        self.camera_y = self.player_pos.y
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def _create_particles(self, x, y, color, count, gravity):
        for _ in range(count):
            self.particles.append(
                Particle(
                    x, y, color,
                    life=self.rng.integers(15, 30),
                    size=self.rng.uniform(2, 5),
                    gravity=gravity,
                    dx=self.rng.uniform(-2, 2),
                    dy=self.rng.uniform(-3, 0)
                )
            )

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.01  # Survival reward
        terminated = False

        # --- Handle Input ---
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else: # None, Up, Down
            self.player_vel.x = 0

        # Jump on press (not hold)
        if space_held and not self.last_space_held and self.is_grounded:
            self.player_vel.y = self.JUMP_STRENGTH
            self.is_grounded = False
            # sfx: jump
            self._create_particles(self.player_pos.x, self.player_pos.y + self.PLAYER_SIZE, self.COLOR_PLATFORM, 10, 0.2)
        self.last_space_held = space_held

        # --- Update Physics & Game State ---
        self.timer -= 1 / self.FPS
        self.steps += 1

        # Apply gravity
        if not self.is_grounded:
            self.player_vel.y += self.GRAVITY
        
        # Update player position
        self.player_pos.x += self.player_vel.x
        self.player_pos.y += self.player_vel.y

        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Update obstacles
        for obs in self.obstacles:
            obs['rect'].x += obs['speed']
            if obs['rect'].left < obs['bounds'][0] or obs['rect'].right > obs['bounds'][1]:
                obs['speed'] *= -1

        # --- Collision Detection ---
        self.is_grounded = False
        player_rect_ground_check = player_rect.copy()
        player_rect_ground_check.height += 1 # Check slightly below for grounding
        
        for i, platform in enumerate(self.platforms):
            if player_rect_ground_check.colliderect(platform) and self.player_vel.y >= 0:
                # Check if player was above the platform in the previous frame
                if self.player_pos.y + self.PLAYER_SIZE - self.player_vel.y <= platform.top + 1: # +1 for tolerance
                    self.player_pos.y = platform.top - self.PLAYER_SIZE
                    self.player_vel.y = 0
                    if not self.is_grounded:
                        # sfx: land
                        self._create_particles(self.player_pos.x + self.PLAYER_SIZE/2, self.player_pos.y + self.PLAYER_SIZE, self.COLOR_PLATFORM, 5, 0.1)
                    self.is_grounded = True
                    
                    if i > self.highest_platform_index:
                        reward += 10.0 # New platform reward
                        self.score += 10.0
                        self.highest_platform_index = i
                    break
        
        # --- Check Termination Conditions ---
        # 1. Reached goal
        if player_rect.colliderect(self.platforms[-1]):
            reward += 100.0 + (self.timer * 2) # Time bonus
            self.score += 100.0 + (self.timer * 2)
            terminated = True
            self.level += 1 # Progress to next difficulty
            # sfx: level complete

        # 2. Obstacle collision
        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                reward = -100.0
                terminated = True
                self._create_particles(self.player_pos.x + self.PLAYER_SIZE/2, self.player_pos.y + self.PLAYER_SIZE/2, self.COLOR_OBSTACLE, 20, 0.2)
                # sfx: fail

        # 3. Fell off screen
        if self.player_pos.y > self.camera_y + self.HEIGHT:
            reward = -100.0
            terminated = True
            # sfx: fall

        # 4. Timer ran out
        if self.timer <= 0:
            reward = -50.0
            terminated = True
            
        self.score += reward
        
        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _get_observation(self):
        # Smooth camera movement
        target_camera_y = self.player_pos.y - self.HEIGHT / 2.5
        self.camera_y += (target_camera_y - self.camera_y) * 0.1
        camera_offset_y = -self.camera_y

        # --- Render ---
        self._render_background()
        
        # Render platforms
        for i, platform in enumerate(self.platforms):
            color = self.COLOR_GOAL if i == len(self.platforms) - 1 else self.COLOR_PLATFORM
            render_rect = platform.move(0, camera_offset_y)
            pygame.draw.rect(self.screen, color, render_rect, border_radius=3)
        
        # Render obstacles
        for obs in self.obstacles:
            render_rect = obs['rect'].move(0, camera_offset_y)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, render_rect)

        # Render particles
        for p in self.particles:
            p.draw(self.screen, camera_offset_y)

        # Render player
        player_render_x = int(self.player_pos.x)
        player_render_y = int(self.player_pos.y + camera_offset_y)
        player_rect = pygame.Rect(player_render_x, player_render_y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Glow effect
        glow_size = self.PLAYER_SIZE * 1.8
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 80), (glow_size/2, glow_size/2), glow_size/2)
        self.screen.blit(glow_surf, (player_render_x - (glow_size - self.PLAYER_SIZE)/2, player_render_y - (glow_size - self.PLAYER_SIZE)/2))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # Render UI
        timer_text = self.font.render(f"Time: {max(0, int(self.timer))}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (10, 10))

        score_text = self.font.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))
        
        level_text = self.font.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "level": self.level,
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            print("Cannot run playable example in headless mode. Exiting.")
            exit()
            
        env = GameEnv(render_mode="rgb_array")
        env.reset()
        
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Minimalist Platformer")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            # --- Human Controls ---
            movement = 0 # no-op
            space_held = 0
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
                
            if keys[pygame.K_SPACE]:
                space_held = 1
            
            action = [movement, space_held, 0] # shift is unused

            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # --- Pygame Rendering ---
            # The observation is already a rendered frame
            # We just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Episode finished! Total reward: {total_reward:.2f}, Info: {info}")
                total_reward = 0
                env.reset()

            # --- Event Handling & Clock ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            clock.tick(GameEnv.FPS)

        env.close()
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("This might be because you are running in a headless environment.")
        print("Try unsetting the SDL_VIDEODRIVER environment variable to run the playable example.")