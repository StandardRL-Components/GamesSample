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

    user_guide = (
        "Controls: Use ←→ to set jump distance and ↑↓ to set jump height. Press space to jump. "
        "Avoid red obstacles and land on the green platform at the end."
    )

    game_description = (
        "Guide a hopping spaceship through obstacle-laden space in this side-scrolling arcade game. "
        "Time your jumps perfectly to reach the end of the level within 30 seconds."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GROUND_Y = self.HEIGHT - 40
        self.LEVEL_LENGTH_PX = 6000
        self.MAX_STEPS = self.FPS * 30  # 30 seconds

        # --- Colors ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 200)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (200, 40, 40)
        self.COLOR_PLATFORM = (50, 255, 50)
        self.COLOR_PLATFORM_GLOW = (40, 200, 40)
        self.COLOR_GROUND = (30, 40, 60)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_INDICATOR = (255, 255, 0)

        # --- Fonts ---
        self.font_ui = pygame.font.Font(None, 32)
        self.font_indicator = pygame.font.Font(None, 20)

        # --- Physics ---
        self.GRAVITY = 0.5
        self.JUMP_VELOCITIES = {
            'height': {'low': -8, 'medium': -10, 'high': -12},
            'length': {'short': 6, 'medium': 8, 'long': 10}
        }

        # --- Game State (persistent across resets) ---
        self.level = 1

        # --- Game State (reset every episode) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.level_complete = False

        self.player_pos = [0, 0]
        self.player_vel = [0, 0]
        self.player_rect = pygame.Rect(0, 0, 0, 0)
        self.is_jumping = False
        self.jump_height_setting = 'medium'
        self.jump_length_setting = 'medium'

        self.world_scroll_x = 0
        self.last_world_x = 0

        self.obstacles = []
        self.cleared_obstacles = set()
        self.end_platform = None
        self.particles = []
        self.stars = []
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.level_complete = False

        self.player_pos = [100, self.GROUND_Y]
        self.player_vel = [0, 0]
        self.player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 20, 20, 20)
        self.is_jumping = False
        self.jump_height_setting = 'medium'
        self.jump_length_setting = 'medium'

        self.world_scroll_x = 0
        self.last_world_x = self.player_pos[0]

        self.obstacles = []
        self.cleared_obstacles = set()
        self.end_platform = None
        self.particles = []

        self._generate_stars()
        self._generate_obstacles()
        self._generate_platform()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)

        self._handle_input(action)
        self._update_player()
        self._update_particles()
        
        # Store position before collision checks for reward calculation
        current_world_x = self.player_pos[0] + self.world_scroll_x
        
        collision, level_complete = self._check_collisions()

        reward = self._calculate_reward(current_world_x, collision, level_complete)
        self.score += reward

        self.last_world_x = current_world_x
        
        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        terminated = collision or level_complete

        if terminated or truncated:
            self.game_over = True
            if level_complete:
                self.level_complete = True
                self.level += 1 # Difficulty increases for the next game
            if collision:
                self._create_explosion(self.player_pos, 50)
                # sfx: explosion_sound

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action

        if not self.is_jumping:
            # Set jump power based on movement keys
            if movement == 1: self.jump_height_setting = 'high'
            elif movement == 2: self.jump_height_setting = 'low'
            else: self.jump_height_setting = 'medium'

            if movement == 3: self.jump_length_setting = 'short'
            elif movement == 4: self.jump_length_setting = 'long'
            else: self.jump_length_setting = 'medium'

            # Initiate jump with spacebar
            if space_held == 1:
                self.is_jumping = True
                self.player_vel[1] = self.JUMP_VELOCITIES['height'][self.jump_height_setting]
                self.player_vel[0] = self.JUMP_VELOCITIES['length'][self.jump_length_setting]
                self._create_jump_particles(self.player_pos)
                # sfx: jump_sound

    def _update_player(self):
        if self.is_jumping:
            self.player_vel[1] += self.GRAVITY
            self.player_pos[1] += self.player_vel[1]
            self.world_scroll_x += self.player_vel[0]
        
        # Clamp player to screen vertically to prevent going off-top
        self.player_pos[1] = max(0, self.player_pos[1])

        self.player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1  # Decrement life

    def _check_collisions(self):
        # Ground collision
        if self.player_pos[1] >= self.GROUND_Y and self.player_vel[1] > 0:
            self.player_pos[1] = self.GROUND_Y
            self.player_vel = [0, 0]
            if self.is_jumping:
                self._create_land_particles(self.player_pos)
                # sfx: land_sound
            self.is_jumping = False

        # Obstacle collision
        for obs_rect in self.obstacles:
            # Adjust obstacle rect for camera scroll
            screen_obs_rect = obs_rect.move(-self.world_scroll_x, 0)
            if self.player_rect.colliderect(screen_obs_rect):
                return True, False

        # End platform collision
        if self.end_platform:
            screen_platform_rect = self.end_platform.move(-self.world_scroll_x, 0)
            if self.player_rect.colliderect(screen_platform_rect) and not self.is_jumping:
                # sfx: success_sound
                return False, True
        
        return False, False

    def _calculate_reward(self, current_world_x, collision, level_complete):
        reward = 0.0
        # Time penalty
        reward -= 0.01

        # Progress reward
        progress = current_world_x - self.last_world_x
        if progress > 0 and self.is_jumping:
            reward += progress * 0.02 # ~+0.16 for medium jump

        # Event-based rewards
        if collision:
            reward = -10.0
        if level_complete:
            reward = 100.0

        # Obstacle clear reward
        for i, obs_rect in enumerate(self.obstacles):
            if i not in self.cleared_obstacles and self.player_rect.left + self.world_scroll_x > obs_rect.right:
                reward += 1.0
                self.cleared_obstacles.add(i)
        
        return reward

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append([
                random.randint(0, self.WIDTH),
                random.randint(0, self.HEIGHT),
                random.uniform(0.5, 2.0), # size
                random.uniform(0.1, 0.5)  # parallax speed factor
            ])

    def _generate_obstacles(self):
        self.obstacles = []
        current_x = 400
        difficulty_factor = 1.0 - (self.level - 1) * 0.1
        min_gap = int(250 * difficulty_factor)
        max_gap = int(400 * difficulty_factor)
        
        while current_x < self.LEVEL_LENGTH_PX - 500:
            gap = self.np_random.integers(min_gap, max_gap)
            current_x += gap

            width = self.np_random.integers(40, 80)
            height = self.np_random.integers(50, 150)
            
            obstacle_rect = pygame.Rect(current_x, self.GROUND_Y - height, width, height)
            self.obstacles.append(obstacle_rect)
            current_x += width

    def _generate_platform(self):
        self.end_platform = pygame.Rect(self.LEVEL_LENGTH_PX, self.GROUND_Y - 20, 150, 20)
    
    def _create_jump_particles(self, pos):
        for _ in range(10):
            self.particles.append([
                pos[0], pos[1], 
                random.uniform(-1, 1), random.uniform(0, 2), # velocity
                random.randint(10, 20), # life
                (200, 200, 200) # color
            ])

    def _create_land_particles(self, pos):
        for _ in range(15):
            self.particles.append([
                pos[0], self.GROUND_Y,
                random.uniform(-2, 2), random.uniform(-1, 0),
                random.randint(15, 25),
                (150, 150, 150)
            ])

    def _create_explosion(self, pos, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 6)
            self.particles.append([
                pos[0], pos[1],
                math.cos(angle) * speed, math.sin(angle) * speed,
                random.randint(20, 40),
                random.choice([self.COLOR_PLAYER, self.COLOR_OBSTACLE, (255, 255, 0)])
            ])

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_stars()
        self._render_ground()
        self._render_obstacles_and_platform()
        self._render_particles()
        if not (self.game_over and not self.level_complete):
             self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for star in self.stars:
            x = (star[0] - self.world_scroll_x * star[3]) % self.WIDTH
            y = star[1]
            size = star[2]
            brightness = int(128 + 127 * star[3]) # Faster stars are brighter
            color = (brightness, brightness, brightness)
            radius = int(size)
            if radius > 0:
                pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)

    def _render_ground(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

    def _render_obstacles_and_platform(self):
        # Obstacles
        for obs_rect in self.obstacles:
            screen_rect = obs_rect.move(-self.world_scroll_x, 0)
            if screen_rect.right > 0 and screen_rect.left < self.WIDTH:
                pygame.gfxdraw.box(self.screen, screen_rect, self.COLOR_OBSTACLE_GLOW)
                pygame.gfxdraw.box(self.screen, screen_rect.inflate(-8, -8), self.COLOR_OBSTACLE)
        
        # Platform
        if self.end_platform:
            screen_rect = self.end_platform.move(-self.world_scroll_x, 0)
            if screen_rect.right > 0 and screen_rect.left < self.WIDTH:
                pygame.gfxdraw.box(self.screen, screen_rect, self.COLOR_PLATFORM_GLOW)
                pygame.gfxdraw.box(self.screen, screen_rect.inflate(-8, -8), self.COLOR_PLATFORM)

    def _render_player(self):
        # Pulsating glow
        glow_radius = int(15 + 3 * math.sin(self.steps * 0.2))
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (self.player_rect.centerx - glow_radius, self.player_rect.centery - glow_radius))

        # Main ship body
        p1 = (self.player_rect.centerx, self.player_rect.top)
        p2 = (self.player_rect.left, self.player_rect.bottom)
        p3 = (self.player_rect.right, self.player_rect.bottom)
        pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)
        pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p[4] / 20.0
            size = max(1, 3 * life_ratio)
            color = (
                min(255, int(p[5][0] * life_ratio)),
                min(255, int(p[5][1] * life_ratio)),
                min(255, int(p[5][2] * life_ratio))
            )
            pos = [p[0] - self.world_scroll_x, p[1]]
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), int(size))

    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        # Level
        level_text = f"LEVEL: {self.level}"
        level_surf = self.font_ui.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, 10))

        # Jump indicators
        if not self.is_jumping:
            h_text = f"H: {self.jump_height_setting.upper()}"
            l_text = f"L: {self.jump_length_setting.upper()}"
            h_surf = self.font_indicator.render(h_text, True, self.COLOR_INDICATOR)
            l_surf = self.font_indicator.render(l_text, True, self.COLOR_INDICATOR)
            self.screen.blit(h_surf, (self.player_pos[0] - h_surf.get_width() // 2, self.player_pos[1] + 10))
            self.screen.blit(l_surf, (self.player_pos[0] - l_surf.get_width() // 2, self.player_pos[1] + 25))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "progress_px": self.world_scroll_x
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen for human rendering
    pygame.display.init()
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Space Hopper")
    
    done = False
    total_reward = 0
    
    print("--- Space Hopper ---")
    print(env.user_guide)

    # Game loop
    while not done:
        # Human input mapping
        keys = pygame.key.get_pressed()
        move = 0
        if keys[pygame.K_UP]: move = 1
        elif keys[pygame.K_DOWN]: move = 2
        elif keys[pygame.K_LEFT]: move = 3
        elif keys[pygame.K_RIGHT]: move = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        done = terminated or truncated

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()