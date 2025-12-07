import os
import os
import pygame

os.environ['SDL_VIDEODRIVER'] = 'dummy'
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, Space to jump. ↑+Space for high jump, ↓+Space for low jump. Hold Shift in air to fall faster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap between procedurally generated platforms to reach the top before time runs out in this side-scrolling arcade hopper."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GOAL_HEIGHT = 8000
        self.TIME_LIMIT_SECONDS = 45

        # Colors
        self.COLOR_BG_TOP = pygame.Color(135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = pygame.Color(0, 0, 139)  # Dark Blue
        self.COLOR_PLAYER = pygame.Color(255, 255, 0)  # Yellow
        self.COLOR_PLATFORM = pygame.Color(34, 139, 34)  # Forest Green
        self.COLOR_PLATFORM_TOP = pygame.Color(50, 205, 50) # Lime Green
        self.COLOR_UI_TEXT = pygame.Color(255, 255, 255)
        self.COLOR_UI_BAR = pygame.Color(255, 165, 0) # Orange
        self.COLOR_UI_BAR_BG = pygame.Color(100, 100, 100, 150)

        # Physics constants
        self.GRAVITY = 0.8
        self.PLAYER_HORIZONTAL_ACCEL = 1.2
        self.PLAYER_FRICTION = -0.15
        self.JUMP_NORMAL = -15
        self.JUMP_HIGH = -18
        self.JUMP_LOW = -12
        self.FAST_FALL_MULTIPLIER = 2.5
        self.MAX_VEL_X = 8

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
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # Etc...        
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.on_ground = None
        self.can_jump = None
        self.platforms = None
        self.particles = None
        self.camera_y = None
        self.highest_platform_idx = None
        self.steps = None
        self.score = None
        self.time_left = None
        self.game_over = None
        self.reward_cache = None
        
        self.bg_surface = self._create_gradient_surface()
        
        # Initialize state variables
        # self.reset() is called by the environment wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_size = pygame.Vector2(20, 20)
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 60)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.can_jump = True

        self.camera_y = 0  # FIX: Initialize camera_y before generating platforms
        self.platforms = []
        self._generate_initial_platforms()
        self.particles = []
        self.highest_platform_idx = 0

        self.steps = 0
        self.score = 0
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.game_over = False
        self.reward_cache = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_cache = 0
        self.clock.tick(self.FPS)

        # Unpack factorized action
        self._handle_input(action)
        
        # Update game logic
        self._update_physics()
        self._update_world()

        self.steps += 1
        self.time_left -= 1
        
        self.reward_cache -= 0.01 # Constant time penalty
        reward = self.reward_cache
        self.score += reward
        
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            terminal_reward = 0
            if self.camera_y >= self.GOAL_HEIGHT:
                terminal_reward = 50 # Goal-oriented reward
            elif self.player_pos.y - self.camera_y > self.HEIGHT:
                terminal_reward = -5 # Failure penalty
            reward += terminal_reward
            self.score += terminal_reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_HORIZONTAL_ACCEL
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_HORIZONTAL_ACCEL

        # Jumping
        if space_held and self.on_ground and self.can_jump:
            # sfx: jump
            if movement == 1: # Up for high jump
                self.player_vel.y = self.JUMP_HIGH
            elif movement == 2: # Down for low jump
                self.player_vel.y = self.JUMP_LOW
            else: # Normal jump
                self.player_vel.y = self.JUMP_NORMAL
            self.on_ground = False
            self.can_jump = False

        if not space_held:
            self.can_jump = True

        # Fast fall
        if shift_held and not self.on_ground and self.player_vel.y > 0:
            # sfx: whoosh
            self.player_vel.y += self.GRAVITY * (self.FAST_FALL_MULTIPLIER - 1)
        
    def _update_physics(self):
        # Apply gravity
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY

        # Apply friction
        self.player_vel.x *= (1.0 + self.PLAYER_FRICTION)
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0

        # Clamp horizontal velocity
        self.player_vel.x = max(-self.MAX_VEL_X, min(self.MAX_VEL_X, self.player_vel.x))
        
        self.player_pos += self.player_vel

        # Screen wrap horizontal
        if self.player_pos.x > self.WIDTH: self.player_pos.x = -self.player_size.x
        elif self.player_pos.x < -self.player_size.x: self.player_pos.x = self.WIDTH

        # Collision detection
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        
        for i, plat in enumerate(self.platforms):
            if self.player_vel.y > 0 and player_rect.colliderect(plat):
                if abs(player_rect.bottom - plat.top) < self.player_vel.y + 1:
                    self.player_pos.y = plat.top - self.player_size.y
                    self.player_vel.y = 0
                    self.on_ground = True
                    self._create_particles(pygame.Vector2(player_rect.centerx, plat.top)) # sfx: land

                    if i > self.highest_platform_idx:
                        self.reward_cache += 1.0 # Reward for landing on a new platform
                        self.highest_platform_idx = i
                    break

    def _update_world(self):
        # Update camera
        previous_camera_y = self.camera_y
        target_camera_y = self.player_pos.y - self.HEIGHT * 0.75
        if target_camera_y > self.camera_y:
            self.camera_y += (target_camera_y - self.camera_y) * 0.1

        height_gain = self.camera_y - previous_camera_y
        if height_gain > 0:
            self.reward_cache += height_gain * 0.1 # Continuous reward for ascent

        # Update and remove old particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # Procedurally generate/cull platforms
        if self.platforms and self.platforms[-1].y - self.camera_y > -self.player_size.y:
            self._generate_new_platforms()
        self.platforms = [p for p in self.platforms if p.y - self.camera_y < self.HEIGHT]

    def _check_termination(self):
        return (
            self.player_pos.y - self.camera_y > self.HEIGHT or
            self.time_left <= 0 or
            self.camera_y >= self.GOAL_HEIGHT
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.bg_surface, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for plat in self.platforms:
            screen_y = plat.y - self.camera_y
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (int(plat.x), int(screen_y), plat.width, plat.height))
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, (int(plat.x), int(screen_y), plat.width, 4))
            
        for p in self.particles:
            screen_pos = p['pos'] - pygame.Vector2(0, self.camera_y)
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            pygame.gfxdraw.pixel(self.screen, int(screen_pos.x), int(screen_pos.y), (*p['color'], alpha))

        player_screen_pos = self.player_pos - pygame.Vector2(0, self.camera_y)
        player_rect = pygame.Rect(player_screen_pos, self.player_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        eye_x = player_rect.centerx + (5 if self.player_vel.x > 0.5 else -5 if self.player_vel.x < -0.5 else 0)
        eye_y = player_rect.centery - 3
        pygame.draw.circle(self.screen, (0,0,0), (int(eye_x), int(eye_y)), 2)

    def _render_ui(self):
        time_str = f"TIME: {self.time_left / self.FPS:.1f}"
        time_surf = self.font_large.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        score_str = f"SCORE: {self.score:.0f}"
        score_surf = self.font_small.render(score_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 50))
        
        bar_height = self.HEIGHT - 40
        progress = min(1.0, self.camera_y / self.GOAL_HEIGHT)
        
        bg_rect = pygame.Rect(15, 20, 20, bar_height)
        s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BAR_BG)
        self.screen.blit(s, (bg_rect.x, bg_rect.y))
        
        fill_height = bar_height * progress
        fill_rect = pygame.Rect(15, 20 + (bar_height - fill_height), 20, fill_height)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, fill_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.camera_y,
            "time_left": self.time_left / self.FPS
        }

    def _create_gradient_surface(self):
        surf = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = self.COLOR_BG_BOTTOM.lerp(self.COLOR_BG_TOP, ratio)
            pygame.draw.line(surf, color, (0, y), (self.WIDTH, y))
        return surf

    def _generate_initial_platforms(self):
        start_plat = pygame.Rect(self.WIDTH/2 - 75, self.HEIGHT - 40, 150, 20)
        self.platforms.append(start_plat)
        for _ in range(10): # Ensure a safe start
             plat = self._get_next_platform(self.platforms[-1], 0)
             self.platforms.append(plat)
        self._generate_new_platforms()

    def _generate_new_platforms(self):
        while len(self.platforms) < 50:
            new_platform = self._get_next_platform(self.platforms[-1], self.camera_y)
            self.platforms.append(new_platform)
            
    def _get_next_platform(self, last_platform, current_height):
        progress = min(1.0, current_height / self.GOAL_HEIGHT)
        min_w = int(120 * (1 - progress * 0.7))
        max_w = int(200 * (1 - progress * 0.6))
        min_y_gap = int(60 * (1 + progress * 0.2))
        max_y_gap = int(140 * (1 + progress * 0.5))
        max_x_offset = int(150 * (1 + progress * 0.8))
        
        width = self.np_random.integers(min_w, max_w)
        y_gap = self.np_random.integers(min_y_gap, max_y_gap)
        x_offset = self.np_random.integers(-max_x_offset, max_x_offset)
        
        x = last_platform.centerx + x_offset - width / 2
        x = max(20, min(self.WIDTH - width - 20, x))
        y = last_platform.y - y_gap
        
        return pygame.Rect(x, y, width, 20)

    def _create_particles(self, pos):
        for _ in range(15):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 0)),
                'life': self.np_random.integers(10, 25),
                'max_life': 25,
                'color': (255, 255, 255)
            })

    def close(self):
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == '__main__':
    try:
        # Check if running in a headless environment
        if 'SDL_VIDEODRIVER' in os.environ and os.environ['SDL_VIDEODRIVER'] == 'dummy':
             raise ImportError("Cannot run interactive demo in a headless environment.")

        # Re-initialize pygame with default video driver for interactive mode
        pygame.quit()
        os.environ.pop('SDL_VIDEODRIVER', None)
        pygame.init()

        env = GameEnv()
        obs, info = env.reset()
        running = True
        human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Leap Hopper")
        
        action = env.action_space.sample()
        action.fill(0)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()

            keys = pygame.key.get_pressed()
            action.fill(0)
            
            # This logic combines keys for the MultiDiscrete action
            # Note: This allows e.g. UP and LEFT to be pressed, but only one is used for action[0]
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Height: {info['height']:.0f}")
                obs, info = env.reset()

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()

        env.close()

    except (ImportError, pygame.error) as e:
        print(f"Could not run interactive demo: {e}. This is normal in a headless environment.")
        print("Running a short non-interactive test...")
        
        env = GameEnv()
        obs, info = env.reset()
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode finished after {i+1} steps. Final info: {info}")
                obs, info = env.reset()
        env.close()
        print("Non-interactive test complete.")