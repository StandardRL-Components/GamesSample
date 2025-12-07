# Generated: 2025-08-28T00:17:57.209414
# Source Brief: brief_03748.md
# Brief Index: 3748

        
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
        "Controls: Press Space to jump over the red obstacles. Survive all three levels to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling arcade game where you control a space hopper. Jump over procedurally generated obstacles to reach the end of each level. The game gets faster and harder with each level."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.LEVEL_LENGTH = self.WIDTH * 10 # 10 screens wide
        self.MAX_STEPS = 5400 # 3 levels * 60s/level * 30fps

        # Colors
        self.COLOR_BG = (15, 23, 42)
        self.COLOR_STAR = (203, 213, 225)
        self.COLOR_PLAYER = (250, 204, 21)
        self.COLOR_PLAYER_SHADOW = (202, 138, 4)
        self.COLOR_OBSTACLE = (220, 38, 38)
        self.COLOR_OBSTACLE_SHADOW = (153, 27, 27)
        self.COLOR_GROUND = (30, 41, 59)
        self.COLOR_UI_TEXT = (241, 245, 249)
        self.COLOR_HEART = (225, 29, 72)
        
        # Physics Constants
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.GROUND_Y = self.HEIGHT - 50

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
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel_y = None
        self.player_rect = None
        self.player_squash = None
        self.obstacles = None
        self.stars = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.lives = None
        self.level = None
        self.level_progress = None
        self.level_timer = None
        self.obstacle_speed = None
        self.obstacle_height_variation = None
        self.prev_space_held = None
        
        self.reset()
        # self.validate_implementation() # Commented out for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = 3
        self.level = 1
        self.level_progress = 0
        self.prev_space_held = False
        
        self.player_pos = pygame.Vector2(self.WIDTH // 4, self.GROUND_Y)
        self.player_vel_y = 0
        self.player_squash = 1.0 # 1.0 = normal, <1.0 = squashed, >1.0 = stretched

        self.particles = []
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT-50))
            for _ in range(100)
        ]
        
        self._setup_level()
        
        # Initialize player rect
        w = 30 * (2.0 - self.player_squash)
        h = 30 * self.player_squash
        self.player_rect = pygame.Rect(self.player_pos.x - w / 2, self.player_pos.y - h, w, h)
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self.level_progress = 0
        self.level_timer = 60 * self.FPS # 60 seconds

        if self.level == 1:
            self.obstacle_speed = 4.0
            self.obstacle_height_variation = 40
        elif self.level == 2:
            self.obstacle_speed = 5.0
            self.obstacle_height_variation = 50
        elif self.level == 3:
            self.obstacle_speed = 6.0
            self.obstacle_height_variation = 60
        
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        obstacles = []
        current_x = self.WIDTH * 1.5
        while current_x < self.LEVEL_LENGTH:
            height = self.np_random.integers(20, 20 + self.obstacle_height_variation)
            width = self.np_random.integers(30, 60)
            y = self.GROUND_Y - height
            obstacles.append(pygame.Rect(current_x, y, width, height))
            min_gap = int(self.player_pos.x * 1.5) # ensure jumpable
            max_gap = int(self.player_pos.x * 2.5)
            current_x += width + self.np_random.integers(min_gap, max_gap)
        return obstacles

    def step(self, action):
        reward = 0
        terminated = False
        
        if not self.game_over:
            # Unpack factorized action
            space_pressed = (action[1] == 1) and not self.prev_space_held
            self.prev_space_held = (action[1] == 1)
            
            # --- Update Game Logic ---
            self._update_player(space_pressed)
            self._update_obstacles()
            self._update_particles()
            
            self.level_progress += self.obstacle_speed
            self.score = int(self.level_progress / 10) # score is distance
            self.steps += 1
            self.level_timer -= 1
            
            # --- Handle Events and Rewards ---
            # Continuous reward for survival
            reward += 0.01 

            # Collision check
            if self._handle_collisions():
                reward -= 5
                self.lives -= 1
                # sfx: player_hit.wav
                self._create_particles(self.player_pos, 20, self.COLOR_PLAYER)

            # Level completion check
            if self.level_progress >= self.LEVEL_LENGTH:
                reward += 10
                self.level += 1
                if self.level > 3:
                    self.win = True
                    self.game_over = True
                    reward += 100 # Big reward for winning
                    # sfx: game_win.wav
                else:
                    # sfx: level_up.wav
                    self._setup_level()

        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self, jump_action):
        on_ground = self.player_pos.y >= self.GROUND_Y
        
        if jump_action and on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.player_squash = 1.5 # Stretch on jump
            # sfx: jump.wav
        
        self.player_vel_y += self.GRAVITY
        self.player_pos.y += self.player_vel_y
        
        if self.player_pos.y > self.GROUND_Y:
            if not on_ground: # Just landed
                self.player_squash = 0.7 # Squash on land
                self._create_particles(pygame.Vector2(self.player_pos.x, self.GROUND_Y), 10, (255, 255, 255))
                # sfx: land.wav
            self.player_pos.y = self.GROUND_Y
            self.player_vel_y = 0
        
        # Animate squash/stretch back to normal
        self.player_squash += (1.0 - self.player_squash) * 0.2

        # Update player rect for collision and rendering
        w = 30 * (2.0 - self.player_squash)
        h = 30 * self.player_squash
        self.player_rect = pygame.Rect(self.player_pos.x - w / 2, self.player_pos.y - h, w, h)

    def _update_obstacles(self):
        for obstacle in self.obstacles:
            obstacle.x -= self.obstacle_speed
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_collisions(self):
        for obstacle in self.obstacles:
            if self.player_rect.colliderect(obstacle):
                return True
        return False

    def _check_termination(self):
        if self.lives <= 0 or self.level_timer <= 0:
            self.game_over = True
            # sfx: game_over.wav
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        return self.game_over

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.np_random.integers(15, 30),
                'color': color,
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_ground()
        self._render_obstacles()
        self._render_particles()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()

        if self.game_over:
            self._render_game_over_screen()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for x, y in self.stars:
            pygame.gfxdraw.pixel(self.screen, x, y, self.COLOR_STAR)

    def _render_ground(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

    def _render_obstacles(self):
        for obs in self.obstacles:
            if obs.right > 0 and obs.left < self.WIDTH:
                # Draw shadow
                shadow_rect = obs.copy()
                shadow_rect.height = 8
                shadow_rect.top = self.GROUND_Y
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_SHADOW, shadow_rect, border_radius=3)
                # Draw main obstacle
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs, border_radius=3)

    def _render_player(self):
        if self.player_rect:
            # Shadow
            shadow_radius = int(self.player_rect.width * 0.5)
            shadow_center_y = self.GROUND_Y + 4
            shadow_dist_factor = 1 - min(1, (self.GROUND_Y - self.player_pos.y) / 200)
            shadow_radius = int(shadow_radius * shadow_dist_factor)
            shadow_alpha = int(100 * shadow_dist_factor)
            
            if shadow_radius > 1:
                shadow_surf = pygame.Surface((shadow_radius*2, shadow_radius*2), pygame.SRCALPHA)
                pygame.draw.ellipse(shadow_surf, (0,0,0,shadow_alpha), (0,0,shadow_radius*2, shadow_radius*2))
                self.screen.blit(shadow_surf, (self.player_pos.x - shadow_radius, shadow_center_y - shadow_radius))

            # Body
            pygame.draw.ellipse(self.screen, self.COLOR_PLAYER_SHADOW, self.player_rect.move(0, 2))
            pygame.draw.ellipse(self.screen, self.COLOR_PLAYER, self.player_rect)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color_with_alpha = p['color'] + (alpha,)
            size = max(1, int(p['life'] / 5))
            # Create a temporary surface for the particle to handle alpha
            particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color_with_alpha, (size, size), size)
            self.screen.blit(particle_surf, (int(p['pos'].x - size), int(p['pos'].y - size)))

    def _render_ui(self):
        # Score / Distance
        score_text = f"SCORE: {self.score}"
        text_surf = self.font_medium.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, 10))

        # Level
        level_text = f"LEVEL: {self.level}"
        text_surf = self.font_medium.render(level_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 20, 10))
        
        # Lives
        for i in range(self.lives):
            self._draw_heart(25 + i * 35, 25, 12)

        # Timer
        timer_text = f"TIME: {self.level_timer // self.FPS:02d}"
        text_surf = self.font_medium.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, 40))

        # Level Progress Bar
        progress_ratio = self.level_progress / self.LEVEL_LENGTH
        bar_width = self.WIDTH - 40
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (20, self.HEIGHT - 20, bar_width, 10), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (20, self.HEIGHT - 20, bar_width * progress_ratio, 10), border_radius=5)

    def _draw_heart(self, x, y, size):
        points = [
            (x, y - size * 0.4),
            (x + size * 0.5, y - size),
            (x + size, y - size * 0.4),
            (x, y + size * 0.8),
            (x - size, y - size * 0.4),
            (x - size * 0.5, y - size),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "YOU WIN!" if self.win else "GAME OVER"
        text_surf = self.font_large.render(message, True, self.COLOR_PLAYER if self.win else self.COLOR_OBSTACLE)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
        self.screen.blit(text_surf, text_rect)

        final_score_text = f"Final Score: {self.score}"
        score_surf = self.font_medium.render(final_score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 40))
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "level": self.level,
            "game_over": self.game_over,
            "win": self.win
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        super().close()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Set up a window to display the game
    # This part requires a display and will not run in a headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    pygame.display.init()
    pygame.display.set_caption("Space Hopper")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Get keyboard input
        keys = pygame.key.get_pressed()
        action[0] = 0 # Movement is not used
        action[1] = 1 if keys[pygame.K_SPACE] else 0 # Space for jump
        action[2] = 0 # Shift is not used

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)

    env.close()