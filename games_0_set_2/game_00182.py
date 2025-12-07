
# Generated: 2025-08-27T12:50:58.513880
# Source Brief: brief_00182.md
# Brief Index: 182

        
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
        "Controls: Use ↑ and ↓ to move the snail up and down to dodge the rocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide your speedy snail to the finish line! Dodge rocks and race against the clock in this fast-paced side-scroller."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.TRACK_LENGTH = 10000  # Total distance to the finish line

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_SKY = (135, 206, 235)
        self.COLOR_SKY_DARK = (100, 160, 200)
        self.COLOR_TRACK_TOP = (124, 252, 0)
        self.COLOR_TRACK_BOTTOM = (34, 139, 34)
        self.COLOR_TRACK_LINE = (107, 220, 0)
        self.COLOR_SNAIL_BODY = (255, 255, 102)
        self.COLOR_SNAIL_SHELL = (210, 105, 30)
        self.COLOR_SNAIL_SHELL_DARK = (139, 69, 19)
        self.COLOR_OBSTACLE = (139, 115, 85)
        self.COLOR_OBSTACLE_OUTLINE = (80, 60, 40)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.COLOR_PROGRESS_BAR = (255, 215, 0)
        self.COLOR_PROGRESS_BG = (50, 50, 50)

        # Game constants
        self.MAX_TIME = 60  # in seconds
        self.MAX_STEPS = self.MAX_TIME * 30  # Assuming 30 FPS
        self.SNAIL_Y_SPEED = 5
        self.WORLD_SCROLL_SPEED = self.TRACK_LENGTH / self.MAX_STEPS
        self.TRACK_Y_START = 200
        self.TRACK_HEIGHT = 180
        
        # State variables are initialized in reset()
        self.snail_rect = None
        self.world_progress = 0
        self.time_left_frames = 0
        self.obstacles = []
        self.particles = []
        self.track_lines = []
        self.clouds = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        # Initialize state variables
        self.reset()

        # Run validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False

        snail_width, snail_height = 50, 30
        self.snail_rect = pygame.Rect(
            100, 
            self.TRACK_Y_START + (self.TRACK_HEIGHT - snail_height) // 2, 
            snail_width, 
            snail_height
        )
        
        self.world_progress = 0
        self.time_left_frames = self.MAX_TIME * 30

        self.obstacles = []
        self.particles = []

        # Generate procedural background elements
        self.track_lines = [self.np_random.integers(0, self.SCREEN_WIDTH) for _ in range(20)]
        self.clouds = [(self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(20, 100)) for _ in range(5)]

        # Spawn initial obstacles
        for i in range(5):
            self._spawn_obstacle(spawn_x=self.SCREEN_WIDTH + i * 200)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- UPDATE GAME LOGIC ---
        self.steps += 1
        
        # 1. Handle player input
        if movement == 1:  # Up
            self.snail_rect.y -= self.SNAIL_Y_SPEED
        elif movement == 2:  # Down
            self.snail_rect.y += self.SNAIL_Y_SPEED
        
        # Clamp snail position to track
        self.snail_rect.y = np.clip(
            self.snail_rect.y,
            self.TRACK_Y_START,
            self.TRACK_Y_START + self.TRACK_HEIGHT - self.snail_rect.height
        )
        
        # 2. Update world state
        self.world_progress += self.WORLD_SCROLL_SPEED
        self.time_left_frames -= 1
        
        self._update_obstacles()
        self._update_particles()
        self._update_background_elements()
        
        # 3. Check for game events and calculate reward
        reward = 0.01  # Small reward for surviving
        
        # Collision check
        if self._check_collisions():
            self.game_over = True
            reward = -10
            # sfx: explosion
            self._add_particles(self.snail_rect.center, 50, (255, 69, 0))
        
        # Win condition check
        if self.world_progress >= self.TRACK_LENGTH and not self.game_over:
            self.game_over = True
            self.win_condition_met = True
            time_bonus = max(0, self.time_left_frames / (self.MAX_TIME * 30))
            reward = 100 + (100 * time_bonus) # Scaled bonus reward
            # sfx: victory fanfare
        
        # Time out check
        if self.time_left_frames <= 0 and not self.game_over:
            self.game_over = True
            reward = -50 # Hefty penalty for timing out
            # sfx: sad trombone

        # Max steps check
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            reward = -50 # Also a failure state
        
        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_obstacles(self):
        # Move existing obstacles
        for obstacle in self.obstacles:
            obstacle['rect'].x -= self.WORLD_SCROLL_SPEED
        
        # Remove off-screen obstacles
        self.obstacles = [o for o in self.obstacles if o['rect'].right > 0]
        
        # Spawn new obstacles
        progress_ratio = self.steps / self.MAX_STEPS
        # Spawn rate increases from 1/50 to 1/25 over the episode
        spawn_chance = np.interp(progress_ratio, [0, 1], [0.02, 0.04])
        if self.np_random.random() < spawn_chance:
            self._spawn_obstacle()

    def _spawn_obstacle(self, spawn_x=None):
        if spawn_x is None:
            spawn_x = self.SCREEN_WIDTH + 50
        
        height = self.np_random.integers(30, 80)
        width = self.np_random.integers(30, 60)
        y_pos = self.np_random.integers(self.TRACK_Y_START, self.TRACK_Y_START + self.TRACK_HEIGHT - height)
        
        # Create a procedural rock shape
        points = []
        num_points = 8
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            radius_variance = self.np_random.uniform(0.7, 1.0)
            px = (width / 2) * math.cos(angle) * radius_variance + (width / 2)
            py = (height / 2) * math.sin(angle) * radius_variance + (height / 2)
            points.append((px, py))

        self.obstacles.append({
            'rect': pygame.Rect(spawn_x, y_pos, width, height),
            'points': points
        })

    def _check_collisions(self):
        for obstacle in self.obstacles:
            if self.snail_rect.colliderect(obstacle['rect']):
                return True
        return False

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['alpha'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['alpha'] -= 5
            p['size'] = max(0, p['size'] - 0.1)

    def _update_background_elements(self):
        # Update track lines for speed effect
        for i in range(len(self.track_lines)):
            self.track_lines[i] -= self.WORLD_SCROLL_SPEED * 1.5
            if self.track_lines[i] < 0:
                self.track_lines[i] = self.SCREEN_WIDTH
        
        # Update clouds for parallax effect
        for i in range(len(self.clouds)):
            x, y = self.clouds[i]
            x -= self.WORLD_SCROLL_SPEED * 0.2
            if x < -100:
                x = self.SCREEN_WIDTH + self.np_random.integers(50, 150)
                y = self.np_random.integers(20, 100)
            self.clouds[i] = (x, y)

    def _add_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'alpha': 255,
                'size': self.np_random.uniform(2, 6),
                'color': color
            })
    
    def _get_observation(self):
        # --- RENDER ALL ELEMENTS ---
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_SKY)
        pygame.draw.rect(self.screen, self.COLOR_SKY_DARK, (0, 150, self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        # Clouds
        for x, y in self.clouds:
            pygame.gfxdraw.filled_ellipse(self.screen, int(x), int(y), 40, 15, (255, 255, 255, 150))
            pygame.gfxdraw.filled_ellipse(self.screen, int(x+25), int(y+5), 50, 20, (255, 255, 255, 150))
        
        # Track
        track_rect = (0, self.TRACK_Y_START, self.SCREEN_WIDTH, self.TRACK_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_TRACK_BOTTOM, track_rect)
        pygame.draw.rect(self.screen, self.COLOR_TRACK_TOP, (0, self.TRACK_Y_START, self.SCREEN_WIDTH, self.TRACK_HEIGHT-140))
        
        # Track lines for speed effect
        for x in self.track_lines:
            pygame.draw.line(self.screen, self.COLOR_TRACK_LINE, (x, self.TRACK_Y_START), (x, self.TRACK_Y_START + self.TRACK_HEIGHT), 2)
            
        # Finish Line
        finish_line_x = self.TRACK_LENGTH - self.world_progress
        if finish_line_x < self.SCREEN_WIDTH:
            for i in range(0, self.TRACK_HEIGHT // 20):
                for j in range(2):
                    color = (255, 255, 255) if (i + j) % 2 == 0 else (0, 0, 0)
                    pygame.draw.rect(self.screen, color, (finish_line_x + j * 20, self.TRACK_Y_START + i * 20, 20, 20))

        # Obstacles
        for obstacle in self.obstacles:
            points = [(p[0] + obstacle['rect'].x, p[1] + obstacle['rect'].y) for p in obstacle['points']]
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE_OUTLINE)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)

        # Particles
        for p in self.particles:
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], p['alpha']), (p['size'], p['size']), p['size'])
            self.screen.blit(s, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

        # Snail (if not crashed)
        if not (self.game_over and not self.win_condition_met):
            self._render_snail()

    def _render_snail(self):
        body_rect = pygame.Rect(self.snail_rect.x, self.snail_rect.y + 15, 40, 15)
        pygame.draw.ellipse(self.screen, self.COLOR_SNAIL_BODY, body_rect)
        pygame.draw.ellipse(self.screen, (0,0,0), body_rect, 1)

        # Eyes
        eye_stalk_y = self.snail_rect.y + 10
        eye_x_offset = 30
        pygame.draw.line(self.screen, (0,0,0), (self.snail_rect.x + eye_x_offset, eye_stalk_y), (self.snail_rect.x + eye_x_offset, eye_stalk_y+5), 2)
        pygame.draw.circle(self.screen, (0,0,0), (self.snail_rect.x + eye_x_offset, eye_stalk_y), 3)

        # Animated Shell
        shell_center_x = self.snail_rect.x + 15
        shell_center_y = self.snail_rect.y + 15
        angle_offset = (self.steps * 5) % 360
        
        for i in range(5):
            radius = 15 - i * 2.5
            start_angle = math.radians(angle_offset + i * 45)
            end_angle = math.radians(angle_offset + i * 45 + 270)
            arc_rect = pygame.Rect(shell_center_x - radius, shell_center_y - radius, radius*2, radius*2)
            if radius > 1:
                pygame.draw.arc(self.screen, self.COLOR_SNAIL_SHELL_DARK, arc_rect, start_angle, end_angle, 3)
                pygame.draw.arc(self.screen, self.COLOR_SNAIL_SHELL, arc_rect, start_angle, end_angle, 2)

    def _render_ui(self):
        # Helper for shadowed text
        def draw_text(text, font, color, x, y, center=False):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            main = font.render(text, True, color)
            shadow_pos = shadow.get_rect()
            main_pos = main.get_rect()
            if center:
                shadow_pos.center = (x + 2, y + 2)
                main_pos.center = (x, y)
            else:
                shadow_pos.topleft = (x + 1, y + 1)
                main_pos.topleft = (x, y)
            self.screen.blit(shadow, shadow_pos)
            self.screen.blit(main, main_pos)

        # Timer
        time_sec = max(0, self.time_left_frames / 30)
        timer_text = f"TIME: {time_sec:.1f}"
        draw_text(timer_text, self.font_small, self.COLOR_TEXT, self.SCREEN_WIDTH - 100, 10)

        # Progress Bar
        progress_ratio = np.clip(self.world_progress / self.TRACK_LENGTH, 0, 1)
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BG, (10, self.SCREEN_HEIGHT - 25, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, (10, self.SCREEN_HEIGHT - 25, bar_width * progress_ratio, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, self.SCREEN_HEIGHT - 25, bar_width, bar_height), 1)

        # Game Over / Win Message
        if self.game_over:
            if self.win_condition_met:
                draw_text("YOU WIN!", self.font_large, (255, 215, 0), self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20, center=True)
            else:
                draw_text("GAME OVER", self.font_large, (255, 0, 0), self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": round(max(0, self.time_left_frames / 30), 2),
            "progress": round(self.world_progress / self.TRACK_LENGTH * 100, 2),
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Set SDL_VIDEODRIVER to a dummy value to run headless
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a real screen
    pygame.display.set_caption("Snail Race")
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # --- Player Input ---
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                done = False

        # --- Game Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if terminated:
                 print(f"Episode Finished. Final Info: {info}")

        # --- Rendering ---
        # The observation is a numpy array, convert it back to a Pygame surface for display
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        # Control frame rate
        env.clock.tick(30)
        
    env.close()