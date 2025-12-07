
# Generated: 2025-08-28T07:08:43.739393
# Source Brief: brief_03157.md
# Brief Index: 3157

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to steer. Hold Space to accelerate. Dodge the blue obstacles and complete 3 laps as fast as you can."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "A fast-paced, retro vector-graphics racer. Navigate a procedurally generated track, dodge obstacles, and race against the clock to set the best time."
    )

    # Frames auto-advance at 30fps for smooth gameplay.
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_GLOW = (255, 100, 100)
    COLOR_OBSTACLE = (50, 150, 255)
    COLOR_OBSTACLE_GLOW = (100, 200, 255)
    COLOR_TRACK = (220, 220, 220)
    COLOR_FINISH_1 = (50, 200, 50)
    COLOR_FINISH_2 = (40, 160, 40)
    COLOR_PARTICLE = (255, 255, 200)
    COLOR_NEAR_MISS = (255, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_SPEED_LINE = (50, 50, 70)

    # Player
    PLAYER_Y_POS = SCREEN_HEIGHT - 80
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 25
    PLAYER_ACCEL = 1.2
    PLAYER_FRICTION = 0.92
    MAX_PLAYER_SPEED_X = 15

    # Track
    TRACK_WIDTH = 400
    LAP_LENGTH = 15000  # Pixels
    SECTION_LENGTH = 800 # Generate new obstacles every section
    
    # Gameplay
    BASE_SCROLL_SPEED = 5
    ACCEL_BOOST = 10
    MAX_SCROLL_SPEED = 20
    FPS = 30
    MAX_EPISODE_STEPS = 10000
    LAP_TIME_SECONDS = 60
    TOTAL_LAPS = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel_x = 0.0
        self.scroll_speed = 0.0
        self.track_progress = 0.0
        self.lap = 1
        self.total_time_steps = 0
        self.obstacles = []
        self.particles = []
        self.speed_lines = []
        self.obstacle_density = 0.0
        self.next_section_y = 0.0
        self.track_left_x = 0
        self.track_right_x = 0
        self.finish_line_y = float('inf')

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.PLAYER_Y_POS)
        self.player_vel_x = 0.0
        
        self.scroll_speed = self.BASE_SCROLL_SPEED
        self.track_progress = 0.0
        self.lap = 1
        self.total_time_steps = self.LAP_TIME_SECONDS * self.FPS * self.TOTAL_LAPS
        
        self.obstacles = []
        self.particles = []
        self.speed_lines = []
        
        self.obstacle_density = 0.1
        self.next_section_y = 0

        self.track_left_x = (self.SCREEN_WIDTH - self.TRACK_WIDTH) / 2
        self.track_right_x = self.track_left_x + self.TRACK_WIDTH
        
        # Pre-populate the first few sections of the track
        for i in range(3):
            self._generate_track_section(i * -self.SECTION_LENGTH)

        self.finish_line_y = self.LAP_LENGTH
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            # If game is over, just return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        accelerate_held = action[1] == 1
        
        # --- Update Player ---
        if movement == 3:  # Left
            self.player_vel_x -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel_x += self.PLAYER_ACCEL
        
        self.player_vel_x *= self.PLAYER_FRICTION
        self.player_vel_x = np.clip(self.player_vel_x, -self.MAX_PLAYER_SPEED_X, self.MAX_PLAYER_SPEED_X)
        self.player_pos.x += self.player_vel_x
        self.player_pos.x = np.clip(self.player_pos.x, self.track_left_x, self.track_right_x)

        # --- Update Scroll Speed (Acceleration) ---
        if accelerate_held:
            self.scroll_speed += (self.MAX_SCROLL_SPEED - self.scroll_speed) * 0.1
            # Sound: engine_rev.wav
        else:
            self.scroll_speed += (self.BASE_SCROLL_SPEED - self.scroll_speed) * 0.05

        self.scroll_speed = np.clip(self.scroll_speed, self.BASE_SCROLL_SPEED, self.MAX_SCROLL_SPEED)

        # --- Update Game World ---
        self.track_progress += self.scroll_speed
        self.total_time_steps -= 1

        # Update entities
        self._update_obstacles()
        self._update_particles()
        self._update_speed_lines()
        
        # Generate new track content
        if self.track_progress > self.next_section_y + self.SECTION_LENGTH:
            self.next_section_y += self.SECTION_LENGTH
            self._generate_track_section(-self.SCREEN_HEIGHT) # Generate off-screen
            reward += 1.0 # Reward for passing a section

        # --- Check Game State & Calculate Rewards ---
        reward += 0.1  # Survival reward

        # Collision & Near-Miss Check
        player_rect = self._get_player_rect()
        collided = False
        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                reward = -100.0
                terminated = True
                self.game_over = True
                collided = True
                self._create_explosion(self.player_pos)
                # Sound: explosion.wav
                break
            # Near-miss check
            near_miss_rect = player_rect.inflate(60, 60)
            if not obs.get('near_missed', False) and near_miss_rect.colliderect(obs['rect']):
                reward -= 5.0
                obs['near_missed'] = True
                self._create_near_miss_effect( (player_rect.centerx + obs['rect'].centerx)/2, (player_rect.centery + obs['rect'].centery)/2 )
                # Sound: near_miss_whoosh.wav
        
        # Lap Completion
        if self.track_progress >= self.LAP_LENGTH and not collided:
            self.lap += 1
            self.track_progress = 0
            self.next_section_y = 0
            self.obstacle_density += 0.05
            
            if self.lap > self.TOTAL_LAPS:
                reward += 300.0
                terminated = True
                self.game_over = True
                self.win_state = True
                # Sound: win_jingle.wav
            else:
                reward += 100.0
                # Sound: lap_complete.wav
        
        # Timeout Check
        if (self.total_time_steps <= 0 or self.steps >= self.MAX_EPISODE_STEPS) and not terminated:
            reward = -50.0
            terminated = True
            self.game_over = True
            # Sound: timeout_buzzer.wav

        self.score += reward
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "lap": self.lap,
            "time_left": self.total_time_steps / self.FPS,
        }

    # --- Helper Methods for Updates ---

    def _generate_track_section(self, y_offset):
        num_obstacles = int(self.obstacle_density * (self.SECTION_LENGTH / 100))
        for _ in range(num_obstacles):
            size = self.np_random.integers(20, 50)
            obs_x = self.np_random.uniform(self.track_left_x, self.track_right_x)
            obs_y = y_offset - self.np_random.uniform(0, self.SECTION_LENGTH)
            shape = 'rect' if self.np_random.random() > 0.5 else 'circle'
            self.obstacles.append({
                'pos': pygame.Vector2(obs_x, obs_y),
                'size': size,
                'shape': shape,
                'rect': pygame.Rect(obs_x - size/2, obs_y - size/2, size, size)
            })

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['pos'].y += self.scroll_speed
            obs['rect'].center = obs['pos']
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].top < self.SCREEN_HEIGHT]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def _update_speed_lines(self):
        # Create new speed lines
        if self.np_random.random() < self.scroll_speed / self.MAX_SCROLL_SPEED:
            line_x = self.np_random.uniform(0, self.SCREEN_WIDTH)
            self.speed_lines.append({
                'pos': pygame.Vector2(line_x, 0),
                'len': self.np_random.uniform(20, 60) * (self.scroll_speed / self.BASE_SCROLL_SPEED)
            })
        
        # Update existing ones
        for line in self.speed_lines:
            line['pos'].y += self.scroll_speed * 1.5 # Faster than track for parallax
        self.speed_lines = [line for line in self.speed_lines if line['pos'].y < self.SCREEN_HEIGHT]

    # --- Helper Methods for Effects ---

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 10)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(15, 30),
                'color': random.choice([self.COLOR_PLAYER, (255,150,0), (255,255,0)])
            })

    def _create_near_miss_effect(self, x, y):
         self.particles.append({
            'pos': pygame.Vector2(x, y),
            'vel': pygame.Vector2(0,0),
            'life': 8,
            'color': self.COLOR_NEAR_MISS,
            'type': 'flash'
        })

    # --- Helper Methods for Rendering ---

    def _render_game(self):
        # Speed Lines
        for line in self.speed_lines:
            start_pos = (int(line['pos'].x), int(line['pos'].y))
            end_pos = (int(line['pos'].x), int(line['pos'].y - line['len']))
            pygame.draw.line(self.screen, self.COLOR_SPEED_LINE, start_pos, end_pos, 2)
            
        # Track Edges
        pygame.draw.line(self.screen, self.COLOR_TRACK, (self.track_left_x, 0), (self.track_left_x, self.SCREEN_HEIGHT), 5)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (self.track_right_x, 0), (self.track_right_x, self.SCREEN_HEIGHT), 5)

        # Finish Line
        finish_y = self.LAP_LENGTH - self.track_progress
        if -50 < finish_y < self.SCREEN_HEIGHT:
            check_size = 20
            for i in range(int(self.track_left_x), int(self.track_right_x), check_size):
                color = self.COLOR_FINISH_1 if (i // check_size) % 2 == 0 else self.COLOR_FINISH_2
                pygame.draw.rect(self.screen, color, (i, finish_y, check_size, 10))

        # Obstacles
        for obs in self.obstacles:
            if obs['rect'].bottom > 0 and obs['rect'].top < self.SCREEN_HEIGHT:
                if obs['shape'] == 'rect':
                    pygame.gfxdraw.box(self.screen, obs['rect'], self.COLOR_OBSTACLE)
                    pygame.gfxdraw.rectangle(self.screen, obs['rect'], self.COLOR_OBSTACLE_GLOW)
                else: # circle
                    center = (int(obs['rect'].centerx), int(obs['rect'].centery))
                    radius = int(obs['size'] / 2)
                    pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_OBSTACLE)
                    pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_OBSTACLE_GLOW)
        
        # Player
        if not (self.game_over and not self.win_state):
            p = self.player_pos
            points = [
                (p.x, p.y - self.PLAYER_HEIGHT / 2),
                (p.x - self.PLAYER_WIDTH / 2, p.y + self.PLAYER_HEIGHT / 2),
                (p.x + self.PLAYER_WIDTH / 2, p.y + self.PLAYER_HEIGHT / 2)
            ]
            # Glow
            pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER)

        # Particles
        for p in self.particles:
            if p.get('type') == 'flash':
                radius = (8 - p['life']) * 5
                alpha = int(max(0, (p['life'] / 8) * 255))
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, p['color'] + (alpha,), (radius, radius), radius)
                self.screen.blit(s, (p['pos'].x - radius, p['pos'].y - radius))
            else:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), max(1, int(p['life']/5)))

    def _render_ui(self):
        # Time
        time_text = f"TIME: {max(0, self.total_time_steps / self.FPS):.1f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Lap
        lap_text = f"LAP: {min(self.lap, self.TOTAL_LAPS)} / {self.TOTAL_LAPS}"
        lap_surf = self.font_small.render(lap_text, True, self.COLOR_TEXT)
        self.screen.blit(lap_surf, (self.SCREEN_WIDTH - lap_surf.get_width() - 10, 10))

        # Speed
        speed_text = f"{(self.scroll_speed * 10):.0f} KPH"
        speed_surf = self.font_small.render(speed_text, True, self.COLOR_TEXT)
        self.screen.blit(speed_surf, (self.SCREEN_WIDTH - speed_surf.get_width() - 10, self.SCREEN_HEIGHT - 30))

        # Game Over / Win Message
        if self.game_over:
            if self.win_state:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_player_rect(self):
        return pygame.Rect(
            self.player_pos.x - self.PLAYER_WIDTH / 2,
            self.player_pos.y - self.PLAYER_HEIGHT / 2,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT
        )

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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