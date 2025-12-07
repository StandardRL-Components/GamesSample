import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set SDL_VIDEODRIVER to dummy for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a fast-paced arcade racer. The player must
    navigate a procedurally generated track, avoid obstacles, and complete
    three laps before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use ← and → to steer. Hold Space to boost."
    )
    game_description = (
        "A retro-vector arcade racer. Dodge obstacles, complete three laps, "
        "and race against the clock for the high score."
    )

    # Frame advance behavior
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment, including Pygame, spaces, and constants.
        """
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Screen & Timing ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30

        # --- Pygame Setup (headless) ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Fonts ---
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_tiny = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            # Fallback if default font is not found
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 24)
            self.font_tiny = pygame.font.SysFont("sans", 16)


        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_TRACK = (220, 220, 255)
        self.COLOR_CAR = (255, 80, 80)
        self.COLOR_CAR_GLOW = (255, 120, 120, 100)
        self.COLOR_OBSTACLE = (80, 180, 255)
        self.COLOR_OBSTACLE_GLOW = (120, 200, 255, 100)
        self.COLOR_NEAR_MISS = (255, 255, 100)
        self.COLOR_COLLISION = (255, 150, 50)
        self.COLOR_BOOST = (100, 150, 255)
        self.COLOR_UI_TEXT = (240, 240, 240)
        
        # --- Game Constants ---
        self.MAX_TIME_SECONDS = 60.0
        self.MAX_STEPS = int(self.MAX_TIME_SECONDS * self.FPS)
        self.LAPS_TO_WIN = 3
        
        # --- Player Car ---
        self.CAR_WIDTH = 20
        self.CAR_HEIGHT = 30
        self.CAR_STEER_SPEED = 8.0
        self.PLAYER_Y_POS = self.SCREEN_HEIGHT - 80

        # --- Track & Scrolling ---
        self.TRACK_WIDTH_PERCENT = 0.8
        self.TRACK_SEGMENT_LENGTH = 10
        self.TRACK_TURN_RATE = 4.0
        self.LAP_LENGTH_SEGMENTS = 300
        self.TRACK_TOTAL_LENGTH = self.LAP_LENGTH_SEGMENTS * self.TRACK_SEGMENT_LENGTH

        # --- Obstacles ---
        self.OBSTACLE_SIZE = 25
        self.OBSTACLE_SPAWN_INTERVAL = 150 # pixels of progress

        # --- Particles ---
        self.particles = []

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.laps_completed = 0
        self.time_elapsed = 0.0
        self.player_x = 0
        self.base_scroll_speed = 0
        self.current_scroll_speed = 0
        self.boost_active = False
        self.track_progress = 0
        self.track_segments = []
        self.obstacles = []
        self.last_obstacle_spawn_progress = 0
        self.stars = []

        # --- Initialize state and validate ---
        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.laps_completed = 0
        self.time_elapsed = 0.0
        
        self.player_x = self.SCREEN_WIDTH / 2
        self.base_scroll_speed = 5.0
        self.current_scroll_speed = self.base_scroll_speed # FIX: Initialize current_scroll_speed
        self.boost_active = False
        
        self.track_progress = 0
        self.last_obstacle_spawn_progress = -self.OBSTACLE_SPAWN_INTERVAL
        
        self.obstacles = []
        self.particles = []
        
        self._generate_track()
        self._generate_parallax_stars()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # --- Unpack Actions ---
        movement = action[0]  # 0:none, 1:up, 2:down, 3:left, 4:right
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused

        self.reward_this_step = 0.01 # Small reward for surviving
        
        # --- Update Game Logic ---
        self._handle_player_movement(movement)
        self._update_scroll_speed(space_held)
        self._update_track_and_obstacles()
        self._update_particles()
        self._check_collisions()
        self._check_lap_completion()

        # --- Update Timers and Counters ---
        self.steps += 1
        self.time_elapsed += 1.0 / self.FPS
        
        # --- Check Termination Conditions ---
        terminated = self.game_over
        if self.laps_completed >= self.LAPS_TO_WIN:
            self.reward_this_step += 50 # Win bonus
            self.score += 50
            terminated = True
        elif self.time_elapsed >= self.MAX_TIME_SECONDS:
            self.reward_this_step -= 10 # Timeout penalty
            self.score -= 10
            terminated = True
        
        self.game_over = terminated
        self.score += self.reward_this_step

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if movement == 3: # Left
            self.player_x -= self.CAR_STEER_SPEED
        elif movement == 4: # Right
            self.player_x += self.CAR_STEER_SPEED
        
        # Clamp player position to stay within screen bounds
        self.player_x = np.clip(self.player_x, self.CAR_WIDTH/2, self.SCREEN_WIDTH - self.CAR_WIDTH/2)

    def _update_scroll_speed(self, space_held):
        self.boost_active = space_held
        if self.boost_active:
            # sfx: boost_sound.play()
            self.current_scroll_speed = self.base_scroll_speed * 1.75
            self._create_particle_effect(
                count=1,
                pos=(self.player_x, self.PLAYER_Y_POS + self.CAR_HEIGHT / 2),
                color=self.COLOR_BOOST,
                lifespan=(5, 15),
                size=(3, 6),
                velocity_angle_range=(math.pi/2 - 0.2, math.pi/2 + 0.2),
                velocity_mag_range=(1, 3)
            )
        else:
            self.current_scroll_speed = self.base_scroll_speed

    def _update_track_and_obstacles(self):
        self.track_progress += self.current_scroll_speed
        
        # Update obstacles
        for obs in self.obstacles:
            obs['y'] += self.current_scroll_speed
        self.obstacles = [obs for obs in self.obstacles if obs['y'] < self.SCREEN_HEIGHT + self.OBSTACLE_SIZE]
        
        # Spawn new obstacles
        if self.track_progress > self.last_obstacle_spawn_progress + self.OBSTACLE_SPAWN_INTERVAL:
            self._spawn_obstacle()
            self.last_obstacle_spawn_progress = self.track_progress
            
    def _check_collisions(self):
        player_rect = pygame.Rect(
            self.player_x - self.CAR_WIDTH / 2,
            self.PLAYER_Y_POS - self.CAR_HEIGHT / 2,
            self.CAR_WIDTH,
            self.CAR_HEIGHT
        )
        
        for obs in self.obstacles:
            obs_rect = pygame.Rect(
                obs['x'] - self.OBSTACLE_SIZE / 2,
                obs['y'] - self.OBSTACLE_SIZE / 2,
                self.OBSTACLE_SIZE,
                self.OBSTACLE_SIZE
            )
            if player_rect.colliderect(obs_rect):
                # sfx: collision_sound.play()
                self.game_over = True
                self.reward_this_step -= 50 # Collision penalty
                self.score -= 50
                self._create_particle_effect(
                    count=40, pos=(self.player_x, self.PLAYER_Y_POS), color=self.COLOR_COLLISION,
                    lifespan=(20, 40), size=(3, 7), velocity_angle_range=(0, 2 * math.pi),
                    velocity_mag_range=(2, 8)
                )
                return

            # Near miss check
            if not obs.get('near_missed', False):
                expanded_rect = player_rect.inflate(self.CAR_WIDTH * 2, 0)
                if expanded_rect.colliderect(obs_rect):
                    # sfx: near_miss_sound.play()
                    self.reward_this_step += 0.5
                    obs['near_missed'] = True
                    self._create_particle_effect(
                        count=5, pos=(self.player_x, self.PLAYER_Y_POS), color=self.COLOR_NEAR_MISS,
                        lifespan=(10, 20), size=(2, 4), velocity_angle_range=(-math.pi/4, math.pi/4),
                        velocity_mag_range=(1, 4)
                    )

    def _check_lap_completion(self):
        if self.track_progress >= self.TRACK_TOTAL_LENGTH:
            # sfx: lap_complete_sound.play()
            self.laps_completed += 1
            self.reward_this_step += 10 # Lap bonus
            self.score += 10
            self.track_progress = self.track_progress % self.TRACK_TOTAL_LENGTH
            self.base_scroll_speed += 0.5 # Increase difficulty
            self._generate_track() # New track for next lap
            self.last_obstacle_spawn_progress = -self.OBSTACLE_SPAWN_INTERVAL

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.laps_completed,
            "time_left": max(0, self.MAX_TIME_SECONDS - self.time_elapsed)
        }
        
    def _render_game(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        self._render_parallax_stars()

        # --- Track ---
        self._render_track()

        # --- Obstacles ---
        for obs in self.obstacles:
            self._render_glowing_rect(
                int(obs['x']), int(obs['y']), self.OBSTACLE_SIZE, self.OBSTACLE_SIZE,
                self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW
            )

        # --- Particles ---
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            p_color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['size']), p_color)

        # --- Player Car ---
        if not self.game_over:
            self._render_player_car()

        # --- UI Overlay ---
        self._render_ui()

    def _render_player_car(self):
        car_points = [
            (self.player_x, self.PLAYER_Y_POS - self.CAR_HEIGHT / 2),
            (self.player_x - self.CAR_WIDTH / 2, self.PLAYER_Y_POS + self.CAR_HEIGHT / 2),
            (self.player_x + self.CAR_WIDTH / 2, self.PLAYER_Y_POS + self.CAR_HEIGHT / 2)
        ]
        car_points_int = [(int(p[0]), int(p[1])) for p in car_points]
        
        # Glow effect
        glow_surface = pygame.Surface((self.CAR_WIDTH * 3, self.CAR_HEIGHT * 3), pygame.SRCALPHA)
        glow_points = [
            (self.CAR_WIDTH * 1.5, 0),
            (0, self.CAR_HEIGHT * 3),
            (self.CAR_WIDTH * 3, self.CAR_HEIGHT * 3)
        ]
        pygame.draw.polygon(glow_surface, self.COLOR_CAR_GLOW, glow_points)
        try: # gaussian_blur might not be available on all pygame versions
            glow_surface = pygame.transform.gaussian_blur(glow_surface, 8)
        except AttributeError:
            pass # just render without blur if not available
        self.screen.blit(glow_surface, (self.player_x - self.CAR_WIDTH * 1.5, self.PLAYER_Y_POS - self.CAR_HEIGHT))

        # Main car shape
        pygame.gfxdraw.aapolygon(self.screen, car_points_int, self.COLOR_CAR)
        pygame.gfxdraw.filled_polygon(self.screen, car_points_int, self.COLOR_CAR)

    def _render_track(self):
        track_width = self.SCREEN_WIDTH * self.TRACK_WIDTH_PERCENT
        
        start_segment_idx = int(self.track_progress / self.TRACK_SEGMENT_LENGTH)
        
        for i in range(-2, int(self.SCREEN_HEIGHT / self.TRACK_SEGMENT_LENGTH) + 2):
            current_idx = (start_segment_idx + i) % self.LAP_LENGTH_SEGMENTS
            next_idx = (start_segment_idx + i + 1) % self.LAP_LENGTH_SEGMENTS
            
            y_offset = (self.track_progress % self.TRACK_SEGMENT_LENGTH)
            
            y1 = self.SCREEN_HEIGHT - (i * self.TRACK_SEGMENT_LENGTH) + y_offset
            y2 = self.SCREEN_HEIGHT - ((i + 1) * self.TRACK_SEGMENT_LENGTH) + y_offset
            
            center_x1 = self.track_segments[current_idx]
            center_x2 = self.track_segments[next_idx]

            left1 = (int(center_x1 - track_width / 2), int(y1))
            right1 = (int(center_x1 + track_width / 2), int(y1))
            left2 = (int(center_x2 - track_width / 2), int(y2))
            right2 = (int(center_x2 + track_width / 2), int(y2))
            
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, left1, left2, 2)
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, right1, right2, 2)

    def _render_ui(self):
        # Laps
        lap_text = self.font_small.render(f"LAP: {min(self.laps_completed + 1, self.LAPS_TO_WIN)} / {self.LAPS_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_text, (10, 10))

        # Time
        time_left = max(0, self.MAX_TIME_SECONDS - self.time_elapsed)
        secs = int(time_left % 60)
        time_text = self.font_small.render(f"TIME: {secs:02d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # Speed
        speed_bar_width = 150
        speed_bar_height = 10
        current_speed_percent = min(1.0, self.current_scroll_speed / (self.base_scroll_speed * 1.75))
        
        speed_label = self.font_tiny.render("SPEED", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_label, (self.SCREEN_WIDTH - speed_bar_width - 10, self.SCREEN_HEIGHT - 35))
        
        # Background bar
        bg_rect = pygame.Rect(self.SCREEN_WIDTH - speed_bar_width - 10, self.SCREEN_HEIGHT - 20, speed_bar_width, speed_bar_height)
        pygame.draw.rect(self.screen, (50,50,80), bg_rect, border_radius=3)
        # Fill bar
        fill_width = int(speed_bar_width * current_speed_percent)
        fill_rect = pygame.Rect(self.SCREEN_WIDTH - speed_bar_width - 10, self.SCREEN_HEIGHT - 20, fill_width, speed_bar_height)
        fill_color = self.COLOR_BOOST if self.boost_active else self.COLOR_UI_TEXT
        pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=3)

    def _generate_track(self):
        self.track_segments = []
        current_x = self.SCREEN_WIDTH / 2
        track_width = self.SCREEN_WIDTH * self.TRACK_WIDTH_PERCENT
        min_x = track_width / 2 + 20
        max_x = self.SCREEN_WIDTH - track_width / 2 - 20

        for _ in range(self.LAP_LENGTH_SEGMENTS):
            turn = self.np_random.uniform(-self.TRACK_TURN_RATE, self.TRACK_TURN_RATE)
            current_x += turn
            current_x = np.clip(current_x, min_x, max_x)
            self.track_segments.append(current_x)

    def _spawn_obstacle(self):
        spawn_segment_idx = int((self.track_progress / self.TRACK_SEGMENT_LENGTH)) % self.LAP_LENGTH_SEGMENTS
        track_center_x = self.track_segments[spawn_segment_idx]
        track_width = self.SCREEN_WIDTH * self.TRACK_WIDTH_PERCENT
        
        min_spawn_x = track_center_x - track_width / 2 + self.OBSTACLE_SIZE
        max_spawn_x = track_center_x + track_width / 2 - self.OBSTACLE_SIZE
        
        if min_spawn_x >= max_spawn_x: return # Avoid spawning in very narrow sections

        spawn_x = self.np_random.uniform(min_spawn_x, max_spawn_x)
        self.obstacles.append({'x': spawn_x, 'y': -self.OBSTACLE_SIZE})
        
    def _create_particle_effect(self, count, pos, color, lifespan, size, velocity_angle_range, velocity_mag_range):
        for _ in range(count):
            angle = self.np_random.uniform(*velocity_angle_range)
            magnitude = self.np_random.uniform(*velocity_mag_range)
            life = self.np_random.integers(*lifespan)
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * magnitude, 'vy': math.sin(angle) * magnitude,
                'lifespan': life, 'max_lifespan': life,
                'size': self.np_random.uniform(*size),
                'color': color
            })
            
    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        
    def _generate_parallax_stars(self):
        self.stars = []
        for i in range(100):
            depth = self.np_random.uniform(0.1, 1.0)
            self.stars.append({
                'x': self.np_random.uniform(0, self.SCREEN_WIDTH),
                'y': self.np_random.uniform(0, self.SCREEN_HEIGHT),
                'depth': depth, # 0.1 is far, 1.0 is near
                'color': tuple(int(c * (0.4 + 0.6 * depth)) for c in (150, 150, 200))
            })

    def _render_parallax_stars(self):
        for star in self.stars:
            star['y'] = (star['y'] + self.current_scroll_speed * star['depth']) % self.SCREEN_HEIGHT
            size = int(1 + star['depth'] * 2)
            pygame.draw.rect(self.screen, star['color'], (int(star['x']), int(star['y']), size, size))
            
    def _render_glowing_rect(self, x, y, w, h, color, glow_color):
        rect = pygame.Rect(x - w / 2, y - h / 2, w, h)
        
        # Glow
        glow_surface = pygame.Surface((w * 2, h * 2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, glow_color, (w/2, h/2, w, h), border_radius=int(w/4))
        try: # gaussian_blur might not be available on all pygame versions
            glow_surface = pygame.transform.gaussian_blur(glow_surface, 5)
        except AttributeError:
            pass # just render without blur if not available
        self.screen.blit(glow_surface, (rect.x - w/2, rect.y - h/2))
        
        # Main rect
        pygame.gfxdraw.box(self.screen, rect, color)
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Un-comment the line below to run with a visible display
    # os.environ["SDL_VIDEODRIVER"] = "x11"

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Setup ---
    # Re-initialize pygame with a display
    pygame.display.init()
    pygame.display.set_caption("Arcade Racer")
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        # --- Event Handling (for manual play) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Action Mapping (for manual play) ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        # if keys[pygame.K_UP]: movement = 1 # up/down not used in this game
        # if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Not used
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # Need to transpose it back for pygame's display format
        frame_to_render = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_to_render)
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        env.clock.tick(env.FPS)
        
        if terminated and (info['laps'] >= env.LAPS_TO_WIN or info['time_left'] <= 0):
            print(f"Game Over! Final Score: {info['score']}, Laps: {info['laps']}")
            # Wait a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            # terminated = False # Uncomment to auto-restart

    env.close()