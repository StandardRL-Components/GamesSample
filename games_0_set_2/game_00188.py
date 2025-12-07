import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to decelerate, ←→ to steer. "
        "Hold space for a speed boost. Hold shift to brake hard."
    )

    game_description = (
        "A fast-paced, neon-infused arcade racer. "
        "Navigate a twisting, procedurally generated track against the clock. "
        "Stay on the line to score points, but three mistakes and you're out!"
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500  # 50 seconds at 30 FPS
    TIME_LIMIT_SECONDS = 30.0
    FINISH_LINE_DISTANCE = 12000 # Pixels to travel to win

    # Colors
    COLOR_BG = (10, 0, 25)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_PARTICLE = (255, 50, 50)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_FINISH_LIGHT = (255, 255, 255)
    COLOR_FINISH_DARK = (100, 100, 100)

    # Player Physics
    PLAYER_TURN_RATE = 0.08
    ACCELERATION = 0.1
    DECELERATION = 0.05
    FRICTION = 0.015
    MIN_SPEED = 1.0
    MAX_SPEED = 6.0
    BOOST_POWER = 0.4
    BRAKE_POWER = 0.3
    BOOST_COOLDOWN_FRAMES = 15
    BRAKE_COOLDOWN_FRAMES = 15

    # Track
    TRACK_WIDTH = 50
    TRACK_SEGMENT_LENGTH = 20
    INITIAL_CURVATURE = 0.01
    CURVATURE_INCREASE_RATE = 0.005 / FPS # Per step

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.np_random = None
        self.track_points = deque()
        self.particles = deque()

        # This is called to set initial state, but without a seed
        # A seed will be passed in the main reset call by the test harness
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos_on_screen = np.array([self.WIDTH / 2, self.HEIGHT / 2 + 50], dtype=float)
        self.player_angle = -math.pi / 2 # Pointing up
        self.player_speed = self.MIN_SPEED
        
        self.misses = 0
        self.time_remaining = self.TIME_LIMIT_SECONDS
        
        self.track_points.clear()
        self.track_angle = -math.pi / 2
        self.track_curvature = self.INITIAL_CURVATURE
        self.track_color_hue = self.np_random.uniform(0, 360)
        
        # Generate initial track
        current_point = np.array([self.WIDTH / 2, self.HEIGHT + self.TRACK_SEGMENT_LENGTH], dtype=float)
        for _ in range(int(self.HEIGHT / self.TRACK_SEGMENT_LENGTH * 2)):
            self.track_points.append(current_point.copy())
            self.track_angle += self.np_random.uniform(-self.track_curvature, self.track_curvature)
            current_point[0] += math.cos(self.track_angle) * self.TRACK_SEGMENT_LENGTH
            current_point[1] += math.sin(self.track_angle) * self.TRACK_SEGMENT_LENGTH
        
        self.world_offset = np.array([0.0, 0.0])
        self.distance_travelled = 0.0

        self.particles.clear()
        self.boost_cooldown = 0
        self.brake_cooldown = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = self._update_game_state(movement, space_held, shift_held)
        
        self.steps += 1
        self.score += reward
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, movement, space_held, shift_held):
        # --- Update timers and cooldowns ---
        self.time_remaining = max(0, self.time_remaining - 1.0 / self.FPS)
        self.track_curvature += self.CURVATURE_INCREASE_RATE
        self.boost_cooldown = max(0, self.boost_cooldown - 1)
        self.brake_cooldown = max(0, self.brake_cooldown - 1)

        # --- Handle player input ---
        if movement == 1: # Up
            self.player_speed += self.ACCELERATION
        elif movement == 2: # Down
            self.player_speed -= self.DECELERATION
        if movement == 3: # Left
            self.player_angle -= self.PLAYER_TURN_RATE
        if movement == 4: # Right
            self.player_angle += self.PLAYER_TURN_RATE
        
        if space_held and self.boost_cooldown == 0:
            self.player_speed += self.BOOST_POWER
            self.boost_cooldown = self.BOOST_COOLDOWN_FRAMES
            # Sfx: boost sound
        if shift_held and self.brake_cooldown == 0:
            self.player_speed -= self.BRAKE_POWER
            self.brake_cooldown = self.BRAKE_COOLDOWN_FRAMES
            # Sfx: brake screech
            
        # --- Apply physics ---
        self.player_speed -= self.FRICTION
        self.player_speed = np.clip(self.player_speed, self.MIN_SPEED, self.MAX_SPEED)
        self.player_angle %= (2 * math.pi)

        # --- Move the world ---
        move_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * self.player_speed
        self.world_offset += move_vec
        self.distance_travelled += self.player_speed

        # --- Update track ---
        self._generate_track_segments()
        
        # --- Check if on track and calculate reward ---
        on_track, _ = self._is_on_track()
        reward = 0
        if on_track:
            reward += 0.1 # Reward for staying on track
        else:
            reward -= 5.0 # Penalty for leaving track
            self.misses += 1
            self._create_miss_particles()
            self._reset_player_on_track()
            # Sfx: crash sound
        
        # --- Win condition reward ---
        if self.distance_travelled >= self.FINISH_LINE_DISTANCE:
            reward += 100.0

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Render Track ---
        self.track_color_hue = (self.track_color_hue + 0.5) % 360
        track_color = pygame.Color(0)
        track_color.hsla = (self.track_color_hue, 100, 50, 100)
        track_glow_color = pygame.Color(0)
        track_glow_color.hsla = (self.track_color_hue, 100, 70, 30)

        # Transform track points to screen space
        screen_track_points = [
            (p - self.world_offset + self.player_pos_on_screen).astype(int)
            for p in self.track_points
        ]

        if len(screen_track_points) > 1:
            pygame.draw.lines(self.screen, track_glow_color, False, screen_track_points, width=int(self.TRACK_WIDTH * 1.5))
            pygame.draw.lines(self.screen, track_color, False, screen_track_points, width=int(self.TRACK_WIDTH))
        
        # --- Render Finish Line ---
        finish_y_screen = self.FINISH_LINE_DISTANCE - self.world_offset[1] + self.player_pos_on_screen[1]
        if 0 < finish_y_screen < self.HEIGHT:
            for i in range(0, self.WIDTH, 20):
                color = self.COLOR_FINISH_LIGHT if (i // 20) % 2 == 0 else self.COLOR_FINISH_DARK
                pygame.draw.rect(self.screen, color, (i, int(finish_y_screen), 20, 10))

        # --- Render Particles ---
        self._update_and_draw_particles()

        # --- Render Player ---
        self._draw_player()
        
    def _draw_player(self):
        # Player is a triangle
        p1 = (
            self.player_pos_on_screen[0] + math.cos(self.player_angle) * 15,
            self.player_pos_on_screen[1] + math.sin(self.player_angle) * 15
        )
        p2 = (
            self.player_pos_on_screen[0] + math.cos(self.player_angle + 2.5) * 10,
            self.player_pos_on_screen[1] + math.sin(self.player_angle + 2.5) * 10
        )
        p3 = (
            self.player_pos_on_screen[0] + math.cos(self.player_angle - 2.5) * 10,
            self.player_pos_on_screen[1] + math.sin(self.player_angle - 2.5) * 10
        )
        
        # Glow effect
        for i in range(10, 0, -2):
            alpha = 100 - i * 10
            color = (*self.COLOR_PLAYER_GLOW, alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(self.player_pos_on_screen[0]), int(self.player_pos_on_screen[1]), 10 + i, color
            )
        
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        
        # Boost/Brake indicators
        if self.boost_cooldown > self.BOOST_COOLDOWN_FRAMES - 5:
            # Sfx: boost particles
            self._create_boost_particles()
        if self.brake_cooldown > self.BRAKE_COOLDOWN_FRAMES - 5:
            self._create_brake_particles()

    def _render_ui(self):
        # Time
        time_text = f"TIME: {self.time_remaining:.1f}"
        time_surf = self.font_medium.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Misses
        miss_text = f"MISSES: {self.misses} / 3"
        miss_surf = self.font_medium.render(miss_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(miss_surf, (self.WIDTH - miss_surf.get_width() - 10, 10))

        # Speed
        speed_kmh = int(self.player_speed * 30)
        speed_text = f"{speed_kmh} KPH"
        speed_surf = self.font_large.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_surf, (self.WIDTH / 2 - speed_surf.get_width() / 2, self.HEIGHT - 50))
        
        # Progress
        progress = min(1.0, self.distance_travelled / self.FINISH_LINE_DISTANCE)
        bar_width = self.WIDTH - 20
        pygame.draw.rect(self.screen, (50, 50, 80), (10, self.HEIGHT - 15, bar_width, 5))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, self.HEIGHT - 15, bar_width * progress, 5))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "time_remaining": self.time_remaining,
            "distance_to_goal": max(0, self.FINISH_LINE_DISTANCE - self.distance_travelled)
        }

    def _check_termination(self):
        # The error "The truth value of an array with more than one element is ambiguous"
        # occurs when a numpy array is used in a boolean context (e.g., `if some_array:`).
        # This can happen if one of the state variables (like self.misses) accidentally
        # becomes a single-element numpy array. Using `bool()` on each condition
        # ensures that the `if` statement receives a standard Python boolean,
        # preventing the error.
        if bool(self.misses >= 3):
            return True
        if bool(self.time_remaining <= 0):
            return True
        if bool(self.steps >= self.MAX_STEPS):
            return True
        if bool(self.distance_travelled >= self.FINISH_LINE_DISTANCE):
            return True
        return False

    def _generate_track_segments(self):
        # Add new segments if the last one is getting close to the top of the screen
        last_point_screen = self.track_points[-1] - self.world_offset + self.player_pos_on_screen
        while last_point_screen[1] > -self.TRACK_SEGMENT_LENGTH:
            new_point = self.track_points[-1].copy()
            self.track_angle += self.np_random.uniform(-self.track_curvature, self.track_curvature)
            # Clamp track angle to prevent it from doubling back on itself too sharply
            self.track_angle = np.clip(self.track_angle, -math.pi * 0.8, -math.pi * 0.2)
            
            new_point += np.array([math.cos(self.track_angle), math.sin(self.track_angle)]) * self.TRACK_SEGMENT_LENGTH
            self.track_points.append(new_point)
            last_point_screen = new_point - self.world_offset + self.player_pos_on_screen
        
        # Remove old segments
        while len(self.track_points) > 2:
            p1_screen = self.track_points[0] - self.world_offset + self.player_pos_on_screen
            p2_screen = self.track_points[1] - self.world_offset + self.player_pos_on_screen
            if p1_screen[1] > self.HEIGHT and p2_screen[1] > self.HEIGHT:
                self.track_points.popleft()
            else:
                break

    def _is_on_track(self):
        min_dist_sq = float('inf')
        closest_point_on_track = None

        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            
            # Convert to world space relative to player
            p1_rel = p1 - self.world_offset
            p2_rel = p2 - self.world_offset
            
            # Simplified check: only consider segments near the player
            if not (min(p1_rel[1], p2_rel[1]) < 0 and max(p1_rel[1], p2_rel[1]) > -self.HEIGHT/2):
                continue
                
            line_vec = p2_rel - p1_rel
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue
            
            t = np.dot(-p1_rel, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            
            closest_point = p1_rel + t * line_vec
            dist_sq = np.dot(closest_point, closest_point)
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_point_on_track = closest_point + self.world_offset
        
        if min_dist_sq == float('inf'):
            return False, None
            
        return math.sqrt(min_dist_sq) < self.TRACK_WIDTH / 2, closest_point_on_track

    def _reset_player_on_track(self):
        _, closest_point = self._is_on_track()
        if closest_point is not None:
            # Adjust world offset to snap the player back to the track
            # The goal is to move the world so that 'closest_point' is now at the player's world position
            player_world_pos = self.world_offset
            correction = player_world_pos - closest_point
            self.world_offset -= correction
            self.player_speed *= 0.5 # Lose some speed

    def _create_miss_particles(self):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': self.player_pos_on_screen.copy(),
                'vel': vel, 'lifespan': lifespan, 'max_life': lifespan,
                'color': (255, self.np_random.integers(50, 150), 0)
            })
            
    def _create_boost_particles(self):
        for _ in range(2):
            angle = self.player_angle + math.pi + self.np_random.uniform(-0.3, 0.3)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(10, 20)
            pos = self.player_pos_on_screen - np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * 10
            self.particles.append({
                'pos': pos, 'vel': vel, 'lifespan': lifespan, 'max_life': lifespan,
                'color': (100, 200, 255)
            })

    def _create_brake_particles(self):
        for side in [-1, 1]:
            angle = self.player_angle + math.pi/2 * side + self.np_random.uniform(-0.2, 0.2)
            speed = self.np_random.uniform(0.5, 1.5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(8, 15)
            pos = self.player_pos_on_screen - np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * 5
            self.particles.append({
                'pos': pos, 'vel': vel, 'lifespan': lifespan, 'max_life': lifespan,
                'color': (200, 200, 200)
            })

    def _update_and_draw_particles(self):
        particles_to_keep = deque()
        while self.particles:
            p = self.particles.popleft()
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Air resistance
            p['lifespan'] -= 1
            
            if p['lifespan'] > 0:
                particles_to_keep.append(p)
                alpha = 255 * (p['lifespan'] / p['max_life'])
                size = 5 * (p['lifespan'] / p['max_life'])
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), int(size), color
                )
        self.particles = particles_to_keep


    def close(self):
        pygame.quit()

    def _validate_implementation(self):
        # This is an internal check and not part of the standard API
        # It's useful for development but can be removed.
        try:
            assert self.action_space.shape == (3,)
            assert self.action_space.nvec.tolist() == [5, 2, 2]
            
            test_obs = self._get_observation()
            assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
            assert test_obs.dtype == np.uint8
            
            obs, info = self.reset(seed=123)
            assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
            assert isinstance(info, dict)
            
            test_action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(test_action)
            assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert isinstance(trunc, bool)
            assert isinstance(info, dict)
        except Exception as e:
            # Re-raise with a more informative message for debugging
            raise RuntimeError(f"Implementation validation failed: {e}") from e


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    # The validation call is useful for dev, but we can bypass it for playing
    # env._validate_implementation()
    obs, info = env.reset(seed=42)
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Line Racer")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    total_reward = 0
    clock = pygame.time.Clock()
    
    # Game loop
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
    env.close()