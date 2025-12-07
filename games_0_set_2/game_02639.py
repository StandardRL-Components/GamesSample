import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    metadata = {"render_modes": ["rgb_array", "human"]}

    user_guide = (
        "Controls: Use ← and → arrow keys to steer your car. Stay on the neon green track to survive."
    )

    game_description = (
        "A fast-paced, side-view line racer. Race against the clock on a procedurally generated, winding neon "
        "track. Reach the checkpoints to increase your speed and advance to the next stage. Fall off or run out "
        "of time, and it's game over."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_STAGE_1 = (10, 0, 30)
    COLOR_BG_STAGE_2 = (30, 0, 10)
    COLOR_BG_STAGE_3 = (0, 20, 20)
    COLOR_OFF_TRACK = (60, 0, 10)
    COLOR_TRACK_GLOW = (0, 255, 100, 30)
    COLOR_TRACK_MAIN = (150, 255, 200)
    COLOR_CAR_GLOW = (0, 150, 255, 40)
    COLOR_CAR_MAIN = (150, 220, 255)
    COLOR_PARTICLE_EDGE = (255, 200, 0)
    COLOR_PARTICLE_CHECKPOINT = (200, 200, 255)
    COLOR_UI_TEXT = (220, 220, 240)

    # Game Parameters
    FPS = 30
    TOTAL_TIME = 180  # 60 seconds per stage * 3 stages
    MAX_STEPS = TOTAL_TIME * FPS
    NUM_STAGES = 3
    
    # Car Physics
    CAR_ACCEL = 0.8
    CAR_FRICTION = 0.90
    MAX_CAR_SPEED = 10
    CAR_Y_POS = SCREEN_HEIGHT * 0.75 # Fixed vertical position
    CAR_SCREEN_X_POS = SCREEN_WIDTH * 0.25 # Car stays at this horizontal screen position

    # Track Generation
    TRACK_WIDTH = 50
    TRACK_SEGMENT_LENGTH = 15
    TRACK_INITIAL_AMPLITUDE = 20
    TRACK_INITIAL_FREQ = 0.02
    TRACK_DIFFICULTY_SCALING_INTERVAL = 50
    TRACK_AMPLITUDE_INCREMENT = 0.05
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.render_mode = render_mode
        self.np_random = None
        # reset() is called by the gym wrapper, but we can call it to initialize state
        # self.reset() is not called here to avoid seeding issues; gym.make handles it.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TOTAL_TIME
        
        self.car_pos_y = self.CAR_Y_POS
        self.car_vel_x = 0.0
        
        self.camera_x = 0.0
        self.scroll_speed = 5.0
        
        self.current_stage = 1
        self.checkpoints_passed = 0
        self.stage_length = 3000 # World units for one stage
        self.checkpoints = [self.stage_length * (i + 1) for i in range(self.NUM_STAGES)]

        self.track_points = deque()
        self.track_base_y = self.CAR_Y_POS  # FIX: Generate track around car's Y position
        self.track_amplitude = self.TRACK_INITIAL_AMPLITUDE
        self.track_frequency = self.TRACK_INITIAL_FREQ
        self.track_phase_offset = self.np_random.uniform(0, 2 * math.pi)

        self.particles = []

        # Generate initial track
        for x in range(-self.SCREEN_WIDTH, self.SCREEN_WIDTH * 2, self.TRACK_SEGMENT_LENGTH):
             self._add_track_point(x)
        
        # Initial car x position is relative to camera
        self.car_world_x = self.CAR_SCREEN_X_POS

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0
        
        # 1. Update Game Logic
        self._handle_input(movement)
        self._update_car()
        self._update_camera()
        self._update_track()
        self._update_particles()

        self.steps += 1
        self.time_remaining -= 1 / self.FPS

        # 2. Check Game State & Calculate Rewards
        on_track, distance_from_center = self._check_on_track()

        if on_track:
            reward += 0.1  # Reward for staying on track
            if distance_from_center > self.TRACK_WIDTH * 0.4:
                reward -= 0.2  # Penalty for being near the edge
                if self.steps % 3 == 0: # Create sparks near edge
                    self._create_edge_particles()
        else:
            reward = -50  # Large penalty for falling off
            self.game_over = True
            # sfx: car falling sound

        self._check_checkpoints()
        
        # Add score from checkpoints to step reward
        # This is a temporary buffer for rewards from events within the step
        reward += self.score 
        self.score = 0 # Reset score buffer

        # 3. Check Termination Conditions
        terminated = self.game_over or self.time_remaining <= 0
        # The test suite might run for more steps than MAX_STEPS, so we use truncated for this
        truncated = self.steps >= self.MAX_STEPS

        if self.checkpoints_passed >= self.NUM_STAGES and not self.game_over:
            reward += 50 # Bonus for finishing
            terminated = True
        
        if terminated and not self.game_over: # Time ran out
            reward -= 25 # Penalty for not finishing in time

        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.car_vel_x -= self.CAR_ACCEL
        elif movement == 4:  # Right
            self.car_vel_x += self.CAR_ACCEL

    def _update_car(self):
        self.car_vel_x = np.clip(self.car_vel_x, -self.MAX_CAR_SPEED, self.MAX_CAR_SPEED)
        self.car_world_x += self.car_vel_x
        self.car_vel_x *= self.CAR_FRICTION
        
        # Ensure car stays within screen horizontal bounds
        min_x = self.camera_x
        max_x = self.camera_x + self.SCREEN_WIDTH
        self.car_world_x = np.clip(self.car_world_x, min_x, max_x)

    def _update_camera(self):
        self.camera_x += self.scroll_speed
        
        # Make car catch up to its designated screen position
        target_world_x = self.camera_x + self.CAR_SCREEN_X_POS
        self.car_world_x += (target_world_x - self.car_world_x) * 0.05


    def _update_track(self):
        # Add new points
        last_x = self.track_points[-1][0]
        while last_x < self.camera_x + self.SCREEN_WIDTH + 100:
            self._add_track_point(last_x + self.TRACK_SEGMENT_LENGTH)
            last_x = self.track_points[-1][0]
        
        # Remove old points
        while self.track_points[0][0] < self.camera_x - 100:
            self.track_points.popleft()

        # Increase difficulty
        if self.steps > 0 and self.steps % self.TRACK_DIFFICULTY_SCALING_INTERVAL == 0:
            self.track_amplitude += self.TRACK_AMPLITUDE_INCREMENT

    def _add_track_point(self, x):
        y_oscillation = self.track_amplitude * math.sin(x * self.track_frequency + self.track_phase_offset)
        y = self.track_base_y + y_oscillation
        self.track_points.append((x, y))

    def _get_track_y_at(self, x_pos):
        # Find two surrounding points
        p1, p2 = None, None
        for i in range(len(self.track_points) - 1):
            if self.track_points[i][0] <= x_pos < self.track_points[i+1][0]:
                p1 = self.track_points[i]
                p2 = self.track_points[i+1]
                break
        
        if p1 is None or p2 is None:
            # Fallback if x_pos is outside the current track deque range
            if len(self.track_points) > 0:
                if x_pos < self.track_points[0][0]: return self.track_points[0][1]
                else: return self.track_points[-1][1]
            return self.track_base_y

        # Linear interpolation
        x1, y1 = p1
        x2, y2 = p2
        if x2 == x1: return y1
        t = (x_pos - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)

    def _check_on_track(self):
        track_center_y = self._get_track_y_at(self.car_world_x)
        distance_from_center = abs(self.car_pos_y - track_center_y)
        on_track = distance_from_center < self.TRACK_WIDTH / 2
        return on_track, distance_from_center

    def _check_checkpoints(self):
        if self.checkpoints_passed < len(self.checkpoints):
            if self.car_world_x >= self.checkpoints[self.checkpoints_passed]:
                self.checkpoints_passed += 1
                self.current_stage = min(self.current_stage + 1, self.NUM_STAGES)
                self.score += 5 # Event-based reward
                self.scroll_speed += 1.0 # Increase speed
                self._create_checkpoint_particles()
                # sfx: checkpoint reached
    
    def _get_observation(self):
        self._render_background()
        self._render_track()
        self._render_particles()
        self._render_car()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, 3), we need (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        # The 'score' in info reflects the total score, while the step reward is instantaneous
        current_total_score = sum(5 for i in range(self.checkpoints_passed))
        return {
            "score": current_total_score,
            "steps": self.steps,
            "time_remaining": int(self.time_remaining),
            "stage": self.current_stage,
        }
        
    def _render_background(self):
        # Off-track area
        self.screen.fill(self.COLOR_OFF_TRACK)
        
        # Stage-specific gradient
        bg_color = self.COLOR_BG_STAGE_1
        if self.current_stage == 2: bg_color = self.COLOR_BG_STAGE_2
        elif self.current_stage >= 3: bg_color = self.COLOR_BG_STAGE_3
        
        top_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT * 0.6)
        bottom_rect = pygame.Rect(0, self.SCREEN_HEIGHT * 0.6, self.SCREEN_WIDTH, self.SCREEN_HEIGHT * 0.4)
        
        pygame.draw.rect(self.screen, (0,0,0), top_rect)
        pygame.draw.rect(self.screen, bg_color, bottom_rect)

    def _render_track(self):
        if len(self.track_points) < 2:
            return

        screen_points = [(p[0] - self.camera_x, p[1]) for p in self.track_points]

        # Draw glow
        pygame.draw.aalines(self.screen, self.COLOR_TRACK_GLOW, False, screen_points, int(self.TRACK_WIDTH * 1.5))
        # Draw main track
        pygame.draw.aalines(self.screen, self.COLOR_TRACK_MAIN, False, screen_points, int(self.TRACK_WIDTH))
        
        # Draw centerline for visual flair
        pygame.draw.aalines(self.screen, self.COLOR_BG_STAGE_1, False, screen_points, 1)

    def _render_car(self):
        car_screen_x = self.car_world_x - self.camera_x
        car_points = [
            (car_screen_x, self.car_pos_y - 10),
            (car_screen_x - 5, self.car_pos_y + 5),
            (car_screen_x + 5, self.car_pos_y + 5)
        ]
        
        # Glow effect
        for i in range(5, 0, -1):
            glow_points = [
                (car_screen_x, self.car_pos_y - (10 + i)),
                (car_screen_x - (5 + i), self.car_pos_y + (5 + i)),
                (car_screen_x + (5 + i), self.car_pos_y + (5 + i))
            ]
            pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_CAR_GLOW)

        # Main car body
        pygame.gfxdraw.aapolygon(self.screen, car_points, self.COLOR_CAR_MAIN)
        pygame.gfxdraw.filled_polygon(self.screen, car_points, self.COLOR_CAR_MAIN)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = list(p['color'])
            if len(color) == 4:
                color[3] = alpha
            
            pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            
            # Create a temporary surface for the particle to handle alpha
            radius = int(p['size'])
            if radius <= 0: continue
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_edge_particles(self):
        # sfx: sparks
        track_y = self._get_track_y_at(self.car_world_x)
        y_dir = 1 if self.car_pos_y > track_y else -1
        
        for _ in range(2):
            particle = {
                'pos': pygame.Vector2(self.car_world_x, self.car_pos_y),
                'vel': pygame.Vector2(self.np_random.uniform(-1, 1) - self.scroll_speed, self.np_random.uniform(-1, 1) * y_dir),
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'color': self.COLOR_PARTICLE_EDGE,
                'size': self.np_random.uniform(1, 3)
            }
            self.particles.append(particle)
            
    def _create_checkpoint_particles(self):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            particle = {
                'pos': pygame.Vector2(self.car_world_x, self.car_pos_y),
                'vel': pygame.Vector2(math.cos(angle) * speed - self.scroll_speed, math.sin(angle) * speed),
                'life': self.np_random.integers(20, 40),
                'max_life': 40,
                'color': self.COLOR_PARTICLE_CHECKPOINT,
                'size': self.np_random.uniform(2, 5)
            }
            self.particles.append(particle)

    def _render_ui(self):
        # Stage
        stage_text = f"STAGE {self.current_stage}/{self.NUM_STAGES}"
        self._draw_text(stage_text, (20, 20), self.font_small)

        # Time
        time_text = f"TIME: {max(0, int(self.time_remaining))}"
        self._draw_text(time_text, (self.SCREEN_WIDTH - 120, 20), self.font_small)

        # Speed
        speed_val = int(self.scroll_speed * 20) # Arbitrary speed unit
        speed_text = f"SPD: {speed_val}"
        self._draw_text(speed_text, (self.SCREEN_WIDTH - 120, self.SCREEN_HEIGHT - 40), self.font_small)
        
        # Game Over
        if self.game_over:
            if self.checkpoints_passed >= self.NUM_STAGES:
                msg = "FINISH!"
            else:
                msg = "GAME OVER"
            
            self._draw_text(msg, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20), self.font_large, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_UI_TEXT, center=False):
        surface = font.render(text, True, color)
        rect = surface.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.screen.blit(surface, rect)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # To run and play the game manually
    # This requires the "human" render mode and a display.
    # The environment itself is headless by default.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for human play
    
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Neon Line Racer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0

        action = [movement, 0, 0] # space and shift are unused
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(GameEnv.FPS)

    env.close()