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
    """
    An isometric arcade racing game where the player must complete a lap
    as fast as possible, using drifts to gain speed and score points,
    while avoiding crashes.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 1200, 2000
        self.FPS = 30
        
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
        
        # Fonts and Colors
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 48)
        self.font_l = pygame.font.Font(None, 72)
        
        self.COLOR_BG = (15, 19, 41)
        self.COLOR_TRACK = (60, 60, 80)
        self.COLOR_WALL = (255, 50, 50)
        self.COLOR_FINISH = (50, 255, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (100, 200, 255)
        self.COLOR_DRIFT = (255, 165, 0)
        self.COLOR_SPARK = (255, 255, 100)
        self.COLOR_LASER = (255, 0, 255)
        self.COLOR_UI = (255, 255, 255)

        # Track dimensions
        self.TRACK_WIDTH = 400.0
        self.TRACK_HEIGHT = 700.0
        self.TRACK_CENTER_X = self.WORLD_WIDTH / 2
        self.TRACK_CENTER_Y = 900.0  # Adjusted to place track around the start/finish line

        # Game state variables are initialized in reset()
        self.car_pos = None
        self.car_vel = None
        self.car_angle = None
        self.car_speed = None
        self.is_drifting = None
        self.drift_timer = None
        self.last_space_held = None
        
        self.lives = None
        self.score = None
        self.time_elapsed = None
        self.lap_completed = None
        self.crossed_midpoint = None
        
        self.particles = []
        self.lasers = []
        
        self.game_over = False
        self.win_message = ""
        self.steps = 0
        
        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Car initial state
        self.car_pos = np.array([self.WORLD_WIDTH / 2, 250.0])
        self.car_vel = np.array([0.0, 0.0])
        self.car_angle = -math.pi / 2  # Pointing "up" in isometric view
        self.car_speed = 0.0
        self.is_drifting = False
        self.drift_timer = 0
        self.last_space_held = False

        # Game progress state
        self.lives = 3
        self.score = 0
        self.time_elapsed = 0.0
        self.steps = 0
        self.lap_completed = False
        self.crossed_midpoint = False # To ensure lap is completed in correct direction
        
        # Effects lists
        self.particles = []
        self.lasers = []
        
        # Game flow state
        self.game_over = False
        self.win_message = ""
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.time_elapsed += 1 / self.FPS
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        prev_pos = self.car_pos.copy()
        
        self._update_car_physics(movement, shift_held)
        self._update_effects(space_held)
        
        collided = self._handle_collisions()
        lap_this_frame = self._check_lap_completion(prev_pos)
        
        reward = self._calculate_reward(collided, lap_this_frame)
        self.score += reward
        
        terminated = self._check_termination(collided, lap_this_frame)
        truncated = False # No truncation condition in this game
        if terminated:
            self.game_over = True
            if self.lap_completed:
                self.win_message = "YOU WIN!"
            elif self.lives <= 0:
                self.win_message = "CRASHED!"
            else: # Time out
                self.win_message = "TIME UP!"

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_car_physics(self, movement, shift_held):
        # Constants for physics
        ACCEL = 1.5
        BRAKE = 2.0
        TURN_SPEED = 0.05
        DRIFT_TURN_MULT = 1.8
        MAX_SPEED = 20.0
        FRICTION = 0.96
        DRIFT_FRICTION = 0.98
        DRIFT_SLIDE = 0.90

        # Determine if drifting
        self.is_drifting = shift_held and self.car_speed > 5.0
        if self.is_drifting:
            self.drift_timer += 1
        else:
            self.drift_timer = 0
            
        # Turning
        turn_input = 0
        if movement == 3: turn_input = -1 # Left
        if movement == 4: turn_input = 1  # Right
        
        turn_rate = TURN_SPEED
        if self.is_drifting:
            turn_rate *= DRIFT_TURN_MULT
        
        # Reduce turn ability at high speed unless drifting
        if not self.is_drifting:
            turn_rate *= max(0.2, 1 - self.car_speed / MAX_SPEED)
            
        self.car_angle += turn_input * turn_rate

        # Acceleration and Braking
        forward_vec = np.array([math.cos(self.car_angle), math.sin(self.car_angle)])
        if movement == 1: # Accelerate
            self.car_vel += forward_vec * ACCEL
        elif movement == 2: # Brake
            self.car_vel *= 0.90 # Stronger friction for braking
            
        # Speed Limiter
        self.car_speed = np.linalg.norm(self.car_vel)
        if self.car_speed > MAX_SPEED:
            self.car_vel = (self.car_vel / self.car_speed) * MAX_SPEED

        # Friction
        if self.is_drifting:
            # Decompose velocity into forward and sideways components
            forward_vel_mag = np.dot(self.car_vel, forward_vec)
            forward_vel = forward_vec * forward_vel_mag
            sideways_vel = self.car_vel - forward_vel
            
            # Apply different friction to each component
            forward_vel *= DRIFT_FRICTION
            sideways_vel *= DRIFT_SLIDE
            self.car_vel = forward_vel + sideways_vel
        else:
            self.car_vel *= FRICTION
        
        # Update position
        self.car_pos += self.car_vel

    def _update_effects(self, space_held):
        # Spawn new particles/lasers
        if self.is_drifting:
            # sound: drift_screech.wav
            for _ in range(2):
                self.particles.append(self._create_particle(self.car_pos, self.COLOR_DRIFT, 1.0, 20))
        
        if space_held and not self.last_space_held:
            # sound: laser_shoot.wav
            self._fire_laser()
        self.last_space_held = space_held
        
        # Update existing particles and lasers
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] *= 0.95
            
        self.lasers = [l for l in self.lasers if l['life'] > 0]
        for l in self.lasers:
            l['pos'] += l['vel']
            l['life'] -= 1

    def _fire_laser(self):
        forward_vec = np.array([math.cos(self.car_angle), math.sin(self.car_angle)])
        laser_vel = forward_vec * 30.0 + self.car_vel
        self.lasers.append({
            'pos': self.car_pos.copy(),
            'vel': laser_vel,
            'life': 30,
        })
        
    def _create_particle(self, pos, color, speed_mult, life):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3) * speed_mult
        return {
            'pos': pos.copy() + np.array([random.uniform(-5, 5), random.uniform(-5, 5)]),
            'vel': np.array([math.cos(angle), math.sin(angle)]) * speed - self.car_vel * 0.2,
            'life': random.randint(life // 2, life),
            'size': random.uniform(2, 5),
            'color': color
        }

    def _handle_collisions(self):
        # Track boundaries (oval shape)
        dx = self.car_pos[0] - self.TRACK_CENTER_X
        dy = self.car_pos[1] - self.TRACK_CENTER_Y
        
        # Simple ellipse collision detection
        dist_norm = (dx / self.TRACK_WIDTH)**2 + (dy / self.TRACK_HEIGHT)**2
        
        if dist_norm > 1.0:
            # sound: crash.wav
            self.lives -= 1
            # Push car back towards center
            self.car_pos -= self.car_vel
            self.car_vel *= -0.7 # Bounce effect
            
            # Create spark particles
            for _ in range(20):
                self.particles.append(self._create_particle(self.car_pos, self.COLOR_SPARK, 3.0, 30))
            return True
        return False

    def _check_lap_completion(self, prev_pos):
        FINISH_LINE_Y = 200
        was_before = prev_pos[1] < FINISH_LINE_Y
        is_after = self.car_pos[1] >= FINISH_LINE_Y
        
        midpoint_y = self.TRACK_CENTER_Y
        if self.car_pos[1] > midpoint_y:
            self.crossed_midpoint = True
            
        if was_before and is_after and self.crossed_midpoint:
            # sound: lap_complete.wav
            self.lap_completed = True
            self.crossed_midpoint = False
            return True
        return False

    def _calculate_reward(self, collided, lap_this_frame):
        reward = 0
        
        # Penalty for collision is primary
        if collided:
            return -5.0
        
        # Reward for moving forward
        forward_vec = np.array([math.cos(self.car_angle), math.sin(self.car_angle)])
        forward_speed = np.dot(self.car_vel, forward_vec)
        if forward_speed > 1.0:
            reward += 0.1
        
        # Reward for drifting
        if self.is_drifting:
            reward += 0.5 + self.drift_timer * 0.01 # Reward longer drifts
            
        # Terminal rewards
        if lap_this_frame:
            time_bonus = max(0, 60 - self.time_elapsed) * (100 / 60.0)
            reward += time_bonus
        
        if self.lives <= 0:
            reward = -100.0
            
        return reward
        
    def _check_termination(self, collided, lap_this_frame):
        return lap_this_frame or self.lives <= 0 or self.time_elapsed >= 60.0 or self.steps >= 1800

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "time": self.time_elapsed,
            "lap_completed": self.lap_completed
        }

    def _world_to_screen(self, x, y, cam_x, cam_y):
        iso_x = (x - y) * 0.707
        iso_y = (x + y) * 0.4
        return int(iso_x - cam_x + self.WIDTH / 2), int(iso_y - cam_y + self.HEIGHT / 2)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Camera follows car
        cam_x_world, cam_y_world = self._world_to_screen(self.car_pos[0], self.car_pos[1], 0, 0)
        cam_offset_x = cam_x_world - self.WIDTH / 2
        cam_offset_y = cam_y_world - self.HEIGHT / 2 - 50 # Look slightly ahead

        # Render track
        self._render_track(cam_offset_x, cam_offset_y)

        # Render particles and lasers
        for p in self.particles:
            sx, sy = self._world_to_screen(p['pos'][0], p['pos'][1], cam_offset_x, cam_offset_y)
            if 0 < sx < self.WIDTH and 0 < sy < self.HEIGHT:
                try:
                    pygame.gfxdraw.filled_circle(self.screen, sx, sy, int(p['size']), p['color'] + (int(255 * (p['life'] / 30.0)),))
                except (ValueError, TypeError): # Handle potential color alpha issues
                    pass
        
        for l in self.lasers:
            start_x, start_y = self._world_to_screen(l['pos'][0], l['pos'][1], cam_offset_x, cam_offset_y)
            end_pos = l['pos'] - l['vel'] * 0.5
            end_x, end_y = self._world_to_screen(end_pos[0], end_pos[1], cam_offset_x, cam_offset_y)
            pygame.draw.line(self.screen, self.COLOR_LASER, (start_x, start_y), (end_x, end_y), 3)

        # Render car
        self._render_car(cam_offset_x, cam_offset_y)
        
        # Render speed lines
        if self.car_speed > 15.0:
            self._render_speed_lines()

    def _render_track(self, cam_x, cam_y):
        # Draw track base
        points_outer = []
        points_inner = []
        for i in range(101):
            angle = i / 100 * 2 * math.pi
            points_outer.append((self.TRACK_CENTER_X + math.cos(angle) * self.TRACK_WIDTH, self.TRACK_CENTER_Y + math.sin(angle) * self.TRACK_HEIGHT))
            points_inner.append((self.TRACK_CENTER_X + math.cos(angle) * (self.TRACK_WIDTH - 80), self.TRACK_CENTER_Y + math.sin(angle) * (self.TRACK_HEIGHT - 80)))
        
        iso_points_outer = [self._world_to_screen(p[0], p[1], cam_x, cam_y) for p in points_outer]
        iso_points_inner = [self._world_to_screen(p[0], p[1], cam_x, cam_y) for p in points_inner]

        pygame.gfxdraw.filled_polygon(self.screen, iso_points_outer, self.COLOR_TRACK)
        pygame.gfxdraw.aapolygon(self.screen, iso_points_outer, self.COLOR_WALL)
        pygame.gfxdraw.filled_polygon(self.screen, iso_points_inner, self.COLOR_BG)
        pygame.gfxdraw.aapolygon(self.screen, iso_points_inner, self.COLOR_WALL)

        # Draw finish line
        fx1, fy1 = self._world_to_screen(self.TRACK_CENTER_X - self.TRACK_WIDTH, 200, cam_x, cam_y)
        fx2, fy2 = self._world_to_screen(self.TRACK_CENTER_X + self.TRACK_WIDTH, 200, cam_x, cam_y)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (fx1, fy1), (fx2, fy2), 5)

    def _render_car(self, cam_x, cam_y):
        car_points_local = [
            np.array([-15, -8]), np.array([15, -8]),
            np.array([15, 8]), np.array([-15, 8])
        ]
        
        rot_matrix = np.array([
            [math.cos(self.car_angle), -math.sin(self.car_angle)],
            [math.sin(self.car_angle), math.cos(self.car_angle)]
        ])
        
        car_points_world = [self.car_pos + p @ rot_matrix for p in car_points_local]
        car_points_screen = [self._world_to_screen(p[0], p[1], cam_x, cam_y) for p in car_points_world]
        
        # Glow effect
        car_center_screen = self._world_to_screen(self.car_pos[0], self.car_pos[1], cam_x, cam_y)
        for i in range(15, 0, -2):
            alpha = 50 * (1 - i / 15.0)
            try:
                pygame.gfxdraw.filled_circle(self.screen, car_center_screen[0], car_center_screen[1], i, self.COLOR_PLAYER_GLOW + (int(alpha),))
            except (ValueError, TypeError): # Handle potential color alpha issues
                pass

        # Car body
        pygame.gfxdraw.filled_polygon(self.screen, car_points_screen, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, car_points_screen, self.COLOR_PLAYER_GLOW)
        
    def _render_speed_lines(self):
        num_lines = int((self.car_speed - 15.0) / (20.0 - 15.0) * 20)
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        for _ in range(num_lines):
            angle = random.uniform(0, 2 * math.pi)
            length = random.uniform(50, self.WIDTH)
            start_x = center_x + math.cos(angle) * (length - 20)
            start_y = center_y + math.sin(angle) * (length - 20)
            end_x = center_x + math.cos(angle) * length
            end_y = center_y + math.sin(angle) * length
            pygame.draw.line(self.screen, (200, 220, 255, 100), (start_x, start_y), (end_x, end_y), 1)

    def _render_ui(self):
        # Time
        time_text = self.font_m.render(f"TIME: {max(0, 60 - self.time_elapsed):05.2f}", True, self.COLOR_UI)
        self.screen.blit(time_text, (10, 10))
        
        # Lives
        lives_text = self.font_m.render(f"LIVES: {self.lives}", True, self.COLOR_UI)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_m.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT - 40))
        
        # Drift indicator
        if self.is_drifting:
            drift_text = self.font_s.render(f"DRIFTING!", True, self.COLOR_DRIFT)
            self.screen.blit(drift_text, (self.WIDTH // 2 - drift_text.get_width() // 2, self.HEIGHT - 70))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_l.render(self.win_message, True, self.COLOR_UI)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - 50))
            
            final_score_text = self.font_m.render(f"Final Score: {int(self.score)}", True, self.COLOR_UI)
            self.screen.blit(final_score_text, (self.WIDTH // 2 - final_score_text.get_width() // 2, self.HEIGHT // 2 + 20))

    def close(self):
        pygame.quit()
        

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    running = True
    terminated = False
    
    # Use a dictionary to track held keys for smooth input
    keys_held = {
        pygame.K_UP: False, pygame.K_DOWN: False,
        pygame.K_LEFT: False, pygame.K_RIGHT: False,
        pygame.K_SPACE: False, pygame.K_LSHIFT: False, pygame.K_RSHIFT: False
    }

    # Create a window to display the game
    pygame.display.set_caption("GameEnv Test")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset on 'r' key
                    terminated = False
                    env.reset()
                if event.key in keys_held:
                    keys_held[event.key] = True
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False
        
        if terminated:
            # On game over, wait for reset by pressing 'r'
            obs = env._get_observation() # Keep rendering the final screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            continue

        # Map keyboard state to action space
        movement = 0 # no-op
        if keys_held[pygame.K_UP]: movement = 1
        elif keys_held[pygame.K_DOWN]: movement = 2
        elif keys_held[pygame.K_LEFT]: movement = 3
        elif keys_held[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys_held[pygame.K_SPACE] else 0
        shift = 1 if keys_held[pygame.K_LSHIFT] or keys_held[pygame.K_RSHIFT] else 0

        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()