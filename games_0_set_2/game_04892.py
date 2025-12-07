
# Generated: 2025-08-28T03:19:27.546483
# Source Brief: brief_04892.md
# Brief Index: 4892

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to steer. Press space to use a boost."
    )

    game_description = (
        "Fast-paced arcade racer. Steer through a neon track, use boosts, and race for the best time."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.NUM_LAPS = 3
        self.LAP_LENGTH = 8000  # pixels
        self.CAR_X_POS = self.WIDTH // 4

        # Colors
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_TRACK = (0, 150, 255)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 0, 40)
        self.COLOR_OBSTACLE = (255, 20, 50)
        self.COLOR_FINISH_LINE = (50, 255, 50)
        self.COLOR_BOOST = (0, 255, 255)
        self.COLOR_UI_TEXT = (255, 255, 255)

        # Physics
        self.MIN_SPEED = 3.0
        self.MAX_SPEED = 8.0
        self.ACCEL_RATE = 0.1
        self.BOOST_SPEED_BONUS = 8.0
        self.BOOST_DURATION = 60  # frames
        self.STEER_FORCE = 0.4
        self.Y_VEL_DAMPING = 0.9
        self.INITIAL_BOOSTS = 5

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.ui_font = pygame.font.SysFont("monospace", 18, bold=True)
        except pygame.error:
            self.ui_font = pygame.font.Font(None, 24)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rng = None
        self.car_pos = None
        self.car_y_velocity = None
        self.base_speed = None
        self.current_speed = None
        self.boost_count = None
        self.boost_timer = None
        self.lap = None
        self.world_progress = None
        self.particles = None
        self.track_center_points = None
        self.obstacles = None
        self.reward_this_step = 0
        self.last_space_press = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.last_space_press = False

        self.car_pos = [self.CAR_X_POS, self.HEIGHT / 2]
        self.car_y_velocity = 0.0
        self.base_speed = self.MIN_SPEED
        self.current_speed = self.base_speed
        self.boost_count = self.INITIAL_BOOSTS
        self.boost_timer = 0
        self.lap = 0
        self.world_progress = 0.0
        self.particles = []

        self._generate_track_and_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.reward_this_step = 0

        if not self.game_over:
            self._handle_input(action)
            self._update_physics()
            self._check_events()
            self.reward_this_step += 0.01  # Small reward for surviving
            self.score += self.reward_this_step

        self.steps += 1
        terminated = self._check_termination()

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_pressed, _ = action
        space_pressed = space_pressed == 1

        # Steering
        if movement == 3:  # Left -> Steer Up
            self.car_y_velocity -= self.STEER_FORCE
        if movement == 4:  # Right -> Steer Down
            self.car_y_velocity += self.STEER_FORCE

        # Acceleration / Braking
        if self.boost_timer <= 0:
            if movement == 1:  # Up -> Accelerate
                self.base_speed = min(self.MAX_SPEED, self.base_speed + self.ACCEL_RATE)
            if movement == 2:  # Down -> Brake
                self.base_speed = max(self.MIN_SPEED, self.base_speed - self.ACCEL_RATE)

        # Boost
        if space_pressed and not self.last_space_press and self.boost_count > 0 and self.boost_timer <= 0:
            # sfx: Boost activation
            self.boost_count -= 1
            self.boost_timer = self.BOOST_DURATION
            self.reward_this_step -= 0.2

        self.last_space_press = space_pressed

    def _update_physics(self):
        # Update speed
        if self.boost_timer > 0:
            self.boost_timer -= 1
            self.current_speed = self.base_speed + self.BOOST_SPEED_BONUS
            # Add boost particles
            if self.steps % 2 == 0:
                self._create_particles(2, self.COLOR_BOOST)
        else:
            self.current_speed = self.base_speed

        # Update world scroll
        self.world_progress += self.current_speed

        # Update car vertical position
        self.car_y_velocity *= self.Y_VEL_DAMPING
        self.car_pos[1] += self.car_y_velocity

        # Update particles
        self._update_particles()

    def _check_events(self):
        # Check track boundaries
        track_y, track_width = self._get_track_properties_at(self.world_progress)
        half_width = track_width / 2
        min_y, max_y = track_y - half_width, track_y + half_width
        
        car_height = 10
        if self.car_pos[1] - car_height/2 < min_y or self.car_pos[1] + car_height/2 > max_y:
            # sfx: Crash sound
            self.reward_this_step -= 100
            self.game_over = True
            return

        # Check obstacle collisions
        car_rect = pygame.Rect(self.car_pos[0] - 5, self.car_pos[1] - 5, 10, 10)
        for obs in self.obstacles:
            obs_world_x = obs['x']
            # Only check obstacles that are on screen
            if abs(obs_world_x - self.world_progress - self.car_pos[0]) < 50:
                screen_x = self.car_pos[0] + (obs_world_x - self.world_progress)
                obs_rect = pygame.Rect(screen_x - obs['w']/2, obs['y'] - obs['h']/2, obs['w'], obs['h'])
                if car_rect.colliderect(obs_rect):
                    # sfx: Crash sound
                    self.reward_this_step -= 100
                    self.game_over = True
                    return

        # Check lap completion
        current_lap = int(self.world_progress // self.LAP_LENGTH)
        if current_lap > self.lap:
            self.lap = current_lap
            if self.lap < self.NUM_LAPS:
                # sfx: Lap complete chime
                self.reward_this_step += 10
            # Increase difficulty
            self.MIN_SPEED += 0.2
            self.MAX_SPEED += 0.2


    def _check_termination(self):
        if self.game_over:
            return True
        if self.lap >= self.NUM_LAPS:
            # sfx: Race finished fanfare
            time_bonus = max(0, (self.MAX_STEPS - self.steps) / 100.0)
            self.reward_this_step += 50 + time_bonus
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render track
        self._render_track()

        # Render obstacles
        for obs in self.obstacles:
            obs_world_x = obs['x']
            screen_x = self.car_pos[0] + (obs_world_x - self.world_progress)
            if 0 <= screen_x < self.WIDTH:
                rect = (int(screen_x - obs['w']/2), int(obs['y'] - obs['h']/2), obs['w'], obs['h'])
                pygame.gfxdraw.box(self.screen, rect, self.COLOR_OBSTACLE)

        # Render particles
        self._render_particles()

        # Render car
        self._render_car()

    def _render_track(self):
        # Draw track segments
        for x in range(0, self.WIDTH, 10):
            world_x1 = self.world_progress + (x - self.car_pos[0])
            world_x2 = self.world_progress + (x + 10 - self.car_pos[0])
            
            y1, w1 = self._get_track_properties_at(world_x1)
            y2, w2 = self._get_track_properties_at(world_x2)

            # Top boundary
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, (x, y1 - w1/2), (x + 10, y2 - w2/2))
            # Bottom boundary
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, (x, y1 + w1/2), (x + 10, y2 + w2/2))

        # Render finish line
        for i in range(1, self.NUM_LAPS + 1):
            finish_world_x = self.LAP_LENGTH * i
            screen_x = self.car_pos[0] + (finish_world_x - self.world_progress)
            if 0 <= screen_x < self.WIDTH:
                track_y, track_width = self._get_track_properties_at(finish_world_x)
                pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, 
                                 (screen_x, track_y - track_width/2), 
                                 (screen_x, track_y + track_width/2), 5)

    def _render_car(self):
        x, y = int(self.car_pos[0]), int(self.car_pos[1])
        angle = self.car_y_velocity * 2 # Tilt based on vertical speed
        
        # Car shape points (a triangle)
        p1 = (x + 12, y)
        p2 = (x - 8, y - 7)
        p3 = (x - 8, y + 7)

        # Rotate points for tilt effect
        def rotate(p, angle_deg, anchor):
            angle_rad = math.radians(angle_deg)
            px, py = p
            ox, oy = anchor
            qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
            qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
            return int(qx), int(qy)

        rp1, rp2, rp3 = rotate(p1, angle, (x,y)), rotate(p2, angle, (x,y)), rotate(p3, angle, (x,y))

        # Draw glow
        pygame.gfxdraw.filled_trigon(self.screen, rp1[0], rp1[1], rp2[0], rp2[1], rp3[0], rp3[1], self.COLOR_PLAYER_GLOW)
        # Draw car
        pygame.gfxdraw.filled_trigon(self.screen, rp1[0], rp1[1], rp2[0], rp2[1], rp3[0], rp3[1], self.COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(self.screen, rp1[0], rp1[1], rp2[0], rp2[1], rp3[0], rp3[1], self.COLOR_PLAYER)

    def _render_ui(self):
        lap_text = f"LAP: {min(self.lap + 1, self.NUM_LAPS)}/{self.NUM_LAPS}"
        time_text = f"TIME: {self.steps / self.FPS:.2f}s"
        boost_text = f"BOOST: {self.boost_count}"
        speed_text = f"SPEED: {int(self.current_speed * 10)} KPH"

        texts = [lap_text, time_text, boost_text, speed_text]
        for i, text in enumerate(texts):
            surf = self.ui_font.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(surf, (10, 10 + i * 22))
        
        if self.game_over:
            msg = "RACE FINISHED!" if self.lap >= self.NUM_LAPS else "CRASHED!"
            msg_surf = self.ui_font.render(msg, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.lap,
            "boosts_left": self.boost_count,
        }

    def _generate_track_and_obstacles(self):
        # Generate track center line using sine waves
        self.track_center_points = []
        total_length = self.LAP_LENGTH * self.NUM_LAPS + self.WIDTH
        
        # Use seeded RNG for deterministic track generation
        freq1, freq2 = self.rng.uniform(0.001, 0.002, 2)
        amp1, amp2 = self.rng.uniform(40, 80, 2)
        phase1, phase2 = self.rng.uniform(0, 2 * math.pi, 2)
        
        for x in range(int(total_length)):
            y_offset = amp1 * math.sin(freq1 * x + phase1) + amp2 * math.sin(freq2 * x + phase2)
            center_y = self.HEIGHT / 2 + y_offset
            width = 120 - 30 * math.cos(0.0005 * x) # Varying width
            self.track_center_points.append((center_y, width))

        # Generate obstacles
        self.obstacles = []
        for x in range(500, int(total_length - 500), 200):
            if self.rng.random() < 0.7:
                track_y, track_width = self._get_track_properties_at(x)
                obs_y_offset = self.rng.uniform(-track_width/2.5, track_width/2.5)
                obs_y = track_y + obs_y_offset
                obs_w = self.rng.integers(20, 40)
                obs_h = self.rng.integers(20, 40)
                self.obstacles.append({'x': x, 'y': obs_y, 'w': obs_w, 'h': obs_h})
    
    def _get_track_properties_at(self, world_x):
        idx = int(max(0, min(len(self.track_center_points) - 1, world_x)))
        return self.track_center_points[idx]

    def _create_particles(self, count, color):
        for _ in range(count):
            particle = {
                'pos': [self.car_pos[0] - 10, self.car_pos[1] + self.rng.uniform(-5, 5)],
                'vel': [self.rng.uniform(-3, -1), self.rng.uniform(-1, 1)],
                'life': self.rng.integers(15, 25),
                'color': color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 25.0))
            color = p['color'] + (alpha,)
            size = max(1, int(p['life'] / 5))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

    def validate_implementation(self):
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example of how to run the environment ---
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Control ---
    # 0=none, 1=up, 2=down, 3=left, 4=right
    # [movement, space, shift]
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # To render to screen, we need to create a display
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # Create action from keyboard input
        action = [0, 0, 0] # [none, no-space, no-shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                action[0] = move_val
                break # Prioritize first key found
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_SHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
            # Wait a bit before restarting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
    
    env.close()