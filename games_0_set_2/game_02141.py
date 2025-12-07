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


# Set Pygame to run in a headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to steer. Stay on the neon line to survive. "
        "Reach checkpoints to gain more time."
    )

    game_description = (
        "A fast-paced, top-down arcade racer. Steer your car to stay on a "
        "procedurally generated neon track. Reach checkpoints to extend your time "
        "and score points for precision driving."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_CAR = (255, 255, 0)
    COLOR_CAR_GLOW = (255, 255, 0, 50)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_TIMER_GOOD = (0, 255, 100)
    COLOR_TIMER_WARN = (255, 255, 0)
    COLOR_TIMER_BAD = (255, 50, 50)
    
    # Game parameters
    FPS = 30
    MAX_STEPS = 1000
    INITIAL_TIME = 60.0
    CHECKPOINT_TIME_BONUS = 5.0
    
    # Car parameters
    CAR_Y_POS = 350
    CAR_SPEED = 6.0
    TURN_SPEED = 4.0
    
    # Track parameters
    TRACK_WIDTH = 35
    TRACK_SEGMENT_LENGTH = 15
    INITIAL_MAX_CURVATURE = 10.0
    DIFFICULTY_INTERVAL = 200
    CURVATURE_INCREASE = 5.0
    CHECKPOINT_INTERVAL = 20 # number of segments between checkpoints

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)
        
        self.track = deque()
        self.checkpoints = deque()
        self.particles = deque()
        
        self.car_angle = 0.0
        self.steps = 0
        self.score = 0
        self.time_left = 0.0
        self.game_over = False
        self.current_max_curvature = 0.0
        self.segments_since_checkpoint = 0
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.INITIAL_TIME
        self.car_angle = 0.0
        self.current_max_curvature = self.INITIAL_MAX_CURVATURE
        
        self.track.clear()
        self.checkpoints.clear()
        self.particles.clear()
        
        self.segments_since_checkpoint = 0
        self._generate_initial_track()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            reward = 0
            terminated = True
            return (
                self._get_observation(), reward, terminated, False, self._get_info()
            )

        # --- Update game state ---
        self._handle_input(action)
        self._update_world()
        self._update_particles()
        
        self.steps += 1
        self.time_left -= 1.0 / self.FPS
        
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.current_max_curvature += self.CURVATURE_INCREASE

        # --- Calculate rewards and check for termination ---
        dist_from_center, on_track = self._get_car_track_distance()
        
        reward = self._calculate_reward(dist_from_center, on_track)
        self.score += reward
        
        terminated = self._check_termination(on_track)
        if terminated and not self.game_over:
            self.game_over = True
            # Apply terminal penalty
            self.score -= 100
            reward -= 100
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )
    
    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 3:  # Left
            self.car_angle -= self.TURN_SPEED
        elif movement == 4:  # Right
            self.car_angle += self.TURN_SPEED
        
        self.car_angle %= 360

    def _update_world(self):
        # Simulate forward movement by scrolling the world
        dx = -self.CAR_SPEED * math.sin(math.radians(self.car_angle))
        dy = self.CAR_SPEED * math.cos(math.radians(self.car_angle))
        
        # Update track points
        for i in range(len(self.track)):
            self.track[i] = (self.track[i][0] + dx, self.track[i][1] + dy, self.track[i][2])
            
        # Update checkpoints
        for i in range(len(self.checkpoints)):
            self.checkpoints[i] = (self.checkpoints[i][0] + dx, self.checkpoints[i][1] + dy)
            
        # Prune old track segments and checkpoints
        while self.track and self.track[0][1] > self.SCREEN_HEIGHT + 50:
            self.track.popleft()
        
        while self.checkpoints and self.checkpoints[0][1] > self.SCREEN_HEIGHT + 50:
            self.checkpoints.popleft()
            
        # Generate new track segments
        while self.track[-1][1] > -50:
            self._generate_next_segment()

    def _generate_initial_track(self):
        # Start track from the car's position
        x, y = self.SCREEN_WIDTH / 2, self.CAR_Y_POS
        angle = -90 # Start pointing straight up
        
        # Generate enough points to fill the screen
        for _ in range(int(self.SCREEN_HEIGHT / self.TRACK_SEGMENT_LENGTH) + 5):
            self.track.append((x, y, angle))
            angle_change = self.np_random.uniform(-self.INITIAL_MAX_CURVATURE / 5, self.INITIAL_MAX_CURVATURE / 5)
            angle += angle_change
            x += self.TRACK_SEGMENT_LENGTH * math.cos(math.radians(angle))
            y += self.TRACK_SEGMENT_LENGTH * math.sin(math.radians(angle))
        self.segments_since_checkpoint = 0


    def _generate_next_segment(self):
        last_x, last_y, last_angle = self.track[-1]
        
        # Introduce random curvature
        angle_change = self.np_random.uniform(-self.current_max_curvature, self.current_max_curvature)
        new_angle = last_angle + angle_change
        
        # Clamp angle to prevent extreme turns
        prev_angle = self.track[-2][2] if len(self.track) > 1 else last_angle
        angle_diff = (new_angle - prev_angle + 180) % 360 - 180
        if abs(angle_diff) > self.current_max_curvature * 1.5:
             new_angle = prev_angle + np.sign(angle_diff) * self.current_max_curvature * 1.5

        # Calculate new point
        new_x = last_x + self.TRACK_SEGMENT_LENGTH * math.cos(math.radians(new_angle))
        new_y = last_y + self.TRACK_SEGMENT_LENGTH * math.sin(math.radians(new_angle))
        
        self.track.append((new_x, new_y, new_angle))
        
        # Add a checkpoint if interval is reached
        self.segments_since_checkpoint += 1
        if self.segments_since_checkpoint >= self.CHECKPOINT_INTERVAL:
            self.checkpoints.append((new_x, new_y))
            self.segments_since_checkpoint = 0
            
    def _get_car_track_distance(self):
        car_pos = np.array([self.SCREEN_WIDTH / 2, self.CAR_Y_POS])
        min_dist = float('inf')
        
        if len(self.track) < 2:
            return min_dist, False

        for i in range(len(self.track) - 1):
            p1 = np.array(self.track[i][:2])
            p2 = np.array(self.track[i+1][:2])
            
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0.0:
                dist = np.linalg.norm(car_pos - p1)
            else:
                t = max(0, min(1, np.dot(car_pos - p1, p2 - p1) / l2))
                projection = p1 + t * (p2 - p1)
                dist = np.linalg.norm(car_pos - projection)
            
            if dist < min_dist:
                min_dist = dist
        
        on_track = min_dist <= self.TRACK_WIDTH / 2
        return min_dist, on_track

    def _calculate_reward(self, dist_from_center, on_track):
        reward = 0.0
        
        if on_track:
            reward += 0.1 * (1.0 - (dist_from_center / (self.TRACK_WIDTH / 2)))
            if dist_from_center < self.TRACK_WIDTH / 4:
                self._spawn_particles(5)
        else:
            reward -= 0.2
            
        car_pos = np.array([self.SCREEN_WIDTH / 2, self.CAR_Y_POS])
        new_checkpoints = deque()
        checkpoint_hit_this_step = False
        for chk_pos_tuple in self.checkpoints:
            if not checkpoint_hit_this_step and np.linalg.norm(car_pos - np.array(chk_pos_tuple)) < self.TRACK_WIDTH:
                reward += 5.0
                self.time_left += self.CHECKPOINT_TIME_BONUS
                checkpoint_hit_this_step = True
            else:
                new_checkpoints.append(chk_pos_tuple)
        self.checkpoints = new_checkpoints
        
        if self.steps >= self.MAX_STEPS:
            reward += 50
        
        return reward

    def _check_termination(self, on_track):
        if not on_track and self._get_car_track_distance()[0] > self.TRACK_WIDTH:
            return True
        if self.time_left <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

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
            "time_left": self.time_left,
        }

    def _render_game(self):
        self._render_track()
        self._render_checkpoints()
        self._render_particles()
        self._render_car()

    def _render_track(self):
        if len(self.track) < 2:
            return
        
        hue_shift = self.steps * 0.5
        r = int(127 * math.sin(0.01 * hue_shift + 0) + 128)
        g = int(127 * math.sin(0.01 * hue_shift + 2) + 128)
        b = int(127 * math.sin(0.01 * hue_shift + 4) + 128)
        bright_color = (r, g, b)
        glow_color = (r // 2, g // 2, b // 2, 100)

        points = [(int(p[0]), int(p[1])) for p in self.track]
        
        pygame.draw.lines(self.screen, glow_color, False, points, width=self.TRACK_WIDTH)
        pygame.draw.lines(self.screen, bright_color, False, points, width=int(self.TRACK_WIDTH * 0.4))

    def _render_checkpoints(self):
        for pos in self.checkpoints:
            x, y = int(pos[0]), int(pos[1])
            radius = self.TRACK_WIDTH // 2
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            glow_radius = int(radius * (1.2 + pulse * 0.4))
            
            pygame.gfxdraw.filled_circle(self.screen, x, y, glow_radius, (255, 255, 255, 30))
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, (200, 200, 255))
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, (255, 255, 255))
    
    def _render_car(self):
        car_pos = (self.SCREEN_WIDTH // 2, self.CAR_Y_POS)
        car_size = 12
        
        points = [
            (0, -car_size),
            (-car_size // 2, car_size // 2),
            (car_size // 2, car_size // 2)
        ]
        
        angle_rad = math.radians(self.car_angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        rotated_points = [
            (p[0] * cos_a - p[1] * sin_a + car_pos[0],
             p[0] * sin_a + p[1] * cos_a + car_pos[1])
            for p in points
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_CAR_GLOW)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_CAR)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_CAR)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(p['size'] * life_ratio)
            if radius > 0:
                color = (
                    int(p['color'][0] * life_ratio),
                    int(p['color'][1] * life_ratio),
                    int(p['color'][2] * life_ratio),
                    int(p['color'][3] * life_ratio)
                )
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _spawn_particles(self, count):
        car_pos = np.array([self.SCREEN_WIDTH / 2, self.CAR_Y_POS])
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.0)
            self.particles.append({
                'pos': car_pos + self.np_random.uniform(-5, 5, 2),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'size': self.np_random.integers(2, 5),
                'color': (255, 255, 255, 150)
            })

    def _update_particles(self):
        surviving_particles = deque()
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                surviving_particles.append(p)
        self.particles = surviving_particles

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_ratio = max(0, self.time_left / self.INITIAL_TIME)
        if time_ratio > 0.5:
            time_color = self.COLOR_TIMER_GOOD
        elif time_ratio > 0.2:
            time_color = self.COLOR_TIMER_WARN
        else:
            time_color = self.COLOR_TIMER_BAD
        
        time_text = self.font_ui.render(f"TIME: {self.time_left:.1f}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        if self.game_over:
            msg = "FINISH!" if self.steps >= self.MAX_STEPS else "GAME OVER"
            over_text = self.font_game_over.render(msg, True, self.COLOR_UI_TEXT)
            over_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, over_rect)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # To run and play the game, you might need to unset the dummy video driver
    # depending on your OS. For example:
    # del os.environ["SDL_VIDEODRIVER"]
    # Or set it to your display driver e.g., "x11", "windows"
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Neon Racer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] 
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()