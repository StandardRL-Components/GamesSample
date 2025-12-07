
# Generated: 2025-08-27T14:45:10.867389
# Source Brief: brief_00780.md
# Brief Index: 780

        
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
        "Controls: Use ↑ and ↓ to steer your car vertically. Avoid the red obstacles."
    )

    game_description = (
        "Race against the clock in a procedurally generated side-view line racer, dodging obstacles to reach the finish line."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (200, 0, 0)
    COLOR_TRACK = (100, 200, 255)
    COLOR_CHECKPOINT = (50, 255, 50)
    COLOR_PARTICLE = (200, 220, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BG = (20, 30, 60, 180)

    # Player
    PLAYER_X_POS = 120
    PLAYER_SIZE = 12
    PLAYER_ACCEL = 1.5
    PLAYER_FRICTION = 0.90
    PLAYER_MAX_SPEED = 15

    # Track
    TRACK_TOP_Y = 50
    TRACK_BOTTOM_Y = 350
    TRACK_SCROLL_SPEED = 10.0
    LAP_DISTANCE = 15000 # pixels
    TOTAL_LAPS = 3

    # Obstacles
    INITIAL_OBSTACLE_SPEED = 7.0
    OBSTACLE_SPEED_INCREASE_PER_LAP = 1.5
    OBSTACLE_SIZE = 15
    MAX_OBSTACLES = 8

    # Game
    TOTAL_TIME = 180.0
    MAX_STEPS = int(TOTAL_TIME * FPS)

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

        try:
            self.ui_font = pygame.font.SysFont("Consolas", 24, bold=True)
            self.game_over_font = pygame.font.SysFont("Consolas", 64, bold=True)
        except pygame.error:
            self.ui_font = pygame.font.SysFont(None, 28)
            self.game_over_font = pygame.font.SysFont(None, 72)
        
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = 0.0
        self.track_progress = 0.0
        self.lap = 0
        self.time_remaining = 0.0
        self.obstacles = []
        self.particles = []
        self.stars = []
        self.obstacle_speed = 0.0
        self.game_over = False
        self.game_won = False
        self.game_over_reason = ""
        self.last_reward = 0.0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.game_over_reason = ""
        self.last_reward = 0.0

        self.player_pos = pygame.Vector2(self.PLAYER_X_POS, self.SCREEN_HEIGHT / 2)
        self.player_vel = 0.0

        self.track_progress = 0.0
        self.lap = 1
        self.time_remaining = self.TOTAL_TIME

        self.obstacles = []
        self.particles = []
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        
        # Initialize background stars
        self.stars = [
            (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.integers(1, 3))
            for _ in range(100)
        ]
        
        # Pre-populate initial obstacles
        for i in range(self.MAX_OBSTACLES):
            self._spawn_obstacle(x_pos=self.SCREEN_WIDTH + i * self.SCREEN_WIDTH / self.MAX_OBSTACLES)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        self._handle_input(movement)
        self._update_player()
        self._update_world()
        self._update_particles()
        
        self._check_collisions()
        self._check_laps()
        self._update_timer()

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            self.game_over_reason = "TIMEOUT"

        reward = self._calculate_reward()
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        if movement == 1: # Up
            self.player_vel -= self.PLAYER_ACCEL
        elif movement == 2: # Down
            self.player_vel += self.PLAYER_ACCEL

    def _update_player(self):
        self.player_vel *= self.PLAYER_FRICTION
        self.player_vel = np.clip(self.player_vel, -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)
        self.player_pos.y += self.player_vel
        self.player_pos.y = np.clip(self.player_pos.y, self.TRACK_TOP_Y, self.TRACK_BOTTOM_Y)

        # Add particles for player trail
        if self.np_random.random() < 0.8:
            p_size = self.np_random.integers(2, 5)
            p_life = self.np_random.integers(10, 20)
            p_vel_x = -self.TRACK_SCROLL_SPEED * 0.5
            p_vel_y = self.np_random.uniform(-0.5, 0.5)
            self.particles.append([pygame.Vector2(self.player_pos.x - self.PLAYER_SIZE, self.player_pos.y), pygame.Vector2(p_vel_x, p_vel_y), p_size, p_life])
            # sfx: player_whoosh

    def _update_world(self):
        self.track_progress += self.TRACK_SCROLL_SPEED

        # Update obstacles
        for obs in self.obstacles:
            obs['pos'].x -= self.obstacle_speed
            obs['pos'].x += obs['h_speed']
            if obs['pos'].x < -self.OBSTACLE_SIZE:
                self.obstacles.remove(obs)
                self._spawn_obstacle()
        
        # Update stars
        for i, (x, y, speed) in enumerate(self.stars):
            x -= self.TRACK_SCROLL_SPEED / (4 - speed)
            if x < 0:
                x = self.SCREEN_WIDTH
                y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            self.stars[i] = (x, y, speed)

    def _spawn_obstacle(self, x_pos=None):
        if x_pos is None:
            x_pos = self.SCREEN_WIDTH + self.np_random.integers(50, 200)
        
        y_pos = self.np_random.integers(self.TRACK_TOP_Y, self.TRACK_BOTTOM_Y)
        h_speed = self.np_random.uniform(-1.0, 1.0) * (self.lap) # Horizontal drift
        self.obstacles.append({'pos': pygame.Vector2(x_pos, y_pos), 'h_speed': h_speed})

    def _update_particles(self):
        for p in self.particles[:]:
            p[0] += p[1] # pos += vel
            p[3] -= 1    # lifetime--
            if p[3] <= 0:
                self.particles.remove(p)

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['pos'].x - self.OBSTACLE_SIZE / 2, obs['pos'].y - self.OBSTACLE_SIZE / 2, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
            if player_rect.colliderect(obs_rect):
                self.game_over = True
                self.game_over_reason = "CRASHED"
                # sfx: explosion
                # Create explosion particles
                for _ in range(50):
                    p_angle = self.np_random.uniform(0, 2 * math.pi)
                    p_speed = self.np_random.uniform(2, 10)
                    p_vel = pygame.Vector2(math.cos(p_angle) * p_speed, math.sin(p_angle) * p_speed)
                    p_size = self.np_random.integers(2, 6)
                    p_life = self.np_random.integers(20, 40)
                    self.particles.append([pygame.Vector2(self.player_pos), p_vel, p_size, p_life])
                break

    def _check_laps(self):
        if self.track_progress >= self.lap * self.LAP_DISTANCE:
            self.lap += 1
            if self.lap > self.TOTAL_LAPS:
                self.game_won = True
                self.game_over = True
                self.game_over_reason = "YOU WIN!"
                # sfx: win_fanfare
            else:
                self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED + (self.lap - 1) * self.OBSTACLE_SPEED_INCREASE_PER_LAP
                # sfx: checkpoint
    
    def _update_timer(self):
        self.time_remaining -= 1 / self.FPS
        if self.time_remaining <= 0:
            self.time_remaining = 0
            self.game_over = True
            self.game_over_reason = "TIMEOUT"
            # sfx: timeout_buzzer

    def _calculate_reward(self):
        reward = 0
        if self.game_over:
            if self.game_won:
                reward = 100.0
            elif self.game_over_reason == "CRASHED":
                reward = -10.0
            elif self.game_over_reason == "TIMEOUT":
                reward = -10.0
        elif int(self.track_progress / self.LAP_DISTANCE) + 1 > self.lap -1: # just completed a lap
             if self.lap > int(self.track_progress / self.LAP_DISTANCE): # lap has just been incremented
                reward = 20.0
        else:
            reward = 0.01  # Survival reward

        # Prevent multiple rewards for the same event
        final_reward = reward
        if reward != 0 and self.last_reward == reward and reward < 0.1: # Don't block survival reward
            final_reward = 0
        self.last_reward = reward
        return final_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_background()
        self._render_track()
        self._render_particles()
        self._render_obstacles()
        if not (self.game_over and self.game_over_reason == "CRASHED"):
            self._render_player()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, speed in self.stars:
            color_val = 50 + speed * 20
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(x), int(y)), speed-1)

    def _render_track(self):
        # Top and bottom boundaries
        for y in [self.TRACK_TOP_Y, self.TRACK_BOTTOM_Y]:
            pygame.draw.line(self.screen, self.COLOR_TRACK, (0, y), (self.SCREEN_WIDTH, y), 2)
        
        # Lap progress markers
        marker_interval = self.LAP_DISTANCE / 10
        for i in range(1, 11):
            marker_progress = (self.lap - 1) * self.LAP_DISTANCE + i * marker_interval
            x_pos = self.SCREEN_WIDTH - (marker_progress - self.track_progress)
            if 0 < x_pos < self.SCREEN_WIDTH:
                pygame.draw.line(self.screen, self.COLOR_TRACK, (x_pos, self.TRACK_TOP_Y), (x_pos, self.TRACK_BOTTOM_Y), 1)

        # Checkpoint line
        checkpoint_progress = self.lap * self.LAP_DISTANCE
        x_pos = self.SCREEN_WIDTH - (checkpoint_progress - self.track_progress)
        if 0 < x_pos < self.SCREEN_WIDTH + 20:
             for i in range(10):
                alpha = 255 - i * 25
                pygame.draw.line(self.screen, (*self.COLOR_CHECKPOINT, alpha), (x_pos + i*2, self.TRACK_TOP_Y), (x_pos + i*2, self.TRACK_BOTTOM_Y), 3)

    def _render_particles(self):
        for pos, vel, size, life in self.particles:
            alpha = max(0, min(255, int(255 * (life / 20.0))))
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(size), color)

    def _render_obstacles(self):
        for obs in self.obstacles:
            x, y = int(obs['pos'].x), int(obs['pos'].y)
            s = int(self.OBSTACLE_SIZE)
            rect = pygame.Rect(x - s // 2, y - s // 2, s, s)

            # Glow effect
            for i in range(4, 0, -1):
                glow_s = s + i * 3
                glow_alpha = 80 - i * 20
                glow_rect = pygame.Rect(x - glow_s // 2, y - glow_s // 2, glow_s, glow_s)
                glow_surf = pygame.Surface((glow_s, glow_s), pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, (*self.COLOR_OBSTACLE_GLOW, glow_alpha), glow_surf.get_rect(), border_radius=3)
                self.screen.blit(glow_surf, glow_rect.topleft)

            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)

    def _render_player(self):
        x, y = int(self.player_pos.x), int(self.player_pos.y)
        s = int(self.PLAYER_SIZE)
        
        # Points of the triangle
        p1 = (x + s, y)
        p2 = (x - s // 2, y - int(s * 0.866))
        p3 = (x - s // 2, y + int(s * 0.866))
        points = [p1, p2, p3]

        # Glow effect
        for i in range(5, 0, -1):
            glow_s = s + i * 2
            glow_alpha = 100 - i * 20
            gp1 = (x + glow_s, y)
            gp2 = (x - glow_s // 2, y - int(glow_s * 0.866))
            gp3 = (x - glow_s // 2, y + int(glow_s * 0.866))
            pygame.gfxdraw.aapolygon(self.screen, [gp1, gp2, gp3], (*self.COLOR_PLAYER_GLOW, glow_alpha))
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Lap counter
        lap_text = self.ui_font.render(f"LAP: {self.lap}/{self.TOTAL_LAPS}", True, self.COLOR_TEXT)
        self.screen.blit(lap_text, (10, 10))

        # Timer
        time_str = f"TIME: {self.time_remaining:.1f}"
        time_text = self.ui_font.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Score
        score_text = self.ui_font.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        text_surface = self.game_over_font.render(self.game_over_reason, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.lap,
            "time_remaining": self.time_remaining,
            "track_progress": self.track_progress,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Play Controls ---
    # ↑: Move Up
    # ↓: Move Down
    # Any other key: No-op
    # Close window to quit
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Line Racer")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Default action is no-op
        movement_action = 0
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        
        action[0] = movement_action
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()
    print(f"Game Over. Final Score: {info['score']}")