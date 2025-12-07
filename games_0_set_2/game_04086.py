
# Generated: 2025-08-28T01:22:41.003514
# Source Brief: brief_04086.md
# Brief Index: 4086

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → arrow keys to steer the car. Stay on the white line to maintain speed and complete the race."
    )

    game_description = (
        "A fast-paced retro racer. Navigate a twisting, procedurally generated track. Going off-track costs you a life and speed. Complete all three stages before time runs out to win."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.TOTAL_TIME_SECONDS = 180
        self.MAX_STEPS = self.TOTAL_TIME_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_TRACK = (240, 240, 255)
        self.COLOR_PLAYER = (255, 60, 60)
        self.COLOR_PLAYER_GLOW = (255, 60, 60)
        self.COLOR_FINISH = (60, 255, 60)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEART = (255, 80, 80)
        self.COLOR_PARTICLE = (200, 200, 255)

        # Player Physics
        self.PLAYER_Y_POS = self.HEIGHT * 0.8
        self.PLAYER_ACCEL = 0.5
        self.PLAYER_FRICTION = 0.92
        self.PLAYER_MAX_VEL = 6.0
        
        # Track Properties
        self.TRACK_WIDTH = 40
        self.TRACK_SEGMENT_LENGTH = 10
        self.DIFFICULTY_PER_STAGE = 0.15

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.player_x = 0
        self.player_x_vel = 0
        self.lives = 0
        self.stage = 0
        self.stage_transition_reward = 0
        self.is_off_track = False
        self.off_track_penalty_applied = False
        self.scroll_speed = 0
        self.track_points = deque()
        self.track_phase = 0
        self.track_amplitude = 0
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.player_x = self.WIDTH / 2
        self.player_x_vel = 0.0
        self.lives = 3
        self.stage = 1
        self.stage_transition_reward = 0
        self.is_off_track = False
        self.off_track_penalty_applied = False
        
        self.scroll_speed = 3.0
        self.track_phase = self.np_random.uniform(0, 2 * math.pi)
        self.track_amplitude = 1.0
        
        self.particles = []
        self._generate_initial_track()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        self._handle_input(movement)

        # --- Game Logic Updates ---
        self._update_player()
        self._update_world()
        self._update_particles()
        
        # --- State Checks ---
        prev_stage = self.stage
        self._check_player_state()
        self._update_game_state()
        
        # --- Reward Calculation ---
        reward = self._calculate_reward(prev_stage)
        self.score += reward
        
        # --- Termination Check ---
        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            terminal_reward = 0
            if self.game_won:
                terminal_reward = 100
            else:
                terminal_reward = -100
            reward += terminal_reward
            self.score += terminal_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.player_x_vel -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_x_vel += self.PLAYER_ACCEL

    def _update_player(self):
        self.player_x_vel *= self.PLAYER_FRICTION
        self.player_x_vel = np.clip(self.player_x_vel, -self.PLAYER_MAX_VEL, self.PLAYER_MAX_VEL)
        self.player_x += self.player_x_vel
        self.player_x = np.clip(self.player_x, 0, self.WIDTH)

    def _update_world(self):
        # Scroll existing track points
        for i in range(len(self.track_points)):
            self.track_points[i] = (self.track_points[i][0], self.track_points[i][1] - self.scroll_speed)
        
        # Remove points that are off-screen
        while self.track_points and self.track_points[0][1] < -self.TRACK_WIDTH:
            self.track_points.popleft()
            
        # Add new points to the top
        while self.track_points[-1][1] < self.HEIGHT + self.TRACK_WIDTH:
            self._extend_track()

    def _update_particles(self):
        # Spawn new particles
        if not self.is_off_track and self.np_random.random() < 0.8:
            p_vel = self.scroll_speed * 1.5
            p_life = self.np_random.integers(15, 30)
            p_size = self.np_random.uniform(1, 3)
            self.particles.append([self.player_x, self.PLAYER_Y_POS + 10, p_vel, p_life, p_size])
        
        # Update and remove old particles
        new_particles = []
        for p in self.particles:
            p[1] -= p[2] * 0.5 # Move particle
            p[3] -= 1 # Decrease lifetime
            p[4] -= 0.05 # Decrease size
            if p[3] > 0 and p[4] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _check_player_state(self):
        track_center_x = -1
        # Find track segment at player's y-level
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            if p1[1] <= self.PLAYER_Y_POS < p2[1]:
                # Linear interpolation to find track's x at player's y
                dy = p2[1] - p1[1]
                if dy > 0:
                    ratio = (self.PLAYER_Y_POS - p1[1]) / dy
                    track_center_x = p1[0] + ratio * (p2[0] - p1[0])
                break
        
        if track_center_x != -1:
            distance_from_center = abs(self.player_x - track_center_x)
            self.is_off_track = distance_from_center > self.TRACK_WIDTH / 2
        else:
            self.is_off_track = True

    def _update_game_state(self):
        # Update stage based on steps
        steps_per_stage = self.MAX_STEPS // 3
        current_stage = min(3, (self.steps // steps_per_stage) + 1)
        if current_stage > self.stage:
            self.stage = current_stage
            self.track_amplitude += self.DIFFICULTY_PER_STAGE
            # sound: stage_complete.wav
        
        # Handle off-track penalty
        if self.is_off_track:
            self.scroll_speed = max(1.0, self.scroll_speed * 0.98)
            if not self.off_track_penalty_applied:
                self.lives -= 1
                self.off_track_penalty_applied = True
                # sound: lose_life.wav
        else:
            self.scroll_speed = min(8.0, self.scroll_speed * 1.005)
            self.off_track_penalty_applied = False

    def _calculate_reward(self, prev_stage):
        reward = 0
        if self.is_off_track:
            reward -= 0.2
        else:
            reward += 0.1
            
        if self.stage > prev_stage:
            reward += 1.0
            
        return reward
        
    def _check_termination(self):
        if self.lives <= 0:
            return True
        if self.steps >= self.MAX_STEPS - 1:
            # If time runs out but we are on stage 3, it's a win
            if self.stage == 3:
                self.game_won = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p[0]), int(p[1])), int(p[4]), 1)

        # Render track
        if len(self.track_points) > 1:
            for i in range(len(self.track_points)):
                point = self.track_points[i]
                pygame.gfxdraw.filled_circle(self.screen, int(point[0]), int(point[1]), self.TRACK_WIDTH // 2, self.COLOR_TRACK)
            
            # Anti-aliased edges for smoothness
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, self.track_points, 1)

        # Render finish lines
        steps_per_stage = self.MAX_STEPS // 3
        for i in range(1, 3):
            stage_step = i * steps_per_stage
            # Calculate y position based on how many steps away the finish line is
            y_pos = self.PLAYER_Y_POS - ((stage_step - self.steps) * self.scroll_speed)
            if 0 < y_pos < self.HEIGHT:
                pygame.draw.line(self.screen, self.COLOR_FINISH, (0, y_pos), (self.WIDTH, y_pos), 5)

        # Render player
        player_rect = pygame.Rect(0, 0, 12, 24)
        player_rect.center = (self.player_x, self.PLAYER_Y_POS)
        
        # Glow effect
        glow_radius = 20
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        alpha = 100 - 80 * abs(self.player_x_vel / self.PLAYER_MAX_VEL)
        pygame.draw.circle(glow_surface, (*self.COLOR_PLAYER_GLOW, int(alpha)), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (player_rect.centerx - glow_radius, player_rect.centery - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Car body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Off-track flash
        if self.off_track_penalty_applied and (self.steps % 10 < 5):
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((255,0,0,30))
            self.screen.blit(s, (0,0))


    def _render_ui(self):
        # Time remaining
        time_left = self.TOTAL_TIME_SECONDS - (self.steps / self.FPS)
        mins, secs = divmod(max(0, time_left), 60)
        time_text = f"TIME {int(mins):02}:{int(secs):02}"
        self._draw_text(time_text, (self.WIDTH - 10, 10), self.font_ui, self.COLOR_UI_TEXT, align="topright")
        
        # Stage
        stage_text = f"STAGE {self.stage}/3"
        self._draw_text(stage_text, (10, 10), self.font_ui, self.COLOR_UI_TEXT, align="topleft")
        
        # Lives
        for i in range(self.lives):
            self._draw_heart((self.WIDTH // 2 - 30 + i * 30, 25), 10)

        # Game Over / Win Message
        if self.game_over:
            msg = "RACE COMPLETE" if self.game_won else "GAME OVER"
            color = self.COLOR_FINISH if self.game_won else self.COLOR_PLAYER
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2), self.font_msg, color, align="center")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }
    
    def _generate_initial_track(self):
        self.track_points.clear()
        y = self.HEIGHT + self.TRACK_WIDTH
        while y > -self.TRACK_WIDTH:
            self._extend_track(prepend=True)
            y -= self.TRACK_SEGMENT_LENGTH

    def _extend_track(self, prepend=False):
        if prepend:
            last_y = self.track_points[0][1] if self.track_points else self.HEIGHT + self.TRACK_WIDTH
            y = last_y - self.TRACK_SEGMENT_LENGTH
        else:
            last_y = self.track_points[-1][1] if self.track_points else -self.TRACK_WIDTH
            y = last_y + self.TRACK_SEGMENT_LENGTH
        
        # Generate x using sine waves for smooth curves
        wave1 = 80 * self.track_amplitude * math.sin(y / 150 + self.track_phase)
        wave2 = 60 * self.track_amplitude * math.sin(y / 80 + self.track_phase * 1.5)
        x_offset = wave1 + wave2
        x = self.WIDTH / 2 + x_offset
        
        if prepend:
            self.track_points.appendleft((x, y))
        else:
            self.track_points.append((x, y))

    def _draw_text(self, text, pos, font, color, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        elif align == "center":
            text_rect.center = pos
        self.screen.blit(text_surface, text_rect)

    def _draw_heart(self, pos, size):
        x, y = pos
        points = [
            (x, y - size * 0.3),
            (x - size * 0.5, y - size * 0.8),
            (x - size, y - size * 0.3),
            (x - size, y + size * 0.2),
            (x, y + size),
            (x + size, y + size * 0.2),
            (x + size, y - size * 0.3),
            (x + size * 0.5, y - size * 0.8),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Line Racer")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # --- Event Handling ---
        action = env.action_space.sample() # Default action
        action[0] = 0 # No movement
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            total_reward = 0
            terminated = False

        # --- Step Environment ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to show it
        # Pygame uses (width, height), numpy uses (height, width)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    pygame.quit()