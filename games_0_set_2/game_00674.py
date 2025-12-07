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
        "Controls: Use ↑ and ↓ to steer your car and stay on the white line. Finish the track in under 30 seconds!"
    )

    game_description = (
        "Steer a speeding car along a procedurally generated line track as fast as possible without falling off."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 5000
        self.WIN_TIME = 30.0

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)

        # Colors
        self.COLOR_BG_TOP = (10, 20, 50)
        self.COLOR_BG_BOTTOM = (40, 60, 120)
        self.COLOR_CAR = (255, 50, 50)
        self.COLOR_CAR_GLOW = (255, 150, 150)
        self.COLOR_TRACK = (255, 255, 255)
        self.COLOR_CHECKPOINT = (50, 255, 50)
        self.COLOR_PARTICLE = (200, 200, 255)
        self.COLOR_TEXT = (255, 255, 255)

        # Game constants
        self.CAR_X_POS = self.SCREEN_WIDTH // 4
        self.CAR_WIDTH = 20
        self.CAR_HEIGHT = 10
        self.CAR_STEER_SPEED = 4
        self.CAR_FORWARD_SPEED = 8  # Pixels per step
        self.TRACK_LENGTH = self.CAR_FORWARD_SPEED * self.MAX_STEPS
        
        # Initialize state variables
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.lap_time = None
        self.car_y = None
        self.track_scroll = None
        self.line_width = None
        self.track_points = None
        self.checkpoints = None
        self.particles = None
        self.np_random = None

    def _generate_track(self):
        """Generates a smooth, curving track using a sum of sine waves."""
        y_offset = self.SCREEN_HEIGHT / 2
        amplitude1 = self.np_random.uniform(low=self.SCREEN_HEIGHT * 0.1, high=self.SCREEN_HEIGHT * 0.35)
        freq1 = self.np_random.uniform(low=0.0005, high=0.001)
        phase1 = self.np_random.uniform(low=0, high=2 * math.pi)
        
        amplitude2 = self.np_random.uniform(low=self.SCREEN_HEIGHT * 0.05, high=self.SCREEN_HEIGHT * 0.15)
        freq2 = self.np_random.uniform(low=0.002, high=0.004)
        phase2 = self.np_random.uniform(low=0, high=2 * math.pi)

        self.track_points = []
        min_y = self.SCREEN_HEIGHT * 0.1
        max_y = self.SCREEN_HEIGHT * 0.9

        # The stability test runs for 60 no-op steps.
        # At step 60, the car's x-position on the track is track_scroll + CAR_X_POS
        # = (60 * CAR_FORWARD_SPEED) + CAR_X_POS = (60 * 8) + 160 = 640.
        # The initial straight section must be longer than this to prevent falling off.
        initial_straight_length = 700

        for x in range(self.TRACK_LENGTH):
            # Start and end straight
            if x < initial_straight_length or x > self.TRACK_LENGTH - self.SCREEN_WIDTH:
                 y = y_offset
            else:
                y = y_offset + \
                    amplitude1 * math.sin(freq1 * x + phase1) + \
                    amplitude2 * math.sin(freq2 * x + phase2)
            
            self.track_points.append(np.clip(y, min_y, max_y))

    def _generate_checkpoints(self):
        """Places checkpoints along the generated track."""
        self.checkpoints = []
        num_checkpoints = 4
        for i in range(1, num_checkpoints + 1):
            x_pos = int(self.TRACK_LENGTH * (i / (num_checkpoints + 1)))
            self.checkpoints.append({"x": x_pos, "passed": False})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.lap_time = 0.0
        
        self.car_y = self.SCREEN_HEIGHT // 2
        self.track_scroll = 0.0
        self.line_width = 10.0
        
        self.particles = deque(maxlen=100)
        
        self._generate_track()
        self._generate_checkpoints()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(30) # Maintain 30 FPS for smooth visuals

        movement = action[0]
        self._update_game_state(movement)
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 100
                self.score += 100
            else:
                reward -= 100
                self.score -= 100
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, movement):
        """Handles all state changes for a single step."""
        # Player movement
        if movement == 1:  # Up
            self.car_y -= self.CAR_STEER_SPEED
        elif movement == 2:  # Down
            self.car_y += self.CAR_STEER_SPEED
        
        self.car_y = np.clip(self.car_y, 0, self.SCREEN_HEIGHT)
        
        # World scroll
        self.track_scroll += self.CAR_FORWARD_SPEED

        # Time - use a fixed time step for determinism
        self.lap_time += 1.0 / 30.0
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.line_width = max(2.0, self.line_width - 0.1)

        # Particle effects
        # Add new particle
        self.particles.append([self.CAR_X_POS, self.car_y, self.np_random.integers(10, 20)])
        # Update existing particles
        for p in self.particles:
            p[0] -= self.CAR_FORWARD_SPEED * 1.2 # Move slightly faster than scroll
            p[2] -= 1 # Decrement lifetime

    def _calculate_reward(self):
        """Calculates reward based on car's position relative to the track."""
        reward = 0
        
        # Get track position at car's location
        track_index = int(self.track_scroll + self.CAR_X_POS)
        if track_index >= len(self.track_points):
            return 0 # Off the end of the track

        track_center_y = self.track_points[track_index]
        distance = abs(self.car_y - track_center_y)
        
        # Reward for staying on the line
        if distance <= self.line_width / 2.0:
            reward += 0.1
            # Penalty for being near the edge
            if (self.line_width / 2.0 - 1) < distance <= self.line_width / 2.0:
                reward -= 0.2 # Net -0.1
        
        # Reward for checkpoints
        for cp in self.checkpoints:
            if not cp["passed"] and self.track_scroll > cp["x"]:
                reward += 1
                self.score += 1
                cp["passed"] = True
        
        return reward

    def _check_termination(self):
        """Checks for all game-ending conditions."""
        # 1. Fallen off the track
        track_index = int(self.track_scroll + self.CAR_X_POS)
        if track_index >= len(self.track_points): # Handle case where car is past the track
            track_center_y = self.track_points[-1]
        else:
            track_center_y = self.track_points[track_index]
        
        distance = abs(self.car_y - track_center_y)
        if distance > self.line_width / 2.0:
            self.game_over = True
            self.win = False
            return True

        # 2. Reached the end of the track
        if self.track_scroll + self.CAR_X_POS >= self.TRACK_LENGTH - self.CAR_FORWARD_SPEED:
            self.game_over = True
            self.win = self.lap_time < self.WIN_TIME
            return True

        # 3. Time limit exceeded
        if self.lap_time >= self.WIN_TIME:
            self.game_over = True
            self.win = False # Failed by timeout
            return True
            
        # 4. Max steps reached
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False # Failed by step limit
            return True

        return False

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lap_time": self.lap_time}

    def _render_background(self):
        """Draws a vertical color gradient for the background."""
        for y in range(self.SCREEN_HEIGHT):
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.SCREEN_HEIGHT
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.SCREEN_HEIGHT
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.SCREEN_HEIGHT
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        """Renders all dynamic game elements."""
        # Render particles
        for x, y, life in self.particles:
            if life > 0:
                alpha = max(0, min(255, int(255 * (life / 20))))
                radius = int(3 * (life / 20))
                if radius > 0:
                    color = (*self.COLOR_PARTICLE, alpha)
                    temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                    self.screen.blit(temp_surf, (int(x - radius), int(y - radius)))

        # Render track
        upper_edge, lower_edge = [], []
        for x in range(self.SCREEN_WIDTH):
            track_index = int(self.track_scroll + x)
            if 0 <= track_index < len(self.track_points):
                track_y = self.track_points[track_index]
                upper_edge.append((x, track_y - self.line_width / 2))
                lower_edge.append((x, track_y + self.line_width / 2))
        
        if len(upper_edge) > 1:
            polygon_points = upper_edge + lower_edge[::-1]
            pygame.gfxdraw.aapolygon(self.screen, polygon_points, self.COLOR_TRACK)
            pygame.gfxdraw.filled_polygon(self.screen, polygon_points, self.COLOR_TRACK)

        # Render checkpoints
        for cp in self.checkpoints:
            if not cp["passed"]:
                screen_x = cp["x"] - self.track_scroll
                if 0 <= screen_x <= self.SCREEN_WIDTH:
                    # Ensure cp["x"] is a valid index for track_points
                    if 0 <= cp["x"] < len(self.track_points):
                        track_y = self.track_points[cp["x"]]
                        half_width = self.line_width / 2 + 10
                        pygame.draw.rect(self.screen, self.COLOR_CHECKPOINT, (screen_x - 2, track_y - half_width, 4, half_width * 2))

        # Render car
        car_points = [
            (self.CAR_X_POS + self.CAR_WIDTH / 2, self.car_y),
            (self.CAR_X_POS - self.CAR_WIDTH / 2, self.car_y - self.CAR_HEIGHT / 2),
            (self.CAR_X_POS - self.CAR_WIDTH / 2, self.car_y + self.CAR_HEIGHT / 2),
        ]
        # Glow effect
        pygame.gfxdraw.filled_trigon(self.screen, 
            int(car_points[0][0]), int(car_points[0][1]),
            int(car_points[1][0]), int(car_points[1][1]),
            int(car_points[2][0]), int(car_points[2][1]),
            self.COLOR_CAR_GLOW
        )
        pygame.gfxdraw.aapolygon(self.screen, car_points, self.COLOR_CAR_GLOW)
        # Main car body
        pygame.gfxdraw.filled_polygon(self.screen, car_points, self.COLOR_CAR)
        pygame.gfxdraw.aapolygon(self.screen, car_points, self.COLOR_CAR)


    def _render_ui(self):
        """Renders UI elements like time and game over text."""
        # Lap time
        time_text = f"{self.lap_time:.2f}s"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))
        
        # Game Over/Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            msg_surf = self.font_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.display.init()
    pygame.display.set_caption("Arcade Racer")
    screen_display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Get player input
        movement = 0 # no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_ESCAPE]:
            running = False

        if keys[pygame.K_r]:
             obs, info = env.reset()
             total_reward = 0
             terminated = False

        # Construct the action
        action = [movement, 0, 0] # Space and Shift are not used

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()