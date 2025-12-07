
# Generated: 2025-08-27T13:09:03.895307
# Source Brief: brief_00278.md
# Brief Index: 278

        
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
        "Controls: Use arrow keys to aim your drawing cursor. "
        "Hold Space to draw a line in the aimed direction. "
        "Hold Shift to draw longer lines."
    )

    game_description = (
        "A physics-based puzzle game. Draw lines to create a track for the sled. "
        "Guide the sled from the green start to the blue finish, passing yellow checkpoints, "
        "before the timer runs out."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (255, 255, 255, 40)
    COLOR_TRAIL = (255, 255, 255)
    COLOR_TRACK = (255, 50, 50)
    COLOR_START = (50, 255, 50)
    COLOR_FINISH = (50, 50, 255)
    COLOR_CHECKPOINT = (255, 255, 50)
    COLOR_UI_TEXT = (220, 220, 220)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Mechanics
    MAX_STEPS = 900  # 30 seconds at 30 FPS
    GRAVITY = 0.1
    FRICTION = 0.01
    BOUNCE_DAMPENING = 0.8
    SLED_RADIUS = 5
    LINE_LENGTH_SHORT = 20
    LINE_LENGTH_LONG = 50
    TRAIL_LENGTH = 50

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        self.render_mode = render_mode
        self.rng = None

        # Persistent state across resets for difficulty scaling
        self.successful_episodes = 0
        self.difficulty_distance_multiplier = 0.0
        self.difficulty_roughness = 10.0

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.sled_pos = np.array([0.0, 0.0])
        self.sled_vel = np.array([0.0, 0.0])
        self.lines = []
        self.sled_trail = []
        self.start_pos = (0, 0)
        self.finish_pos = (0, 0)
        self.checkpoints = []
        self.unclaimed_checkpoints = []
        self.last_draw_point = (0, 0)
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.rng is None or seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self._generate_level()

        self.sled_pos = np.array(self.start_pos, dtype=float)
        self.sled_vel = np.array([0.0, 0.0])
        self.lines = []
        self.sled_trail = []
        self.last_draw_point = self.start_pos

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.start_pos = (50, 200)
        
        finish_x = 550 + self.difficulty_distance_multiplier
        finish_y = self.rng.integers(150, 250)
        self.finish_pos = (min(finish_x, self.SCREEN_WIDTH - 40), finish_y)

        num_checkpoints = self.rng.integers(1, 4)
        self.checkpoints = []
        x_points = np.linspace(self.start_pos[0] + 100, self.finish_pos[0] - 100, num_checkpoints)
        for x in x_points:
            y_offset = self.rng.uniform(-self.difficulty_roughness, self.difficulty_roughness)
            self.checkpoints.append((int(x), int(self.start_pos[1] + y_offset)))
        self.unclaimed_checkpoints = self.checkpoints[:]

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        prev_x_pos = self.sled_pos[0]
        
        self._handle_drawing(movement, space_held, shift_held)
        self._update_physics()
        
        reward = self._calculate_reward(prev_x_pos)
        self.score += reward
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_drawing(self, movement, space_held, shift_held):
        if not space_held or movement == 0:
            return

        length = self.LINE_LENGTH_LONG if shift_held else self.LINE_LENGTH_SHORT
        direction_map = {
            1: np.array([0, -1]),  # Up
            2: np.array([0, 1]),   # Down
            3: np.array([-1, 0]),  # Left
            4: np.array([1, 0]),   # Right
        }
        direction = direction_map.get(movement)
        
        if direction is not None:
            start_point = self.last_draw_point
            end_point = (
                start_point[0] + direction[0] * length,
                start_point[1] + direction[1] * length
            )
            
            # Clamp to screen boundaries
            end_point = (
                max(0, min(self.SCREEN_WIDTH, end_point[0])),
                max(0, min(self.SCREEN_HEIGHT, end_point[1]))
            )

            # sfx: draw_line.wav
            self.lines.append((start_point, end_point))
            self.last_draw_point = end_point

    def _update_physics(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY

        # Update position
        self.sled_pos += self.sled_vel

        # Collision detection and response
        self._handle_collisions()

        # Update trail
        self.sled_trail.append(tuple(self.sled_pos.astype(int)))
        if len(self.sled_trail) > self.TRAIL_LENGTH:
            self.sled_trail.pop(0)

    def _handle_collisions(self):
        collided = False
        for p1, p2 in self.lines:
            p1 = np.array(p1)
            p2 = np.array(p2)
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)

            if line_len_sq == 0:
                continue

            point_vec = self.sled_pos - p1
            t = np.dot(point_vec, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)

            closest_point = p1 + t * line_vec
            dist_vec = self.sled_pos - closest_point
            dist_sq = np.dot(dist_vec, dist_vec)

            if dist_sq < self.SLED_RADIUS ** 2:
                # sfx: sled_scrape.wav
                collided = True
                dist = math.sqrt(dist_sq)
                overlap = self.SLED_RADIUS - dist
                
                # Move sled out of collision
                if dist > 0:
                    self.sled_pos += (dist_vec / dist) * overlap

                # Calculate reflection
                normal = dist_vec / dist if dist > 0 else np.array([0, -1])
                
                # Apply bounce and friction
                vel_dot_normal = np.dot(self.sled_vel, normal)
                if vel_dot_normal < 0: # Moving towards the line
                    self.sled_vel -= 2 * vel_dot_normal * normal * self.BOUNCE_DAMPENING
                
                # Apply friction parallel to the surface
                tangent = np.array([-normal[1], normal[0]])
                vel_dot_tangent = np.dot(self.sled_vel, tangent)
                friction_force = tangent * vel_dot_tangent * self.FRICTION
                self.sled_vel -= friction_force

    def _calculate_reward(self, prev_x_pos):
        reward = -0.01  # Step penalty

        # Reward for forward progress
        progress = self.sled_pos[0] - prev_x_pos
        if progress > 0:
            reward += 0.1 * progress

        # Reward for hitting checkpoints
        for i in range(len(self.unclaimed_checkpoints) - 1, -1, -1):
            cp = self.unclaimed_checkpoints[i]
            dist_sq = (self.sled_pos[0] - cp[0])**2 + (self.sled_pos[1] - cp[1])**2
            if dist_sq < (self.SLED_RADIUS + 10)**2:
                # sfx: checkpoint.wav
                reward += 10.0
                self.unclaimed_checkpoints.pop(i)
        
        return reward

    def _check_termination(self):
        self.steps += 1
        
        # Win condition
        finish_dist_sq = (self.sled_pos[0] - self.finish_pos[0])**2 + (self.sled_pos[1] - self.finish_pos[1])**2
        if finish_dist_sq < (self.SLED_RADIUS + 15)**2:
            # sfx: victory.wav
            self.score += 100.0
            self._update_difficulty()
            return True

        # Loss conditions
        if not (0 < self.sled_pos[0] < self.SCREEN_WIDTH and 0 < self.sled_pos[1] < self.SCREEN_HEIGHT):
            # sfx: crash.wav
            self.score -= 10.0
            return True
        
        if self.steps >= self.MAX_STEPS:
            # sfx: timeout.wav
            return True
        
        return False

    def _update_difficulty(self):
        self.successful_episodes += 1
        self.difficulty_distance_multiplier += 10.0
        if self.successful_episodes % 5 == 0:
            self.difficulty_roughness = min(self.difficulty_roughness + 5.0, 150.0)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_level_markers()
        self._draw_trail()
        self._draw_lines()
        self._draw_sled()

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_level_markers(self):
        # Start
        pygame.draw.rect(self.screen, self.COLOR_START, (self.start_pos[0] - 2, 0, 4, self.SCREEN_HEIGHT))
        # Finish
        pygame.draw.rect(self.screen, self.COLOR_FINISH, (self.finish_pos[0] - 2, 0, 4, self.SCREEN_HEIGHT))
        # Checkpoints
        for cp in self.checkpoints:
            color = self.COLOR_CHECKPOINT if cp in self.unclaimed_checkpoints else self.COLOR_GRID
            pygame.gfxdraw.filled_circle(self.screen, int(cp[0]), int(cp[1]), 10, color)
            pygame.gfxdraw.aacircle(self.screen, int(cp[0]), int(cp[1]), 10, color)

    def _draw_trail(self):
        if len(self.sled_trail) > 1:
            for i in range(len(self.sled_trail) - 1):
                alpha = int(255 * (i / len(self.sled_trail)))
                color = self.COLOR_TRAIL + (alpha,)
                try:
                    pygame.draw.aaline(self.screen, color, self.sled_trail[i], self.sled_trail[i+1])
                except TypeError: # Handle potential alpha issues with aaline
                     pygame.draw.line(self.screen, self.COLOR_TRAIL, self.sled_trail[i], self.sled_trail[i+1])


    def _draw_lines(self):
        for p1, p2 in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 3)

    def _draw_sled(self):
        x, y = int(self.sled_pos[0]), int(self.sled_pos[1])
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.SLED_RADIUS * 2, self.COLOR_PLAYER_GLOW)
        # Sled body
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.SLED_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.SLED_RADIUS, self.COLOR_PLAYER)
        # Rider
        rider_pos = (x, y - self.SLED_RADIUS)
        pygame.draw.line(self.screen, self.COLOR_BG, rider_pos, (rider_pos[0], rider_pos[1] - 5), 2)
        pygame.gfxdraw.filled_circle(self.screen, rider_pos[0], rider_pos[1] - 7, 3, self.COLOR_BG)

    def _render_ui(self):
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 30.0
        timer_text = f"TIME: {time_left:.1f}"
        timer_surf = self.font_small.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (10, 10))

        # Speed
        speed = np.linalg.norm(self.sled_vel) * 10
        speed_text = f"SPEED: {speed:.0f}"
        speed_surf = self.font_small.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_surf, (self.SCREEN_WIDTH - speed_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"SCORE: {self.score:.0f}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH // 2 - score_surf.get_width() // 2, 10))

        if self.game_over:
            finish_dist_sq = (self.sled_pos[0] - self.finish_pos[0])**2 + (self.sled_pos[1] - self.finish_pos[1])**2
            if finish_dist_sq < (self.SLED_RADIUS + 15)**2:
                 msg = "SUCCESS!"
            else:
                 msg = "FAILED!"
            msg_surf = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            self.screen.blit(msg_surf, (self.SCREEN_WIDTH // 2 - msg_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_surf.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_pos": self.sled_pos.tolist(),
            "sled_vel": self.sled_vel.tolist(),
            "checkpoints_remaining": len(self.unclaimed_checkpoints),
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


# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This setup allows a human to play the game.
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Get player input
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Step the environment
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle quitting and restarting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False
                
        # Control the frame rate
        env.clock.tick(30)

    env.close()