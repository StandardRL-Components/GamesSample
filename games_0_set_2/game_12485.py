import gymnasium as gym
import os
import pygame
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:50:40.882327
# Source Brief: brief_02485.md
# Brief Index: 2485
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls five interconnected bouncing balls
    to collect stars within a time limit. The game prioritizes visual quality and
    satisfying physics-based gameplay.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a group of five interconnected bouncing balls to collect all the stars before time runs out."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to apply force to all balls and guide them towards the stars."
    )
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (200, 200, 220)
    COLOR_TEXT = (240, 240, 240)
    BALL_COLORS = [
        (255, 80, 80),  # Red
        (80, 255, 80),  # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
    ]
    STAR_COLOR = (255, 220, 50)

    # Game parameters
    NUM_BALLS = 5
    NUM_STARS = 20
    BALL_RADIUS = 12
    STAR_RADIUS = 10
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    PLAYER_ACCELERATION = 0.25
    INITIAL_BALL_SPEED = 1.0
    STAR_COLLECT_SPEED_BOOST = 1.03  # 3% speed increase
    COLLISION_DAMPING = 0.95  # Slight energy loss on wall bounce

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        # The observation space is the screen pixels
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        # Action space: [movement, unused, unused]
        # movement: 0=No-op, 1=Up, 2=Down, 3=Left, 4=Right
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_end = pygame.font.SysFont("monospace", 50, bold=True)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.balls = []
        self.stars = []
        self.stars_collected = 0

        # Note: reset() is called by the parent gym.Env constructor, but we may need to call it again
        # if some initialization depends on attributes set after super().__init__()
        # In this case, it's fine.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stars_collected = 0

        self._initialize_balls()
        self._initialize_stars()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        self._apply_player_action(movement)

        # --- Game Logic ---
        self._update_physics()

        # --- Rewards and Termination ---
        reward = 0.01  # Small reward for surviving

        collected_this_step = self._handle_star_collection()
        if collected_this_step > 0:
            reward += collected_this_step * 1.0  # Reward for each star
            self.score += collected_this_step * 1.0
            # SFX: Star collect sound

        self.steps += 1
        terminated = self._check_termination()
        truncated = False  # This environment does not truncate based on conditions other than termination

        if terminated:
            if self.stars_collected == self.NUM_STARS:
                reward += 100.0  # Big reward for winning
                self.score += 100.0
            else: # Time ran out
                reward -= 50.0  # Penalty for losing
                self.score -= 50.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _initialize_balls(self):
        self.balls = []
        for i in range(self.NUM_BALLS):
            angle = (2 * math.pi / self.NUM_BALLS) * i
            pos_x = self.SCREEN_WIDTH / 2 + math.cos(angle) * 50
            pos_y = self.SCREEN_HEIGHT / 2 + math.sin(angle) * 50

            vel_angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.math.Vector2(
                math.cos(vel_angle) * self.INITIAL_BALL_SPEED,
                math.sin(vel_angle) * self.INITIAL_BALL_SPEED
            )

            self.balls.append({
                "pos": pygame.math.Vector2(pos_x, pos_y),
                "vel": vel,
                "color": self.BALL_COLORS[i],
                "radius": self.BALL_RADIUS,
                "trail": deque(maxlen=10)
            })

    def _initialize_stars(self):
        self.stars = []
        spawn_margin = 30

        while len(self.stars) < self.NUM_STARS:
            pos = pygame.math.Vector2(
                self.np_random.uniform(spawn_margin, self.SCREEN_WIDTH - spawn_margin),
                self.np_random.uniform(spawn_margin, self.SCREEN_HEIGHT - spawn_margin)
            )

            is_overlapping = False
            # Check overlap with other stars
            for star in self.stars:
                if pos.distance_to(star["pos"]) < self.STAR_RADIUS * 2.5:
                    is_overlapping = True
                    break
            if is_overlapping: continue

            # Check overlap with initial ball positions
            for ball in self.balls:
                if pos.distance_to(ball["pos"]) < self.BALL_RADIUS + self.STAR_RADIUS + 20:
                    is_overlapping = True
                    break
            if is_overlapping: continue

            self.stars.append({
                "pos": pos,
                "twinkle_phase": self.np_random.uniform(0, 2 * math.pi)
            })

    def _apply_player_action(self, movement):
        accel = pygame.math.Vector2(0, 0)
        if movement == 1: accel.y = -1  # Up
        elif movement == 2: accel.y = 1  # Down
        elif movement == 3: accel.x = -1  # Left
        elif movement == 4: accel.x = 1  # Right

        if accel.length() > 0:
            accel = accel.normalize() * self.PLAYER_ACCELERATION
            for ball in self.balls:
                ball["vel"] += accel

    def _update_physics(self):
        # Update positions
        for ball in self.balls:
            ball["trail"].append(ball["pos"].copy())
            ball["pos"] += ball["vel"]

        # Handle wall collisions
        for ball in self.balls:
            if ball["pos"].x - ball["radius"] < 0:
                ball["pos"].x = ball["radius"]
                ball["vel"].x *= -self.COLLISION_DAMPING
            elif ball["pos"].x + ball["radius"] > self.SCREEN_WIDTH:
                ball["pos"].x = self.SCREEN_WIDTH - ball["radius"]
                ball["vel"].x *= -self.COLLISION_DAMPING

            if ball["pos"].y - ball["radius"] < 0:
                ball["pos"].y = ball["radius"]
                ball["vel"].y *= -self.COLLISION_DAMPING
            elif ball["pos"].y + ball["radius"] > self.SCREEN_HEIGHT:
                ball["pos"].y = self.SCREEN_HEIGHT - ball["radius"]
                ball["vel"].y *= -self.COLLISION_DAMPING

        # Handle inter-ball collisions
        for i in range(self.NUM_BALLS):
            for j in range(i + 1, self.NUM_BALLS):
                b1 = self.balls[i]
                b2 = self.balls[j]

                dist_vec = b1["pos"] - b2["pos"]
                dist = dist_vec.length()
                min_dist = b1["radius"] + b2["radius"]

                if dist < min_dist:
                    # Resolve overlap
                    overlap = min_dist - dist
                    if dist == 0: dist_vec = pygame.math.Vector2(1, 0); dist = 1
                    correction = dist_vec.normalize() * overlap
                    b1["pos"] += correction / 2
                    b2["pos"] -= correction / 2

                    # Elastic collision response
                    normal = dist_vec.normalize()
                    tangent = pygame.math.Vector2(-normal.y, normal.x)

                    v1n = b1["vel"].dot(normal)
                    v1t = b1["vel"].dot(tangent)
                    v2n = b2["vel"].dot(normal)
                    v2t = b2["vel"].dot(tangent)

                    # Swap normal velocities
                    b1["vel"] = v2n * normal + v1t * tangent
                    b2["vel"] = v1n * normal + v2t * tangent

    def _handle_star_collection(self):
        collected_count = 0
        remaining_stars = []
        for star in self.stars:
            is_collected = False
            for ball in self.balls:
                if ball["pos"].distance_to(star["pos"]) < ball["radius"] + self.STAR_RADIUS:
                    is_collected = True
                    break
            if is_collected:
                self.stars_collected += 1
                collected_count += 1
            else:
                remaining_stars.append(star)

        if collected_count > 0:
            for _ in range(collected_count):
                for ball in self.balls:
                    ball["vel"] *= self.STAR_COLLECT_SPEED_BOOST

        self.stars = remaining_stars
        return collected_count

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        if self.stars_collected == self.NUM_STARS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        # Pygame's coordinate system (x, y) maps to array indices (j, i).
        # To get the standard (height, width, channels) shape, we need to transpose.
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Render trails
        for ball in self.balls:
            for i, pos in enumerate(ball["trail"]):
                alpha = int(255 * (i / len(ball["trail"])) * 0.3)
                color = ball["color"]
                radius = int(ball["radius"] * (i / len(ball["trail"])))
                if radius > 1:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, (*color, alpha))

        # Render stars
        for star in self.stars:
            twinkle = (math.sin(star["twinkle_phase"] + self.steps * 0.1) + 1) / 2
            size = self.STAR_RADIUS * (0.8 + twinkle * 0.4)
            points = []
            for i in range(5):
                angle = (i * 2 * math.pi / 5) - math.pi / 2
                outer_p = star["pos"] + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * size
                points.append((int(outer_p.x), int(outer_p.y)))

                angle += math.pi / 5
                inner_p = star["pos"] + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * size * 0.5
                points.append((int(inner_p.x), int(inner_p.y)))

            pygame.gfxdraw.filled_polygon(self.screen, points, self.STAR_COLOR)
            pygame.gfxdraw.aapolygon(self.screen, points, self.STAR_COLOR)

        # Render balls
        for ball in self.balls:
            pos_int = (int(ball["pos"].x), int(ball["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], ball["radius"], ball["color"])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], ball["radius"], ball["color"])

    def _render_ui(self):
        # Time remaining
        time_left = (self.MAX_STEPS - self.steps) / 30
        time_text = f"TIME: {time_left:.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Stars collected
        stars_text = f"STARS: {self.stars_collected}/{self.NUM_STARS}"
        stars_surf = self.font_ui.render(stars_text, True, self.COLOR_TEXT)
        self.screen.blit(stars_surf, (self.SCREEN_WIDTH - stars_surf.get_width() - 10, 10))

        # End game message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.stars_collected == self.NUM_STARS:
                end_text = "YOU WIN!"
                end_color = (150, 255, 150)
            else:
                end_text = "TIME UP!"
                end_color = (255, 150, 150)

            end_surf = self.font_end.render(end_text, True, end_color)
            text_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stars_collected": self.stars_collected,
            "time_left_steps": self.MAX_STEPS - self.steps,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block will not be executed in the testing environment, but is useful for manual testing.
    # We need to unset the dummy video driver to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Bouncing Balls Star Collector")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement_action = 0  # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4

        action = [movement_action, 0, 0]  # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Transpose obs for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000)  # Pause for 2 seconds on game over
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)  # Run at 30 FPS

    env.close()