import gymnasium as gym
import os
import pygame
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:04:56.816772
# Source Brief: brief_00202.md
# Brief Index: 202
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a swinging pendulum to intercept falling blocks.
    The goal is to achieve a high score by hitting blocks and avoiding misses.

    **Visuals:**
    - A minimalist, clean aesthetic with a dark background.
    - The player's pendulum is bright white for high contrast.
    - Falling blocks are colored according to their speed (Red=Fast, Green=Medium, Blue=Slow).
    - Particle effects provide satisfying feedback on block interceptions.

    **Gameplay:**
    - The pendulum swings naturally under simulated gravity.
    - The player can apply force to the left or right to alter the pendulum's angular velocity.
    - Intercepting a block adds to the score and gives the pendulum a speed boost.
    - Missing a block (letting it fall off-screen) subtracts from the score.
    - The game's difficulty increases over time as blocks fall faster.

    **Termination:**
    - The episode ends if the score reaches 1000 (win) or drops below -500 (loss).
    - It also ends after 20 blocks are intercepted or a maximum of 2000 steps is reached.
    """

    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Control a swinging pendulum to intercept falling blocks and achieve a high score."
    user_guide = "Controls: ←→ to apply force and swing the pendulum."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30  # Assumed frame rate for physics calculations

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (255, 255, 255, 40)
    COLOR_UI = (220, 220, 220)
    BLOCK_COLORS = {
        "slow": (102, 153, 255),  # Blue
        "medium": (102, 255, 153),  # Green
        "fast": (255, 102, 102),  # Red
    }

    # Pendulum Physics
    PIVOT_POS = (WIDTH // 2, 50)
    PENDULUM_LENGTH = 150
    BOB_RADIUS = 15
    GRAVITY = 0.05
    DAMPING = 0.998
    PLAYER_FORCE = math.radians(1.0)  # Force applied per action
    IMPACT_BOOST = math.radians(5.0)  # Velocity boost on successful interception

    # Block Mechanics
    BLOCK_SIZE = (30, 30)
    BLOCK_SPAWN_INTERVAL = 60  # Ticks between new blocks
    MAX_BLOCKS = 5

    # Game Rules
    SCORE_WIN = 1000
    SCORE_LOSE = -500
    MAX_INTERCEPTS = 20
    MAX_STEPS = 2000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 30)
            self.font_game_over = pygame.font.Font(None, 60)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""

        self.pendulum_angle = 0.0
        self.pendulum_av = 0.0  # Angular Velocity

        self.blocks = []
        self.intercepted_blocks = 0
        self.base_block_speed = 0.0
        self.block_spawn_timer = 0

        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""

        # Reset pendulum
        self.pendulum_angle = math.pi / 2  # Start hanging straight down
        self.pendulum_av = 0.0

        # Reset blocks
        self.blocks = []
        self.intercepted_blocks = 0
        self.base_block_speed = 1.0
        self.block_spawn_timer = 0
        self._spawn_block()

        # Reset effects
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            reward = 0
            terminated = True
            truncated = self.steps >= self.MAX_STEPS
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        if movement == 3:  # Left
            self.pendulum_av -= self.PLAYER_FORCE
            # SFX: whoosh_left.wav
        elif movement == 4:  # Right
            self.pendulum_av += self.PLAYER_FORCE
            # SFX: whoosh_right.wav

        # --- Game Logic Update ---
        self._update_pendulum()
        self._update_blocks()
        self._update_particles()

        # --- Collision and Scoring ---
        reward = self._handle_collisions_and_score()

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.base_block_speed += 0.1

        # --- Termination Check ---
        terminated, term_reward = self._check_termination()
        reward += term_reward
        if terminated:
            self.game_over = True

        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, truncated episodes are also terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_pendulum(self):
        # Apply gravity: angular acceleration is proportional to sin(angle)
        angular_acceleration = (
            -self.GRAVITY * math.sin(self.pendulum_angle) / self.PENDULUM_LENGTH
        )
        self.pendulum_av += angular_acceleration
        # Apply damping
        self.pendulum_av *= self.DAMPING
        # Update angle
        self.pendulum_angle += self.pendulum_av

    def _update_blocks(self):
        # Spawn new blocks
        self.block_spawn_timer -= 1
        if self.block_spawn_timer <= 0 and len(self.blocks) < self.MAX_BLOCKS:
            self._spawn_block()
            self.block_spawn_timer = self.BLOCK_SPAWN_INTERVAL

        # Move existing blocks
        for block in self.blocks:
            block["pos"][1] += block["speed"]

    def _handle_collisions_and_score(self):
        reward = 0
        bob_pos = self._get_bob_position()

        # Check for block interceptions
        remaining_blocks = []
        for block in self.blocks:
            block_rect = pygame.Rect(block["pos"], self.BLOCK_SIZE)
            if self._check_circle_rect_collision(bob_pos, self.BOB_RADIUS, block_rect):
                # --- Interception ---
                self.score += 100
                reward += 10.0  # Event reward
                self.intercepted_blocks += 1
                self._create_particles(block_rect.center, block["color"])
                # SFX: intercept_success.wav

                # Boost pendulum velocity in the direction of the swing
                if self.pendulum_av > 0:
                    self.pendulum_av += self.IMPACT_BOOST
                else:
                    self.pendulum_av -= self.IMPACT_BOOST
            else:
                remaining_blocks.append(block)

        # Check for missed blocks
        missed_blocks = [b for b in remaining_blocks if b["pos"][1] > self.HEIGHT]
        if missed_blocks:
            # SFX: miss.wav
            self.score -= 50 * len(missed_blocks)
            reward -= 5.0 * len(missed_blocks)  # Event reward

        self.blocks = [b for b in remaining_blocks if b["pos"][1] <= self.HEIGHT]

        # --- Shaping Reward ---
        # +/- 1 for moving towards/away from the nearest block
        nearest_block = self._get_nearest_block(bob_pos)
        if nearest_block:
            bob_x_velocity = (
                self.pendulum_av * self.PENDULUM_LENGTH * math.cos(self.pendulum_angle)
            )
            block_is_right = nearest_block["pos"][0] > bob_pos[0]

            if (block_is_right and bob_x_velocity > 0) or (
                not block_is_right and bob_x_velocity < 0
            ):
                reward += 1.0  # Moving towards
            else:
                reward -= 1.0  # Moving away

        return reward

    def _check_termination(self):
        reward = 0
        terminated = False
        if self.score >= self.SCORE_WIN:
            terminated = True
            reward = 100.0
            self.game_over_message = "VICTORY!"
        elif self.score <= self.SCORE_LOSE:
            terminated = True
            reward = -100.0
            self.game_over_message = "DEFEAT"
        elif self.intercepted_blocks >= self.MAX_INTERCEPTS:
            terminated = True
            self.game_over_message = "LEVEL COMPLETE"
            # Final reward depends on score
            reward = self.score / 10.0
        elif self.steps >= self.MAX_STEPS:
            # This case is now handled by truncation
            terminated = False
            self.game_over_message = "TIME UP"

        return terminated, reward

    def _get_observation(self):
        # --- Main Render Loop ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "intercepts": self.intercepted_blocks,
            "difficulty": self.base_block_speed,
        }

    def _render_game(self):
        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"], int(p["radius"]))

        # Render pendulum
        bob_pos = self._get_bob_position()
        # Arm
        pygame.draw.aaline(self.screen, self.COLOR_PLAYER, self.PIVOT_POS, bob_pos, 2)
        # Pivot
        pygame.gfxdraw.filled_circle(
            self.screen, self.PIVOT_POS[0], self.PIVOT_POS[1], 5, self.COLOR_PLAYER
        )
        # Bob Glow
        pygame.gfxdraw.filled_circle(
            self.screen,
            int(bob_pos[0]),
            int(bob_pos[1]),
            self.BOB_RADIUS + 5,
            self.COLOR_PLAYER_GLOW,
        )
        # Bob
        pygame.gfxdraw.filled_circle(
            self.screen, int(bob_pos[0]), int(bob_pos[1]), self.BOB_RADIUS, self.COLOR_PLAYER
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(bob_pos[0]), int(bob_pos[1]), self.BOB_RADIUS, self.COLOR_PLAYER
        )

        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(
                self.screen, block["color"], pygame.Rect(block["pos"], self.BLOCK_SIZE)
            )

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        intercept_text = self.font_ui.render(
            f"Intercepts: {self.intercepted_blocks}/{self.MAX_INTERCEPTS}",
            True,
            self.COLOR_UI,
        )
        self.screen.blit(intercept_text, (10, 40))

        if self.game_over:
            over_text = self.font_game_over.render(
                self.game_over_message, True, self.COLOR_PLAYER
            )
            text_rect = over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _get_bob_position(self):
        x = self.PIVOT_POS[0] + self.PENDULUM_LENGTH * math.sin(self.pendulum_angle)
        y = self.PIVOT_POS[1] + self.PENDULUM_LENGTH * math.cos(self.pendulum_angle)
        return (x, y)

    def _spawn_block(self):
        speed_type = self.np_random.choice(["slow", "medium", "fast"], p=[0.4, 0.4, 0.2])
        speed_multipliers = {"slow": 0.8, "medium": 1.0, "fast": 1.3}

        block = {
            "pos": [
                self.np_random.uniform(0, self.WIDTH - self.BLOCK_SIZE[0]),
                -self.BLOCK_SIZE[1],
            ],
            "color": self.BLOCK_COLORS[speed_type],
            "speed": self.base_block_speed * speed_multipliers[speed_type],
        }
        self.blocks.append(block)

    def _get_nearest_block(self, pos):
        if not self.blocks:
            return None

        min_dist = float("inf")
        nearest = None
        for block in self.blocks:
            dist = math.hypot(block["pos"][0] - pos[0], block["pos"][1] - pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest = block
        return nearest

    def _check_circle_rect_collision(self, circle_pos, radius, rect):
        # Find the closest point on the rect to the circle's center
        closest_x = max(rect.left, min(circle_pos[0], rect.right))
        closest_y = max(rect.top, min(circle_pos[1], rect.bottom))

        # Calculate the distance between the circle's center and this closest point
        distance_x = circle_pos[0] - closest_x
        distance_y = circle_pos[1] - closest_y

        # If the distance is less than the circle's radius, there's a collision
        return (distance_x**2 + distance_y**2) < (radius**2)

    def _create_particles(self, pos, color):
        # SFX: explosion.wav
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append(
                {
                    "pos": list(pos),
                    "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                    "radius": self.np_random.uniform(2, 6),
                    "lifespan": self.np_random.integers(15, 30),
                    "color": color,
                }
            )

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            p["radius"] *= 0.95  # Shrink
            if p["lifespan"] > 0 and p["radius"] > 0.5:
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")

    # --- Manual Play ---
    # This loop allows a human to play the game.
    # Use Left/Right arrow keys to control the pendulum.

    obs, info = env.reset()
    done = False

    # Re-initialize pygame for display window
    # The dummy driver must be unset to show a window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    pygame.display.init()
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Pendulum Interceptor")

    running = True
    total_reward = 0

    while running:
        # --- Event Handling (for human player) ---
        action = env.action_space.sample()  # Start with a random action
        action[0] = 0  # Default to no-op for movement

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
            running = False
        if keys[pygame.K_r]:  # Reset
            obs, info = env.reset()
            total_reward = 0
            done = False
            continue

        if not done:
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering to Display ---
        # The observation is (H, W, C), but pygame needs (W, H)
        # and surfarray wants (W, H, C). So we need to transpose back.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(GameEnv.FPS)

    env.close()