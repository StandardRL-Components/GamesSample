import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:36:57.733795
# Source Brief: brief_03435.md
# Brief Index: 3435
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball to clear colored blocks.

    The player adjusts the launch angle of a stationary ball, selects its color (red, green, blue),
    and launches it. The ball bounces off walls and destroys blocks of the same color upon contact.
    The goal is to clear all 25 blocks within 120 seconds.

    Visuals are clean and geometric, with particle effects for block destruction to provide
    satisfying feedback.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) - Adjusts launch angle.
    - actions[1]: Space button (0=released, 1=held) - Cycles ball color on press.
    - actions[2]: Shift button (0=released, 1=held) - Launches the ball on press.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Aim and launch a colored ball to break matching blocks. "
        "Clear all the blocks before time runs out to win."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to aim the launcher. "
        "Press space to cycle the ball's color and shift to launch."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_TIME_SECONDS = 120
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS

        # --- Colors ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_RED = (255, 80, 80)
        self.COLOR_GREEN = (80, 255, 80)
        self.COLOR_BLUE = (80, 80, 255)
        self.BALL_COLORS = [self.COLOR_RED, self.COLOR_GREEN, self.COLOR_BLUE]
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_LAUNCH_INDICATOR = (200, 200, 200, 150)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.ball_radius = 12
        self.ball_speed = 8
        self.ball_color_index = 0
        self.ball_launched = False
        self.launch_angle = 0.0
        self.blocks = []
        self.block_size = (32, 32)
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.blocks_destroyed_this_bounce = 0

        # self.reset() # reset is called by the wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.ball_launched = False
        self.ball_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.ball_color_index = self.np_random.integers(0, 3)
        self.launch_angle = -math.pi / 2  # Straight up

        self.prev_space_held = False
        self.prev_shift_held = False

        self.blocks_destroyed_this_bounce = 0

        self.particles.clear()
        self._generate_blocks()

        return self._get_observation(), self._get_info()

    def _generate_blocks(self):
        self.blocks.clear()
        grid_w, grid_h = 7, 4
        total_blocks = 25
        
        start_x = (self.SCREEN_WIDTH - (grid_w * self.block_size[0] + (grid_w - 1) * 10)) / 2
        start_y = 50
        
        positions = [(col, row) for row in range(grid_h) for col in range(grid_w)]
        selected_indices = self.np_random.choice(len(positions), total_blocks, replace=False)

        for i in selected_indices:
            col, row = positions[i]
            x = start_x + col * (self.block_size[0] + 10)
            y = start_y + row * (self.block_size[1] + 10)
            color_index = self.np_random.integers(0, 3)
            block_rect = pygame.Rect(x, y, self.block_size[0], self.block_size[1])
            self.blocks.append({"rect": block_rect, "color_index": color_index})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        if not self.ball_launched:
            angle_change_speed = 0.05
            if movement == 3:  # Left
                self.launch_angle -= angle_change_speed
            if movement == 4:  # Right
                self.launch_angle += angle_change_speed
            self.launch_angle = max(-math.pi + 0.1, min(-0.1, self.launch_angle))

        if space_held and not self.prev_space_held:
            self.ball_color_index = (self.ball_color_index + 1) % 3
            # sfx: color_change_sound

        if shift_held and not self.prev_shift_held and not self.ball_launched:
            self.ball_launched = True
            self.ball_vel.x = self.ball_speed * math.cos(self.launch_angle)
            self.ball_vel.y = self.ball_speed * math.sin(self.launch_angle)
            self.blocks_destroyed_this_bounce = 0
            # sfx: launch_sound

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self.steps += 1

        if self.ball_launched:
            self.ball_pos += self.ball_vel
            bounced = self._handle_bounces()
            if bounced:
                if self.blocks_destroyed_this_bounce > 1:
                    reward += 1.0  # Chain reaction reward
                self.blocks_destroyed_this_bounce = 0
            reward += self._handle_block_collisions()

        self._update_particles()

        # --- Check Termination ---
        terminated = False
        truncated = False
        if not self.blocks:
            terminated = True
            reward += 100.0  # Victory reward
            # sfx: victory_sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Time limit is a terminal condition, not truncation
            reward -= 100.0  # Time out penalty
            # sfx: failure_sound

        self.game_over = terminated or truncated
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_bounces(self):
        bounced = False
        if self.ball_pos.x <= self.ball_radius or self.ball_pos.x >= self.SCREEN_WIDTH - self.ball_radius:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.ball_radius, min(self.SCREEN_WIDTH - self.ball_radius, self.ball_pos.x))
            bounced = True
            # sfx: bounce_sound
        if self.ball_pos.y <= self.ball_radius:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.ball_radius
            bounced = True
            # sfx: bounce_sound
        if self.ball_pos.y >= self.SCREEN_HEIGHT + self.ball_radius:
            self.ball_launched = False
            self.ball_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40)
            self.ball_vel = pygame.math.Vector2(0, 0)
            bounced = True
        return bounced

    def _handle_block_collisions(self):
        collision_reward = 0.0
        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            if block["rect"].collidepoint(self.ball_pos) and block["color_index"] == self.ball_color_index:
                collision_reward += 0.1
                self.blocks_destroyed_this_bounce += 1
                self._create_particles(block["rect"].center, self.BALL_COLORS[block["color_index"]])
                self.blocks.pop(i)
                # sfx: block_destroy_sound
        return collision_reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({"pos": pygame.math.Vector2(pos), "vel": vel, "lifetime": lifetime, "color": color})

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"] += p["vel"]
            p["vel"] *= 0.95  # Damping
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.pop(i)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "blocks_remaining": len(self.blocks)}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifetime"] / 30.0))))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), 2, (*p["color"], alpha))

        for block in self.blocks:
            color = self.BALL_COLORS[block["color_index"]]
            pygame.draw.rect(self.screen, color, block["rect"], border_radius=4)
            highlight_color = tuple(min(255, c + 40) for c in color)
            pygame.draw.rect(self.screen, highlight_color, (block["rect"].x + 2, block["rect"].y + 2, block["rect"].width - 4, block["rect"].height - 4), 2, border_radius=4)

        if not self.ball_launched:
            end_pos_x = self.ball_pos.x + 50 * math.cos(self.launch_angle)
            end_pos_y = self.ball_pos.y + 50 * math.sin(self.launch_angle)
            pygame.draw.line(self.screen, self.COLOR_LAUNCH_INDICATOR, (int(self.ball_pos.x), int(self.ball_pos.y)), (int(end_pos_x), int(end_pos_y)), 3)

        ball_color = self.BALL_COLORS[self.ball_color_index]
        glow_radius = int(self.ball_radius * 1.5)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surface, glow_radius, glow_radius, glow_radius, (*ball_color, 50))
        self.screen.blit(glow_surface, (int(self.ball_pos.x - glow_radius), int(self.ball_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.ball_radius, ball_color)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.ball_radius, ball_color)

    def _render_ui(self):
        time_left = self.MAX_TIME_SECONDS - (self.steps / self.FPS)
        time_text = f"Time: {max(0, int(time_left))}"
        text_surface = self.font.render(time_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 10, 10))

        blocks_text = f"Blocks: {len(self.blocks)}"
        text_surface = self.font.render(blocks_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (10, self.SCREEN_HEIGHT - text_surface.get_height() - 10))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to remove the dummy video driver environment variable to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Color Bounce")
    clock = pygame.time.Clock()

    running = True
    while running:
        movement = 0 # No-op
        space_pressed = 0 # Use pressed, not held, for single actions
        shift_pressed = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_pressed = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_pressed = 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        # Note: The original code had up/down for movement, but logic only uses left/right.
        # I'm keeping the manual controls consistent with the action logic.

        action = [movement, space_pressed, shift_pressed]
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()