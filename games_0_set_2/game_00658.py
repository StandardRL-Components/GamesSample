import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the drawing cursor. Space to draw a line. Shift to reset the cursor."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw lines to guide a sled down a slope to the finish line before time runs out. "
        "Optimize your lines for speed, but be careful not to crash!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    TIME_LIMIT_SECONDS = 15

    # Colors
    COLOR_BG = (26, 26, 46)
    COLOR_SLED = (255, 255, 255)
    COLOR_LINE = (0, 255, 255)
    COLOR_GHOST_LINE = (0, 128, 128)
    COLOR_START = (0, 255, 0)
    COLOR_FINISH = (255, 0, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (220, 220, 220)

    # Game parameters
    SLED_SIZE = 10
    GRAVITY = pygame.math.Vector2(0, 0.3)
    FINISH_LINE_X = 600
    MAX_STEPS = 1000
    CURSOR_SPEED = 8
    MIN_LINE_LENGTH = 10
    MAX_LINE_LENGTH = 150

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)

        # Initialize state variables
        self.sled_pos = pygame.math.Vector2(0, 0)
        self.sled_vel = pygame.math.Vector2(0, 0)
        self.sled_trail = []
        self.lines = []
        self.particles = []
        self.draw_start_pos = pygame.math.Vector2(0, 0)
        self.cursor_pos = pygame.math.Vector2(0, 0)
        self.last_space_held = False
        self.time_remaining = 0
        self.stuck_frames = 0
        self.game_over_message = ""
        self.np_random = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""

        # Sled state
        # A long, gentle slope to prevent falling off during no-op stability tests
        start_platform = [(20, 150), (500, 250)]
        self.sled_pos = pygame.math.Vector2(start_platform[0][0] + 20, start_platform[0][1] - self.SLED_SIZE)
        self.sled_vel = pygame.math.Vector2(0, 0)
        self.sled_trail = [self.sled_pos.copy() for _ in range(10)]

        # Drawing state
        self.lines = [(pygame.math.Vector2(p1), pygame.math.Vector2(p2)) for p1, p2 in [start_platform]]
        self.draw_start_pos = self.lines[0][1].copy()
        self.cursor_pos = self.draw_start_pos + pygame.math.Vector2(50, 0)
        self.last_space_held = False

        # Game flow
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS
        self.stuck_frames = 0
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        reward = 0
        terminated = False
        truncated = False

        if not self.game_over:
            self.steps += 1
            self.time_remaining -= 1

            # --- 1. Handle Input ---
            self._handle_input(movement, space_held, shift_held)

            # --- 2. Update Physics ---
            prev_sled_x = self.sled_pos.x
            self._update_physics()

            # --- 3. Update Particles ---
            self._update_particles()

            # --- 4. Calculate Reward ---
            dx = self.sled_pos.x - prev_sled_x
            if dx > 0.1:
                reward += 0.1  # Moving towards finish
            elif not terminated:
                reward -= 0.01  # Stuck or moving away
            self.score += reward

            # --- 5. Check Termination Conditions ---
            terminated, term_reward = self._check_termination()
            if terminated:
                reward += term_reward
                self.score += term_reward
                self.game_over = True
            
            if self.steps >= self.MAX_STEPS:
                truncated = True
                self.game_over = True


        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED

        # Clamp cursor to screen
        self.cursor_pos.x = max(0, min(self.WIDTH, self.cursor_pos.x))
        self.cursor_pos.y = max(0, min(self.HEIGHT, self.cursor_pos.y))

        # Reset cursor with Shift
        if shift_held:
            self.cursor_pos = self.draw_start_pos + pygame.math.Vector2(50, 0)

        # Draw line with Space
        if space_held and not self.last_space_held:
            line_vec = self.cursor_pos - self.draw_start_pos
            if line_vec.length() > self.MIN_LINE_LENGTH:
                # Clamp line length
                if line_vec.length() > self.MAX_LINE_LENGTH:
                    line_vec.scale_to_length(self.MAX_LINE_LENGTH)
                    self.cursor_pos = self.draw_start_pos + line_vec

                # Add the new line
                new_line = (self.draw_start_pos.copy(), self.cursor_pos.copy())
                self.lines.append(new_line)
                # SFX: Draw line sound

                # Update for next line
                self.draw_start_pos = self.cursor_pos.copy()
                self.cursor_pos += pygame.math.Vector2(50, 0)  # Default next ghost line

        self.last_space_held = space_held

    def _update_physics(self):
        # Apply gravity
        self.sled_vel += self.GRAVITY

        # Move sled
        self.sled_pos += self.sled_vel

        # Collision detection and response
        collided = False
        for p1, p2 in self.lines:
            line_vec = p2 - p1
            line_len_sq = line_vec.length_squared()

            if line_len_sq == 0: continue

            # Find closest point on line segment
            sled_to_p1 = self.sled_pos - p1
            t = sled_to_p1.dot(line_vec) / line_len_sq
            t = max(0, min(1, t))  # Clamp to segment

            closest_point = p1 + t * line_vec
            dist_vec = self.sled_pos - closest_point

            if dist_vec.length() < self.SLED_SIZE / 2:
                collided = True
                # SFX: Sled grind/impact sound

                # Resolve penetration
                penetration_depth = self.SLED_SIZE / 2 - dist_vec.length()
                if dist_vec.length() > 0:
                    self.sled_pos += dist_vec.normalize() * penetration_depth

                # Calculate response
                normal = dist_vec.normalize()

                # Friction
                friction_impulse = self.sled_vel.dot(normal) * normal
                self.sled_vel -= friction_impulse * 0.5  # Lose some normal velocity

                # Slide
                tangent = pygame.math.Vector2(-normal.y, normal.x)
                slide_component = self.sled_vel.dot(tangent) * tangent
                self.sled_vel = slide_component * 0.99  # Apply kinetic friction

                # Spawn particles on collision
                for _ in range(2):
                    self._create_particle(self.sled_pos, 2, 3)
                break

        # Update stuck timer
        if self.sled_vel.length() < 0.2:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0

        # Update trail for motion blur
        self.sled_trail.pop(0)
        self.sled_trail.append(self.sled_pos.copy())

    def _check_termination(self):
        # Win condition
        if self.sled_pos.x >= self.FINISH_LINE_X:
            self.game_over_message = "FINISH!"
            # SFX: Win fanfare
            return True, 50.0

        # Crash condition: off-screen
        if not (0 < self.sled_pos.y < self.HEIGHT and 0 < self.sled_pos.x < self.WIDTH):
            self.game_over_message = "CRASH!"
            # SFX: Crash/Explosion sound
            return True, -100.0

        # Crash condition: stuck
        if self.stuck_frames > self.FPS * 1.5:  # Stuck for 1.5 seconds
            self.game_over_message = "STUCK!"
            # SFX: Failure sound
            return True, -100.0

        # Timeout condition
        if self.time_remaining <= 0:
            self.game_over_message = "TIME'S UP!"
            # SFX: Timeout buzzer
            return True, -10.0

        return False, 0.0

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Start and Finish lines
        pygame.draw.line(self.screen, self.COLOR_START, (20, 0), (20, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_LINE_X, 0), (self.FINISH_LINE_X, self.HEIGHT), 3)

        # Drawn lines
        for p1, p2 in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_LINE, p1, p2, 2)

        # Ghost line and cursor
        if not self.game_over:
            self._draw_dashed_line(self.draw_start_pos, self.cursor_pos, self.COLOR_GHOST_LINE)
            pygame.gfxdraw.filled_circle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 5, self.COLOR_LINE)
            pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 5, self.COLOR_LINE)

        # Particles
        for p in self.particles:
            p_pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.draw.circle(self.screen, p['color'], p_pos, int(p['size']))

        # Sled motion trail
        for i, pos in enumerate(self.sled_trail):
            alpha = int(255 * (i / len(self.sled_trail)) * 0.5)
            trail_surf = pygame.Surface((self.SLED_SIZE, self.SLED_SIZE), pygame.SRCALPHA)
            trail_surf.fill((self.COLOR_SLED[0], self.COLOR_SLED[1], self.COLOR_SLED[2], alpha))
            self.screen.blit(trail_surf, (pos.x - self.SLED_SIZE / 2, pos.y - self.SLED_SIZE / 2))

        # Sled
        sled_rect = pygame.Rect(0, 0, self.SLED_SIZE, self.SLED_SIZE)
        sled_rect.center = (int(self.sled_pos.x), int(self.sled_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_SLED, sled_rect)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_str = f"{self.time_remaining / self.FPS:.1f}"
        time_color = self.COLOR_TEXT if self.time_remaining / self.FPS > 5 else self.COLOR_FINISH
        timer_text = self.font_small.render(time_str, True, time_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            end_text = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _draw_dashed_line(self, p1, p2, color, dash_length=5):
        line_vec = p2 - p1
        length = line_vec.length()
        if length == 0: return

        unit_vec = line_vec.normalize()

        current_pos = p1.copy()
        drawn_length = 0
        drawing = True
        while drawn_length < length:
            end_segment = current_pos + unit_vec * dash_length
            if (end_segment - p1).length() > length:
                end_segment = p2

            if drawing:
                pygame.draw.aaline(self.screen, color, current_pos, end_segment)

            current_pos = end_segment
            drawn_length = (current_pos - p1).length()
            drawing = not drawing

    def _create_particle(self, pos, count, speed_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(10, 20, endpoint=True),
                'size': self.np_random.uniform(1, 3),
                'color': self.COLOR_PARTICLE
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95  # Air resistance
            p['lifetime'] -= 1
            p['size'] -= 0.05
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['size'] > 0]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": round(self.time_remaining / self.FPS, 2),
            "sled_pos": (self.sled_pos.x, self.sled_pos.y),
            "lines_drawn": len(self.lines) - 1
        }

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to see the rendering.
    # For training, 'rgb_array' is used.
    render_mode = "human"

    if render_mode == "human":
        import sys
        # For human mode, we need a display
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.display.init()
        pygame.display.set_caption("Line Rider Gym Environment")
        human_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))

    env = GameEnv()
    seed = 123
    obs, info = env.reset(seed=seed)

    running = True
    total_reward = 0.0

    while running:
        action = np.array([0, 0, 0])  # Default no-op action

        if render_mode == "human":
            keys = pygame.key.get_pressed()

            # Combine movement keys (only one at a time)
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4

            # Button presses
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    total_reward = 0.0
                    obs, info = env.reset(seed=seed)
        else:  # For automated testing, use random actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render_mode == "human":
            # Blit the environment's screen to the display screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.FPS)

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            if render_mode == "human":
                # Wait a bit before resetting on termination
                pygame.time.wait(2000)
                total_reward = 0.0
                obs, info = env.reset(seed=seed)
            else:
                running = False  # End if not in human mode

    env.close()