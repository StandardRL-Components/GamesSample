import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control two pendulums swinging from a central pivot. Manipulate their swing and change one "
        "pendulum's length to guide both bobs into the goal zone while avoiding obstacles."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to swing the blue pendulum and ←→ to swing the yellow one. "
        "Hold space to shorten the blue pendulum and shift to lengthen it."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG_TOP = (10, 15, 30)
    COLOR_BG_BOTTOM = (30, 25, 50)
    COLOR_P1 = (50, 150, 255)
    COLOR_P2 = (255, 200, 50)
    COLOR_OBSTACLE = (255, 80, 80)
    COLOR_GOAL = (80, 255, 80)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (255, 255, 255)

    # Physics
    GRAVITY = 0.4
    DAMPING = 0.998
    FORCE_MAGNITUDE = 0.0015

    # Pendulum 1 (Transformable)
    P1_LEN_MIN = 80
    P1_LEN_MAX = 200
    P1_LEN_CHANGE_SPEED = 0.1

    # Pendulum 2 (Static)
    P2_LEN = 140

    # Game
    MAX_STEPS = 1800  # 60 seconds * 30 FPS
    NUM_OBSTACLES = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)

        self.pivot_pos = (self.SCREEN_WIDTH // 2, 50)
        self.goal_rect = pygame.Rect(self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT - 50, self.SCREEN_WIDTH * 0.5, 40)

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.goal_bonus_awarded = False
        self.p1_angle = 0
        self.p1_angle_vel = 0
        self.p1_length = 0
        self.p1_target_length = 0
        self.p2_angle = 0
        self.p2_angle_vel = 0
        self.obstacles = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_left = self.MAX_STEPS
        self.game_over = False
        self.goal_bonus_awarded = False

        # Pendulum 1 State
        self.p1_angle = math.pi + self.np_random.uniform(-0.5, 0.5)
        self.p1_angle_vel = 0
        self.p1_length = self.P1_LEN_MAX
        self.p1_target_length = self.P1_LEN_MAX

        # Pendulum 2 State
        self.p2_angle = math.pi + self.np_random.uniform(-0.5, 0.5)
        self.p2_angle_vel = 0

        # Particles
        self.particles = []

        # Procedurally generate obstacles
        self._generate_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        reward = 0.1  # Survival reward
        self.steps += 1
        self.time_left -= 1

        # --- 1. Handle Player Input ---
        force_p1 = 0
        force_p2 = 0

        # Action 1/2: Force on P1
        if movement == 1:  # Up (swing higher)
            force_p1 = -self.FORCE_MAGNITUDE
        elif movement == 2:  # Down (swing lower)
            force_p1 = self.FORCE_MAGNITUDE

        # Action 3/4: Force on P2
        if movement == 3:  # Left (P2 swing higher)
            force_p2 = -self.FORCE_MAGNITUDE
        elif movement == 4:  # Right (P2 swing lower)
            force_p2 = self.FORCE_MAGNITUDE

        # Action 5/6 (Space/Shift): Transform P1 length
        is_transforming = False
        if space_held:
            self.p1_target_length = self.P1_LEN_MIN
            is_transforming = True
        elif shift_held:
            self.p1_target_length = self.P1_LEN_MAX
            is_transforming = True

        # --- 2. Update Game State & Physics ---

        # Interpolate P1 length
        len_diff = self.p1_target_length - self.p1_length
        if abs(len_diff) > 0.1:
            self.p1_length += len_diff * self.P1_LEN_CHANGE_SPEED
            # Transformation particles
            if is_transforming and self.steps % 2 == 0:
                self._create_particles(self._get_bob_pos(self.p1_angle, self.p1_length), 3)

        # Pendulum 1 Physics
        p1_accel = (self.GRAVITY / self.p1_length) * math.sin(self.p1_angle) + force_p1
        self.p1_angle_vel += p1_accel
        self.p1_angle_vel *= self.DAMPING
        self.p1_angle += self.p1_angle_vel

        # Pendulum 2 Physics
        p2_accel = (self.GRAVITY / self.P2_LEN) * math.sin(self.p2_angle) + force_p2
        self.p2_angle_vel += p2_accel
        self.p2_angle_vel *= self.DAMPING
        self.p2_angle += self.p2_angle_vel

        # Update particles
        self._update_particles()

        # --- 3. Check Game Conditions ---

        p1_bob_pos = self._get_bob_pos(self.p1_angle, self.p1_length)
        p2_bob_pos = self._get_bob_pos(self.p2_angle, self.P2_LEN)

        # Collision Check
        for obs in self.obstacles:
            if obs.collidepoint(p1_bob_pos) or obs.collidepoint(p2_bob_pos):
                reward = -100
                self.game_over = True
                break

        # Win/Goal Check
        if not self.game_over:
            p1_in_goal = self.goal_rect.collidepoint(p1_bob_pos)
            p2_in_goal = self.goal_rect.collidepoint(p2_bob_pos)

            if p1_in_goal and p2_in_goal:
                if not self.goal_bonus_awarded:
                    reward += 5
                    self.goal_bonus_awarded = True

                if self.time_left > 0:
                    reward += 100
                    self.game_over = True

        # Time Out Check
        if self.time_left <= 0 and not self.game_over:
            self.game_over = True

        self.score += reward
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "p1_length": self.p1_length,
        }

    def _generate_obstacles(self):
        self.obstacles = []
        spawn_area = pygame.Rect(20, self.pivot_pos[1] + 40, self.SCREEN_WIDTH - 40,
                                 self.goal_rect.top - self.pivot_pos[1] - 60)

        for _ in range(self.NUM_OBSTACLES):
            for _ in range(100):  # Max 100 attempts to place an obstacle
                w = self.np_random.integers(40, 100)
                h = self.np_random.integers(15, 30)
                x = self.np_random.integers(spawn_area.left, spawn_area.right - w)
                y = self.np_random.integers(spawn_area.top, spawn_area.bottom - h)
                new_obs = pygame.Rect(x, y, w, h)

                # Check for overlap with other obstacles
                if not any(new_obs.colliderect(obs) for obs in self.obstacles):
                    self.obstacles.append(new_obs)
                    break

    def _get_bob_pos(self, angle, length):
        x = self.pivot_pos[0] + length * math.sin(angle)
        y = self.pivot_pos[1] + length * math.cos(angle)
        return (int(x), int(y))

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Goal Area
        goal_surface = pygame.Surface((self.goal_rect.width, self.goal_rect.height), pygame.SRCALPHA)
        goal_surface.fill((*self.COLOR_GOAL, 50))
        self.screen.blit(goal_surface, self.goal_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal_rect, 2)

        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs, 2)

        # Pendulums
        p1_bob_pos = self._get_bob_pos(self.p1_angle, self.p1_length)
        p2_bob_pos = self._get_bob_pos(self.p2_angle, self.P2_LEN)

        # Pendulum 2 (draw first to be in background)
        pygame.draw.aaline(self.screen, self.COLOR_P2, self.pivot_pos, p2_bob_pos, 1)
        self._draw_glowing_circle(self.screen, p2_bob_pos, 10, self.COLOR_P2)

        # Pendulum 1
        pygame.draw.aaline(self.screen, self.COLOR_P1, self.pivot_pos, p1_bob_pos, 1)
        self._draw_glowing_circle(self.screen, p1_bob_pos, 12, self.COLOR_P1)

        # Pivot
        pygame.gfxdraw.filled_circle(self.screen, self.pivot_pos[0], self.pivot_pos[1], 5, (200, 200, 220))
        pygame.gfxdraw.aacircle(self.screen, self.pivot_pos[0], self.pivot_pos[1], 5, (200, 200, 220))

        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), p['color'])

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {self.time_left / self.FPS:.1f}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 15, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Pendulum Lengths display
        p1_len_text = f"P1 Len: {int(self.p1_length)}"
        p2_len_text = f"P2 Len: {int(self.P2_LEN)}"
        p1_surf = self.font_small.render(p1_len_text, True, self.COLOR_P1)
        p2_surf = self.font_small.render(p2_len_text, True, self.COLOR_P2)
        self.screen.blit(p1_surf, (self.SCREEN_WIDTH - p1_surf.get_width() - 15, 40))
        self.screen.blit(p2_surf, (self.SCREEN_WIDTH - p2_surf.get_width() - 15, 58))

    def _draw_glowing_circle(self, surface, pos, radius, color):
        pos = (int(pos[0]), int(pos[1]))
        for i in range(4):
            alpha = 150 - i * 35
            rad = radius + i * 2
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], rad, (*color, alpha // 4))

        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': random.uniform(2, 4),
                'color': (*self.COLOR_PARTICLE, 200),
                'life': random.randint(10, 20)
            })

    def _update_particles(self):
        # Use a list comprehension to filter out dead particles, which is safer than removing while iterating
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] -= 0.1

    def close(self):
        pygame.quit()


# Example usage:
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset(seed=42)

    terminated = False
    total_reward = 0
    step_count = 0

    print(f"Game: Pendulum Gauntlet")
    print(f"Description: {GameEnv.game_description}")
    print(f"Controls: {GameEnv.user_guide}")
    print("-" * 30)

    while step_count < env.MAX_STEPS:
        action = env.action_space.sample()  # Sample a random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if (step_count % 100) == 0:
            print(f"Step: {step_count}, Reward: {reward:.2f}, Info: {info}")

        if terminated or truncated:
            print(f"\nEpisode finished after {step_count} steps.")
            print(f"Final Score: {info['score']:.2f}")
            break

    if not (terminated or truncated):
        print(f"\nEpisode timed out after {step_count} steps.")
        print(f"Final Score: {info['score']:.2f}")

    env.close()