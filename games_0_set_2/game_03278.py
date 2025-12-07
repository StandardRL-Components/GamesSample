
# Generated: 2025-08-27T22:54:18.297701
# Source Brief: brief_03278.md
# Brief Index: 3278

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a side-scrolling robot platformer.

    The player controls a procedurally animated robot that runs automatically.
    The goal is to jump over obstacles to reach the finish line.
    The episode ends upon collision, reaching the finish line, or running out of time.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]`: Movement (unused, as the robot runs automatically).
    - `action[1]`: Jump button (0=released, 1=pressed).
    - `action[2]`: High-jump modifier (0=released, 1=held). A high jump is performed if both jump and this modifier are active.

    **Observation Space:** `Box(0, 255, (400, 640, 3), uint8)`
    - An RGB image of the game screen.

    **Rewards:**
    - `+0.1` for each step survived.
    - `+5` for each successfully cleared obstacle.
    - `+100` for reaching the finish line.
    - `-100` for colliding with an obstacle.

    **Termination:**
    - The robot collides with an obstacle.
    - The robot reaches the finish line.
    - The maximum number of steps (2700, equivalent to 45s at 60fps) is reached.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Press Space to jump. Hold Shift while pressing Space for a high jump."
    )

    game_description = (
        "Guide a procedurally animated robot through a side-scrolling obstacle course to reach the finish line as quickly as possible."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 45 * self.FPS
        self.GROUND_Y = 350
        self.ROBOT_X_POS = 160
        self.GRAVITY = 0.5
        self.JUMP_POWER = -11
        self.HIGH_JUMP_POWER = -15
        self.INITIAL_WORLD_SPEED = 4.0
        self.FINISH_LINE_X = 18000

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_ROBOT = (0, 150, 255)
        self.COLOR_ROBOT_GLOW = (100, 200, 255)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (255, 150, 150)
        self.COLOR_FINISH = (50, 255, 50)
        self.COLOR_TEXT = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_large = pygame.font.Font(None, 64)
        self.font_small = pygame.font.Font(None, 32)

        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.won = False
        self.robot_pos = [0, 0]
        self.robot_vel_y = 0.0
        self.is_jumping = False
        self.world_scroll_x = 0.0
        self.world_speed = 0.0
        self.obstacles = []
        self.next_obstacle_spawn_x = 0.0
        self.particles = []
        self.run_cycle_angle = 0.0
        self.body_bob_offset = 0.0

        self.reset()
        # self.validate_implementation() # For development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.won = False

        self.robot_pos = [self.ROBOT_X_POS, self.GROUND_Y]
        self.robot_vel_y = 0.0
        self.is_jumping = False

        self.world_scroll_x = 0.0
        self.world_speed = self.INITIAL_WORLD_SPEED
        self.obstacles = []
        self.particles = []

        self.next_obstacle_spawn_x = self.WIDTH * 1.5
        self._generate_initial_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over or self.won:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.steps += 1

        self._update_world_scroll()
        self._handle_input(space_pressed, shift_held)
        self._update_robot_physics()

        reward += self._update_and_reward_obstacles()
        self._check_collisions()
        self._manage_obstacles()
        self._update_particles()

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward

        if not terminated:
            reward += 0.1
            self.score += 0.1

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_world_scroll(self):
        self.world_scroll_x += self.world_speed
        if self.steps > 0 and self.steps % 60 == 0: # Every second at 60 FPS
            self.world_speed += 0.1

    def _handle_input(self, space_pressed, shift_held):
        if space_pressed and not self.is_jumping:
            self.is_jumping = True
            self.robot_vel_y = self.HIGH_JUMP_POWER if shift_held else self.JUMP_POWER
            # sfx: jump

    def _update_robot_physics(self):
        if self.is_jumping:
            self.robot_pos[1] += self.robot_vel_y
            self.robot_vel_y += self.GRAVITY
            if self.robot_pos[1] >= self.GROUND_Y:
                self.robot_pos[1] = self.GROUND_Y
                self.robot_vel_y = 0
                self.is_jumping = False
                self._create_particles(self.robot_pos, 10, self.COLOR_ROBOT)
                # sfx: land

    def _update_and_reward_obstacles(self):
        reward = 0
        robot_rect = self._get_robot_rect()
        for obs in self.obstacles:
            obs_right_edge = obs['rect'].x - self.world_scroll_x + obs['rect'].width
            if not obs['cleared'] and robot_rect.x > obs_right_edge:
                obs['cleared'] = True
                self.score += 5
                reward += 5
        return reward

    def _check_collisions(self):
        robot_rect = self._get_robot_rect()
        for obs in self.obstacles:
            obs_screen_rect = obs['rect'].move(-self.world_scroll_x, 0)
            if robot_rect.colliderect(obs_screen_rect):
                self.game_over = True
                self._create_particles(robot_rect.center, 30, self.COLOR_OBSTACLE, is_explosion=True)
                # sfx: explosion
                break

    def _check_termination(self):
        if self.game_over:
            self.score -= 100
            return True, -100.0
        if self.world_scroll_x >= self.FINISH_LINE_X:
            self.won = True
            self.score += 100
            return True, 100.0
        if self.steps >= self.MAX_STEPS:
            return True, 0.0
        return False, 0.0

    def _generate_initial_obstacles(self):
        for _ in range(5):
            self._spawn_obstacle()

    def _spawn_obstacle(self):
        gap = self.np_random.integers(300, 600)
        x_pos = self.next_obstacle_spawn_x + gap
        width = self.np_random.integers(40, 80)
        height = self.np_random.integers(50, 150)
        y_pos = self.GROUND_Y - height + 1

        if self.np_random.random() < 0.25:
            y_pos -= self.np_random.integers(40, 100)

        new_obstacle = {'rect': pygame.Rect(x_pos, y_pos, width, height), 'cleared': False}
        self.obstacles.append(new_obstacle)
        self.next_obstacle_spawn_x = x_pos + width

    def _manage_obstacles(self):
        last_obs_x = self.obstacles[-1]['rect'].right if self.obstacles else 0
        if self.world_scroll_x + self.WIDTH > last_obs_x - self.WIDTH:
            self._spawn_obstacle()
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > self.world_scroll_x]

    def _get_robot_rect(self):
        return pygame.Rect(self.robot_pos[0] - 15, self.robot_pos[1] - 50, 30, 50)

    def _create_particles(self, pos, count, color, is_explosion=False):
        for _ in range(count):
            if is_explosion:
                vel = [self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5)]
                life = self.np_random.integers(20, 40)
                radius = self.np_random.uniform(3, 8)
            else: # Landing dust
                vel = [self.np_random.uniform(-1.5, 0.5), self.np_random.uniform(-1, 0)]
                life = self.np_random.integers(15, 30)
                radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': list(pos), 'vel': vel, 'radius': radius, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_finish_line()
        self._draw_obstacles()
        self._draw_particles()
        self._draw_robot()
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 2)

    def _draw_grid(self):
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)
        offset = (self.world_scroll_x * 0.5) % 40
        for i in range(0, self.WIDTH + 40, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i - offset, 0), (i - offset, self.HEIGHT), 1)

    def _draw_finish_line(self):
        finish_screen_x = self.FINISH_LINE_X - self.world_scroll_x
        if 0 < finish_screen_x < self.WIDTH:
            check_size = 20
            for y in range(0, self.HEIGHT, check_size):
                for x_offset in range(0, 40, check_size * 2):
                    color1 = self.COLOR_FINISH if (y // check_size) % 2 == 0 else (255, 255, 255)
                    color2 = (255, 255, 255) if (y // check_size) % 2 == 0 else self.COLOR_FINISH
                    pygame.draw.rect(self.screen, color1, (finish_screen_x + x_offset, y, check_size, check_size))
                    pygame.draw.rect(self.screen, color2, (finish_screen_x + x_offset + check_size, y, check_size, check_size))
            pygame.draw.line(self.screen, (255,255,255), (finish_screen_x, 0), (finish_screen_x, self.HEIGHT), 3)

    def _draw_obstacles(self):
        for obs in self.obstacles:
            obs_screen_rect = obs['rect'].move(-self.world_scroll_x, 0)
            if obs_screen_rect.right > 0 and obs_screen_rect.left < self.WIDTH:
                glow_rect = obs_screen_rect.inflate(10, 10)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, border_radius=8)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_screen_rect, border_radius=5)

    def _draw_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                alpha = int(255 * (p['life'] / p['max_life']))
                s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p['color'], alpha), (radius, radius), radius)
                self.screen.blit(s, (pos[0] - radius, pos[1] - radius))

    def _draw_robot(self):
        x, y = int(self.robot_pos[0]), int(self.robot_pos[1])
        if self.is_jumping:
            self.run_cycle_angle = 0
            self.body_bob_offset = 0
            leg_angle1, leg_angle2 = math.radians(45), math.radians(60)
            shin_angle1, shin_angle2 = math.radians(-60), math.radians(-70)
        else:
            self.run_cycle_angle += self.world_speed * 0.05
            self.body_bob_offset = math.sin(self.run_cycle_angle * 2) * 2
            leg_angle1 = math.sin(self.run_cycle_angle) * 0.6
            leg_angle2 = math.sin(self.run_cycle_angle + math.pi) * 0.6
            shin_angle1 = -abs(math.sin(self.run_cycle_angle) * 1.2) - 0.2
            shin_angle2 = -abs(math.sin(self.run_cycle_angle + math.pi) * 1.2) - 0.2

        body_y = y - 35 + self.body_bob_offset
        torso = pygame.Rect(x - 10, body_y, 20, 25)
        head_pos = (x, body_y - 10)

        glow_surf = pygame.Surface((80, 100), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_ROBOT_GLOW, 50), (40, 50), 40)
        self.screen.blit(glow_surf, (x - 40, y - 70))

        self._draw_limb(x, body_y + 20, 18, 15, leg_angle2, shin_angle2, 4)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, torso, border_radius=4)
        pygame.draw.circle(self.screen, self.COLOR_ROBOT, head_pos, 10)
        pygame.draw.circle(self.screen, (255, 255, 255), (head_pos[0]+3, head_pos[1]), 2)
        self._draw_limb(x, body_y + 20, 18, 15, leg_angle1, shin_angle1, 6)

    def _draw_limb(self, x, y, len1, len2, ang1, ang2, width):
        joint1_x = x + math.cos(ang1 + math.pi/2) * len1
        joint1_y = y + math.sin(ang1 + math.pi/2) * len1
        end_x = joint1_x + math.cos(ang1 + ang2 + math.pi/2) * len2
        end_y = joint1_y + math.sin(ang1 + ang2 + math.pi/2) * len2
        pygame.draw.line(self.screen, self.COLOR_ROBOT, (int(x), int(y)), (int(joint1_x), int(joint1_y)), width)
        pygame.draw.line(self.screen, self.COLOR_ROBOT, (int(joint1_x), int(joint1_y)), (int(end_x), int(end_y)), width - 1)

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        timer_surf = self.font_small.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (10, 10))

        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)

        if self.game_over:
            msg, color = "GAME OVER", self.COLOR_OBSTACLE
        elif self.won:
            msg, color = "FINISH!", self.COLOR_FINISH
        else:
            return

        msg_surf = self.font_large.render(msg, True, color)
        msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "won": self.won,
            "distance_traveled": self.world_scroll_x,
            "distance_to_goal": self.FINISH_LINE_X - self.world_scroll_x,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")