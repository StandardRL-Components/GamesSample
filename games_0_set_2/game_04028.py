import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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

    user_guide = (
        "Controls: Use arrows to set draw direction. Hold Space to draw a short line, or Shift to draw a long line. The line is drawn from the rider's current position."
    )

    game_description = (
        "A physics-based puzzle game. Draw lines in real-time to guide a sled rider from the start to the finish line, creating a path for them to follow."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (240, 240, 240)
    COLOR_TRACK = (20, 20, 20)
    COLOR_START = (220, 50, 50)
    COLOR_FINISH = (50, 220, 50)
    COLOR_RIDER_BODY = (60, 60, 200)
    COLOR_RIDER_SLED = (150, 100, 50)
    COLOR_TEXT = (10, 10, 10)

    # Physics
    GRAVITY = 0.15
    FRICTION = 0.005
    BOUNCE_RESTITUTION = 0.5
    RIDER_RADIUS = 6
    PHYSICS_SUBSTEPS = 5

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.start_pos = pygame.math.Vector2(80, 350)
        self.finish_pos = pygame.math.Vector2(self.SCREEN_WIDTH - 80, 350)

        self.rider_pos = pygame.math.Vector2(0, 0)
        self.rider_vel = pygame.math.Vector2(0, 0)
        self.rider_angle = 0.0
        self.track_lines = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Seeding for reproducibility
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.rider_pos = pygame.math.Vector2(self.start_pos)
        self.rider_vel = pygame.math.Vector2(1.0, 0)  # Initial push
        self.rider_angle = 0.0

        # Initial flat ground
        self.track_lines = [
            (pygame.math.Vector2(0, 380), pygame.math.Vector2(self.SCREEN_WIDTH, 380))
        ]

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._handle_action(movement, space_held, shift_held)
        self._update_physics()

        self.steps += 1
        terminated = self._check_termination()
        reward = self._calculate_reward(terminated)
        self.score += reward
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, movement, space_held, shift_held):
        length = 0
        if shift_held:
            length = 20
        elif space_held:
            length = 10

        if length > 0 and movement > 0:
            direction_map = {
                1: pygame.math.Vector2(0, -1),  # Up
                2: pygame.math.Vector2(0, 1),  # Down
                3: pygame.math.Vector2(-1, 0),  # Left
                4: pygame.math.Vector2(1, 0),  # Right
            }
            direction = direction_map.get(movement)
            if direction:
                start_point = pygame.math.Vector2(self.rider_pos)
                end_point = start_point + direction * length
                # Prevent drawing lines outside the screen
                end_point.x = max(0, min(self.SCREEN_WIDTH, end_point.x))
                end_point.y = max(0, min(self.SCREEN_HEIGHT, end_point.y))
                self.track_lines.append((start_point, end_point))

    def _update_physics(self):
        dt = 1.0 / self.PHYSICS_SUBSTEPS

        for _ in range(self.PHYSICS_SUBSTEPS):
            if self.game_over: break

            self.rider_vel.y += self.GRAVITY * dt

            # Apply friction
            if self.rider_vel.length() > 0:
                friction_force = self.rider_vel.normalize() * self.FRICTION
                if friction_force.length_squared() < self.rider_vel.length_squared():
                    self.rider_vel -= friction_force
                else:
                    self.rider_vel.update(0,0)


            potential_pos = self.rider_pos + self.rider_vel * dt

            for line_start, line_end in self.track_lines:
                closest_point, distance_sq, on_segment = self._get_closest_point_on_segment(potential_pos, line_start, line_end)

                if on_segment and distance_sq < self.RIDER_RADIUS ** 2:
                    # Collision response
                    collision_normal = (potential_pos - closest_point).normalize() if (potential_pos - closest_point).length() > 0 else pygame.math.Vector2(0, -1)

                    # Resolve penetration
                    penetration_depth = self.RIDER_RADIUS - math.sqrt(distance_sq)
                    self.rider_pos += collision_normal * penetration_depth
                    potential_pos = self.rider_pos # Update potential pos after resolving penetration

                    # Reflect velocity
                    velocity_component_normal = self.rider_vel.dot(collision_normal)
                    if velocity_component_normal < 0:  # Moving towards the surface
                        restitution_vec = - (1 + self.BOUNCE_RESTITUTION) * velocity_component_normal * collision_normal
                        self.rider_vel += restitution_vec

                    break  # Handle one collision per substep for stability

            self.rider_pos += self.rider_vel * dt

        # Update rider angle to align with velocity
        if self.rider_vel.length() > 0.1:
            self.rider_angle = self.rider_vel.angle_to(pygame.math.Vector2(1, 0))

    def _get_closest_point_on_segment(self, p, a, b):
        ap = p - a
        ab = b - a
        ab_len_sq = ab.length_squared()

        if ab_len_sq == 0:
            return a, (p - a).length_squared(), True

        t = ap.dot(ab) / ab_len_sq

        if t < 0.0:
            closest_point = a
            on_segment = False
        elif t > 1.0:
            closest_point = b
            on_segment = False
        else:
            closest_point = a + ab * t
            on_segment = True

        return closest_point, (p - closest_point).length_squared(), on_segment

    def _check_termination(self):
        if self.game_over:
            return True

        is_out_of_bounds = not (0 <= self.rider_pos.x <= self.SCREEN_WIDTH and 0 <= self.rider_pos.y <= self.SCREEN_HEIGHT)

        if is_out_of_bounds:
            self.game_over = True
            self.win = False
            self._create_crash_particles()
            return True

        if self.rider_pos.x >= self.finish_pos.x:
            self.game_over = True
            self.win = True
            return True

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            return True

        return False

    def _calculate_reward(self, terminated):
        reward = 0.0

        if terminated and not self.steps >= self.MAX_STEPS:
            if self.win:
                # Win bonus + time bonus
                reward += 50.0 + max(0, 10 * (self.MAX_STEPS - self.steps) / 100)
            else:
                # Crash penalty
                reward -= 20.0
        else:
            # Survival reward
            reward += 0.1

            # Penalty for distance from a straight line to the finish
            y_deviation = abs(self.rider_pos.y - self.finish_pos.y)
            reward -= 0.01 * (y_deviation / 100)

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw start and finish zones
        pygame.draw.circle(self.screen, self.COLOR_START, (int(self.start_pos.x), int(self.start_pos.y)), 10)
        pygame.draw.circle(self.screen, self.COLOR_FINISH, (int(self.finish_pos.x), int(self.finish_pos.y)), 10)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_pos.x, 0), (self.finish_pos.x, self.SCREEN_HEIGHT), 2)

        # Draw track lines
        for start, end in self.track_lines:
            pygame.draw.line(self.screen, self.COLOR_TRACK, (int(start.x), int(start.y)), (int(end.x), int(end.y)), 3)

        # Draw particles
        self._update_and_draw_particles()

        # Draw rider if not crashed out of bounds
        if not (self.game_over and not self.win):
            self._draw_rider()

    def _draw_rider(self):
        pos = self.rider_pos

        # Sled
        sled_p1 = pos + pygame.math.Vector2(-10, 5).rotate(-self.rider_angle)
        sled_p2 = pos + pygame.math.Vector2(10, 5).rotate(-self.rider_angle)
        pygame.draw.line(self.screen, self.COLOR_RIDER_SLED, sled_p1, sled_p2, 3)

        # Body
        body_start = pos + pygame.math.Vector2(0, 2).rotate(-self.rider_angle)
        body_end = pos + pygame.math.Vector2(0, -12).rotate(-self.rider_angle)
        pygame.draw.line(self.screen, self.COLOR_RIDER_BODY, body_start, body_end, 4)

        # Head
        head_pos = pos + pygame.math.Vector2(0, -15).rotate(-self.rider_angle)
        pygame.gfxdraw.filled_circle(self.screen, int(head_pos.x), int(head_pos.y), 5, self.COLOR_RIDER_BODY)
        pygame.gfxdraw.aacircle(self.screen, int(head_pos.x), int(head_pos.y), 5, self.COLOR_RIDER_BODY)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_main.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        text_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_text, text_rect)

        if self.game_over:
            status_text_str = "FINISH!" if self.win else "CRASHED"
            if self.steps >= self.MAX_STEPS and not self.win:
                status_text_str = "TIME UP"
            color = self.COLOR_FINISH if self.win else self.COLOR_START
            status_text = self.font_main.render(status_text_str, True, color)
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
        }

    def _create_crash_particles(self):
        # sound: "wood_snap.wav"
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = random.randint(20, 50)
            self.particles.append([pygame.math.Vector2(self.rider_pos), vel, lifetime, self.COLOR_RIDER_BODY])

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p[0] += p[1]  # pos += vel
            p[2] -= 1  # lifetime -= 1
            if p[2] <= 0:
                self.particles.remove(p)
            else:
                size = max(1, int(p[2] / 10))
                pygame.draw.circle(self.screen, p[3], (int(p[0].x), int(p[0].y)), size)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment
    # Make sure to unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()

    running = True
    game_over = False
    total_reward = 0.0

    # To control the game manually
    # Keys map to the MultiDiscrete action space
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right

    action = [0, 0, 0]

    # Use a separate pygame screen for human rendering
    pygame.display.init()
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Line Rider Gym Environment")
    clock = pygame.time.Clock()

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Press R to reset.")
    print("Press Q to quit.")
    print("----------------------")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    game_over = False
                    action = [0, 0, 0]  # Reset action on env reset
                if event.key == pygame.K_q:
                    running = False

        if not game_over:
            keys = pygame.key.get_pressed()

            # Update action based on key presses
            movement = 0
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space, shift]

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            game_over = terminated or truncated

        # Render the observation from the environment to the human-visible screen
        # Need to transpose it back to pygame's (width, height, channels) format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(60)  # Control the speed of manual play

    env.close()