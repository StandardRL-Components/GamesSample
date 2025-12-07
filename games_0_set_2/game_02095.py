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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.W, self.H = 640, 400
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_TRACK = (60, 60, 65)
        self.COLOR_OFFROAD = (40, 43, 47)
        self.COLOR_PLAYER = (43, 255, 156)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_CHECKPOINT = (255, 220, 0)
        self.COLOR_FINISH = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SMOKE = (150, 150, 150)
        self.COLOR_BOOST = (255, 150, 0)

        # Game parameters
        self.FPS = 30
        self.MAX_TIME = 30  # seconds
        self.MAX_STEPS = self.MAX_TIME * self.FPS

        # Player physics
        self.MAX_SPEED = 12.0
        self.ACCELERATION = 0.25
        self.BRAKE_FORCE = 0.5
        self.DRAG = 0.98  # Natural deceleration
        self.TURN_RATE = 0.05
        self.DRIFT_TURN_MULT = 2.0
        self.DRIFT_SLIP = 0.85

        # Track parameters
        self.TRACK_WIDTH = 100
        self.NUM_WAYPOINTS = 20
        self.WAYPOINT_MIN_DIST = 250
        self.WAYPOINT_MAX_DIST = 400

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.waypoints = []
        self.track_segments = []
        self.obstacles = []
        self.checkpoints = []
        self.particles = []
        self.next_checkpoint_idx = 0
        self.obstacle_density = 0.1

        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.game_won = False

        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_left = self.MAX_STEPS
        self.game_over = False
        self.game_won = False
        self.obstacle_density = 0.1

        self._generate_track()

        start_pos = self.waypoints[0]
        second_pos = self.waypoints[1]

        self.player_pos = np.array(start_pos, dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_angle = math.atan2(second_pos[1] - start_pos[1], second_pos[0] - start_pos[0])

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1
            shift_held = action[2] == 1

            # Update game logic
            self._update_player(movement, shift_held, space_held)
            self._update_particles()

            # Check game events
            reward += self._check_events()

            # Time penalty
            self.time_left -= 1

            # Continuous reward for survival
            reward += 0.1

        self.steps += 1

        # Check termination conditions
        terminated = self._check_termination()
        if terminated and not self.game_won:
            reward = -50.0  # Penalty for losing

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_track(self):
        self.waypoints = []
        self.track_segments = []
        self.obstacles = []
        self.checkpoints = []

        # Start position at the bottom center of the view
        current_pos = np.array([self.W / 2, self.H + self.WAYPOINT_MAX_DIST])
        current_angle = -math.pi / 2  # Start moving up

        for i in range(self.NUM_WAYPOINTS):
            self.waypoints.append(tuple(current_pos))
            turn = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            current_angle += turn
            current_angle = np.clip(current_angle, -math.pi * 0.8, -math.pi * 0.2)
            dist = self.np_random.uniform(self.WAYPOINT_MIN_DIST, self.WAYPOINT_MAX_DIST)
            current_pos += np.array([math.cos(current_angle), math.sin(current_angle)]) * dist

        for i in range(len(self.waypoints) - 1):
            p1 = np.array(self.waypoints[i])
            p2 = np.array(self.waypoints[i + 1])
            self.track_segments.append((p1, p2))

            # Place checkpoints every 4 segments
            if i > 0 and i % 4 == 0:
                self.checkpoints.append({"pos": p1, "active": True, "radius": self.TRACK_WIDTH / 2})

            # Place obstacles
            num_obstacles = self.np_random.integers(0, int(3 * self.obstacle_density) + 1)
            for _ in range(num_obstacles):
                # Place obstacles along the segment
                lerp_factor = self.np_random.uniform(0.1, 0.9)
                center_pos = p1 + (p2 - p1) * lerp_factor

                # Offset perpendicular to track direction
                norm_p2_p1 = np.linalg.norm(p2 - p1)
                if norm_p2_p1 > 1e-6:
                    track_dir = (p2 - p1) / norm_p2_p1
                    perp_dir = np.array([-track_dir[1], track_dir[0]])
                    offset = self.np_random.uniform(-self.TRACK_WIDTH * 0.8, self.TRACK_WIDTH * 0.8)
                    
                    obstacle_pos = center_pos + perp_dir * offset
                    obstacle_radius = self.np_random.uniform(10, 20)
                    self.obstacles.append({"pos": obstacle_pos, "radius": obstacle_radius})

        self.next_checkpoint_idx = 0

    def _update_player(self, movement, is_drifting, is_boosting):
        # --- Steering ---
        turn_input = 0
        if movement == 3:  # Left
            turn_input = -1
        elif movement == 4:  # Right
            turn_input = 1

        # --- Acceleration ---
        accel_input = 0
        if movement == 1:  # Up
            accel_input = 1
        elif movement == 2:  # Down
            accel_input = -1

        # --- Physics ---
        speed = np.linalg.norm(self.player_vel)

        # Apply acceleration/braking along the car's direction
        if accel_input > 0:
            force = np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * self.ACCELERATION
            self.player_vel += force
        elif accel_input < 0:
            brake_force = min(speed, self.BRAKE_FORCE)
            self.player_vel -= self.player_vel / max(speed, 1e-6) * brake_force

        # Apply drag
        self.player_vel *= self.DRAG

        # Limit max speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.MAX_SPEED

        # Update angle based on steering input and speed
        turn_multiplier = self.DRIFT_TURN_MULT if is_drifting else 1.0
        # More responsive steering at higher speeds
        speed_factor = min(1.0, speed / (self.MAX_SPEED * 0.5))
        self.player_angle += turn_input * self.TURN_RATE * turn_multiplier * speed_factor

        # Drift mechanics: make velocity vector lag behind car angle
        if is_drifting and speed > 2.0:
            target_vel_dir = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
            current_vel_dir = self.player_vel / max(speed, 1e-6)

            # Interpolate velocity direction towards car direction
            new_vel_dir = (current_vel_dir * self.DRIFT_SLIP + target_vel_dir * (1 - self.DRIFT_SLIP))
            new_vel_dir /= np.linalg.norm(new_vel_dir)
            self.player_vel = new_vel_dir * speed

            # Add drift particles
            if self.steps % 2 == 0:
                self._add_particle(self.player_pos, self.COLOR_SMOKE, 15, 20, -self.player_vel * 0.1)
        else:
            # If not drifting, align velocity with car angle
            self.player_vel = np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * speed

        # Boost/Fire visual effect
        if is_boosting:  # sfx: rocket_boost
            self._add_particle(self.player_pos, self.COLOR_BOOST, 10, 15, -self.player_vel * 0.5, 3)

        # Update position
        self.player_pos += self.player_vel

    def _add_particle(self, pos, color, life, size, vel_offset, size_decay=1):
        particle = {
            "pos": pos.copy() + self.np_random.uniform(-3, 3, 2),
            "vel": self.np_random.uniform(-0.5, 0.5, 2) + vel_offset,
            "life": life,
            "max_life": life,
            "size": size,
            "color": color,
            "size_decay": size_decay
        }
        self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] -= p["size_decay"]
        self.particles = [p for p in self.particles if p["life"] > 0 and p["size"] > 0]

    def _check_events(self):
        reward = 0

        # Obstacle collision
        player_circle = (self.player_pos, 10)  # Approximate player as a circle
        for obs in self.obstacles:
            dist = np.linalg.norm(player_circle[0] - obs["pos"])
            if dist < player_circle[1] + obs["radius"]:
                self.game_over = True  # sfx: crash
                self._add_particle(self.player_pos, self.COLOR_OBSTACLE, 30, 30, np.array([0, 0]), 0.5)
                self._add_particle(self.player_pos, (255, 255, 255), 30, 20, np.array([0, 0]), 1)
                return 0  # Final reward handled in step()

        # Checkpoint collision
        if self.next_checkpoint_idx < len(self.checkpoints):
            checkpoint = self.checkpoints[self.next_checkpoint_idx]
            dist = np.linalg.norm(self.player_pos - checkpoint["pos"])
            if dist < checkpoint["radius"]:
                reward += 1.0  # sfx: checkpoint_ding
                self.checkpoints[self.next_checkpoint_idx]["active"] = False
                self.next_checkpoint_idx += 1
                self.obstacle_density += 0.05  # Increase difficulty
                self.time_left += 5 * self.FPS  # Add 5 seconds for reaching checkpoint

        # Finish line
        dist_to_finish = np.linalg.norm(self.player_pos - np.array(self.waypoints[-1]))
        if dist_to_finish < self.TRACK_WIDTH:
            self.game_over = True
            self.game_won = True
            reward += 50.0  # sfx: win_fanfare

        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.time_left <= 0:
            self.game_over = True  # sfx: time_out_buzzer
            return True
        if self.steps >= self.MAX_STEPS * 2:  # Failsafe
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_OFFROAD)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Camera transform
        cam_x = self.W / 2 - self.player_pos[0]
        cam_y = self.H / 2 - self.player_pos[1]

        # Render track
        for p1, p2 in self.track_segments:
            direction_norm = np.linalg.norm(p2 - p1)
            if direction_norm < 1e-6: continue
            direction = (p2 - p1) / direction_norm
            perp = np.array([-direction[1], direction[0]]) * self.TRACK_WIDTH / 2
            p1_left = p1 - perp + np.array([cam_x, cam_y])
            p1_right = p1 + perp + np.array([cam_x, cam_y])
            p2_left = p2 - perp + np.array([cam_x, cam_y])
            p2_right = p2 + perp + np.array([cam_x, cam_y])
            pygame.draw.polygon(self.screen, self.COLOR_TRACK, [p1_left, p2_left, p2_right, p1_right])

        # Render checkpoints
        for cp in self.checkpoints:
            if cp["active"]:
                pos = (int(cp["pos"][0] + cam_x), int(cp["pos"][1] + cam_y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(cp["radius"]), (*self.COLOR_CHECKPOINT, 80))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(cp["radius"]), self.COLOR_CHECKPOINT)

        # Render finish line (checkered)
        if len(self.waypoints) >= 2:
            finish_pos = np.array(self.waypoints[-1])
            finish_dir_vec = finish_pos - np.array(self.waypoints[-2])
            norm_finish_dir = np.linalg.norm(finish_dir_vec)
            if norm_finish_dir > 1e-6:
                finish_dir = finish_dir_vec / norm_finish_dir
                perp = np.array([-finish_dir[1], finish_dir[0]])
                for i in range(-5, 5):
                    p1 = finish_pos + perp * i * (self.TRACK_WIDTH / 10)
                    p2 = p1 + finish_dir * 20
                    color = self.COLOR_FINISH if i % 2 == 0 else (0, 0, 0)
                    pygame.draw.line(self.screen, color, p1 + np.array([cam_x, cam_y]),
                                     p2 + np.array([cam_x, cam_y]), int(self.TRACK_WIDTH / 10) + 1)

        # Render obstacles
        for obs in self.obstacles:
            pos = (int(obs["pos"][0] + cam_x), int(obs["pos"][1] + cam_y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(obs["radius"]), self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(obs["radius"]), (255, 255, 255))

        # Render particles
        for p in self.particles:
            pos = (int(p["pos"][0] + cam_x), int(p["pos"][1] + cam_y))
            alpha = int(200 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["size"]), color)

        # Render player
        self._render_player()

    def _render_player(self):
        car_center = (self.W // 2, self.H // 2)
        car_size = np.array([30, 15])

        points = np.array([
            [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]
        ]) * car_size

        rotation = np.array([
            [math.cos(self.player_angle), -math.sin(self.player_angle)],
            [math.sin(self.player_angle), math.cos(self.player_angle)]
        ])

        rotated_points = points @ rotation.T + car_center

        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)

        # Windshield
        windshield_points = np.array([
            [0.1, -0.4], [0.4, -0.4], [0.3, 0.4], [0.1, 0.4]
        ]) * car_size
        rotated_windshield = windshield_points @ rotation.T + car_center
        pygame.gfxdraw.filled_polygon(self.screen, rotated_windshield, (20, 20, 20, 150))

    def _render_ui(self):
        # Time
        time_text = f"TIME: {max(0, self.time_left / self.FPS):.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Speed
        speed = np.linalg.norm(self.player_vel)
        speed_text = f"SPEED: {int(speed * 10):03d}"
        speed_surf = self.font_ui.render(speed_text, True, self.COLOR_TEXT)
        self.screen.blit(speed_surf, (self.W - speed_surf.get_width() - 10, 10))

        # Checkpoint progress
        progress_text = f"CHECKPOINT: {self.next_checkpoint_idx}/{len(self.checkpoints)}"
        progress_surf = self.font_ui.render(progress_text, True, self.COLOR_TEXT)
        self.screen.blit(progress_surf, (self.W // 2 - progress_surf.get_width() // 2, 10))

        # Game over message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_OBSTACLE
            msg_surf = self.font_msg.render(msg, True, color)
            self.screen.blit(msg_surf, (self.W // 2 - msg_surf.get_width() // 2, self.H // 2 - msg_surf.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left / self.FPS,
            "checkpoints_passed": self.next_checkpoint_idx
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    # Pygame setup for human play
    pygame.display.init()
    pygame.font.init()

    env = GameEnv()
    obs, info = env.reset()
    done = False

    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Arcade Racer")

    total_reward = 0

    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()

        movement = 0  # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # If both left and right are pressed, prioritize one
        if keys[pygame.K_LEFT] and keys[pygame.K_RIGHT]:
            movement = 0

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        done = terminated or truncated

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()